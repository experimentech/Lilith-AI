"""
FastAPI-based shim exposing MCP-style endpoints for Lilith plus a Xiaozhi-compatible WebSocket.

HTTP endpoints keep the core pipeline untouched and delegate to MCPAdapter. The
WebSocket endpoint (/xiaozhi/ws) speaks the minimal Xiaozhi hello/listen/abort/mcp
envelope and routes mcp payloads to the same adapter. Audio frames are accepted
but ignored unless a future audio bridge is enabled.
"""
import json
import os
import uuid
from typing import Optional

import anyio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from lilith.mcp_adapter import MCPAdapter, WeatherReport
from lilith.speech_bridge import NullSpeechBridge, SpeechBridge
from lilith.xiaozhi_tools import XiaozhiToolRegistry


class ClientContext(BaseModel):
    client_id: str
    context_id: Optional[str] = None


class ChatRequest(ClientContext):
    message: str


class TeachRequest(ClientContext):
    trigger: str
    response: str
    intent: str = "general"


class FeedbackRequest(ClientContext):
    pattern_id: str
    strength: float = 1.0


class WeatherRequest(ClientContext):
    location: str


class NewsRequest(ClientContext):
    topic: str


class ResetRequest(ClientContext):
    keep_backup: bool = True
    bootstrap: bool = False


adapter = MCPAdapter()
app = FastAPI(title="Lilith MCP Adapter", version="0.1")
speech_bridge: SpeechBridge = NullSpeechBridge()
xiaozhi_registry = XiaozhiToolRegistry(adapter)
STRIP_MCP_METADATA = os.getenv("LILITH_MCP_MINIMAL", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
    "minimal",
    "strip",
    "text-only",
}


def _get_auth_token() -> Optional[str]:
    """Fetch the optional Xiaozhi auth token from env each time to honor test overrides."""

    return os.getenv("XIAOZHI_AUTH_TOKEN")


def _is_audio_enabled() -> bool:
    flag = os.getenv("XIAOZHI_AUDIO_ENABLED", "0").lower()
    return flag in {"1", "true", "yes", "on"}


def _build_mcp_payload(resp):
    """Build the MCP payload, optionally omitting metadata when minimal mode is on."""

    payload = {"text": resp.text}
    if not STRIP_MCP_METADATA:
        payload.update(
            {
                "pattern_id": resp.pattern_id,
                "confidence": resp.confidence,
                "is_fallback": resp.is_fallback,
                "is_low_confidence": resp.is_low_confidence,
                "source": resp.source,
            }
        )
    return payload


def _auth_ok(headers) -> bool:
    expected = _get_auth_token()
    if not expected:
        return True
    supplied = headers.get("authorization") or headers.get("Authorization")
    if not supplied or not supplied.startswith("Bearer "):
        return False
    return supplied.split(" ", 1)[1] == expected


def set_speech_bridge(bridge: SpeechBridge) -> None:
    """Allow callers to replace the speech bridge (e.g., in tests or deployments)."""

    global speech_bridge
    speech_bridge = bridge


def _jsonrpc_error(code: int, message: str, rpc_id):
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}


def _jsonrpc_result(rpc_id, result: dict):
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


async def _handle_mcp_jsonrpc(payload: dict, client_id: str, context_id: Optional[str]):
    if payload.get("jsonrpc") != "2.0":
        return _jsonrpc_error(-32600, "Invalid Request", payload.get("id"))

    method = payload.get("method")
    params = payload.get("params") or {}
    rpc_id = payload.get("id")

    if method == "initialize":
        return _jsonrpc_result(
            rpc_id,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "Lilith", "version": "0.1"},
            },
        )

    if method == "tools/list":
        tools = xiaozhi_registry.list_tools()
        return _jsonrpc_result(
            rpc_id,
            {
                "tools": tools,
                "nextCursor": "",
            },
        )

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not name:
            return _jsonrpc_error(-32602, "Missing tool name", rpc_id)
        try:
            text, is_error = await anyio.to_thread.run_sync(
                xiaozhi_registry.call, name, arguments, client_id, context_id
            )
            return _jsonrpc_result(
                rpc_id,
                {
                    "content": [{"type": "text", "text": text}],
                    "isError": bool(is_error),
                },
            )
        except KeyError:
            return _jsonrpc_error(-32601, f"Unknown tool: {name}", rpc_id)
        except Exception as exc:  # pylint: disable=broad-except
            return _jsonrpc_error(-32000, f"Tool error: {exc}", rpc_id)

    return _jsonrpc_error(-32601, f"Unknown method: {method}", rpc_id)


@app.post("/chat")
async def chat(req: ChatRequest):
    resp = await anyio.to_thread.run_sync(adapter.handle_chat, req.client_id, req.message, req.context_id)
    return {
        "text": resp.text,
        "pattern_id": resp.pattern_id,
        "confidence": resp.confidence,
        "is_fallback": resp.is_fallback,
        "is_low_confidence": resp.is_low_confidence,
        "source": resp.source,
        "learned_fact": resp.learned_fact,
    }


@app.post("/teach")
async def teach(req: TeachRequest):
    pattern_id = await anyio.to_thread.run_sync(
        adapter.handle_teach,
        req.client_id,
        req.trigger,
        req.response,
        req.intent,
        req.context_id,
    )
    return {"pattern_id": pattern_id}


@app.post("/feedback/upvote")
async def upvote(req: FeedbackRequest):
    await anyio.to_thread.run_sync(adapter.handle_upvote, req.client_id, req.pattern_id, req.strength, req.context_id)
    return {"status": "ok"}


@app.post("/feedback/downvote")
async def downvote(req: FeedbackRequest):
    await anyio.to_thread.run_sync(adapter.handle_downvote, req.client_id, req.pattern_id, req.strength, req.context_id)
    return {"status": "ok"}


@app.post("/stats")
async def stats(req: ClientContext):
    return await anyio.to_thread.run_sync(adapter.handle_stats, req.client_id, req.context_id, True)


@app.post("/weather")
async def weather(req: WeatherRequest):
    report: WeatherReport = await anyio.to_thread.run_sync(adapter.handle_weather, req.client_id, req.location, req.context_id)
    return {
        "location": report.location,
        "summary": report.summary,
        "temperature_c": report.temperature_c,
    }


@app.post("/news")
async def news(req: NewsRequest):
    headline = await anyio.to_thread.run_sync(adapter.handle_news, req.client_id, req.topic, req.context_id)
    return {"headline": headline}


@app.post("/reset")
async def reset(req: ResetRequest):
    backup = await anyio.to_thread.run_sync(
        adapter.handle_reset_user,
        req.client_id,
        req.keep_backup,
        req.bootstrap,
        req.context_id,
    )
    if backup is None:
        raise HTTPException(status_code=400, detail="Reset not available for this caller")
    return {"backup_path": backup}


@app.websocket("/xiaozhi/ws")
async def xiaozhi_ws(websocket: WebSocket):
    if not _auth_ok(websocket.headers):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    session_id = str(uuid.uuid4())
    client_id = (
        websocket.headers.get("client-id")
        or websocket.headers.get("Client-Id")
        or "xiaozhi-client"
    )
    context_id = websocket.headers.get("context-id") or None

    await websocket.send_json(
        {
            "type": "hello",
            "transport": "websocket",
            "version": 1,
            "session_id": session_id,
            "features": {"mcp": True},
        }
    )

    async def send_ack(message_type: str, state: Optional[str] = None, ok: bool = True, error: Optional[str] = None):
        payload = {
            "type": message_type,
            "session_id": session_id,
            "ok": ok,
        }
        if state:
            payload["state"] = state
        if error:
            payload["error"] = error
        await websocket.send_json(payload)

    try:
        while True:
            msg = await websocket.receive()
            msg_type = msg.get("type")

            if msg_type == "websocket.disconnect":
                break

            text_data = msg.get("text")
            bytes_data = msg.get("bytes")

            if text_data is not None:
                try:
                    data = json.loads(text_data)
                except Exception:
                    await send_ack("error", ok=False, error="invalid_json")
                    continue

                kind = data.get("type")
                if kind == "listen":
                    await send_ack("listen", state=data.get("state"))
                elif kind == "abort":
                    await send_ack("abort", state=data.get("reason"))
                elif kind == "mcp":
                    payload = data.get("payload") or {}
                    if payload.get("jsonrpc"):
                        rpc_resp = await _handle_mcp_jsonrpc(payload, client_id, context_id)
                        await websocket.send_json(
                            {
                                "type": "mcp",
                                "session_id": session_id,
                                "payload": rpc_resp,
                            }
                        )
                    else:
                        message = payload.get("message") or payload.get("text") or ""
                        try:
                            resp = await anyio.to_thread.run_sync(
                                adapter.handle_chat, client_id, message, context_id
                            )
                            await websocket.send_json(
                                {
                                    "type": "mcp",
                                    "session_id": session_id,
                                    "payload": _build_mcp_payload(resp),
                                }
                            )
                        except Exception:
                            await send_ack("mcp", ok=False, error="chat_failed")
                elif kind == "iot":
                    await send_ack("iot")
                else:
                    await send_ack("error", ok=False, error="unsupported_type")

            elif bytes_data is not None:
                if not _is_audio_enabled():
                    await send_ack("audio", ok=False, error="audio_disabled")
                    continue

                if isinstance(speech_bridge, NullSpeechBridge):
                    await send_ack("audio", ok=False, error="audio_not_wired")
                    continue

                # Attempt ASR -> chat -> optional TTS
                try:
                    transcript = await anyio.to_thread.run_sync(
                        speech_bridge.transcribe, bytes_data, session_id, client_id
                    )
                except Exception:
                    await send_ack("audio", ok=False, error="asr_failed")
                    continue

                if not transcript:
                    await send_ack("audio", ok=False, error="no_transcript")
                    continue

                try:
                    resp = await anyio.to_thread.run_sync(
                        adapter.handle_chat, client_id, transcript, context_id
                    )
                    await websocket.send_json(
                        {
                            "type": "mcp",
                            "session_id": session_id,
                            "payload": _build_mcp_payload(resp),
                        }
                    )
                except Exception:
                    await send_ack("mcp", ok=False, error="chat_failed")
                    continue

                if speech_bridge.supports_tts():
                    try:
                        audio_out = await anyio.to_thread.run_sync(
                            speech_bridge.synthesize, resp.text, session_id, client_id
                        )
                        if audio_out:
                            await websocket.send_bytes(audio_out)
                    except Exception:
                        # TTS failure should not fail the session; ignore.
                        pass

            # Ignore other frame types

    except WebSocketDisconnect:
        pass
    finally:
        try:
            if (
                websocket.application_state == WebSocketState.CONNECTED
                and websocket.client_state == WebSocketState.CONNECTED
            ):
                await websocket.close()
        except RuntimeError:
            # Ignore double-close races from the server side.
            pass
