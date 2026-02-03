"""
V2 Xiaozhi Server - MCP-compatible WebSocket endpoint for Lilith V2.

This is the V2 equivalent of lilith/mcp_server.py, using CognitiveStage
instead of LilithSession. The WebSocket protocol is identical to V1 for
compatibility with xiaozhi-esp32 and py-xiaozhi clients.

Usage:
    uvicorn v2.xiaozhi_server:app --reload --port 8001
    
Or programmatically:
    from v2.xiaozhi_server import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

Environment variables:
    XIAOZHI_AUTH_TOKEN - Optional bearer token for auth
    XIAOZHI_AUDIO_ENABLED - Enable audio processing (default: off)
    LILITH_MCP_MINIMAL - Strip metadata from MCP responses
"""
import json
import logging
import os
import uuid
from typing import Any, Dict, Optional

import anyio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from v2.mcp_adapter import V2MCPAdapter, MCPResponse, WeatherReport

logger = logging.getLogger(__name__)


# ===== Request Models =====

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


# ===== Lazy Global Adapter =====
# Adapter is created on first use, not at import time (avoids slow torch import)

_adapter: Optional[V2MCPAdapter] = None
_adapter_lock = None

def get_adapter() -> V2MCPAdapter:
    """Get or create the shared adapter (lazy initialization)."""
    global _adapter, _adapter_lock
    import threading
    if _adapter_lock is None:
        _adapter_lock = threading.Lock()
    with _adapter_lock:
        if _adapter is None:
            _adapter = V2MCPAdapter()
        return _adapter

app = FastAPI(title="Lilith V2 Xiaozhi Server", version="0.2")

STRIP_MCP_METADATA = os.getenv("LILITH_MCP_MINIMAL", "").lower() in {
    "1", "true", "yes", "on", "minimal", "strip", "text-only"
}


# ===== Tool Registry (Xiaozhi MCP tools) =====

class XiaozhiToolRegistry:
    """Tool registry for Xiaozhi MCP JSON-RPC calls."""
    
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._install_defaults()
    
    @property
    def adapter(self) -> V2MCPAdapter:
        return get_adapter()
    
    def _install_defaults(self) -> None:
        """Register default Xiaozhi tools."""
        self.register(
            name="self.get_device_status",
            description="Get device status",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda params, cid, ctx: ("ok", False),
        )
        self.register(
            name="self.audio_speaker.set_volume",
            description="Set speaker volume",
            input_schema={
                "type": "object",
                "properties": {"volume": {"type": "integer", "minimum": 0, "maximum": 100}},
                "required": ["volume"],
            },
            handler=lambda params, cid, ctx: ("true", False),
        )
        self.register(
            name="self.audio_speaker.get_volume",
            description="Get speaker volume",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda params, cid, ctx: ("50", False),
        )
        self.register(
            name="chat.reply",
            description="Chat via Lilith V2",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=self._handle_chat,
        )
    
    def register(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler,
    ) -> None:
        """Register a tool."""
        self._tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "handler": handler,
        }
    
    def list_tools(self) -> list:
        """Return list of available tools."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["inputSchema"],
            }
            for t in self._tools.values()
        ]
    
    def call(
        self,
        name: str,
        arguments: Dict[str, Any],
        client_id: str,
        context_id: Optional[str]
    ) -> tuple:
        """Call a tool by name. Returns (text, is_error)."""
        tool = self._tools.get(name)
        if not tool:
            raise KeyError(name)
        return tool["handler"](arguments or {}, client_id, context_id)
    
    def _handle_chat(
        self,
        params: Dict[str, Any],
        client_id: str,
        context_id: Optional[str]
    ) -> tuple:
        """Handle chat.reply tool call."""
        text = params.get("text") or ""
        resp = self.adapter.handle_chat(client_id, text, context_id)
        return resp.text, False


xiaozhi_registry = XiaozhiToolRegistry()


# ===== Helper Functions =====

def _get_auth_token() -> Optional[str]:
    """Get optional auth token from env."""
    return os.getenv("XIAOZHI_AUTH_TOKEN")


def _is_audio_enabled() -> bool:
    """Check if audio processing is enabled."""
    flag = os.getenv("XIAOZHI_AUDIO_ENABLED", "0").lower()
    return flag in {"1", "true", "yes", "on"}


def _auth_ok(headers) -> bool:
    """Validate authorization header."""
    expected = _get_auth_token()
    if not expected:
        return True
    supplied = headers.get("authorization") or headers.get("Authorization")
    if not supplied or not supplied.startswith("Bearer "):
        return False
    return supplied.split(" ", 1)[1] == expected


def _build_mcp_payload(resp: MCPResponse) -> Dict[str, Any]:
    """Build MCP response payload."""
    payload = {"text": resp.text}
    if not STRIP_MCP_METADATA:
        payload.update({
            "pattern_id": resp.pattern_id,
            "confidence": resp.confidence,
            "is_fallback": resp.is_fallback,
            "is_low_confidence": resp.is_low_confidence,
            "source": resp.source,
        })
        if resp.reasoning_confidence is not None:
            payload["reasoning_confidence"] = resp.reasoning_confidence
        if resp.mood is not None:
            payload["mood"] = {
                "label": resp.mood.label,
                "emoji": resp.mood.emoji,
            }
    return payload


def _jsonrpc_error(code: int, message: str, rpc_id) -> Dict[str, Any]:
    """Build JSON-RPC error response."""
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}


def _jsonrpc_result(rpc_id, result: Dict[str, Any]) -> Dict[str, Any]:
    """Build JSON-RPC success response."""
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


async def _handle_mcp_jsonrpc(
    payload: Dict[str, Any],
    client_id: str,
    context_id: Optional[str]
) -> Dict[str, Any]:
    """Handle MCP JSON-RPC request."""
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
                "serverInfo": {"name": "Lilith-V2", "version": "0.2"},
            },
        )
    
    if method == "tools/list":
        tools = xiaozhi_registry.list_tools()
        return _jsonrpc_result(
            rpc_id,
            {"tools": tools, "nextCursor": ""},
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
        except Exception as exc:
            return _jsonrpc_error(-32000, f"Tool error: {exc}", rpc_id)
    
    return _jsonrpc_error(-32601, f"Unknown method: {method}", rpc_id)


# ===== HTTP Endpoints =====

@app.post("/chat")
async def chat(req: ChatRequest):
    """Chat endpoint."""
    resp = await anyio.to_thread.run_sync(
        get_adapter().handle_chat, req.client_id, req.message, req.context_id
    )
    result = {
        "text": resp.text,
        "confidence": resp.confidence,
        "is_fallback": resp.is_fallback,
        "is_low_confidence": resp.is_low_confidence,
        "source": resp.source,
        "pattern_id": resp.pattern_id,
        "learned_fact": resp.learned_fact,
    }
    if resp.mood is not None:
        result["mood"] = {"label": resp.mood.label, "emoji": resp.mood.emoji}
    return result


@app.post("/teach")
async def teach(req: TeachRequest):
    """Teach a new pattern."""
    pattern_id = await anyio.to_thread.run_sync(
        get_adapter().handle_teach,
        req.client_id,
        req.trigger,
        req.response,
        req.intent,
        req.context_id,
    )
    return {"pattern_id": pattern_id}


@app.post("/feedback/upvote")
async def upvote(req: FeedbackRequest):
    """Upvote a pattern."""
    await anyio.to_thread.run_sync(
        get_adapter().handle_upvote, req.client_id, req.pattern_id, req.strength, req.context_id
    )
    return {"status": "ok"}


@app.post("/feedback/downvote")
async def downvote(req: FeedbackRequest):
    """Downvote a pattern."""
    await anyio.to_thread.run_sync(
        get_adapter().handle_downvote, req.client_id, req.pattern_id, req.strength, req.context_id
    )
    return {"status": "ok"}


@app.post("/stats")
async def stats(req: ClientContext):
    """Get session stats."""
    return await anyio.to_thread.run_sync(
        get_adapter().handle_stats, req.client_id, req.context_id, True
    )


@app.post("/weather")
async def weather(req: WeatherRequest):
    """Get weather info."""
    report = await anyio.to_thread.run_sync(
        get_adapter().handle_weather, req.client_id, req.location, req.context_id
    )
    return {
        "location": report.location,
        "summary": report.summary,
        "temperature_c": report.temperature_c,
    }


@app.post("/news")
async def news(req: NewsRequest):
    """Get news headline."""
    headline = await anyio.to_thread.run_sync(
        get_adapter().handle_news, req.client_id, req.topic, req.context_id
    )
    return {"headline": headline}


# ===== WebSocket Endpoint =====

@app.websocket("/xiaozhi/ws")
async def xiaozhi_ws(websocket: WebSocket):
    """
    Xiaozhi-compatible WebSocket endpoint.
    
    Protocol:
    1. Server sends hello: {type: "hello", transport: "websocket", session_id, features: {mcp: true}}
    2. Client sends messages:
       - {type: "listen", state: "start|stop|detect", mode: "realtime|auto|manual"}
       - {type: "abort", reason: "..."}
       - {type: "mcp", payload: {...}} - JSON-RPC or simple chat
       - Binary audio frames (if audio enabled)
    3. Server responds with appropriate acks and MCP responses
    """
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
    
    await websocket.send_json({
        "type": "hello",
        "transport": "websocket",
        "version": 2,  # V2!
        "session_id": session_id,
        "features": {"mcp": True, "v2": True},
    })
    
    async def send_ack(
        message_type: str,
        state: Optional[str] = None,
        ok: bool = True,
        error: Optional[str] = None
    ):
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
                    
                    # JSON-RPC format
                    if payload.get("jsonrpc") == "2.0":
                        rpc_resp = await _handle_mcp_jsonrpc(payload, client_id, context_id)
                        await websocket.send_json({
                            "type": "mcp",
                            "session_id": session_id,
                            "payload": rpc_resp,
                        })
                    
                    # Simple chat format
                    elif "message" in payload:
                        resp = await anyio.to_thread.run_sync(
                            get_adapter().handle_chat,
                            client_id,
                            payload["message"],
                            context_id,
                        )
                        await websocket.send_json({
                            "type": "mcp",
                            "session_id": session_id,
                            "payload": _build_mcp_payload(resp),
                        })
                    
                    else:
                        await send_ack("mcp", ok=False, error="unknown_mcp_format")
                
                elif kind == "iot":
                    # IoT state/descriptors - ack and log for future use
                    logger.debug(f"IoT message from {client_id}: {data}")
                    await send_ack("iot", state="received")
                
                else:
                    await send_ack("error", ok=False, error=f"unknown_type:{kind}")
            
            elif bytes_data is not None:
                # Audio frame
                if _is_audio_enabled():
                    # Would process audio here when speech bridge is wired
                    await send_ack("audio", state="received")
                else:
                    await send_ack("audio", ok=False, error="audio_disabled")
    
    except WebSocketDisconnect:
        logger.debug(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)


# ===== CLI Entry Point =====

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("XIAOZHI_PORT", "8001"))
    host = os.getenv("XIAOZHI_HOST", "0.0.0.0")
    
    print(f"Starting Lilith V2 Xiaozhi Server on {host}:{port}")
    print(f"WebSocket endpoint: ws://{host}:{port}/xiaozhi/ws")
    print(f"HTTP endpoints: /chat, /teach, /stats, /weather, /news")
    
    uvicorn.run(app, host=host, port=port)
