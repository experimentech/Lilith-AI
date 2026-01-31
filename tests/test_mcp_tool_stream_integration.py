import asyncio
import json
import socket
import threading
from types import SimpleNamespace

import pytest

from lilith.session import LilithSession, SessionConfig


class _StubXiaozhiMCPServer:
    def __init__(self):
        self.tools_list_calls = 0
        self.tools_call_calls = 0
        self.last_called = None
        self._thread = None
        self._loop = None
        self._server = None
        self._port = None

    @property
    def url(self) -> str:
        assert self._port is not None
        return f"ws://127.0.0.1:{self._port}"

    def start(self):
        # Bind a free port first to avoid races.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        _host, port = sock.getsockname()
        sock.close()
        self._port = int(port)

        ready = threading.Event()

        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.create_task(self._serve(ready))
            self._loop.run_forever()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        if not ready.wait(timeout=5):
            raise RuntimeError("stub MCP server failed to start")

    async def _serve(self, ready: threading.Event):
        import websockets

        async def handler(ws):
            session_id = "stub-session"
            await ws.send(
                json.dumps(
                    {
                        "type": "hello",
                        "transport": "websocket",
                        "version": 1,
                        "session_id": session_id,
                        "features": {"mcp": True},
                    }
                )
            )

            async for msg in ws:
                data = json.loads(msg)
                assert data.get("type") == "mcp"
                payload = data.get("payload")
                assert payload.get("jsonrpc") == "2.0"

                method = payload.get("method")
                rpc_id = payload.get("id")

                if method == "tools/list":
                    self.tools_list_calls += 1
                    resp = {
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "result": {
                            "tools": [
                                {
                                    "name": "weather.tool",
                                    "description": "weather forecast",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {"text": {"type": "string"}},
                                        "required": ["text"],
                                    },
                                }
                            ],
                            "nextCursor": "",
                        },
                    }
                elif method == "tools/call":
                    self.tools_call_calls += 1
                    params = payload.get("params") or {}
                    self.last_called = params
                    resp = {
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "stub weather: sunny in London",
                                }
                            ],
                            "isError": False,
                        },
                    }
                else:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "error": {"code": -32601, "message": f"Unknown method: {method}"},
                    }

                await ws.send(json.dumps({"type": "mcp", "session_id": session_id, "payload": resp}))

        self._server = await websockets.serve(handler, "127.0.0.1", self._port)
        ready.set()

    def stop(self):
        if self._loop is None:
            return

        async def _shutdown():
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()

        fut = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
        try:
            fut.result(timeout=5)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5)


@pytest.fixture
def stub_xiaozhi_mcp_server():
    srv = _StubXiaozhiMCPServer()
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def test_session_mcp_tool_stream_enriches_context_with_xiaozhi_envelope(tmp_path, stub_xiaozhi_mcp_server, monkeypatch):
    cfg = SessionConfig(
        data_path=str(tmp_path),
        learning_enabled=False,
        enable_declarative_learning=False,
        enable_auto_learning=False,
        enable_feedback_detection=False,
        plasticity_enabled=False,
        enable_world_model=False,
        enable_reasoning=False,
        enable_pragmatic_templates=False,
        enable_compositional=False,
        enable_knowledge_augmentation=False,
        enable_modal_routing=False,
        use_grammar=False,
        enable_mcp_tool_stream=True,
        mcp_endpoints=[{"name": "stub", "url": stub_xiaozhi_mcp_server.url}],
        mcp_max_tools_per_turn=1,
        mcp_timeout_seconds=2.0,
        mcp_cache_ttl_seconds=0.0,
    )

    session = LilithSession(user_id="pytest", config=cfg)

    captured = {"context": None}
    captured_tool = {"artifact": None}

    def fake_compose_response(*, context: str, user_input: str, tool_artifact=None):
        captured["context"] = context
        captured_tool["artifact"] = tool_artifact
        # Provide minimal attributes session expects.
        session.composer.last_approach = "test"
        return SimpleNamespace(
            text="ok",
            fragment_ids=["pattern_test"],
            confidence=0.9,
            is_fallback=False,
            is_low_confidence=False,
            composition_weights=[1.0],
            trace=None,
        )

    monkeypatch.setattr(session.composer, "compose_response", fake_compose_response)

    resp = session.process_message("weather London")
    assert resp.text == "ok"

    # Verify the MCP observation is passed as a structured stage artifact.
    assert captured_tool["artifact"] is not None
    assert "MCP[stub:weather.tool]: stub weather: sunny in London" in (captured_tool["artifact"].get("summary") or "")

    # Verify the stub server actually saw the JSON-RPC calls.
    assert stub_xiaozhi_mcp_server.tools_list_calls >= 1
    assert stub_xiaozhi_mcp_server.tools_call_calls >= 1
    assert stub_xiaozhi_mcp_server.last_called
    assert stub_xiaozhi_mcp_server.last_called.get("name") == "weather.tool"
    assert stub_xiaozhi_mcp_server.last_called.get("arguments", {}).get("text") == "weather London"
