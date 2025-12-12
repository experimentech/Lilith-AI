import pytest
from starlette.testclient import TestClient

import lilith.mcp_server as mcp_server


def test_xiaozhi_websocket_handshake_and_chat(monkeypatch):
    monkeypatch.delenv("XIAOZHI_AUTH_TOKEN", raising=False)  # allow open access for test
    client = TestClient(mcp_server.app)

    headers = {"Client-Id": "pytest-client"}
    with client.websocket_connect("/xiaozhi/ws", headers=headers) as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert hello["transport"] == "websocket"
        assert hello["session_id"]

        ws.send_json({"type": "mcp", "payload": {"message": "hello"}})
        resp = ws.receive_json()
        assert resp["type"] == "mcp"
        assert resp["session_id"] == hello["session_id"]
        assert "text" in resp["payload"]


def test_xiaozhi_websocket_listen_ack(monkeypatch):
    monkeypatch.delenv("XIAOZHI_AUTH_TOKEN", raising=False)
    client = TestClient(mcp_server.app)

    with client.websocket_connect("/xiaozhi/ws") as ws:
        hello = ws.receive_json()
        ws.send_json({"type": "listen", "state": "start", "mode": "realtime"})
        ack = ws.receive_json()
        assert ack["type"] == "listen"
        assert ack["session_id"] == hello["session_id"]
        assert ack["ok"] is True


def test_xiaozhi_websocket_audio_disabled(monkeypatch):
    monkeypatch.setenv("XIAOZHI_AUDIO_ENABLED", "0")
    client = TestClient(mcp_server.app)

    with client.websocket_connect("/xiaozhi/ws") as ws:
        _hello = ws.receive_json()
        ws.send_bytes(b"opusframe")
        ack = ws.receive_json()
        assert ack["type"] == "audio"
        assert ack["ok"] is False
        assert ack.get("error") == "audio_disabled"


def test_xiaozhi_websocket_auth_guard(monkeypatch):
    monkeypatch.setenv("XIAOZHI_AUTH_TOKEN", "secret")
    client = TestClient(mcp_server.app)

    with pytest.raises(Exception):
        with client.websocket_connect("/xiaozhi/ws"):
            pass


def test_xiaozhi_jsonrpc_initialize(monkeypatch):
    monkeypatch.delenv("XIAOZHI_AUTH_TOKEN", raising=False)
    client = TestClient(mcp_server.app)

    with client.websocket_connect("/xiaozhi/ws") as ws:
        _hello = ws.receive_json()
        ws.send_json({"type": "mcp", "payload": {"jsonrpc": "2.0", "method": "initialize", "id": 1}})
        resp = ws.receive_json()["payload"]
        assert resp["result"]["protocolVersion"]
        assert resp["id"] == 1


def test_xiaozhi_jsonrpc_tools_list(monkeypatch):
    monkeypatch.delenv("XIAOZHI_AUTH_TOKEN", raising=False)
    client = TestClient(mcp_server.app)

    with client.websocket_connect("/xiaozhi/ws") as ws:
        _hello = ws.receive_json()
        ws.send_json({"type": "mcp", "payload": {"jsonrpc": "2.0", "method": "tools/list", "id": 2}})
        resp = ws.receive_json()["payload"]
        tools = resp["result"]["tools"]
        assert isinstance(tools, list) and len(tools) >= 1
        assert resp["id"] == 2


def test_xiaozhi_jsonrpc_tools_call_chat(monkeypatch):
    monkeypatch.delenv("XIAOZHI_AUTH_TOKEN", raising=False)
    client = TestClient(mcp_server.app)

    with client.websocket_connect("/xiaozhi/ws") as ws:
        _hello = ws.receive_json()
        ws.send_json(
            {
                "type": "mcp",
                "payload": {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {"name": "chat.reply", "arguments": {"text": "hello"}},
                    "id": 3,
                },
            }
        )
        resp = ws.receive_json()["payload"]
        content = resp["result"]["content"]
        assert content and content[0]["type"] == "text"
        assert resp["id"] == 3


def test_xiaozhi_jsonrpc_unknown_tool(monkeypatch):
    monkeypatch.delenv("XIAOZHI_AUTH_TOKEN", raising=False)
    client = TestClient(mcp_server.app)

    with client.websocket_connect("/xiaozhi/ws") as ws:
        _hello = ws.receive_json()
        ws.send_json(
            {
                "type": "mcp",
                "payload": {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {"name": "unknown.tool", "arguments": {}},
                    "id": 4,
                },
            }
        )
        resp = ws.receive_json()["payload"]
        assert resp["error"]["code"] == -32601
        assert resp["id"] == 4
