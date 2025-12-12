#!/usr/bin/env python3
"""
Simple Xiaozhi-compatible WebSocket client for local testing.

Usage examples:
  python scripts/xiaozhi_ws_client.py --url ws://localhost:8000/xiaozhi/ws --message "hello"
  python scripts/xiaozhi_ws_client.py --url ws://localhost:8000/xiaozhi/ws --list-tools
  python scripts/xiaozhi_ws_client.py --url ws://localhost:8000/xiaozhi/ws --tool chat.reply --arg text "hi"

Requires the server running (e.g., `uvicorn lilith.mcp_server:app --reload`).
"""
import argparse
import json
import uuid

from websockets.sync.client import connect


def build_jsonrpc_call(method: str, params: dict | None = None, rpc_id: int = 1) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
        "id": rpc_id,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Xiaozhi WebSocket smoke tester")
    parser.add_argument("--url", default="ws://localhost:8000/xiaozhi/ws", help="WebSocket URL")
    parser.add_argument("--token", help="Bearer token for XIAOZHI_AUTH_TOKEN")
    parser.add_argument("--client-id", default="cli-tester", help="Client-Id header")
    parser.add_argument("--message", help="Send a plain MCP chat payload (non-RPC)")
    parser.add_argument("--list-tools", action="store_true", help="Call tools/list JSON-RPC")
    parser.add_argument("--tool", help="Tool name for tools/call")
    parser.add_argument("--arg", nargs=2, action="append", metavar=("KEY", "VALUE"), help="Tool argument key/value pairs")
    return parser.parse_args()


def main():
    args = parse_args()

    headers = [("Client-Id", args.client_id)]
    if args.token:
        headers.append(("Authorization", f"Bearer {args.token}"))

    with connect(args.url, additional_headers=headers) as ws:
        hello = ws.recv()
        print("<--", hello)

        # Tools list
        if args.list_tools:
            rpc_id = 1
            payload = build_jsonrpc_call("tools/list", rpc_id=rpc_id)
            ws.send(json.dumps({"type": "mcp", "payload": payload}))
            resp = ws.recv()
            print("<--", resp)

        # Tool call
        if args.tool:
            rpc_id = 2
            arguments = {k: v for k, v in (args.arg or [])}
            payload = build_jsonrpc_call("tools/call", {"name": args.tool, "arguments": arguments}, rpc_id=rpc_id)
            ws.send(json.dumps({"type": "mcp", "payload": payload}))
            resp = ws.recv()
            print("<--", resp)

        # Plain chat
        if args.message:
            ws.send(json.dumps({"type": "mcp", "payload": {"message": args.message}}))
            resp = ws.recv()
            print("<--", resp)


if __name__ == "__main__":
    main()
