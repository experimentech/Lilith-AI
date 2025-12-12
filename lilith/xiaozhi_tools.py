"""
Xiaozhi tool registry and handlers for MCP JSON-RPC calls.

This mirrors the basic shapes expected by xiaozhi-esp32/py-xiaozhi: tools/list
and tools/call return JSON-RPC-compatible structures.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from lilith.mcp_adapter import MCPAdapter


@dataclass
class XiaozhiTool:
    name: str
    description: str
    input_schema: Dict
    handler: Callable[[Dict, str, Optional[str]], Tuple[str, bool]]
    # handler returns (text, is_error)


class XiaozhiToolRegistry:
    def __init__(self, adapter: MCPAdapter):
        self.adapter = adapter
        self._tools: Dict[str, XiaozhiTool] = {}
        self._install_defaults()

    def _install_defaults(self) -> None:
        self.register(
            XiaozhiTool(
                name="self.get_device_status",
                description="Get device status",
                input_schema={"type": "object", "properties": {}, "required": []},
                handler=lambda params, client_id, context_id: ("ok", False),
            )
        )
        self.register(
            XiaozhiTool(
                name="self.audio_speaker.set_volume",
                description="Set speaker volume",
                input_schema={
                    "type": "object",
                    "properties": {"volume": {"type": "integer", "minimum": 0, "maximum": 100}},
                    "required": ["volume"],
                },
                handler=self._handle_set_volume,
            )
        )
        self.register(
            XiaozhiTool(
                name="self.audio_speaker.get_volume",
                description="Get speaker volume",
                input_schema={"type": "object", "properties": {}, "required": []},
                handler=lambda params, client_id, context_id: ("50", False),
            )
        )
        self.register(
            XiaozhiTool(
                name="chat.reply",
                description="Chat via Lilith",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                handler=self._handle_chat,
            )
        )

    def register(self, tool: XiaozhiTool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> List[Dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def call(self, name: str, arguments: Dict, client_id: str, context_id: Optional[str]) -> Tuple[str, bool]:
        tool = self._tools.get(name)
        if not tool:
            raise KeyError(name)
        return tool.handler(arguments or {}, client_id, context_id)

    def _handle_chat(self, params: Dict, client_id: str, context_id: Optional[str]) -> Tuple[str, bool]:
        text = params.get("text") or ""
        resp = self.adapter.handle_chat(client_id, text, context_id)
        return resp.text, False

    def _handle_set_volume(self, params: Dict, client_id: str, context_id: Optional[str]) -> Tuple[str, bool]:
        # No real device; acknowledge the request
        return "true", False


__all__ = ["XiaozhiTool", "XiaozhiToolRegistry"]
