from typing import Any, Callable, Dict, Optional
from .mcp_transport import MCPTransport

class LocalToolsTransport(MCPTransport):
    """Executes tools locally in the python process."""

    def __init__(self) -> None:
        self.tools: Dict[str, Callable[..., Any]] = {}

    def register_tool(self, name: str, func: Callable[..., Any]) -> None:
        """Register a python function as a callable tool."""
        self.tools[name] = func

    def call(self, name: str, message: Any, meta: Dict[str, Any]) -> Any:
        """
        Executes the tool.
        Expects 'message' to be a dict of kwargs for the function, 
        or the direct argument if the function executes a single payload.
        """
        # Simplistic dispatch: name of the 'limb' action is usually the tool name
        # But here 'name' passed to call() is the descriptor name (e.g. "vscode")
        # So the message must contain the specific action intent.
        
        # We assume message matches the structure: {"action": "tool_name", "args": {...}}
        if not isinstance(message, dict):
             return {"error": "Invalid payload structure. Expected dict with 'action'."}
        
        action = message.get("action")
        args = message.get("args", {})
        
        if not action:
             return {"error": "No action specified."}
             
        tool = self.tools.get(action)
        if not tool:
            return {"error": f"Tool '{action}' not found."}
            
        try:
            # If args is a dict, unpack it. If it's a value, pass as first arg?
            # For robustness, we assume dict.
            if isinstance(args, dict):
                result = tool(**args)
            else:
                result = tool(args)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}
