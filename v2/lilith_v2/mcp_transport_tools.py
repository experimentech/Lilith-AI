from typing import Any, Callable, Dict, List, Optional
from .mcp_transport import MCPTransport

class LocalToolsTransport(MCPTransport):
    """Executes tools locally in the python process."""

    def __init__(self) -> None:
        self.tools: Dict[str, Callable[..., Any]] = {}
        self._tool_descriptions: Dict[str, str] = {}  # Tool name -> description

    def register_tool(self, name: str, func: Callable[..., Any], description: str = "") -> None:
        """Register a python function as a callable tool.
        
        Args:
            name: Tool name for invocation
            func: Python callable to execute
            description: Human-readable description for action learning
        """
        self.tools[name] = func
        # Use docstring if no description provided
        if not description and func.__doc__:
            description = func.__doc__.strip().split('\n')[0]
        self._tool_descriptions[name] = description or f"Execute {name}"
    
    def list_tools(self) -> List[Dict[str, str]]:
        """List available tools with their descriptions.
        
        Returns:
            List of {"name": str, "description": str} dicts
        """
        return [
            {"name": name, "description": self._tool_descriptions.get(name, "")}
            for name in self.tools.keys()
        ]

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
