from typing import Any, Callable, Dict


class MCPTransport:
    """Interface for MCP transports."""

    def call(self, name: str, message: Any, meta: Dict[str, Any]) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


class MockMCPTransport(MCPTransport):
    """Mock transport that dispatches to registered handlers or echoes.

    Supports success and error simulation by allowing handlers to raise.
    """

    def __init__(self) -> None:
        self.handlers: Dict[str, Callable[[Any, Dict[str, Any]], Any]] = {}

    def register(self, name: str, handler: Callable[[Any, Dict[str, Any]], Any]) -> None:
        self.handlers[name] = handler

    def register_error(self, name: str, message: str) -> None:
        def _err(_payload: Any, _meta: Dict[str, Any]) -> Any:
            raise RuntimeError(message)

        self.handlers[name] = _err

    def call(self, name: str, message: Any, meta: Dict[str, Any]) -> Any:
        handler = self.handlers.get(name)
        if handler:
            return handler(message, meta)
        return {"echo": message, "meta": meta}
