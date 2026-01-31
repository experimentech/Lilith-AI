from typing import Any, Dict, Iterable, Protocol


class IOPort(Protocol):
    """Typed I/O port with trace-aware metadata."""

    def send(self, message: Any, meta: Dict[str, Any]) -> None:
        ...

    def receive(self, meta: Dict[str, Any]) -> Iterable[Any]:
        ...

    def ack(self, message_id: str) -> None:
        ...

    def nack(self, message_id: str, reason: str) -> None:
        ...

    def flush(self) -> None:
        ...

    def attach(self, adapter: Any) -> None:
        ...
