from typing import Any, Dict, Iterable, Optional, Protocol, Tuple


class Store(Protocol):
    """Storage contract shared by SQLite/JSON backends."""

    def get(self, key: str) -> Optional[Any]:
        ...

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        ...

    def list(self, prefix: Optional[str] = None) -> Iterable[Tuple[str, Any]]:
        ...

    def delete(self, key: str) -> None:
        ...

    def decay(self, now: float) -> int:
        ...

    def sanitize(self, value: Any) -> Any:
        ...
