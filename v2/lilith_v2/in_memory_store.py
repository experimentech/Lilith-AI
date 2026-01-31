import time
from typing import Any, Dict, Iterable, Optional, Tuple

from .store import Store


class InMemoryStore(Store):
    """Simple in-memory Store with optional TTL and decay."""

    def __init__(self) -> None:
        self._data: Dict[str, Tuple[Any, Optional[float]]] = {}

    def get(self, key: str) -> Optional[Any]:
        item = self._data.get(key)
        if not item:
            return None
        value, expires_at = item
        if expires_at and expires_at <= time.time():
            # Expired; remove and return None
            self._data.pop(key, None)
            return None
        return value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        expires_at = time.time() + ttl if ttl else None
        self._data[key] = (value, expires_at)

    def list(self, prefix: Optional[str] = None) -> Iterable[Tuple[str, Any]]:
        now = time.time()
        for k, (v, exp) in list(self._data.items()):
            if exp and exp <= now:
                self._data.pop(k, None)
                continue
            if prefix and not k.startswith(prefix):
                continue
            yield k, v

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def decay(self, now: float) -> int:
        removed = 0
        for k, (_, exp) in list(self._data.items()):
            if exp and exp <= now:
                self._data.pop(k, None)
                removed += 1
        return removed

    def sanitize(self, value: Any) -> Any:
        return value
