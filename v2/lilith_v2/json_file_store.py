import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from .store import Store


class JsonFileStore(Store):
    """JSON-backed Store with simple TTL support.

    Persisted as a mapping of key -> {"value": ..., "expires_at": float|None}.
    Not concurrency-safe; intended for local testing.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._data: Dict[str, Tuple[Any, Optional[float]]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        text = self._path.read_text(encoding="utf-8")
        if not text.strip():
            return
        raw = json.loads(text)
        for k, entry in raw.items():
            self._data[k] = (entry.get("value"), entry.get("expires_at"))

    def _persist(self) -> None:
        serialized = {
            k: {"value": v, "expires_at": exp} for k, (v, exp) in self._data.items()
        }
        self._path.write_text(json.dumps(serialized), encoding="utf-8")

    def get(self, key: str) -> Optional[Any]:
        item = self._data.get(key)
        if not item:
            return None
        value, exp = item
        if exp and exp <= time.time():
            self._data.pop(key, None)
            self._persist()
            return None
        return value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        expires_at = time.time() + ttl if ttl else None
        self._data[key] = (value, expires_at)
        self._persist()

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
        if key in self._data:
            self._data.pop(key, None)
            self._persist()

    def decay(self, now: float) -> int:
        removed = 0
        for k, (_, exp) in list(self._data.items()):
            if exp and exp <= now:
                self._data.pop(k, None)
                removed += 1
        if removed:
            self._persist()
        return removed

    def sanitize(self, value: Any) -> Any:
        return value
