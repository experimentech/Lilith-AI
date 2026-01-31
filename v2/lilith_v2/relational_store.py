import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from .store import Store


class RelationalStore(Store):
    """SQLite-backed key/value store storing JSON blobs.

    Schema: kv(key TEXT PRIMARY KEY, value TEXT, expires_at REAL)
    TTL support is best-effort; decay removes expired rows.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, expires_at REAL)"
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[Any]:
        row = self._conn.execute("SELECT value, expires_at FROM kv WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        value_text, expires_at = row
        if expires_at and expires_at <= time.time():
            self.delete(key)
            return None
        return json.loads(value_text)

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        expires_at = time.time() + ttl if ttl else None
        value_text = json.dumps(value)
        self._conn.execute(
            "INSERT OR REPLACE INTO kv(key, value, expires_at) VALUES (?, ?, ?)",
            (key, value_text, expires_at),
        )
        self._conn.commit()

    def list(self, prefix: Optional[str] = None) -> Iterable[Tuple[str, Any]]:
        now = time.time()
        if prefix:
            rows = self._conn.execute(
                "SELECT key, value, expires_at FROM kv WHERE key LIKE ?", (f"{prefix}%",)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT key, value, expires_at FROM kv").fetchall()
        for k, v_text, exp in rows:
            if exp and exp <= now:
                self.delete(k)
                continue
            yield k, json.loads(v_text)

    def delete(self, key: str) -> None:
        self._conn.execute("DELETE FROM kv WHERE key=?", (key,))
        self._conn.commit()

    def decay(self, now: float) -> int:
        cur = self._conn.execute("DELETE FROM kv WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,))
        self._conn.commit()
        return cur.rowcount

    def sanitize(self, value: Any) -> Any:
        return value