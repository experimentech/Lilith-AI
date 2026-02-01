import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .persistence import PersistenceWrapper
from .relational_store import RelationalStore


class SQLitePersistenceWrapper(PersistenceWrapper):
    """Lightweight persistence wrapper around SQLite stores.

    Provides transactional helper and hygiene hooks. Uses a dedicated connection
    for tx() while allowing separate RelationalStore instances for KV access.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.commit()

    def open_store(self, path: str, schema_version: int):
        # schema_version is retained for future migrations; unused for now
        return RelationalStore(path)

    def migrate(self, target_version: int) -> None:
        # No-op stub; migrations can be implemented per schema when needed
        return None

    def tx(self, fn: Callable[[], Any]) -> Any:
        cur = self._conn.cursor()
        try:
            result = fn()
            self._conn.commit()
            return result
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def sanitize(self, record: Any) -> Any:
        return record

    def decay(self, now: float) -> int:
        # Provide a best-effort cleanup hook; assumes a kv table if present.
        cur = self._conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='kv'")
        if not cur.fetchone():
            return 0
        cur = self._conn.execute("DELETE FROM kv WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,))
        self._conn.commit()
        return cur.rowcount

    def validate(self, record: Any) -> bool:
        return True

    def close(self) -> None:
        self._conn.close()


__all__ = ["SQLitePersistenceWrapper"]
