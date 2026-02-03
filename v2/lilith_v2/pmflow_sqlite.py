import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .pmflow_state import PMFlowStateStore


class SQLitePMFlowStateStore(PMFlowStateStore):
    """SQLite-backed PMFlow state store with versioning.

    Schema: pmflow_state(branch_id TEXT PRIMARY KEY, version INTEGER, state_json TEXT, updated_at REAL, latent_dims TEXT)
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pmflow_state (
                branch_id TEXT PRIMARY KEY,
                version INTEGER DEFAULT 0,
                state_json TEXT,
                updated_at REAL,
                latent_dims TEXT
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_pmflow_state_updated ON pmflow_state(updated_at)")
        self._conn.commit()

    def load_state(self, branch_id: str, **kwargs) -> Dict[str, Any]:
        row = self._conn.execute(
            "SELECT state_json FROM pmflow_state WHERE branch_id=?",
            (branch_id,),
        ).fetchone()
        if not row or row[0] is None:
            return {}
        try:
            return json.loads(row[0])
        except Exception:
            return {}

    def save_state(self, branch_id: str, state: Dict[str, Any], version: int, **kwargs) -> None:
        now = time.time()
        state_json = json.dumps(state)
        latent_dims = None
        if isinstance(state, dict) and "latent_dims" in state:
            try:
                latent_dims = json.dumps(state["latent_dims"])
            except Exception:
                latent_dims = None
        self._conn.execute(
            """
            INSERT INTO pmflow_state(branch_id, version, state_json, updated_at, latent_dims)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(branch_id) DO UPDATE SET
                version=excluded.version,
                state_json=excluded.state_json,
                updated_at=excluded.updated_at,
                latent_dims=excluded.latent_dims
            """,
            (branch_id, version, state_json, now, latent_dims),
        )
        self._conn.commit()

    def bump_version(self, branch_id: str, **kwargs) -> int:
        cur = self._conn.execute(
            "SELECT version FROM pmflow_state WHERE branch_id=?",
            (branch_id,),
        )
        row = cur.fetchone()
        next_version = (row[0] if row else 0) + 1
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO pmflow_state(branch_id, version, state_json, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(branch_id) DO UPDATE SET
                version=excluded.version,
                updated_at=excluded.updated_at
            """,
            (branch_id, next_version, json.dumps({}), now),
        )
        self._conn.commit()
        return next_version

    def latent_dims(self, branch_id: str, **kwargs) -> Tuple[int, ...]:
        row = self._conn.execute(
            "SELECT latent_dims FROM pmflow_state WHERE branch_id=?",
            (branch_id,),
        ).fetchone()
        if not row or row[0] is None:
            return tuple()
        try:
            dims = json.loads(row[0])
            if isinstance(dims, (list, tuple)):
                return tuple(int(x) for x in dims)
        except Exception:
            return tuple()
        return tuple()

    def compact(self) -> None:
        self._conn.execute("PRAGMA optimize")
        self._conn.execute("VACUUM")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


__all__ = ["SQLitePMFlowStateStore"]
