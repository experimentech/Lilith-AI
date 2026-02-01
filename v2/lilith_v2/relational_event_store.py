import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class RelationalEventStore:
    """SQLite-backed append-only event store scoped by branch_id.

    Columns: branch_id, ts_ns, kind, payload_json, trace_id, tenant, modality
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                branch_id TEXT NOT NULL,
                ts_ns INTEGER NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT,
                trace_id TEXT,
                tenant TEXT,
                modality TEXT
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_branch_kind ON events(branch_id, kind, ts_ns DESC)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts_ns DESC)")
        self._conn.commit()

    def append(
        self,
        branch_id: str,
        kind: str,
        payload: Any,
        *,
        trace_id: Optional[str] = None,
        tenant: Optional[str] = None,
        modality: Optional[str] = None,
        ts_ns: Optional[int] = None,
    ) -> int:
        ts_val = ts_ns or time.time_ns()
        payload_json = json.dumps(payload)
        self._conn.execute(
            """
            INSERT INTO events(branch_id, ts_ns, kind, payload_json, trace_id, tenant, modality)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (branch_id, ts_val, kind, payload_json, trace_id, tenant, modality),
        )
        self._conn.commit()
        return ts_val

    def query(
        self,
        branch_id: str,
        *,
        kind: Optional[str] = None,
        since_ts_ns: Optional[int] = None,
        limit: int = 100,
        tenant: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> Iterable[Dict[str, Any]]:
        clauses: List[str] = ["branch_id = ?"]
        params: List[Any] = [branch_id]
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if since_ts_ns:
            clauses.append("ts_ns >= ?")
            params.append(since_ts_ns)
        if tenant:
            clauses.append("tenant = ?")
            params.append(tenant)
        if modality:
            clauses.append("modality = ?")
            params.append(modality)

        where = " AND ".join(clauses)
        sql = f"SELECT branch_id, ts_ns, kind, payload_json, trace_id, tenant, modality FROM events WHERE {where} ORDER BY ts_ns DESC LIMIT ?"
        params.append(limit)
        cur = self._conn.execute(sql, tuple(params))
        rows = cur.fetchall()
        results: List[Dict[str, Any]] = []
        for row in rows:
            payload = {}
            try:
                payload = json.loads(row[3]) if row[3] is not None else {}
            except Exception:
                payload = {}
            results.append(
                {
                    "branch_id": row[0],
                    "ts_ns": row[1],
                    "kind": row[2],
                    "payload": payload,
                    "trace_id": row[4],
                    "tenant": row[5],
                    "modality": row[6],
                }
            )
        return results

    def delete_before(self, ts_ns: int) -> int:
        cur = self._conn.execute("DELETE FROM events WHERE ts_ns < ?", (ts_ns,))
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self._conn.close()


__all__ = ["RelationalEventStore"]
