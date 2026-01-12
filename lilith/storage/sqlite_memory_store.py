"""SQLite-backed store for (embedding, payload) memory items.

This is intended as a reusable substrate for modality-agnostic memory leaves.
It stores:
- an embedding vector (BLOB)
- a JSON payload (opaque to the store)
- optional metadata (scenario, created_at)

Retrieval is embedding-based (cosine or euclidean), consistent with the
"BioNN using database" philosophy: memory is an addressable set of learned
vectors, not hard-coded slots.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class MemoryRow:
    label: int
    payload: Dict[str, Any]


class SQLiteMemoryStore:
    """Persist embeddings with JSON payloads."""

    def __init__(self, path: Path, *, metric: str = "cosine") -> None:
        if metric not in {"cosine", "euclidean"}:
            raise ValueError(f"Unsupported metric '{metric}'.")
        self.path = path
        self.metric = metric
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scenario TEXT,
                    label INTEGER NOT NULL,
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_items_scenario_label
                ON memory_items(scenario, label)
                """
            )
            conn.commit()

    def clear(self, *, scenario: Optional[str] = None) -> None:
        with self._connect() as conn:
            if scenario is None:
                conn.execute("DELETE FROM memory_items")
            else:
                conn.execute("DELETE FROM memory_items WHERE scenario = ?", (scenario,))
            conn.commit()

    def count(self, *, scenario: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) FROM memory_items"
        params: Sequence[object] = ()
        if scenario is not None:
            query += " WHERE scenario = ?"
            params = (scenario,)
        with self._connect() as conn:
            cur = conn.execute(query, params)
            (count,) = cur.fetchone()
        return int(count)

    def next_label(self, *, scenario: Optional[str] = None) -> int:
        query = "SELECT MAX(label) FROM memory_items"
        params: Sequence[object] = ()
        if scenario is not None:
            query += " WHERE scenario = ?"
            params = (scenario,)
        with self._connect() as conn:
            cur = conn.execute(query, params)
            (max_label,) = cur.fetchone()
        return int(max_label + 1) if max_label is not None else 0

    def add(
        self,
        embeddings: torch.Tensor,
        payloads: List[Dict[str, Any]],
        labels: Optional[torch.Tensor] = None,
        *,
        scenario: Optional[str] = None,
    ) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D (N, D).")
        if embeddings.shape[0] != len(payloads):
            raise ValueError("payloads length must match embeddings rows.")

        n, d = int(embeddings.shape[0]), int(embeddings.shape[1])
        if labels is None:
            base = self.next_label(scenario=scenario)
            labels = torch.arange(base, base + n, dtype=torch.long)
        elif labels.shape[0] != n:
            raise ValueError("labels length must match embeddings rows.")

        embeddings_np = embeddings.detach().cpu().to(torch.float32).numpy()
        labels_np = labels.detach().cpu().numpy()

        now = time.time()
        rows = []
        for idx in range(n):
            label_int = int(labels_np[idx])
            vec_bytes = embeddings_np[idx].tobytes()
            payload_json = json.dumps(payloads[idx], ensure_ascii=False)
            rows.append((scenario, label_int, d, vec_bytes, payload_json, now))

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO memory_items (scenario, label, dim, vector, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

        return labels

    def fetch(self, *, scenario: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        query = "SELECT label, dim, vector, payload_json FROM memory_items"
        params: Sequence[object] = ()
        if scenario is not None:
            query += " WHERE scenario = ?"
            params = (scenario,)

        with self._connect() as conn:
            cur = conn.execute(query, params)
            rows = cur.fetchall()

        if not rows:
            raise RuntimeError("Memory store is empty.")

        labels = [int(row[0]) for row in rows]
        dims = [int(row[1]) for row in rows]
        vectors_list = [np.frombuffer(row[2], dtype=np.float32).copy() for row in rows]
        payloads_raw = [row[3] for row in rows]

        # Defensive: keep only the most common dimensionality.
        target_dim = int(np.bincount(np.array(dims, dtype=np.int64)).argmax())
        kept = [i for i, dim in enumerate(dims) if dim == target_dim and vectors_list[i].shape[0] == target_dim]
        if not kept:
            raise RuntimeError("Memory store contains no consistent vector shapes.")

        kept_labels = [labels[i] for i in kept]
        kept_vectors = np.stack([vectors_list[i] for i in kept], axis=0)
        kept_payloads: List[Dict[str, Any]] = []
        for i in kept:
            try:
                kept_payloads.append(json.loads(payloads_raw[i]))
            except Exception:
                kept_payloads.append({"_raw": str(payloads_raw[i])})

        return torch.from_numpy(kept_vectors), torch.tensor(kept_labels, dtype=torch.long), kept_payloads

    def search(
        self,
        queries: torch.Tensor,
        topk: int = 1,
        *,
        scenario: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[Dict[str, Any]]]]:
        vectors, labels, payloads = self.fetch(scenario=scenario)
        topk = max(1, min(topk, int(vectors.shape[0])))

        if self.metric == "cosine":
            support = torch.nn.functional.normalize(vectors, p=2, dim=1)
            q_norm = torch.nn.functional.normalize(queries, p=2, dim=1)
            scores = q_norm @ support.T
            values, indices = scores.topk(topk, dim=1)
        else:
            distances = torch.cdist(queries, vectors)
            neg_dist = -distances
            values, indices = neg_dist.topk(topk, dim=1)

        top_labels = labels[indices]
        top_payloads: List[List[Dict[str, Any]]] = []
        payload_by_row = payloads
        for row in indices.tolist():
            top_payloads.append([payload_by_row[int(i)] for i in row])

        return values, top_labels, top_payloads
