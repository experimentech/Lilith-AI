"""SQLite-backed symbol storage for retrieval experiments."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import torch

class SQLiteVectorStore:
    """Persist embeddings and labels in a lightweight SQLite database."""

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
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scenario TEXT,
                    label INTEGER NOT NULL,
                    vector BLOB NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embeddings_label
                ON embeddings(label)
                """
            )
            conn.commit()

    def clear(self, *, scenario: Optional[str] = None) -> None:
        with self._connect() as conn:
            if scenario is None:
                conn.execute("DELETE FROM embeddings")
            else:
                conn.execute("DELETE FROM embeddings WHERE scenario = ?", (scenario,))
            conn.commit()

    def add(self, embeddings: torch.Tensor, labels: torch.Tensor, *, scenario: Optional[str] = None) -> None:
        embeddings = embeddings.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        rows = []
        for label, vec in zip(labels, embeddings):
            vector_bytes = torch.as_tensor(vec, dtype=torch.float32).numpy().tobytes()
            rows.append((scenario, int(label), vector_bytes))
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO embeddings (scenario, label, vector) VALUES (?, ?, ?)",
                rows,
            )
            conn.commit()

    def fetch(self, scenario: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        query = "SELECT label, vector FROM embeddings"
        params: Sequence[object] = ()
        if scenario is not None:
            query += " WHERE scenario = ?"
            params = (scenario,)
        with self._connect() as conn:
            cur = conn.execute(query, params)
            rows = cur.fetchall()
        if not rows:
            raise RuntimeError("Vector store is empty.")
        labels = torch.tensor([row[0] for row in rows], dtype=torch.long)
        vectors_np = np.stack(
            [np.frombuffer(row[1], dtype=np.float32).copy() for row in rows],
            axis=0,
        )
        vectors = torch.from_numpy(vectors_np)
        return vectors, labels

    def search(
        self, queries: torch.Tensor, topk: int = 1, *, scenario: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vectors, labels = self.fetch(scenario)
        topk = max(1, min(topk, vectors.shape[0]))
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
        return values, top_labels

    def count(self, *, scenario: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) FROM embeddings"
        params: Sequence[object] = ()
        if scenario is not None:
            query += " WHERE scenario = ?"
            params = (scenario,)
        with self._connect() as conn:
            cur = conn.execute(query, params)
            (count,) = cur.fetchone()
        return int(count)
