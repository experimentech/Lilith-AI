"""Reusable, modality-agnostic memory leaf.

Design goals:
- No chat-specific assumptions.
- DB-backed, embedding-addressable memory items (BioNN/PMFlow-friendly).
- Simple interface: observe() and query().

This is intentionally small; higher-level leaves can layer schemas and policies
(e.g., personal facts, episodic events, tool traces) on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol

import torch

from lilith.storage.sqlite_memory_store import SQLiteMemoryStore


class EmbeddingEncoder(Protocol):
    def encode(self, tokens: Any) -> torch.Tensor:  # tokens may be list[str] or str depending on encoder
        ...


@dataclass(frozen=True)
class MemoryHit:
    score: float
    label: int
    payload: Dict[str, Any]


class MemoryLeaf:
    """A modality-agnostic leaf: store + retrieve (embedding, payload)."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        encoder: Optional[EmbeddingEncoder] = None,
        *,
        scenario: str = "default",
    ) -> None:
        self.store = store
        self.encoder = encoder
        self.scenario = scenario

    def observe_embedding(self, embedding: torch.Tensor, payload: Dict[str, Any]) -> int:
        """Store an already-computed embedding with payload.

        Returns the assigned label.
        """
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        labels = self.store.add(embedding, [payload], scenario=self.scenario)
        return int(labels[0].item())

    def observe_tokens(self, tokens: Any, payload: Dict[str, Any]) -> int:
        """Encode tokens then store embedding + payload."""
        if self.encoder is None:
            raise RuntimeError("MemoryLeaf has no encoder; use observe_embedding().")
        embedding = self.encoder.encode(tokens)
        return self.observe_embedding(embedding, payload)

    def observe_text(self, text: str, payload: Optional[Dict[str, Any]] = None) -> int:
        """Convenience for text modalities.

        Keeps the leaf modality-agnostic by treating text as just another token stream.
        """
        payload_obj: Dict[str, Any] = dict(payload or {})
        payload_obj.setdefault("text", text)
        return self.observe_tokens(text, payload_obj)

    def query_embedding(self, embedding: torch.Tensor, *, topk: int = 5) -> List[MemoryHit]:
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        scores, labels, payloads = self.store.search(embedding, topk=topk, scenario=self.scenario)

        hits: List[MemoryHit] = []
        for score, label, payload in zip(scores[0].tolist(), labels[0].tolist(), payloads[0]):
            hits.append(MemoryHit(score=float(score), label=int(label), payload=dict(payload)))
        return hits

    def query_tokens(self, tokens: Any, *, topk: int = 5) -> List[MemoryHit]:
        if self.encoder is None:
            raise RuntimeError("MemoryLeaf has no encoder; use query_embedding().")
        embedding = self.encoder.encode(tokens)
        return self.query_embedding(embedding, topk=topk)

    def query_text(self, text: str, *, topk: int = 5) -> List[MemoryHit]:
        return self.query_tokens(text, topk=topk)

    def bulk_observe_embeddings(self, embeddings: torch.Tensor, payloads: List[Dict[str, Any]]) -> List[int]:
        labels = self.store.add(embeddings, payloads, scenario=self.scenario)
        return [int(v) for v in labels.tolist()]

    def bulk_observe_texts(self, texts: Iterable[str], *, extra_payload: Optional[Dict[str, Any]] = None) -> List[int]:
        if self.encoder is None:
            raise RuntimeError("MemoryLeaf has no encoder; use bulk_observe_embeddings().")
        payloads: List[Dict[str, Any]] = []
        embeddings: List[torch.Tensor] = []
        base = dict(extra_payload or {})
        for text in texts:
            payload = dict(base)
            payload.setdefault("text", text)
            payloads.append(payload)
            embeddings.append(self.encoder.encode(text))
        emb = torch.cat([e if e.ndim == 2 else e.unsqueeze(0) for e in embeddings], dim=0)
        return self.bulk_observe_embeddings(emb, payloads)
