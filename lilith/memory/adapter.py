"""Adapter interface between generic cognitive events and DB-backed memory leaves.

This keeps memory leaves modality-agnostic while still making them easy to use
from higher-level orchestration code (tree/branches).

Key idea: callers provide a *MemoryEvent* describing the observation or query.
The adapter turns that into the appropriate MemoryLeaf call (embedding/tokens/text).

No chat-specific logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import torch

from lilith.memory.leaf import MemoryHit, MemoryLeaf


@dataclass(frozen=True)
class MemoryEvent:
    """A modality-agnostic observation/query.

    Provide exactly one of:
    - embedding: precomputed embedding tensor (preferred for non-text modalities)
    - tokens: token stream (list[str] or other encoder-compatible object)
    - text: raw text

    payload is stored verbatim for observations and is optional for queries.
    """

    modality: str = "unknown"
    text: Optional[str] = None
    tokens: Any = None
    embedding: Optional[torch.Tensor] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def with_payload(self, **updates: Any) -> "MemoryEvent":
        payload = dict(self.payload)
        payload.update(updates)
        return MemoryEvent(
            modality=self.modality,
            text=self.text,
            tokens=self.tokens,
            embedding=self.embedding,
            payload=payload,
        )


class MemoryLeafAdapter:
    """Small adapter around MemoryLeaf for event-style usage."""

    def __init__(self, leaf: MemoryLeaf) -> None:
        self.leaf = leaf

    def observe(self, event: MemoryEvent) -> int:
        """Store an event in memory and return its assigned label."""
        payload = dict(event.payload)
        payload.setdefault("modality", event.modality)

        if event.embedding is not None:
            return self.leaf.observe_embedding(event.embedding, payload)
        if event.tokens is not None:
            return self.leaf.observe_tokens(event.tokens, payload)
        if event.text is not None:
            payload.setdefault("text", event.text)
            return self.leaf.observe_text(event.text, payload)

        raise ValueError("MemoryEvent must provide embedding, tokens, or text.")

    def query(self, event: MemoryEvent, *, topk: int = 5) -> List[MemoryHit]:
        """Retrieve nearest stored events for this query."""
        if event.embedding is not None:
            return self.leaf.query_embedding(event.embedding, topk=topk)
        if event.tokens is not None:
            return self.leaf.query_tokens(event.tokens, topk=topk)
        if event.text is not None:
            return self.leaf.query_text(event.text, topk=topk)

        raise ValueError("MemoryEvent must provide embedding, tokens, or text.")

    def bulk_observe_texts(
        self,
        texts: Iterable[str],
        *,
        modality: str = "text",
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        base = dict(extra_payload or {})
        base.setdefault("modality", modality)
        return self.leaf.bulk_observe_texts(texts, extra_payload=base)
