from __future__ import annotations

from pathlib import Path

import torch
import pytest

from lilith.embedding import HashedEmbeddingEncoder
from lilith.memory.adapter import MemoryEvent, MemoryLeafAdapter
from lilith.memory.leaf import MemoryLeaf
from lilith.storage.sqlite_memory_store import SQLiteMemoryStore


def test_memory_leaf_adapter_observe_and_query_roundtrip(tmp_path: Path):
    db_path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(db_path)
    encoder = HashedEmbeddingEncoder(dimension=64)
    leaf = MemoryLeaf(store=store, encoder=encoder, scenario="pytest")
    adapter = MemoryLeafAdapter(leaf)

    label = adapter.observe(MemoryEvent(modality="text", text="Alice gave Bob a book.", payload={"kind": "event"}))
    assert isinstance(label, int)

    hits = adapter.query(MemoryEvent(modality="text", text="Alice gave Bob a book."), topk=1)
    assert len(hits) == 1
    assert hits[0].label == label
    assert hits[0].payload.get("text") == "Alice gave Bob a book."
    # Identical hashed embedding -> cosine similarity should be 1.0
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)


def test_memory_leaf_adapter_accepts_precomputed_embedding(tmp_path: Path):
    db_path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(db_path)
    encoder = HashedEmbeddingEncoder(dimension=32)
    leaf = MemoryLeaf(store=store, encoder=encoder, scenario="pytest")
    adapter = MemoryLeafAdapter(leaf)

    emb = encoder.encode("tool_result ok")
    label = adapter.observe(MemoryEvent(modality="tool", embedding=emb, payload={"tool": "demo"}))
    hits = adapter.query(MemoryEvent(modality="tool", embedding=emb), topk=1)

    assert hits[0].label == label
    assert hits[0].payload.get("tool") == "demo"


def test_memory_event_requires_input(tmp_path: Path):
    db_path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(db_path)
    encoder = HashedEmbeddingEncoder(dimension=32)
    leaf = MemoryLeaf(store=store, encoder=encoder, scenario="pytest")
    adapter = MemoryLeafAdapter(leaf)

    try:
        adapter.observe(MemoryEvent(modality="text"))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when event has no text/tokens/embedding")

    try:
        adapter.query(MemoryEvent(modality="text"), topk=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when query has no text/tokens/embedding")
