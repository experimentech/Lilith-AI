"""Tests for SymbolicStore persistence and refresh behaviour."""

from __future__ import annotations

import torch

from experiments.retrieval_sanity.pipeline.base import Utterance
from experiments.retrieval_sanity.pipeline.pipeline import SymbolicPipeline
from experiments.retrieval_sanity.pipeline.storage_bridge import SymbolicStore


def test_refresh_embeddings_updates_vectors(tmp_path) -> None:
    sqlite_path = tmp_path / "vectors.db"
    store = SymbolicStore(sqlite_path, scenario="unit")
    store._frames_path = tmp_path / "frames.json"
    pipeline = SymbolicPipeline(use_pmflow=False)

    artefact = pipeline.process(Utterance(text="visit the museum tomorrow", language="en"))
    store.persist([artefact])

    vectors_before, labels = store.vector_store.fetch(scenario="unit")
    assert vectors_before.shape[0] == 1

    zero_vectors = torch.zeros_like(vectors_before)
    store.vector_store.clear(scenario="unit")
    store.vector_store.add(zero_vectors, labels, scenario="unit")

    refreshed = store.refresh_embeddings(pipeline.encoder)
    assert refreshed == 1

    vectors_after, labels_after = store.vector_store.fetch(scenario="unit")
    assert torch.equal(labels, labels_after)

    expected = pipeline.encoder.encode([token.text for token in artefact.parsed.tokens]).squeeze(0)
    assert torch.allclose(vectors_after[0], expected, atol=1e-5)
