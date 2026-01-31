import numpy as np

from lilith.production_concept_store import ProductionConceptStore


class _DummyEncoder:
    def encode(self, tokens):
        # Shape doesn't matter for this test; similarity is monkeypatched.
        return np.ones(8, dtype=np.float32)


def test_consolidation_does_not_merge_unrelated_terms(tmp_path, monkeypatch):
    db_path = tmp_path / "concepts.db"
    store = ProductionConceptStore(_DummyEncoder(), str(db_path), consolidation_threshold=0.85)

    # Force cosine similarity to be (spuriously) high regardless of input.
    monkeypatch.setattr(store, "_cosine_similarity", lambda a, b: 0.96)

    c1 = store.add_concept("hello", ["a greeting"], source="test")
    c2 = store.add_concept("sydney", ["a city"], source="test")

    # Without the lexical guardrail, these could incorrectly consolidate.
    assert c1 != c2


def test_consolidation_still_merges_lexically_similar_terms(tmp_path, monkeypatch):
    db_path = tmp_path / "concepts.db"
    store = ProductionConceptStore(_DummyEncoder(), str(db_path), consolidation_threshold=0.85)

    # Force cosine similarity to be high.
    monkeypatch.setattr(store, "_cosine_similarity", lambda a, b: 0.96)

    c1 = store.add_concept("new york", ["a place"], source="test")
    c2 = store.add_concept("new york city", ["a city"], source="test")

    # Token overlap should permit consolidation.
    assert c1 == c2
