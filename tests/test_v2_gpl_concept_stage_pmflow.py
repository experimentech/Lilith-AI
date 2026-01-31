from __future__ import annotations

import pytest

pmflow = pytest.importorskip("pmflow")
from pmflow.encoder import PMFlowEmbeddingEncoder  # noqa: E402

from v2.lilith_v2.gpl_concept_stage import GPLConceptStage
from v2.lilith_v2.json_file_store import JsonFileStore


def test_gpl_concept_stage_pmflow_retrieval(tmp_path):
    store = JsonFileStore(str(tmp_path / "gpl_pmflow.json"))
    encoder = PMFlowEmbeddingEncoder(seed=7, dimension=48, latent_dim=24)
    stage = GPLConceptStage(
        "trunk.concepts",
        store,
        encoder=encoder,
        retrieval_config={"pmflow": True, "top_k": 2, "expand_query": True, "hierarchical": True, "min_similarity": -1.0},
    )

    ctx1 = {}
    stage.learn({"text": "hello world", "response": "hi", "intent": "greet"}, ctx1)
    ctx2 = {}
    stage.learn({"text": "goodbye world", "response": "bye", "intent": "farewell"}, ctx2)

    results = list(stage.retrieve("hello", {}))
    assert results, "expected at least one retrieval result"
    assert any(r.get("intent") == "greet" for r in results)
    assert isinstance(results[0].get("pm_latent"), list), "pm_latent should be stored as list"

    stats = stage.stats()
    assert stats["patterns"] == 2
