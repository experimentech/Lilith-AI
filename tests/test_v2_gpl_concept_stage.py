from __future__ import annotations

from v2.lilith_v2.gpl_concept_stage import GPLConceptStage
from v2.lilith_v2.json_file_store import JsonFileStore
from v2.lilith_v2.relational_store import RelationalStore


class DummyVector:
    def __init__(self, data):
        self._data = data

    def squeeze(self):  # mimic tensor API surface
        return self

    def tolist(self):
        return self._data


class DummyEncoder:
    def encode(self, tokens):
        size = float(len(list(tokens)) or 1)
        return DummyVector([size, size])


class DummyPMEncoder(DummyEncoder):
    def __init__(self):
        self.pm_field = None  # Simulate presence without MultiScale to hit fallback

    def encode_with_components(self, tokens):
        combined = self.encode(tokens)
        latent = DummyVector([1.0, 2.0])
        raw = DummyVector([3.0, 4.0])
        return combined, latent, raw


def test_gpl_concept_stage_learn_and_retrieve_json(tmp_path):
    store = JsonFileStore(str(tmp_path / "gpl.json"))
    stage = GPLConceptStage("trunk.concepts", store, encoder=DummyEncoder(), retrieval_config={"top_k": 3})
    ctx = {}
    stage.learn({"text": "hello world", "response": "hi", "intent": "greet"}, ctx)

    results = list(stage.retrieve("hello", {}))
    assert len(results) == 1
    assert results[0]["intent"] == "greet"

    pid = results[0]["id"]
    stage.update_success({"id": pid, "delta": 0.5}, {})
    stats = stage.stats()
    assert stats["patterns"] == 1


def test_gpl_concept_stage_with_relational_store(tmp_path):
    store = RelationalStore(str(tmp_path / "gpl.sqlite"))
    stage = GPLConceptStage(
        "branch.concepts",
        store,
        encoder=DummyPMEncoder(),
        retrieval_config={"top_k": 2, "pmflow": True},
    )
    ctx = {}
    stage.learn({"text": "farewell", "response": "bye", "intent": "exit"}, ctx)

    results = list(stage.retrieve("farewell", {}))
    assert results
    sidecar = stage.relational_sidecar(None)
    assert sidecar and sidecar["table"] == "kv"