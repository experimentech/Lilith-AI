from .concepts_stage import ConceptStage
from .json_file_store import JsonFileStore


def test_concept_stage_learn_merge_and_retrieve(tmp_path):
    store = JsonFileStore(str(tmp_path / "concepts.json"))
    stage = ConceptStage("branch.concepts", store)

    stage.learn({"id": "bird", "properties": {"beak": "short"}}, ctx={})
    stage.learn({"id": "bird", "properties": {"feathers": "yes"}, "relations": ["animal"]}, ctx={})

    hits = list(stage.retrieve({"id": "bird"}, ctx={}))
    assert len(hits) == 1
    concept = hits[0]
    assert concept["properties"]["beak"] == "short"
    assert concept["properties"]["feathers"] == "yes"
    assert "animal" in concept.get("relations", [])

    stats = stage.stats()
    assert stats["concepts"] == 1
