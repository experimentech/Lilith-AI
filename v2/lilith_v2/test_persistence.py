import json
import time
from pathlib import Path

from .json_file_store import JsonFileStore
from .persistent_stage import PersistentStage


def test_json_file_store_round_trip(tmp_path: Path):
    path = tmp_path / "store.json"
    store = JsonFileStore(str(path))
    store.put("a", {"v": 1})
    store.put("b", 2, ttl=0.01)

    assert store.get("a") == {"v": 1}
    assert dict(store.list()) == {"a": {"v": 1}, "b": 2}

    time.sleep(0.02)
    removed = store.decay(time.time())
    assert removed >= 1
    assert store.get("b") is None

    # Re-load from disk and verify persistence of non-expired
    store2 = JsonFileStore(str(path))
    assert store2.get("a") == {"v": 1}


def test_persistent_stage_uses_store(tmp_path: Path):
    path = tmp_path / "store.json"
    store = JsonFileStore(str(path))
    stage = PersistentStage("node.x", store)

    stage.learn({"msg": "one"}, ctx={})
    stage.learn({"msg": "two"}, ctx={})

    items = list(stage.retrieve(query=None, ctx={}))
    assert len(items) == 2

    stage.update_success(feedback=None, ctx={})
    stats = stage.stats()
    assert stats["events"] == 2
    assert stats["successes"] == 1
