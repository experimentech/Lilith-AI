import time

from .in_memory_store import InMemoryStore
from .noop_stage import NoopStage


def test_in_memory_store_put_get_list_decay():
    store = InMemoryStore()
    store.put("a", 1)
    store.put("b", 2, ttl=0.01)

    assert store.get("a") == 1
    assert dict(store.list()) == {"a": 1, "b": 2}

    time.sleep(0.02)
    removed = store.decay(time.time())
    assert removed >= 1
    assert store.get("b") is None
    assert "b" not in dict(store.list())


def test_noop_stage_flow():
    stage = NoopStage("trunk.test")
    stage.learn({"msg": "hello"}, ctx={})
    stage.learn({"msg": "world"}, ctx={})

    encoded = stage.encode({"msg": "x"}, ctx={})
    assert encoded == {"msg": "x"}

    results = list(stage.retrieve(query=None, ctx={}))
    assert len(results) == 2

    stage.update_success(feedback=None, ctx={})
    stats = stage.stats()
    assert stats["events"] == 2
    assert stats["successes"] == 1
