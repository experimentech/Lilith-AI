import time
from pathlib import Path

from .relational_store import RelationalStore


def test_relational_store_round_trip(tmp_path: Path):
    store = RelationalStore(str(tmp_path / "store.sqlite"))
    store.put("a", {"v": 1})
    assert store.get("a") == {"v": 1}
    assert dict(store.list()) == {"a": {"v": 1}}

    store.put("b", 2, ttl=0.01)
    time.sleep(0.02)
    removed = store.decay(time.time())
    assert removed >= 1
    assert store.get("b") is None


def test_relational_store_list_prefix(tmp_path: Path):
    store = RelationalStore(str(tmp_path / "store.sqlite"))
    store.put("x:1", 1)
    store.put("x:2", 2)
    store.put("y:1", 3)

    items = dict(store.list(prefix="x:"))
    assert items == {"x:1": 1, "x:2": 2}