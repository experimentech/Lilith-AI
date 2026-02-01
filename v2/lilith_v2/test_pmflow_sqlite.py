from pathlib import Path

from .pmflow_sqlite import SQLitePMFlowStateStore
from .relational_event_store import RelationalEventStore
from .persistence_sqlite import SQLitePersistenceWrapper


def test_pmflow_state_store_round_trip(tmp_path: Path):
    path = tmp_path / "pmflow.sqlite"
    store = SQLitePMFlowStateStore(str(path))

    assert store.load_state("branch") == {}

    store.save_state("branch", {"foo": "bar", "latent_dims": [4, 8]}, version=1)
    assert store.load_state("branch") == {"foo": "bar", "latent_dims": [4, 8]}
    assert store.latent_dims("branch") == (4, 8)

    bumped = store.bump_version("branch")
    assert bumped == 2

    store.compact()
    store.close()


def test_relational_event_store_append_query(tmp_path: Path):
    path = tmp_path / "events.sqlite"
    events = RelationalEventStore(str(path))

    ts1 = events.append("branch", "learn", {"x": 1}, trace_id="t1", tenant="a", modality="text")
    ts2 = events.append("branch", "learn", {"x": 2}, trace_id="t2", tenant="a", modality="text")
    events.append("branch", "other", {"x": 3}, tenant="b", modality="image")

    rows = list(events.query("branch", kind="learn", since_ts_ns=ts1, limit=10, tenant="a"))
    assert len(rows) == 2
    assert rows[0]["payload"]["x"] == 2
    assert rows[1]["payload"]["x"] == 1

    deleted = events.delete_before(ts2)
    assert deleted >= 1

    events.close()


def test_persistence_wrapper_decay(tmp_path: Path):
    path = tmp_path / "persist.sqlite"
    wrapper = SQLitePersistenceWrapper(str(path))

    # Create kv table with an expired row to exercise decay
    conn = wrapper._conn
    conn.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, expires_at REAL)")
    conn.execute("INSERT OR REPLACE INTO kv(key, value, expires_at) VALUES ('a', 'v', 0.0)")
    conn.commit()

    removed = wrapper.decay(now=1.0)
    assert removed >= 1

    wrapper.close()
