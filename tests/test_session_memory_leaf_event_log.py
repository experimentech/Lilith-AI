from __future__ import annotations

from pathlib import Path

from lilith.session import LilithSession, SessionConfig
from lilith.storage.sqlite_memory_store import SQLiteMemoryStore


def test_session_memory_leaf_event_log_creates_db_and_records_turns(tmp_path: Path):
    cfg = SessionConfig(
        data_path=str(tmp_path),
        enable_knowledge_augmentation=False,
        enable_modal_routing=False,
        use_grammar=False,
        enable_world_model=False,
        enable_feedback_detection=False,
        enable_auto_learning=False,
        enable_declarative_learning=False,
        enable_memory_leaf_event_log=True,
    )

    session = LilithSession(user_id="u1", context_id="ctx", config=cfg)
    session.process_message("Hello")

    db_path = tmp_path / "users" / "u1" / cfg.memory_leaf_db_name
    assert db_path.exists()

    store = SQLiteMemoryStore(db_path)
    # At minimum, we should have logged the user turn; usually we also log assistant.
    assert store.count(scenario="ctx") >= 1
