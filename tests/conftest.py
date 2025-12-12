from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings

import pytest
from _pytest.warning_types import PytestReturnNotNoneWarning

# Silence PytestReturnNotNoneWarning for legacy tests that intentionally return values.
warnings.filterwarnings("ignore", category=PytestReturnNotNoneWarning)

from conversation_loop import ConversationLoop
from lilith.session import LilithSession, SessionConfig


@pytest.fixture
def loop():
    """Lightweight ConversationLoop for teaching/coherence tests."""
    return ConversationLoop(history_window=6, composition_mode="best_match", learning_mode="eager")


@pytest.fixture
def session(tmp_path):
    """LilithSession with isolated temp data dir for inference tests."""
    data_dir = tmp_path / "lilith_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = SessionConfig(
        data_path=str(data_dir),
        enable_knowledge_augmentation=False,
        enable_modal_routing=False,
        use_grammar=False,
    )
    return LilithSession(user_id="test_user", context_id="test_ctx", config=cfg)


@pytest.fixture
def scenario_name():
    return "teaching-baseline"
