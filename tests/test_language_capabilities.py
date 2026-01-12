from __future__ import annotations

import shutil
from pathlib import Path

from lilith.session import LilithSession, SessionConfig


def _make_session(tmp_path: Path) -> LilithSession:
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    cfg = SessionConfig(
        data_path=str(tmp_path),
        enable_knowledge_augmentation=False,
        enable_modal_routing=True,
        use_grammar=True,
        enable_world_model=False,
        enable_reasoning=False,
        enable_feedback_detection=False,
        enable_auto_learning=False,
        learning_enabled=False,
        enable_declarative_learning=False,
        enable_compositional=True,
        enable_pragmatic_templates=True,
        composition_mode="pragmatic",
        enable_personality=False,
        enable_mood=False,
        enable_preferences=False,
    )
    return LilithSession(user_id="test-user", context_id="test", config=cfg)


def test_sentence_about_topic_not_fallback() -> None:
    session = _make_session(Path("tmp/test_language_capabilities/about"))
    out = session.process_message("Can you tell me a short sentence about rain?")

    assert not out.is_fallback
    assert not out.is_low_confidence
    assert "rain" in out.text.lower()


def test_sentence_using_words_not_fallback() -> None:
    session = _make_session(Path("tmp/test_language_capabilities/words"))
    out = session.process_message("Invent a new sentence using the words: lantern, drift, quiet.")

    assert not out.is_fallback
    assert not out.is_low_confidence

    text = out.text.lower()
    for w in ["lantern", "drift", "quiet"]:
        assert w in text


def test_rephrase_returns_rewritten_sentence() -> None:
    session = _make_session(Path("tmp/test_language_capabilities/rephrase"))
    out = session.process_message("Please rephrase: 'The cat sat on the mat.'")

    assert not out.is_fallback
    assert not out.is_low_confidence
    assert "rested" in out.text.lower() or "upon" in out.text.lower()


def test_language_capability_writeback_allows_retrieval_override(tmp_path):
    """When enabled, language capability outputs should be written back as patterns.

    On the next identical prompt, a high-confidence retrieved pattern should override
    the deterministic capability path (pattern_* id instead of capability_*).
    """

    config = SessionConfig(
        data_path=str(tmp_path),
        enable_capability_writeback=True,
        enable_compositional=False,
        enable_pragmatic_templates=True,
        composition_mode="pragmatic",
        learning_enabled=True,
        enable_feedback_detection=False,
        enable_mood=False,
        enable_personality=False,
        enable_preferences=False,
    )

    session = LilithSession(user_id="test_user", context_id="test", config=config)

    prompt = "rephrase: the sky is blue"
    first = session.process_message(prompt)
    assert first.pattern_id is not None
    assert first.pattern_id == "rewrite"

    second = session.process_message(prompt)
    assert second.pattern_id is not None
    assert second.pattern_id.startswith("pattern_")
    assert second.text.strip() == first.text.strip()
