import pytest

from lilith.session import LilithSession, SessionConfig


class DummyResponse:
    def __init__(self, text: str, confidence: float = 0.5, intent: str = ""):
        self.text = text
        self.confidence = confidence
        self.fragment_ids = []
        self.is_fallback = False
        self.is_low_confidence = False
        self.primary_pattern = type("Primary", (), {"intent": intent}) if intent else None


class DummyComposer:
    def __init__(self, *args, **kwargs):
        self.knowledge_augmenter = None
        self.contrastive_learner = None
        self.syntax_stage = None

    def compose_response(self, context, user_input):
        intent = user_input.strip().lower()
        return DummyResponse(text="ok", confidence=0.5, intent=intent)


class DummyStore:
    pass


def _make_session(tmp_path, monkeypatch):
    monkeypatch.setattr("lilith.response_composer.ResponseComposer", DummyComposer)

    cfg = SessionConfig(
        learning_enabled=False,
        enable_auto_learning=False,
        enable_declarative_learning=False,
        enable_feedback_detection=False,
        enable_personality=False,
        enable_mood=False,
        enable_compositional=False,
        enable_pragmatic_templates=False,
        enable_modal_routing=False,
        enable_knowledge_augmentation=False,
        plasticity_enabled=False,
        enable_preferences=True,
        data_path=str(tmp_path),
    )

    session = LilithSession(user_id="tester", config=cfg, store=DummyStore())
    session.state.is_active = lambda: False  # Skip pipeline parsing in tests
    return session


def test_preferences_learn_and_bias(monkeypatch, tmp_path):
    session = _make_session(tmp_path, monkeypatch)

    # Baseline confidence with no interests/aversions
    baseline = session.process_message("ping").confidence

    # Learn an interest
    session.process_message("I like chess and music")
    assert "chess" in session.user_preferences.interests

    boosted = session.process_message("chess").confidence
    assert boosted > baseline

    # Learn an aversion
    session.process_message("please avoid politics")
    assert "politics" in session.user_preferences.aversions

    reduced = session.process_message("politics").confidence
    assert reduced < baseline
