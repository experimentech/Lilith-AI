import pytest

from lilith.personality import PersonalityProfile, MoodState, update_mood_state
from lilith.session import LilithSession, SessionConfig


class DummyResponse:
    def __init__(self, text: str, confidence: float = 0.9):
        self.text = text
        self.confidence = confidence
        self.fragment_ids = []
        self.is_fallback = False
        self.is_low_confidence = False
        self.primary_pattern = None


class DummyComposer:
    def __init__(self, *args, **kwargs):
        self.knowledge_augmenter = None
        self.contrastive_learner = None
        self.syntax_stage = None

    def compose_response(self, context, user_input):
        return DummyResponse(text="Sure thing", confidence=0.9)


class DummySyntaxStage:
    def __init__(self):
        self.patterns = {"p": object()}
        self.last_feedback = None
        self.last_scale = None

    def _apply_plasticity(self, pattern, feedback, contrastive_pairs=None):  # pragma: no cover - inspected via state
        self.last_feedback = feedback

    def apply_contrastive_learning(self, scale=1.0):  # pragma: no cover - inspected via state
        self.last_scale = scale


class DummyComposerWithSyntax(DummyComposer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.syntax_stage = DummySyntaxStage()


class DummyStore:
    pass


def test_update_mood_state_transitions():
    mood = MoodState.neutral()

    mood = update_mood_state(mood, "I feel great and happy today!")
    assert mood.label == "positive"
    assert mood.emoji == "😊"

    mood = update_mood_state(mood, "I'm sad and very upset about this")
    assert mood.label == "concerned"
    assert mood.emoji == "😟"

    # Repeated neutral turns decay back to neutral
    steps = 0
    while mood.label != "neutral" and steps < 20:
        mood = update_mood_state(mood, "just checking in")
        steps += 1
    assert mood.label == "neutral"
    assert mood.intensity < 0.2


def test_session_applies_personality_and_mood(monkeypatch):
    monkeypatch.setattr("lilith.response_composer.ResponseComposer", DummyComposer)

    cfg = SessionConfig(
        learning_enabled=False,
        enable_auto_learning=False,
        enable_declarative_learning=False,
        enable_feedback_detection=False,
        enable_personality=True,
        enable_mood=True,
        enable_compositional=False,
        enable_pragmatic_templates=False,
        enable_modal_routing=False,
        enable_knowledge_augmentation=False,
    )

    session = LilithSession(user_id="tester", config=cfg, store=DummyStore())
    session.state.is_active = lambda: False  # Skip pipeline parsing in tests
    session.personality_profile = PersonalityProfile(
        tone="playful",
        warmth=0.8,
        brevity=0.8,
        humor=0.7,
        proactivity=0.8,
    )

    resp = session.process_message("I am happy and excited about this project")

    assert resp.personality is not None
    assert resp.mood is not None
    assert resp.mood.label == "positive"
    assert "happy to help" in resp.text
    assert "tiny bit of humor" in resp.text
    assert resp.text.endswith("More?")


def test_mood_scales_confidence_and_plasticity(monkeypatch):
    # Use composer with syntax stage to capture feedback/scale
    monkeypatch.setattr("lilith.response_composer.ResponseComposer", DummyComposerWithSyntax)

    cfg = SessionConfig(
        learning_enabled=True,
        enable_auto_learning=False,
        enable_declarative_learning=False,
        enable_feedback_detection=False,
        enable_personality=False,
        enable_mood=True,
        enable_compositional=False,
        enable_pragmatic_templates=False,
        enable_modal_routing=False,
        enable_knowledge_augmentation=False,
        plasticity_enabled=True,
        syntax_plasticity_interval=1,
        contrastive_interval=1,
    )

    session = LilithSession(user_id="tester", config=cfg, store=DummyStore())
    session.state.is_active = lambda: False  # Skip pipeline parsing

    # Start with neutral mood
    session.mood_state = MoodState.neutral()
    resp_neutral = session.process_message("just checking in")
    neutral_conf = resp_neutral.confidence

    # Positive mood should boost confidence and plasticity feedback
    session.mood_state = MoodState(label="positive", emoji="😊", decay=0.9, intensity=1.0)
    resp_positive = session.process_message("great job, I'm happy")
    assert resp_positive.confidence >= neutral_conf

    syntax_stage = session.composer.syntax_stage
    assert syntax_stage.last_feedback is not None
    assert syntax_stage.last_scale is not None
    positive_feedback = syntax_stage.last_feedback
    positive_scale = syntax_stage.last_scale

    # Concerned mood should dampen confidence and plasticity
    session.mood_state = MoodState(label="concerned", emoji="😟", decay=0.9, intensity=1.0)
    resp_concerned = session.process_message("I am worried and upset")

    assert resp_concerned.confidence <= resp_positive.confidence
    assert syntax_stage.last_feedback < positive_feedback
    assert syntax_stage.last_scale <= positive_scale
