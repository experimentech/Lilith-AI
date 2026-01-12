from collections import deque

from lilith.session import LilithSession, SessionConfig


class _StoreCapturingExternalLearn:
    def __init__(self):
        self.calls = []

    def learn_from_wikipedia(self, *, query: str, response_text: str, success_score: float, intent: str):
        self.calls.append(
            {
                "query": query,
                "response_text": response_text,
                "success_score": success_score,
                "intent": intent,
            }
        )
        return "pattern-1"


def test_upvote_external_uses_eligibility_even_if_last_fields_missing():
    # Construct a minimal session instance without running full __init__.
    session = LilithSession.__new__(LilithSession)
    session.config = SessionConfig(
        enable_mood=False,
        enable_personality=False,
        enable_preferences=False,
        learning_enabled=True,
    )
    session.store = _StoreCapturingExternalLearn()

    # Ensure we don't accidentally take the last_* fallback.
    session.last_user_input = None
    session.last_response_text = None
    session.last_pattern_id = "external_123"

    # Minimal dependencies used by upvote.
    session.mood_state = None
    session.personality_profile = None
    session.user_preferences = None

    session.composer = type("C", (), {"world_model": None})()

    # Eligibility buffer contains the correct (query, response_text).
    session._eligibility_buffer = deque(
        [
            {
                "channel": "test",
                "user_input": "What do you know about fog?",
                "response_text": "Fog is a visible aerosol...",
                "fragment_ids": ["external_123"],
                "weights": [1.0],
            }
        ],
        maxlen=25,
    )

    # Use real helper methods from the class.
    session._find_eligibility_record = LilithSession._find_eligibility_record.__get__(session)  # type: ignore[attr-defined]
    session._limbic_reinforcement_scale = LilithSession._limbic_reinforcement_scale.__get__(session)  # type: ignore[attr-defined]

    ok = LilithSession.upvote(session)
    assert ok is True

    assert len(session.store.calls) == 1
    assert session.store.calls[0]["query"] == "What do you know about fog?"
    assert "Fog is" in session.store.calls[0]["response_text"]
