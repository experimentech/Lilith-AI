import pytest

from v2.lilith_v2.linguistic_processor import LinguisticProcessor


class DummyLanguageModel:
    def __init__(self):
        self.scored = []
        self.generated = []

    def score_text(self, text: str) -> float:
        self.scored.append(text)
        return -1.23

    def generate_text(self, prompt: str, max_new_tokens: int = 12) -> str:
        self.generated.append((prompt, max_new_tokens))
        return "clarified suggestion"


class FailingLanguageModel:
    def score_text(self, text: str) -> float:
        raise RuntimeError("simulated lm failure")

    def generate_text(self, prompt: str, max_new_tokens: int = 12) -> str:
        raise RuntimeError("simulated lm failure")


def test_lm_assist_triggers_for_low_confidence_parse():
    lm = DummyLanguageModel()
    processor = LinguisticProcessor(language_model=lm)

    # Mostly unknown tokens -> low parse confidence (< 0.6)
    artifact = processor.process("blorf glarn zqx")

    assert artifact.lm_score == pytest.approx(-1.23)
    assert artifact.lm_suggestion == "clarified suggestion"
    assert artifact.frame.attributes.get("lm_assist") == "clarified suggestion"
    assert len(lm.scored) == 1
    assert len(lm.generated) == 1


def test_lm_generation_is_skipped_for_high_confidence_parse():
    lm = DummyLanguageModel()
    processor = LinguisticProcessor(language_model=lm)

    # Known tokens should parse confidently and not require LM generation assist.
    artifact = processor.process("i like this")

    assert artifact.lm_score == pytest.approx(-1.23)
    assert artifact.lm_suggestion is None
    assert "lm_assist" not in artifact.frame.attributes
    assert len(lm.scored) == 1
    assert len(lm.generated) == 0


def test_lm_errors_do_not_break_linguistic_pipeline():
    processor = LinguisticProcessor(language_model=FailingLanguageModel())

    artifact = processor.process("blorf glarn zqx")

    # Pipeline should continue even if LM scoring/generation fails.
    assert artifact.normalized_text == "blorf glarn zqx"
    assert artifact.lm_score is None
    assert artifact.lm_suggestion is None
