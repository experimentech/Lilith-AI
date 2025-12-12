"""Lightweight personality and mood primitives for Lilith.

These structures are intentionally minimal and neutral by default so they can
be wired into any adapter (CLI, Discord, MCP, etc.) without changing behavior
until explicitly enabled.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PersonalityProfile:
    """Optional personality profile attached to a session/context."""

    tone: str = "neutral"        # e.g., neutral, warm, formal, playful
    brevity: float = 0.5          # 0 (verbose) .. 1 (terse)
    warmth: float = 0.5           # 0 (cool) .. 1 (warm)
    humor: float = 0.0            # 0 (none) .. 1 (lots)
    interests: List[str] = field(default_factory=list)
    aversions: List[str] = field(default_factory=list)
    proactivity: float = 0.0      # 0 (off) .. 1 (short follow-ups allowed)

    @classmethod
    def neutral(cls) -> "PersonalityProfile":
        return cls()


@dataclass
class MoodState:
    """Optional mood metadata for UI/UX (no behavior change by default)."""

    label: str = "neutral"        # e.g., neutral, curious, confident, cautious
    emoji: str = "🙂"
    decay: float = 0.9            # Drift back to neutral
    intensity: float = 0.0        # 0 (neutral) .. 1 (strong)

    @classmethod
    def neutral(cls) -> "MoodState":
        return cls()


def apply_style(text: str, profile: PersonalityProfile) -> str:
    """Lightweight style adjustment. Defaults to no change for neutral profiles."""

    if profile is None:
        return text
    # Keep neutral profiles unchanged
    if profile.tone == "neutral" and profile.brevity == 0.5 and profile.warmth == 0.5 and profile.humor == 0.0:
        return text

    styled = text

    # Warm tone: add a soft closing if not present
    if profile.warmth > 0.6 and not styled.endswith("."):
        styled = f"{styled}."
    if profile.warmth > 0.6:
        styled = f"{styled} (happy to help)"

    # Brevity: lightly trim trailing whitespace (placeholder for future summarization)
    if profile.brevity > 0.7:
        styled = styled.strip()

    # Humor: append a light, short aside if enabled
    if profile.humor > 0.6:
        styled = f"{styled} (just a tiny bit of humor)"

    return styled


def maybe_add_followup(text: str, profile: PersonalityProfile, confidence: float) -> str:
    """Optionally add a short follow-up when proactivity is enabled and confidence is high."""

    if profile is None or profile.proactivity <= 0.0:
        return text
    if confidence < 0.65:
        return text

    followup = " Want to go deeper on that?"
    # Keep it concise if brevity is high
    if profile.brevity > 0.7:
        followup = " More?"

    return f"{text}{followup}"


# Simple lexicon for lightweight mood tracking. This is intentionally
# conservative: only strong cues flip the mood; otherwise it decays toward neutral.
POSITIVE_CUES = {
    "great", "good", "happy", "excited", "glad", "awesome", "amazing", "love", "thanks", "thank you",
    "cool", "nice", "wonderful", "fantastic"
}
NEGATIVE_CUES = {
    "sad", "upset", "angry", "frustrated", "worried", "anxious", "tired", "confused", "bad",
    "hate", "terrible", "awful", "annoyed", "stressed"
}


def _sentiment_score(text: str) -> int:
    """Very small sentiment heuristic based on keyword hits."""

    lowered = text.lower()
    score = 0
    for cue in POSITIVE_CUES:
        if cue in lowered:
            score += 1
    for cue in NEGATIVE_CUES:
        if cue in lowered:
            score -= 1
    return score


def _emoji_for_label(label: str) -> str:
    return {
        "positive": "😊",
        "concerned": "😟",
        "neutral": "🙂",
    }.get(label, "🙂")


def update_mood_state(current: Optional[MoodState], user_text: str) -> MoodState:
    """Update mood using lightweight sentiment cues and decay toward neutral.

    - Strong positive/negative cues set the mood immediately.
    - Otherwise, the previous mood decays toward neutral using ``decay``.
    """

    current = current or MoodState.neutral()
    score = _sentiment_score(user_text)

    # Positive or negative swing overrides decay
    if score >= 2:
        return MoodState(label="positive", emoji=_emoji_for_label("positive"), decay=current.decay, intensity=1.0)
    if score <= -2:
        return MoodState(label="concerned", emoji=_emoji_for_label("concerned"), decay=current.decay, intensity=1.0)

    # No strong signal: decay toward neutral
    decayed_intensity = current.intensity * current.decay
    if decayed_intensity < 0.2:
        return MoodState.neutral()

    # Maintain previous mood but with reduced intensity
    return MoodState(
        label=current.label,
        emoji=_emoji_for_label(current.label),
        decay=current.decay,
        intensity=decayed_intensity,
    )
