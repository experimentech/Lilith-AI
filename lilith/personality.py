"""Lightweight personality and mood primitives for Lilith.

These structures are intentionally minimal and neutral by default so they can
be wired into any adapter (CLI, Discord, MCP, etc.) without changing behavior
until explicitly enabled.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union


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
    """Lightweight style adjustment. Defaults to no change for neutral profiles.
    
    Note: Most personality influence happens at the BNN embedding level during retrieval.
    This function only applies minimal post-processing for extreme settings.
    """

    if profile is None:
        return text
    # Keep neutral profiles unchanged
    if profile.tone == "neutral" and profile.brevity == 0.5 and profile.warmth == 0.5 and profile.humor == 0.0:
        return text

    styled = text

    # Brevity: lightly trim for very terse settings
    if profile.brevity > 0.8:
        styled = styled.strip()

    # Warmth: add a small supportive cue for high-warmth profiles.
    if profile.warmth >= 0.75 and "happy to help" not in styled.lower():
        styled = f"{styled} — happy to help"

    # Humor: keep it very lightweight and explicit.
    if profile.humor >= 0.6 and "tiny bit of humor" not in styled.lower():
        styled = f"{styled} (tiny bit of humor)"

    return styled


def maybe_add_followup(text: str, profile: PersonalityProfile, confidence: float) -> str:
    """Optionally add a short follow-up when proactivity is enabled and confidence is high.
    
    Note: Proactivity primarily influences BNN retrieval, not text appending.
    This is a minimal fallback for backwards compatibility.
    """

    if profile is None or profile.proactivity <= 0.0:
        return text
    if confidence < 0.70:  # Only for high confidence
        return text

    # Simple, non-intrusive followup
    if profile.proactivity >= 0.7:
        return f"{text} More?"

    if profile.brevity < 0.5:
        return f"{text} Anything else you'd like to know?"

    return text


def compute_sentiment_from_embedding(embedding, encoder, positive_anchors=None, negative_anchors=None):
    """Compute sentiment by comparing embedding to learned positive/negative anchors.
    
    This uses BNN embeddings instead of hard-coded word lists.
    Should be called from session/composer with access to encoder.
    
    Args:
        embedding: Query embedding tensor
        encoder: PMFlow encoder to compute anchor embeddings
        positive_anchors: List of positive reference terms (e.g., ['happy', 'excited'])
        negative_anchors: List of negative reference terms (e.g., ['sad', 'frustrated'])
    
    Returns:
        Score from -1.0 (negative) to 1.0 (positive)
    """
    import torch
    import torch.nn.functional as F
    
    if positive_anchors is None:
        positive_anchors = ['good', 'happy', 'excited']
    if negative_anchors is None:
        negative_anchors = ['sad', 'frustrated', 'worried']
    
    # Compute anchor embeddings
    pos_embs = [encoder.encode(term) for term in positive_anchors]
    neg_embs = [encoder.encode(term) for term in negative_anchors]
    
    # Compute similarities
    pos_sims = [F.cosine_similarity(embedding, emb, dim=-1).item() for emb in pos_embs]
    neg_sims = [F.cosine_similarity(embedding, emb, dim=-1).item() for emb in neg_embs]
    
    # Score based on relative similarity
    pos_score = max(pos_sims) if pos_sims else 0.0
    neg_score = max(neg_sims) if neg_sims else 0.0
    
    return pos_score - neg_score


def _emoji_for_label(label: str) -> str:
    return {
        "positive": "😊",
        "concerned": "😟",
        "neutral": "🙂",
    }.get(label, "🙂")


def mood_confidence_scale(mood: Optional[MoodState]) -> float:
    """Scale confidence based on mood intensity and valence.

    - Positive mood: slight boost up to +10%.
    - Concerned mood: slight dampening down to -10%.
    - Neutral/no mood: 1.0 (no change).
    """

    if mood is None or mood.label == "neutral":
        return 1.0

    intensity = min(max(mood.intensity, 0.0), 1.0)
    if mood.label == "positive":
        return 1.0 + 0.1 * intensity
    if mood.label == "concerned":
        return max(0.8, 1.0 - 0.1 * intensity)
    return 1.0


def mood_plasticity_scale(mood: Optional[MoodState]) -> float:
    """Scale plasticity learning rates based on mood.

    - Positive mood: slightly more plastic (up to +15%).
    - Concerned mood: slightly less plastic (down to -15%).
    - Neutral/no mood: 1.0.
    """

    if mood is None or mood.label == "neutral":
        return 1.0

    intensity = min(max(mood.intensity, 0.0), 1.0)
    if mood.label == "positive":
        return 1.0 + 0.15 * intensity
    if mood.label == "concerned":
        return max(0.75, 1.0 - 0.15 * intensity)
    return 1.0


def update_mood_state(current: Optional[MoodState], sentiment_score: Union[float, str] = 0.0) -> MoodState:
    """Update mood using BNN-derived sentiment and decay toward neutral.

    Args:
        current: Current mood state
        sentiment_score: Score from -1.0 (negative) to 1.0 (positive) from BNN embeddings
    
    Returns:
        Updated mood state
    """

    current = current or MoodState.neutral()

    score: float
    if isinstance(sentiment_score, str):
        text = sentiment_score.lower()
        positive = {
            "happy",
            "excited",
            "great",
            "awesome",
            "fantastic",
            "good",
            "love",
            "amazing",
        }
        negative = {
            "sad",
            "angry",
            "upset",
            "frustrated",
            "worried",
            "anxious",
            "bad",
            "hate",
        }
        pos_hits = sum(1 for w in positive if w in text)
        neg_hits = sum(1 for w in negative if w in text)
        if pos_hits == 0 and neg_hits == 0:
            score = 0.0
        else:
            score = (pos_hits - neg_hits) / float(pos_hits + neg_hits)
    else:
        score = float(sentiment_score)

    # Strong positive/negative signals from BNN embeddings (or heuristic text fallback)
    if score > 0.5:
        return MoodState(label="positive", emoji=_emoji_for_label("positive"), decay=current.decay, intensity=0.8)
    if score < -0.5:
        return MoodState(label="concerned", emoji=_emoji_for_label("concerned"), decay=current.decay, intensity=0.8)

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
