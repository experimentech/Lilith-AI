"""
Affective System / Limbic System for Lilith v2.

This module models the internal emotional state and personality of the system.
It provides a "Subjective Context" that influences decision making,
memory retrieval, and response generation.

Concepts:
- Mood: Transient emotional state (Happy, Concerned, Curious).
- Personality: Stable traits that drift slowly (Openness, Conscientiousness).
- Drive: Internal motivations (Curiosity, Helpfulness).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class MoodState:
    valence: float = 0.0      # -1.0 (Negative) to +1.0 (Positive)
    arousal: float = 0.0      # 0.0 (Calm) to 1.0 (Excited/Alert)
    dominance: float = 0.5    # 0.0 (Submissive) to 1.0 (Dominant/Confident)
    label: str = "neutral"

    @property
    def vector(self) -> list[float]:
        return [self.valence, self.arousal, self.dominance]

    @classmethod
    def neutral(cls):
        return cls()

@dataclass
class PersonalityTraits:
    # Big Five (Ocean) or custom
    openness: float = 0.8
    conscientiousness: float = 0.7
    extraversion: float = 0.5
    agreeableness: float = 0.9
    neuroticism: float = 0.1
    
    # Lilith specific
    curiosity: float = 0.8
    warmth: float = 0.7

class AffectiveSystem:
    """
    Manages the 'Limbic' state of the agent.
    
    This system runs in parallel to the Cognitive Stage, maintaining
    a running 'mood' based on interaction outcomes (Success/Failure/Feedback).
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.mood = MoodState.neutral()
        self.personality = PersonalityTraits()
        self.decay_rate = config.get("decay_rate", 0.95)

    def update(self, feedback: float, context: Dict = None):
        """
        Update mood based on feedback (Success/Failure).
        
        Args:
            feedback: -1.0 (Failure/Correction) to +1.0 (Success/Praise)
        """
        # Simple rudimentary dynamics
        # Success increases Valence and Dominance (Confidence)
        target_valence = 0.0
        if feedback > 0:
            target_valence = min(self.mood.valence + (0.2 * feedback), 1.0)
            self.mood.dominance = min(self.mood.dominance + 0.05, 1.0)
        elif feedback < 0:
            target_valence = max(self.mood.valence + (0.3 * feedback), -1.0)
            self.mood.dominance = max(self.mood.dominance - 0.05, 0.0) # Less confident on error

        # Decay towards neutral/personality baseline
        self.mood.valence = (self.mood.valence * self.decay_rate) + (target_valence * (1-self.decay_rate))
        
        # Update label
        self._update_label()

    def _update_label(self):
        v = self.mood.valence
        a = self.mood.arousal
        if v > 0.3:
            self.mood.label = "happy" if a > 0.5 else "content"
        elif v < -0.3:
            self.mood.label = "concerned" if a > 0.5 else "sad"
        else:
            self.mood.label = "neutral"

    def get_state_vector(self) -> Dict[str, float]:
        """Returns a flat dictionary of current affective state for modulating other systems."""
        return {
            "mood_valence": self.mood.valence,
            "mood_arousal": self.mood.arousal,
            "confidence": self.mood.dominance,
            "curiosity_drive": self.personality.curiosity
        }
