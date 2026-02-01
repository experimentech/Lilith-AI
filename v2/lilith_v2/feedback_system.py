"""
Feedback Detection System for Lilith v2.

Analyzes user input for implicit feedback (gratitude, confusion, correction)
to drive the Affective System and Reinforcement Learning loops.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional

class FeedbackSignal(Enum):
    STRONG_POSITIVE = "strong_positive"   # "Great!", "Exactly", "Thanks"
    WEAK_POSITIVE = "weak_positive"       # "Okay", "I see"
    NEUTRAL = "neutral"
    WEAK_NEGATIVE = "weak_negative"       # "Huh?", "What do you mean?"
    STRONG_NEGATIVE = "strong_negative"   # "No", "Wrong", "That's incorrect"

@dataclass
class FeedbackResult:
    signal: FeedbackSignal
    score: float                # -1.0 to 1.0 (for AffectiveSystem)
    confidence: float           # 0.0 to 1.0
    reason: str

class FeedbackDetector:
    """Detects implicit feedback in conversation."""
    
    def __init__(self):
        self.patterns = self._compile_patterns()

    def detect(self, text: str) -> FeedbackResult:
        text = text.strip().lower()
        
        # Check Strong Positive
        if any(p.search(text) for p in self.patterns['strong_positive']):
            return FeedbackResult(FeedbackSignal.STRONG_POSITIVE, 1.0, 0.9, "Strong positive keyword detected")
            
        # Check Strong Negative
        if any(p.search(text) for p in self.patterns['strong_negative']):
            return FeedbackResult(FeedbackSignal.STRONG_NEGATIVE, -1.0, 0.9, "Strong negative keyword detected")
            
        # Check Weak Positive
        if any(p.search(text) for p in self.patterns['weak_positive']):
             return FeedbackResult(FeedbackSignal.WEAK_POSITIVE, 0.3, 0.6, "Weak positive keyword detected")

        # Check Weak Negative
        if any(p.search(text) for p in self.patterns['weak_negative']):
             return FeedbackResult(FeedbackSignal.WEAK_NEGATIVE, -0.3, 0.6, "Weak negative keyword detected")

        return FeedbackResult(FeedbackSignal.NEUTRAL, 0.0, 0.0, "No feedback detected")

    def _compile_patterns(self) -> dict:
        return {
            'strong_positive': [
                re.compile(r'\b(thanks|thank you|thx)\b'),
                re.compile(r'\b(perfect|exactly|precisely|excellent|awesome|amazing)\b'),
                re.compile(r'\b(that\'?s? (right|correct))\b'),
                re.compile(r'\b(helpful|useful)\b'),
            ],
            'weak_positive': [
                re.compile(r'^(ok|okay|cool|nice)\b'),
                re.compile(r'\b(i see|got it|makes sense)\b'),
                re.compile(r'\b(interesting)\b'),
            ],
            'strong_negative': [
                re.compile(r'\b(wrong|incorrect|false)\b'),
                re.compile(r'\b(no[,.]?\s*that\'?s?\s*(not|wrong))\b'),
                re.compile(r'\b(useless|unhelpful)\b'),
            ],
            'weak_negative': [
                re.compile(r'^\s*(what|huh|eh)\s*\?+\s*$'),
                re.compile(r'\b(i don\'?t (get|understand))\b'),
                re.compile(r'\b(confused|confusing)\b'),
                re.compile(r'\b(what do you mean)\b'),
            ]
        }
