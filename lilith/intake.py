"""Basic normalisation and candidate generation for noisy text."""

from __future__ import annotations

import re
from typing import Iterable, List

from .base import Utterance


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _simple_fixups(text: str) -> str:
    replacements = {
        "teh": "the",
        "realy": "really",
        "dont": "don't",
        "cant": "can't",
        "fui": "fui",  # placeholder to keep mapping structure consistent
        "hospitol": "hospital",
        "recieve": "receive",
    }
    tokens = text.split()
    fixed = [replacements.get(token, token) for token in tokens]
    return " ".join(fixed)


class NoiseNormalizer:
    """Apply lightweight cleanup operations to utterances."""

    def __init__(self, *, extra_replacements: Iterable[tuple[str, str]] | None = None) -> None:
        self._extra = dict(extra_replacements or [])
        
        # Filler phrases to strip from queries (language-level preprocessing)
        self.filler_phrases = [
            "i mean", "i meant", "well", "so", "like", "um", "uh",
            "you know", "actually", "basically", "honestly", "literally",
            "just", "okay so", "ok so", "alright so", "anyway",
            "to be honest", "in fact", "the thing is"
        ]

    def normalise(self, utterance: Utterance) -> str:
        text = utterance.text.lower()
        text = _collapse_whitespace(text)
        text = _simple_fixups(text)
        if self._extra:
            for src, dst in self._extra.items():
                text = text.replace(src, dst)
        return text

    def generate_candidates(self, text: str) -> List[str]:
        """Return alternate spellings by applying single-character edits."""

        alphabet = "abcdefghijklmnopqrstuvwxyz"
        candidates = {text}
        chars = list(text)
        for idx, ch in enumerate(chars):
            if ch.isalpha():
                candidates.add("".join(chars[:idx] + [ch, ch] + chars[idx + 1 :]))
                candidates.add("".join(chars[:idx] + chars[idx + 1 :]))
        for idx in range(len(chars) + 1):
            for ch in alphabet[:3]:  # constrain for speed
                candidates.add("".join(chars[:idx] + [ch] + chars[idx:]))
        return sorted({c for c in candidates if c})
    
    def clean_query(self, query: str) -> str:
        """
        Clean a query by stripping filler phrases and discourse markers.
        
        This is LANGUAGE-LEVEL preprocessing that happens BEFORE symbolic processing.
        Resolves cases like "I mean, what can you do?" → "what can you do?"
        
        Args:
            query: Raw user query
            
        Returns:
            Cleaned query with filler phrases removed
        """
        cleaned = query.lower().strip()
        
        # Strip filler phrases from the beginning
        for filler in self.filler_phrases:
            if cleaned.startswith(filler):
                # Remove filler and any following comma/space
                cleaned = cleaned[len(filler):].lstrip(", ")
                
        # Also strip from middle (e.g., "what, like, can you do")
        for filler in self.filler_phrases:
            cleaned = cleaned.replace(f", {filler},", ",")
            cleaned = cleaned.replace(f", {filler} ", " ")
            
        # Restore original case pattern
        original_words = query.split()
        cleaned_words = cleaned.split()
        
        # If cleaned query is shorter, capitalize first letter
        if len(cleaned_words) < len(original_words):
            cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else cleaned
            
        return cleaned