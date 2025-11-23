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