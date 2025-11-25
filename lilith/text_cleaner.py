"""Utility helpers to sanitise plain text sources before feeding the pipeline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator, Sequence

__all__ = [
    "clean_lines",
]


_DROP_PATTERNS = (
    "*** PROJECT GUTENBERG",
    "END OF THE PROJECT GUTENBERG",
    "START OF THIS PROJECT GUTENBERG",
    "End of the Project Gutenberg",
    "Produced by",
    "Transcriber's Note",
)

_HORIZONTAL_RULE = re.compile(r"^[\-=*_~]{3,}$")
_MULTI_WHITESPACE = re.compile(r"\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NON_LETTER = re.compile(r"[^A-Za-z0-9\s']")


def _normalise_whitespace(text: str) -> str:
    return _MULTI_WHITESPACE.sub(" ", text.strip())


def _looks_noisy(text: str, drop_patterns: Sequence[str], min_alpha_ratio: float, min_tokens: int) -> bool:
    if not text.strip():
        return True
    if any(pattern.lower() in text.lower() for pattern in drop_patterns):
        return True
    if _HORIZONTAL_RULE.match(text.strip()):
        return True
    tokens = text.split()
    if len(tokens) < min_tokens:
        return True
    letters = sum(ch.isalpha() for ch in text)
    if not text:
        return True
    alpha_ratio = letters / max(len(text), 1)
    if alpha_ratio < min_alpha_ratio:
        return True
    return False


def _split_sentences(text: str, *, sentence_split: re.Pattern[str]) -> list[str]:
    sentences: list[str] = []
    for fragment in sentence_split.split(text):
        fragment = fragment.strip()
        if fragment:
            sentences.append(fragment)
    if not sentences:
        sentences = [text.strip()]
    return sentences


def clean_lines(
    path: Path,
    *,
    chunk_sentences: bool = True,
    min_tokens: int = 5,
    min_alpha_ratio: float = 0.5,
    drop_patterns: Sequence[str] | None = None,
    skip_headers: bool = True,
    lowercase: bool = False,
    max_tokens: int | None = None,
) -> Iterator[str]:
    """Yield sanitised lines or sentences from ``path``.

    Parameters
    ----------
    path:
        Plain-text or JSONL-like file containing raw text.
    chunk_sentences:
        Split each clean line into sentences heuristically. If False, yield the
        entire line/paragraph as-is.
    min_tokens:
        Minimum number of whitespace-delimited tokens required for a line to be
        considered.
    min_alpha_ratio:
        Reject lines whose proportion of alphabetic characters falls below this
        threshold (helps filter ASCII art, tables, etc.).
    drop_patterns:
        Additional substrings that should cause a line to be skipped when
        matched case-insensitively.
    skip_headers:
        When True, attempt to skip the Project Gutenberg header/footer regions
        by detecting the standard START/END markers.
    lowercase:
        Lowercase the output sentences when True.
    max_tokens:
        If provided, truncate sentences exceeding this many tokens.
    """

    drop_patterns = tuple(drop_patterns or ()) + _DROP_PATTERNS

    text = path.read_text(encoding="utf-8", errors="ignore")

    if skip_headers:
        start_idx = text.lower().find("*** start of this project gutenberg")
        end_idx = text.lower().find("*** end of this project gutenberg")
        if start_idx != -1:
            text = text[start_idx:]
        if end_idx != -1:
            text = text[: end_idx + 1]

    for raw_line in text.splitlines():
        normalised = _normalise_whitespace(raw_line)
        if _looks_noisy(normalised, drop_patterns, min_alpha_ratio, min_tokens):
            continue
        if chunk_sentences:
            candidates = _split_sentences(normalised, sentence_split=_SENTENCE_SPLIT)
        else:
            candidates = [normalised]
        for candidate in candidates:
            candidate = _normalise_whitespace(candidate)
            if _looks_noisy(candidate, drop_patterns, min_alpha_ratio, min_tokens):
                continue
            tokens = candidate.split()
            if max_tokens and len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                candidate = " ".join(tokens)
            if lowercase:
                candidate = candidate.lower()
            yield candidate