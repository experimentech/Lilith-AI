"""
Concept property sanitizer to keep ConceptStore clean (length caps, dedupe, term mention).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def sanitize_properties(
    term: str,
    properties: Iterable[str],
    max_len: int = 200,
    require_term: bool = True,
    dedupe: bool = True,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Clean a property list: cap length, optional term mention requirement, optional dedupe.

    Returns sanitized list and stats dict.
    """
    term_norm = _normalize(term)
    seen = set()
    kept: List[str] = []
    stats = {"removed_long": 0, "removed_term": 0, "removed_dupe": 0}

    for prop in properties:
        if prop is None:
            continue
        raw = str(prop).strip()
        if not raw:
            continue
        if len(raw) > max_len:
            stats["removed_long"] += 1
            continue
        norm = _normalize(raw)
        if require_term and term_norm and term_norm not in norm:
            stats["removed_term"] += 1
            continue
        if dedupe:
            key = norm
            if key in seen:
                stats["removed_dupe"] += 1
                continue
            seen.add(key)
        kept.append(raw)

    return kept, stats


__all__ = ["sanitize_properties"]
