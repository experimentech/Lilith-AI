"""Build symbolic frames from heuristic parses."""

from __future__ import annotations

from typing import Optional

from .base import ParsedSentence, SymbolicFrame, Utterance


def _safe_token(parsed: ParsedSentence, index: Optional[int]) -> Optional[str]:
    if index is None:
        return None
    if 0 <= index < len(parsed.tokens):
        return parsed.tokens[index].text
    return None


def build_frame(utterance: Utterance, parsed: ParsedSentence, normalised_text: str) -> SymbolicFrame:
    actor = _safe_token(parsed, parsed.subject_index)
    action = _safe_token(parsed, parsed.verb_index)
    target = _safe_token(parsed, parsed.object_index)

    modifiers = dict(parsed.modifiers)

    attributes = {
        "token_count": str(len(parsed.tokens)),
        "language_hint": utterance.language,
        "subject_present": str(parsed.subject_index is not None),
        "verb_present": str(parsed.verb_index is not None),
        "object_present": str(parsed.object_index is not None),
    }

    if parsed.location_index is not None and 0 <= parsed.location_index < len(parsed.tokens):
        attributes["location_token"] = parsed.tokens[parsed.location_index].text
    if parsed.time_index is not None and 0 <= parsed.time_index < len(parsed.tokens):
        attributes["time_token"] = parsed.tokens[parsed.time_index].text
    time_phrase = parsed.modifiers.get("time_phrase")
    if time_phrase:
        attributes["time_phrase"] = time_phrase
    if parsed.intent:
        attributes["intent"] = parsed.intent
    if parsed.negation_indices:
        attributes["negated"] = "true"
    for role, index in parsed.role_map.items():
        attributes[f"role_{role}"] = str(index)

    return SymbolicFrame(
        actor=actor,
        action=action,
        target=target,
        modifiers=modifiers,
        attributes=attributes,
        confidence=parsed.confidence,
        raw_text=normalised_text,
        language=utterance.language,
    )