"""Heuristic parser that extracts coarse syntactic roles."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .base import ParsedSentence, Token

# Basic lexicons to guide POS tagging.
PRONOUNS = {
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "me",
    "him",
    "her",
    "nosotros",
    "ellos",
    "ellas",
    "yo",
    "tu",
    "usted",
    "ustedes",
    "nosotras",
    "ellos",
}

COMMON_VERBS = {
    "am",
    "are",
    "is",
    "was",
    "were",
    "be",
    "go",
    "going",
    "went",
    "see",
    "saw",
    "have",
    "has",
    "need",
    "want",
    "visit",
    "visited",
    "schedule",
    "ask",
    "feel",
    "ver",
    "fui",
    "ir",
    "tener",
    "necesitar",
    "querer",
    "visitar",
}

COMMON_NOUNS = {
    "hospital",
    "clinic",
    "mother",
    "padre",
    "madre",
    "doctor",
    "doctora",
    "cita",
    "appointment",
    "cita",
    "medico",
    "salud",
    "family",
    "son",
    "daughter",
    "time",
    "mom",
    "dad",
    "week",
    "month",
}

QUESTION_WORDS = {
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "cual",
    "cuando",
    "donde",
    "por",
    "por que",
    "como",
}

NEGATIONS = {"not", "never", "no", "nunca"}

LOCATION_WORDS = {
    "hospital",
    "clinic",
    "consultorio",
    "home",
    "casa",
    "office",
    "centro",
}

LOCATION_PREPOSITIONS = {"to", "at", "in", "en", "al", "del", "para"}

TIME_WORDS = {
    "today",
    "tomorrow",
    "yesterday",
    "tonight",
    "morning",
    "afternoon",
    "evening",
    "soon",
    "later",
    "next",
    "week",
    "month",
    "year",
    "mañana",
    "ayer",
    "pronto",
    "luego",
    "tarde",
}

TIME_PHRASES = {
    ("next", "week"),
    ("next", "month"),
    ("esta", "semana"),
    ("proxima", "semana"),
    ("esta", "tarde"),
    ("esta", "noche"),
    ("next", "appointment"),
}

TIME_LEAD_WORDS = {"next", "esta", "proxima", "last"}

AUXILIARY_VERBS = {"do", "did", "does", "have", "has", "had", "estar", "ser"}

TOKEN_SPLIT_RE = re.compile(r"[\w']+", re.UNICODE)


def _tokenise(text: str) -> List[str]:
    return TOKEN_SPLIT_RE.findall(text)


def _assign_pos(token: str) -> str:
    if token in PRONOUNS:
        return "PRON"
    if token in COMMON_VERBS or token.endswith("ing") or token.endswith("ed"):
        return "VERB"
    if token in COMMON_NOUNS or token.endswith("tion") or token.endswith("ment") or token.endswith("dad"):
        return "NOUN"
    if token in QUESTION_WORDS:
        return "ADV"
    if token.isdigit():
        return "NUM"
    return "UNK"


def _lemmatise(token: str, pos: str) -> str:
    if pos == "VERB":
        for suffix in ("ing", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                return token[: -len(suffix)]
        if token in {"went"}:
            return "go"
        if token in {"saw"}:
            return "see"
        if token in {"fui"}:
            return "ir"
    if pos == "NOUN":
        for suffix in ("s", "es"):
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                return token[: -len(suffix)]
    if pos == "ADV" and token in QUESTION_WORDS:
        return token
    return token
def _find_first(tokens: List[Token], target_pos: str) -> Optional[int]:
    for token in tokens:
        if token.pos == target_pos:
            return token.position
    return None


def parse(text: str) -> ParsedSentence:
    raw_tokens = _tokenise(text)
    tokens: List[Token] = []
    recognised = 0
    role_map: Dict[str, int] = {}
    negation_indices: List[int] = []
    location_index: Optional[int] = None
    time_index: Optional[int] = None
    time_phrase: Optional[str] = None
    for idx, tok in enumerate(raw_tokens):
        normalised = tok.lower()
        pos = _assign_pos(normalised)
        lemma = _lemmatise(normalised, pos)
        tokens.append(Token(text=normalised, position=idx, pos=pos, lemma=lemma))
        if pos != "UNK":
            recognised += 1
        if normalised in NEGATIONS:
            negation_indices.append(idx)
        if normalised in LOCATION_WORDS:
            location_index = idx
        if time_index is None and len(tokens) >= 2:
            bigram = (tokens[-2].text, tokens[-1].text)
            if bigram in TIME_PHRASES:
                time_index = tokens[-2].position
                time_phrase = " ".join(bigram)
        if normalised in TIME_WORDS and time_index is None and normalised not in TIME_LEAD_WORDS:
            time_index = idx
            time_phrase = normalised

    subject_idx = _find_first(tokens, "PRON")
    if subject_idx is None:
        for token in tokens:
            if token.pos == "NOUN":
                subject_idx = token.position
                break
    verb_idx: Optional[int] = None
    for token in tokens:
        if token.pos == "VERB" and token.text not in AUXILIARY_VERBS:
            verb_idx = token.position
            break
    if verb_idx is None:
        verb_idx = _find_first(tokens, "VERB")

    object_idx: Optional[int] = None
    if verb_idx is not None:
        for token in tokens[verb_idx + 1 :]:
            if token.pos in {"NOUN", "PRON"}:
                object_idx = token.position
                break
            if token.pos == "UNK" and token.text in LOCATION_WORDS:
                object_idx = token.position
                break

    modifiers = {}
    if tokens:
        tail_token = tokens[-1]
        if tail_token.pos == "UNK" and tail_token.text not in LOCATION_WORDS:
            modifiers["tail"] = tail_token.text

    # Detect question or intent phrase.
    intent: Optional[str] = None
    stripped = text.strip().lower()
    if stripped.endswith("?"):
        intent = "polar-question"
    if tokens:
        first_token = tokens[0].text
        if first_token in QUESTION_WORDS:
            intent = first_token
    if intent:
        modifiers["intent"] = intent

    if negation_indices:
        modifiers["negated"] = "true"

    if location_index is not None:
        modifiers["location"] = tokens[location_index].text
        if location_index > 0:
            prev = tokens[location_index - 1].text
            if prev in LOCATION_PREPOSITIONS:
                modifiers["location_prep"] = prev

    if time_phrase:
        modifiers["time_phrase"] = time_phrase
    elif time_index is not None:
        modifiers["time_phrase"] = tokens[time_index].text

    confidence = 0.0
    if tokens:
        confidence = recognised / len(tokens)
    if subject_idx is not None:
        confidence += 0.1
    if verb_idx is not None:
        confidence += 0.1
    if object_idx is not None:
        confidence += 0.05
    if location_index is not None:
        confidence += 0.05
    if time_index is not None:
        confidence += 0.05
    if intent:
        confidence += 0.05
    if negation_indices:
        confidence -= 0.05
    confidence = max(0.0, min(1.0, confidence))

    role_map["subject"] = subject_idx if subject_idx is not None else -1
    role_map["verb"] = verb_idx if verb_idx is not None else -1
    role_map["object"] = object_idx if object_idx is not None else -1
    if location_index is not None:
        role_map["location"] = location_index
    if time_index is not None:
        role_map["time"] = time_index

    filtered_roles = {role: idx for role, idx in role_map.items() if idx is not None and idx >= 0}

    return ParsedSentence(
        tokens=tokens,
        subject_index=subject_idx,
        verb_index=verb_idx,
        object_index=object_idx,
        modifiers=modifiers,
        confidence=confidence,
        location_index=location_index,
        time_index=time_index,
        intent=intent,
        negation_indices=negation_indices,
        role_map=filtered_roles,
    )