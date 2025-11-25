"""Template-driven decoder that turns symbolic frames into text."""

from __future__ import annotations

from typing import Optional

from .base import SymbolicFrame


class TemplateDecoder:
    """Generate a textual summary from a symbolic frame.

    The decoder favours human-friendly sentences while keeping dependencies light.
    """

    def __init__(self, *, default_actor: str = "Someone") -> None:
        self.default_actor = default_actor

    def generate(self, frame: SymbolicFrame) -> str:
        language_hint = (frame.attributes.get("language_hint") or frame.language or "unknown").strip()
        main_sentence = self._build_sentence(frame)
        details_sentence = self._build_details(frame)
        confidence = f"confidence {frame.confidence:.2f}"

        parts = []
        if language_hint and language_hint.lower() != "unknown":
            parts.append(f"[{language_hint}]")
        parts.append(main_sentence)
        if details_sentence:
            parts.append(details_sentence)
        parts.append(f"({confidence})")
        return " ".join(parts).strip()

    def _build_sentence(self, frame: SymbolicFrame) -> str:
        raw = (frame.raw_text or "").strip()
        if raw:
            return self._normalise_raw(raw)

        actor = self._format_capitalised(frame.actor) or self.default_actor
        action = (frame.action or "noted").strip()
        target = (frame.target or "").strip()

        if target:
            clause = f"{actor} {action} {target}"
        else:
            clause = f"{actor} {action}".strip()
        return self._ensure_sentence(clause)

    def _build_details(self, frame: SymbolicFrame) -> str:
        details: list[str] = []

        tail = (frame.modifiers.get("tail") or "").strip()
        if tail:
            details.append(f"tail '{tail}'")

        tokens = (frame.attributes.get("token_count") or "").strip()
        if tokens:
            details.append(f"{tokens} tokens")

        location = (frame.attributes.get("location_token") or frame.modifiers.get("location") or "").strip()
        if location:
            details.append(f"at {location}")

        time_phrase = (frame.attributes.get("time_phrase") or "").strip()
        if time_phrase:
            details.append(f"around {time_phrase}")

        if frame.attributes.get("negated") == "true":
            details.append("negated statement")

        intent = (frame.attributes.get("intent") or frame.modifiers.get("intent") or "").strip()
        if intent:
            details.append(f"intent '{intent}'")

        if not details:
            return ""
        return "Details: " + ", ".join(details) + "."

    @staticmethod
    def _normalise_raw(text: str) -> str:
        cleaned = " ".join(text.replace("\n", " ").split())
        cleaned = cleaned.lstrip("\ufeff")
        return TemplateDecoder._ensure_sentence(cleaned)

    @staticmethod
    def _ensure_sentence(text: str) -> str:
        if not text:
            return ""
        stripped = text.strip()
        if not stripped:
            return ""
        first = stripped[0]
        if first.isalpha():
            stripped = first.upper() + stripped[1:]
        if stripped[-1] not in ".?!":
            stripped += "."
        return stripped

    @staticmethod
    def _format_capitalised(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return value[0].upper() + value[1:] if len(value) > 1 else value.upper()
