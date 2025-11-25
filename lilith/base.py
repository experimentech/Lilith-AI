"""Shared dataclasses and type definitions for the language-to-symbol pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class Utterance:
    """Raw user text plus optional metadata."""

    text: str
    language: str = "unknown"
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Token:
    """Minimal token representation after intake normalisation."""

    text: str
    position: int
    pos: str = "UNK"
    lemma: Optional[str] = None


@dataclass
class ParsedSentence:
    """Heuristic syntactic signal extracted from an utterance."""

    tokens: List[Token]
    subject_index: Optional[int] = None
    verb_index: Optional[int] = None
    object_index: Optional[int] = None
    modifiers: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    location_index: Optional[int] = None
    time_index: Optional[int] = None
    intent: Optional[str] = None
    negation_indices: List[int] = field(default_factory=list)
    role_map: Dict[str, int] = field(default_factory=dict)


@dataclass
class SymbolicFrame:
    """Language-agnostic tuple describing an action and its actors."""

    actor: Optional[str]
    action: Optional[str]
    target: Optional[str]
    modifiers: Dict[str, str]
    attributes: Dict[str, str]
    confidence: float
    raw_text: str
    language: str

    def as_dict(self) -> Dict[str, Optional[str]]:
        payload: Dict[str, Optional[str]] = {
            "actor": self.actor,
            "action": self.action,
            "target": self.target,
            "language": self.language,
            "raw_text": self.raw_text,
            "confidence": f"{self.confidence:.3f}",
        }
        for key, value in self.modifiers.items():
            payload[f"modifier:{key}"] = value
        for key, value in self.attributes.items():
            payload[f"attr:{key}"] = value
        return payload


@dataclass
class PipelineArtifact:
    """Bundle all intermediate artefacts for inspection and logging."""

    utterance: Utterance
    normalised_text: str
    candidates: List[str]
    parsed: ParsedSentence
    frame: SymbolicFrame
    embedding: torch.Tensor
    confidence: float