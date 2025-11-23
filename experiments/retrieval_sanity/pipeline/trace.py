"""Trace logging utilities for conversational plasticity analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import PipelineArtifact

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from .responder import Response


ISO8601 = "%Y-%m-%dT%H:%M:%S.%fZ"


@dataclass
class TraceRecord:
    """Structured payload capturing a single conversational turn."""

    timestamp: str
    scenario: str
    utterance: str
    normalised: str
    language: str
    confidence: float
    frame: Dict[str, Any]
    recall_score: Optional[float]
    recall_actor: Optional[str]
    recall_action: Optional[str]
    recall_target: Optional[str]
    nearest: List[Dict[str, Any]]
    plasticity: Optional[Dict[str, float]]
    working_memory: Optional[Dict[str, Any]]

    @classmethod
    def from_artifact(
        cls,
        *,
        scenario: str,
        artefact: PipelineArtifact,
        response: "Response",
    ) -> "TraceRecord":
        timestamp = datetime.now(timezone.utc).strftime(ISO8601)
        frame_dict = artefact.frame.as_dict()
        nearest = [
            {
                "label": label,
                "score": score,
            }
            for label, score in zip(response.nearest_labels, response.nearest_scores)
        ]
        recall_frame = response.recall_frame
        working_memory_payload: Optional[Dict[str, Any]] = None
        if response.working_memory is not None:
            working_memory_payload = {
                "active": response.working_memory.active,
                "activation_energy": response.working_memory.activation_energy,
                "novelty": response.working_memory.novelty,
                "topics": [
                    {
                        "signature": list(topic.signature),
                        "strength": topic.strength,
                        "summary": topic.summary,
                        "mentions": topic.mentions,
                        "last_strength": topic.last_strength,
                    }
                    for topic in response.working_memory.topics
                ],
            }
        return cls(
            timestamp=timestamp,
            scenario=scenario,
            utterance=artefact.utterance.text,
            normalised=artefact.normalised_text,
            language=artefact.utterance.language,
            confidence=float(artefact.confidence),
            frame=frame_dict,
            recall_score=response.recall_score,
            recall_actor=recall_frame.actor if recall_frame else None,
            recall_action=recall_frame.action if recall_frame else None,
            recall_target=recall_frame.target if recall_frame else None,
            nearest=nearest,
            plasticity=response.plasticity,
            working_memory=working_memory_payload,
        )


class TraceLogger:
    """Append structured JSONL trace records for each responder turn."""

    def __init__(self, path: Path, *, scenario: str) -> None:
        self.path = path
        self.scenario = scenario
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, artefact: PipelineArtifact, response: "Response") -> None:
        record = TraceRecord.from_artifact(scenario=self.scenario, artefact=artefact, response=response)
        payload = json.dumps(asdict(record), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(payload + "\n")