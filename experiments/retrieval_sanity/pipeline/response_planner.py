"""Rule-based response planning for the symbol pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from .base import PipelineArtifact, SymbolicFrame
from .conversation_state import ConversationStateSnapshot


@dataclass
class PlanEvidence:
    label: int
    score: float
    frame: SymbolicFrame | None
    summary: str = ""


@dataclass
class ResponsePlan:
    """Intermediate plan describing how to respond."""

    intent: str
    headline: str
    current_summary: str
    prompt: str
    evidences: List[PlanEvidence] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    memory_highlights: List[str] = field(default_factory=list)


class ResponsePlanner:
    """Lightweight rule engine that maps artefacts and memories to a response plan."""

    def __init__(self, *, recall_threshold: float = 0.0) -> None:
        self.recall_threshold = recall_threshold

    def plan(
        self,
        artefact: PipelineArtifact,
        *,
        nearest: Iterable[PlanEvidence],
        current_summary: str,
        state: ConversationStateSnapshot | None = None,
    ) -> ResponsePlan:
        evidences = list(nearest)
        filtered = [ev for ev in evidences if ev.score >= self.recall_threshold and ev.frame is not None]

        if not evidences:
            notes = ["store-empty"]
            notes.extend(self._build_state_notes(state))
            return ResponsePlan(
                intent="seed",
                headline="No memories stored yet.",
                current_summary=current_summary,
                prompt="Please keep sharing so I can build a baseline.",
                evidences=evidences,
                notes=notes,
                memory_highlights=self._build_memory_highlights(state),
            )

        if filtered:
            top = filtered[0]
            intent = "connect"
            headline = top.summary or _describe_frame(top.frame)
            prompt = "What else would you like me to remember related to this?"
        else:
            intent = "catalogue"
            headline = "Couldn't find a confident match yet."
            prompt = "Give me another hint so I can link this to existing memories."

        notes: List[str] = []
        if len(filtered) > 1:
            notes.append("multiple-support")
        if not filtered and evidences:
            notes.append("low-score")
        memory_highlights = self._build_memory_highlights(state)
        notes.extend(self._build_state_notes(state))

        return ResponsePlan(
            intent=intent,
            headline=headline,
            current_summary=current_summary,
            prompt=prompt,
            evidences=evidences,
            notes=notes,
            memory_highlights=memory_highlights,
        )

    def _build_memory_highlights(self, state: ConversationStateSnapshot | None) -> List[str]:
        if not state or not state.active:
            return []
        dominant = state.dominant
        highlights: List[str] = []
        if dominant:
            highlights.append(
                f"My working memory is anchored on '{dominant.summary}' after {dominant.mentions} mentions."
            )
        if state.novelty >= 0.45:
            highlights.append("This turn feels like a new direction compared to the previous one.")
        return highlights

    def _build_state_notes(self, state: ConversationStateSnapshot | None) -> List[str]:
        if not state or not state.active:
            return []
        notes: List[str] = []
        dominant = state.dominant
        if dominant:
            signature_note = "-".join(str(idx) for idx in dominant.signature[:3])
            notes.append(f"wm-dominant:{signature_note}")
        if state.novelty >= 0.45:
            notes.append("novelty-high")
        return notes


def _describe_frame(frame: SymbolicFrame | None) -> str:
    if frame is None:
        return "Unknown detail."
    actor = frame.actor or "Someone"
    action = frame.action or "did something"
    target = frame.target
    if target:
        clause = f"{actor} {action} {target}"
    else:
        clause = f"{actor} {action}"
    clause = clause.strip()
    if not clause.endswith("."):
        clause += "."
    return clause
