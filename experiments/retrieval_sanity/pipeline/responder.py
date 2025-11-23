"""Conversation responder built on symbolic frames and retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .base import PipelineArtifact, SymbolicFrame
from .decoder import TemplateDecoder
from .storage_bridge import SymbolicStore
from .trace import TraceLogger
from .plasticity import PlasticityController, PlasticityReport
from .response_planner import PlanEvidence, ResponsePlan, ResponsePlanner
from .conversation_state import ConversationState, ConversationStateSnapshot


@dataclass
class Response:
    text: str
    recall_frame: Optional[SymbolicFrame]
    recall_score: Optional[float]
    nearest_labels: List[int] = field(default_factory=list)
    nearest_scores: List[float] = field(default_factory=list)
    plasticity: Optional[Dict[str, float]] = None
    working_memory: Optional[ConversationStateSnapshot] = None


class ConversationResponder:
    """Generate small-talk style responses backed by the symbol store."""

    def __init__(
        self,
        store: SymbolicStore,
        decoder: TemplateDecoder,
        *,
        min_score: float = 0.15,
        topk: int = 3,
        trace_logger: TraceLogger | None = None,
        plasticity_controller: PlasticityController | None = None,
        planner: ResponsePlanner | None = None,
        conversation_state: ConversationState | None = None,
    ) -> None:
        self.store = store
        self.decoder = decoder
        self.min_score = min_score
        self.topk = topk
        self.trace_logger = trace_logger
        self._log = logging.getLogger(__name__)
        self.plasticity_controller = plasticity_controller
        self.planner = planner or ResponsePlanner(recall_threshold=min_score)
        self.conversation_state = conversation_state

    def reply(self, artefact: PipelineArtifact) -> Response:
        snapshot = self._update_state(artefact)
        if self.store.vector_store.count(scenario=self.store.scenario) == 0:
            summary = self.decoder.generate(artefact.frame)
            plan = ResponsePlan(
                intent="seed",
                headline="I'm starting a fresh memory stack.",
                current_summary=summary,
                prompt="Keep describing the scenario so I can store more context.",
            )
            response = self._render_plan(plan, recall_frame=None, recall_score=None, state=snapshot)
            self._apply_plasticity(artefact, response)
            self._log_trace(artefact, response)
            return response

        scores, labels = self.store.vector_store.search(
            artefact.embedding, topk=self.topk, scenario=self.store.scenario
        )
        flat_scores = scores.squeeze(0)
        flat_labels = labels.squeeze(0)
        nearest_scores = flat_scores.tolist()
        nearest_labels = flat_labels.tolist()

        best_frame: Optional[SymbolicFrame] = None
        best_score: Optional[float] = None

        frames = self.store.load_frames()
        for label, score in zip(flat_labels.tolist(), flat_scores.tolist()):
            if score < self.min_score:
                continue
            candidate = _safe_get_frame(frames, label)
            if candidate is None:
                continue
            best_frame = candidate
            best_score = score
            break

        plan_nearest = []
        for lbl, score in zip(nearest_labels, nearest_scores):
            frame = _safe_get_frame(frames, lbl)
            summary = self.decoder.generate(frame) if frame is not None else ""
            plan_nearest.append(PlanEvidence(label=lbl, score=score, frame=frame, summary=summary))

        current_summary = self.decoder.generate(artefact.frame)

        if best_frame is None or best_score is None:
            plan = self.planner.plan(
                artefact,
                nearest=plan_nearest,
                current_summary=current_summary,
                state=snapshot,
            )
            response = self._render_plan(plan, recall_frame=None, recall_score=None, state=snapshot)
            self._apply_plasticity(artefact, response)
            self._log_trace(artefact, response)
            return response

        recall_summary = self.decoder.generate(best_frame)

        plan = self.planner.plan(
            artefact,
            nearest=plan_nearest,
            current_summary=current_summary,
            state=snapshot,
        )
        plan.headline = f"Closest memory: {recall_summary}"
        plan.notes.append(f"current:{current_summary}")
        response = self._render_plan(plan, recall_frame=best_frame, recall_score=best_score, state=snapshot)
        self._apply_plasticity(artefact, response)
        self._log_trace(artefact, response)
        return response

    def _render_plan(
        self,
        plan: ResponsePlan,
        *,
        recall_frame: SymbolicFrame | None,
        recall_score: float | None,
        state: ConversationStateSnapshot | None,
    ) -> Response:
        sentences: list[str] = []

        headline = plan.headline.strip()
        if recall_score is not None and plan.intent == "connect":
            headline = f"{headline} (score {recall_score:.2f})"
        if headline:
            sentences.append(_ensure_sentence(headline))

        if plan.current_summary:
            sentences.append(_ensure_sentence(f"I heard: {plan.current_summary}"))

        evidence_sentence = _summarise_evidences(plan.evidences)
        if evidence_sentence:
            sentences.append(evidence_sentence)

        for highlight in plan.memory_highlights:
            sentences.append(_ensure_sentence(highlight))

        if plan.prompt:
            sentences.append(_ensure_sentence(plan.prompt))

        text = " ".join(sentence.strip() for sentence in sentences if sentence)

        return Response(
            text=text,
            recall_frame=recall_frame,
            recall_score=recall_score,
            nearest_labels=[ev.label for ev in plan.evidences],
            nearest_scores=[ev.score for ev in plan.evidences],
            working_memory=state,
        )

    def _apply_plasticity(self, artefact: PipelineArtifact, response: Response) -> None:
        if not self.plasticity_controller:
            return
        trigger_score: float | None = float(response.recall_score) if response.recall_score is not None else None
        report: Optional[PlasticityReport] = self.plasticity_controller.maybe_update(
            artefact,
            recall_score=trigger_score,
        )
        if report:
            refreshed = self.store.refresh_embeddings(self.plasticity_controller.encoder)
            payload = report.as_dict()
            payload["refreshed"] = float(refreshed)
            response.plasticity = payload
            self.plasticity_controller.encoder.save_state()

    def _log_trace(self, artefact: PipelineArtifact, response: Response) -> None:
        if not self.trace_logger:
            return
        try:
            self.trace_logger.log(artefact, response)
        except Exception as exc:  # pragma: no cover - logging should not break flow
            self._log.warning("Failed to log trace record: %s", exc)

    def _update_state(self, artefact: PipelineArtifact) -> ConversationStateSnapshot | None:
        if not self.conversation_state:
            return None
        try:
            snapshot = self.conversation_state.update(artefact)
        except Exception as exc:  # pragma: no cover - state tracking is best-effort
            self._log.debug("Working memory update failed: %s", exc)
            return None
        return snapshot

def _safe_get_frame(frames: List[Dict[str, object]], label: int) -> SymbolicFrame | None:
    if label < 0 or label >= len(frames):
        return None
    frame_dict = frames[label]
    modifiers = {
        key.split(":", 1)[1]: value
        for key, value in frame_dict.items()
        if isinstance(key, str) and key.startswith("modifier:")
    }
    attributes = {
        key.split(":", 1)[1]: value
        for key, value in frame_dict.items()
        if isinstance(key, str) and key.startswith("attr:")
    }
    confidence_value = frame_dict.get("confidence", 0.0)
    confidence = _coerce_float(confidence_value)
    return SymbolicFrame(
        actor=_coerce_optional_str(frame_dict.get("actor")),
        action=_coerce_optional_str(frame_dict.get("action")),
        target=_coerce_optional_str(frame_dict.get("target")),
        modifiers={k: str(v) for k, v in modifiers.items()},
        attributes={k: str(v) for k, v in attributes.items()},
        confidence=confidence,
        raw_text=str(frame_dict.get("raw_text", "")),
        language=str(frame_dict.get("language", "unknown")),
    )


def _summarise_evidences(evidences: List[PlanEvidence]) -> str:
    if not evidences:
        return ""
    parts: list[str] = []
    for idx, evidence in enumerate(evidences[:3], start=1):
        if evidence.frame is None:
            continue
        summary = evidence.summary or _describe_brief(evidence.frame)
        parts.append(f"#{idx} {summary} (score {evidence.score:.2f})")
    if not parts:
        return ""
    return "Supporting memories: " + "; ".join(parts) + "."


def _describe_brief(frame: SymbolicFrame) -> str:
    actor = frame.actor or "Someone"
    action = frame.action or "did"
    target = frame.target
    if target:
        clause = f"{actor} {action} {target}".strip()
    else:
        clause = f"{actor} {action}".strip()
    return clause


def _ensure_sentence(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped[-1] not in ".?!":
        stripped += "."
    return stripped


def _coerce_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _coerce_optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)