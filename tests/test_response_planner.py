from __future__ import annotations

import torch

from experiments.retrieval_sanity.pipeline.base import (
    PipelineArtifact,
    ParsedSentence,
    SymbolicFrame,
    Token,
    Utterance,
)
from experiments.retrieval_sanity.pipeline.response_planner import (
    PlanEvidence,
    ResponsePlanner,
)
from experiments.retrieval_sanity.pipeline.conversation_state import (
    ConversationStateSnapshot,
    WorkingMemoryTopic,
)


def make_frame(actor: str | None = "alice", action: str | None = "reads", target: str | None = "books") -> SymbolicFrame:
    return SymbolicFrame(
        actor=actor,
        action=action,
        target=target,
        modifiers={},
        attributes={},
        confidence=0.42,
        raw_text=f"{actor or 'someone'} {action or ''} {target or ''}",
        language="en",
    )


def make_artifact(frame: SymbolicFrame) -> PipelineArtifact:
    utterance = Utterance(text=frame.raw_text, language=frame.language)
    parsed = ParsedSentence(tokens=[Token(text="stub", position=0)], confidence=frame.confidence)
    return PipelineArtifact(
        utterance=utterance,
        normalised_text=frame.raw_text,
        candidates=[],
        parsed=parsed,
        frame=frame,
        embedding=torch.zeros(1, 1),
        confidence=frame.confidence,
    )


def test_plan_when_store_empty() -> None:
    planner = ResponsePlanner()
    frame = make_frame("narrator", "observes", "the stars")
    artefact = make_artifact(frame)
    plan = planner.plan(artefact, nearest=[], current_summary="Someone observes the stars.")

    assert plan.intent == "seed"
    assert plan.headline == "No memories stored yet."
    assert plan.current_summary == "Someone observes the stars."
    assert "store-empty" in plan.notes


def test_plan_with_valid_evidence() -> None:
    planner = ResponsePlanner(recall_threshold=0.5)
    frame = make_frame()
    evidences = [PlanEvidence(label=0, score=0.8, frame=frame, summary="Alice reads books.")]

    artefact = make_artifact(make_frame("bob", "mentions", "cthulhu"))
    plan = planner.plan(
        artefact,
        nearest=evidences,
        current_summary="Bob mentions cthulhu.",
    )

    assert plan.intent == "connect"
    assert plan.headline.startswith("Alice")
    assert plan.current_summary.startswith("Bob")
    assert plan.prompt.startswith("What else")


def test_plan_with_low_scores() -> None:
    planner = ResponsePlanner(recall_threshold=0.6)
    evidences = [PlanEvidence(label=1, score=0.4, frame=make_frame())]

    artefact = make_artifact(make_frame("carter", "notes", "the tablet"))
    plan = planner.plan(
        artefact,
        nearest=evidences,
        current_summary="Carter notes the tablet.",
    )

    assert plan.intent == "catalogue"
    assert "Give me another hint" in plan.prompt
    assert "low-score" in plan.notes


def test_plan_uses_conversation_state() -> None:
    planner = ResponsePlanner()
    artefact = make_artifact(make_frame("dana", "mentions", "the lighthouse"))
    dominant = WorkingMemoryTopic(
        signature=(1, 3, 5),
        strength=1.2,
        summary="dana mentions the lighthouse",
        mentions=2,
        last_strength=1.2,
    )
    snapshot = ConversationStateSnapshot(
        active=True,
        activation_energy=1.2,
        novelty=0.6,
        topics=[dominant],
    )

    plan = planner.plan(artefact, nearest=[], current_summary="Dana mentions the lighthouse.", state=snapshot)

    assert "store-empty" in plan.notes
    assert any(note.startswith("wm-dominant") for note in plan.notes)
    assert "novelty-high" in plan.notes
    assert plan.memory_highlights
    assert any("working memory" in highlight.lower() for highlight in plan.memory_highlights)
