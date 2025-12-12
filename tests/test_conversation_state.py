from __future__ import annotations

import pytest

from experiments.retrieval_sanity.pipeline.base import Utterance
from experiments.retrieval_sanity.pipeline.conversation_state import ConversationState
from experiments.retrieval_sanity.pipeline.embedding import PMFlowEmbeddingEncoder
from experiments.retrieval_sanity.pipeline.pipeline import SymbolicPipeline


def test_conversation_state_inactive_without_pmflow() -> None:
    pipeline = SymbolicPipeline(use_pmflow=False)
    state = ConversationState(None)
    artefact = pipeline.process(Utterance(text="we visited the hospital", language="en"))

    snapshot = state.update(artefact)

    assert snapshot.active is False
    assert snapshot.topics == []
    assert snapshot.novelty == 0.0


def test_conversation_state_tracks_topics() -> None:
    pytest.importorskip("pmflow.pmflow", reason="pmflow runtime is required")

    pipeline = SymbolicPipeline()
    assert isinstance(pipeline.encoder, PMFlowEmbeddingEncoder)
    state = ConversationState(pipeline.encoder, decay=0.9)

    first = pipeline.process(Utterance(text="we visited the hospital", language="en"))
    first_snapshot = state.update(first)
    assert first_snapshot.active is True
    assert first_snapshot.topics
    dominant_first = first_snapshot.dominant
    assert dominant_first is not None
    assert dominant_first.mentions == 1

    second = pipeline.process(Utterance(text="we visited the hospital again", language="en"))
    second_snapshot = state.update(second)
    dominant_second = second_snapshot.dominant
    assert dominant_second is not None
    assert dominant_second.summary
    assert len(second_snapshot.topics) <= 5
    assert dominant_second.mentions >= 2
    assert second_snapshot.novelty <= 0.5
