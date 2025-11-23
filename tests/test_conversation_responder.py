"""Unit tests for ConversationResponder metadata useful for plasticity hooks."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.retrieval_sanity.pipeline.base import Utterance
from experiments.retrieval_sanity.pipeline.decoder import TemplateDecoder
from experiments.retrieval_sanity.pipeline.pipeline import SymbolicPipeline
from experiments.retrieval_sanity.pipeline.responder import ConversationResponder
from experiments.retrieval_sanity.pipeline.storage_bridge import SymbolicStore
from experiments.retrieval_sanity.pipeline.trace import TraceLogger
from experiments.retrieval_sanity.pipeline.embedding import PMFlowEmbeddingEncoder
from experiments.retrieval_sanity.pipeline.conversation_state import ConversationState


def test_responder_returns_nearest_metadata(tmp_path) -> None:
    sqlite_path = Path(tmp_path) / "vectors.db"
    store = SymbolicStore(sqlite_path, scenario="unit")
    decoder = TemplateDecoder()
    pipeline = SymbolicPipeline(use_pmflow=False)
    conversation_state = ConversationState(
        pipeline.encoder if isinstance(pipeline.encoder, PMFlowEmbeddingEncoder) else None
    )
    responder = ConversationResponder(store, decoder, topk=2, conversation_state=conversation_state)

    first = pipeline.process(Utterance(text="We visited the hospital today", language="en"))
    first_response = responder.reply(first)
    assert first_response.nearest_labels == []
    assert first_response.nearest_scores == []
    assert first_response.plasticity is None
    assert first_response.working_memory is not None
    assert first_response.working_memory.active is False
    assert "starting a fresh" in first_response.text.lower()
    assert "I heard:" in first_response.text

    store.persist([first])

    second = pipeline.process(Utterance(text="We will visit the hospital again tomorrow", language="en"))
    second_response = responder.reply(second)

    assert second_response.nearest_labels
    assert len(second_response.nearest_labels) == len(second_response.nearest_scores)
    assert second_response.nearest_labels[0] == 0
    assert second_response.recall_frame is not None
    assert second_response.recall_score is not None
    assert second_response.plasticity is None
    assert second_response.working_memory is not None
    assert second_response.working_memory.active is False
    assert "Closest memory" in second_response.text
    assert "I heard:" in second_response.text
    assert "Supporting memories" in second_response.text


def test_trace_logger_writes_jsonl(tmp_path) -> None:
    sqlite_path = Path(tmp_path) / "vectors.db"
    trace_path = Path(tmp_path) / "trace.jsonl"

    store = SymbolicStore(sqlite_path, scenario="unit")
    decoder = TemplateDecoder()
    pipeline = SymbolicPipeline(use_pmflow=False)
    trace_logger = TraceLogger(trace_path, scenario="unit")
    conversation_state = ConversationState(
        pipeline.encoder if isinstance(pipeline.encoder, PMFlowEmbeddingEncoder) else None
    )
    responder = ConversationResponder(
        store, decoder, topk=2, trace_logger=trace_logger, conversation_state=conversation_state
    )

    first = pipeline.process(Utterance(text="I need to visit the clinic today", language="en"))
    responder.reply(first)
    assert trace_path.exists()
    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    first_record = json.loads(lines[0])
    assert first_record["scenario"] == "unit"
    assert first_record["recall_score"] is None
    assert first_record["nearest"] == []
    assert first_record["plasticity"] is None
    assert first_record["working_memory"] is not None
    assert first_record["working_memory"]["active"] is False

    store.persist([first])
    second = pipeline.process(Utterance(text="We're going back tomorrow", language="en"))
    responder.reply(second)
    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    second_record = json.loads(lines[-1])
    assert second_record["recall_score"] is not None
    assert second_record["nearest"]
    assert second_record["nearest"][0]["label"] == 0
    assert second_record["plasticity"] is None
    assert second_record["working_memory"] is not None
    assert second_record["working_memory"]["active"] is False
