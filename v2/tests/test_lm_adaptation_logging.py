import os
import tempfile

import torch

from v2.lilith_v2.cognitive_stage import CognitiveStage
from v2.lilith_v2.communication_planner import CommunicationPlan
from v2.lilith_v2.compositional_realizer import RealizationResult
from v2.lilith_v2.pmflow_sqlite import SQLitePMFlowStateStore
from v2.lilith_v2.relational_graph_store import RelationalGraphStore


class MockEncoder:
    def encode(self, _data):
        return torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)


class DummyLanguageModel:
    def score_text(self, _text: str) -> float:
        return -0.42

    def generate_text(self, _prompt: str, max_new_tokens: int = 12) -> str:
        return "possible intent clarify request"


def _make_stage():
    db_fd, db_path = tempfile.mkstemp()
    os.close(db_fd)
    pm_fd, pm_path = tempfile.mkstemp()
    os.close(pm_fd)

    graph = RelationalGraphStore(db_path)
    pmflow = SQLitePMFlowStateStore(pm_path)

    stage = CognitiveStage(
        node_id="test.cognitive.lm.adapt",
        pmflow_store=pmflow,
        graph_store=graph,
        encoder=MockEncoder(),
        config={"conversational_mode": True, "knowledge_enabled": False, "enable_action_planning": False},
    )

    def sparse_plan(**_kwargs):
        return CommunicationPlan(primary_goal="clarify")

    def sparse_realize(plan, topic_context=None):
        return RealizationResult(text="generic response", frames_used=[], confidence=0.2)

    stage._communication_planner.plan = sparse_plan  # type: ignore[method-assign]
    stage._compositional_realizer.realize = sparse_realize  # type: ignore[method-assign]
    stage.linguistic.language_model = DummyLanguageModel()

    return stage, graph, pmflow, db_path, pm_path


def _cleanup_stage(graph, pmflow, db_path: str, pm_path: str) -> None:
    graph.close()
    pmflow.close()
    os.unlink(db_path)
    os.unlink(pm_path)


def test_lm_adaptation_event_logged_and_finalized_from_feedback():
    stage, graph, pmflow, db_path, pm_path = _make_stage()
    try:
        # Turn 1: weak parse triggers LM assist and logs a pending event.
        stage.learn("blorf glarn zqx")
        event_id = stage._pending_lm_feedback_event_id
        assert event_id is not None

        event = stage.graph.get_node(event_id)
        assert event is not None
        assert event["type"] == "lm_adaptation_event"
        assert event["data"]["status"] == "pending_feedback"

        # Turn 2: positive feedback should finalize event as accepted.
        stage.learn("thanks, that helps")
        assert stage._pending_lm_feedback_event_id is None

        finalized = stage.graph.get_node(event_id)
        assert finalized is not None
        assert finalized["data"]["status"] == "accepted"
        assert finalized["data"]["feedback"]["score"] > 0
    finally:
        _cleanup_stage(graph, pmflow, db_path, pm_path)
