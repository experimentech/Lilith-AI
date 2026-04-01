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


def _make_stage() -> CognitiveStage:
    db_fd, db_path = tempfile.mkstemp()
    os.close(db_fd)
    pm_fd, pm_path = tempfile.mkstemp()
    os.close(pm_fd)

    graph = RelationalGraphStore(db_path)
    pmflow = SQLitePMFlowStateStore(pm_path)

    stage = CognitiveStage(
        node_id="test.cognitive.lm",
        pmflow_store=pmflow,
        graph_store=graph,
        encoder=MockEncoder(),
        config={"conversational_mode": True, "knowledge_enabled": False, "enable_action_planning": False},
    )

    # Keep cleanup handles on object for concise tests.
    stage._test_db_path = db_path
    stage._test_pm_path = pm_path
    stage._test_graph = graph
    stage._test_pmflow = pmflow
    return stage


def _cleanup_stage(stage: CognitiveStage) -> None:
    stage._test_graph.close()
    stage._test_pmflow.close()
    os.unlink(stage._test_db_path)
    os.unlink(stage._test_pm_path)


def _force_sparse_conversation_path(stage: CognitiveStage) -> None:
    def sparse_plan(**_kwargs):
        return CommunicationPlan(primary_goal="clarify")

    def sparse_realize(plan, topic_context=None):
        return RealizationResult(text="generic response", frames_used=[], confidence=0.2)

    stage._communication_planner.plan = sparse_plan
    stage._compositional_realizer.realize = sparse_realize


def test_lm_assist_changes_final_response_for_weak_parse():
    stage = _make_stage()
    try:
        _force_sparse_conversation_path(stage)
        stage.linguistic.language_model = DummyLanguageModel()

        stage.learn("blorf glarn zqx")

        response = stage.last_thought.get("response", "")
        assert "I may be missing your intent" in response
        assert "possible intent clarify request" in response
        assert stage.last_thought.get("lm_assist_applied") is True
    finally:
        _cleanup_stage(stage)


def test_without_lm_assist_response_stays_generic_in_same_path():
    stage = _make_stage()
    try:
        _force_sparse_conversation_path(stage)
        stage.linguistic.language_model = None

        stage.learn("blorf glarn zqx")

        response = stage.last_thought.get("response", "")
        assert "I may be missing your intent" not in response
        assert stage.last_thought.get("lm_assist_applied") is False
    finally:
        _cleanup_stage(stage)
