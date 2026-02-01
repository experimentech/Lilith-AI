import pytest
import torch
import tempfile
import os
from unittest.mock import MagicMock
from v2.lilith_v2.cognitive_stage import CognitiveStage
from v2.lilith_v2.relational_graph_store import RelationalGraphStore
from v2.lilith_v2.pmflow_sqlite import SQLitePMFlowStateStore

class MockEncoder:
    def encode(self, data):
        # Always return a fixed vector for testing
        return torch.tensor([0.1, 0.2, 0.3])

@pytest.fixture
def stage_stack():
    # Setup temporary stores
    db_fd, db_path = tempfile.mkstemp()
    os.close(db_fd)
    
    pmflow_fd, pmflow_path = tempfile.mkstemp()
    os.close(pmflow_fd)

    graph = RelationalGraphStore(db_path)
    pmflow = SQLitePMFlowStateStore(pmflow_path)
    encoder = MockEncoder()

    stage = CognitiveStage(
        node_id="test.cognitive",
        pmflow_store=pmflow,
        graph_store=graph,
        encoder=encoder
    )

    yield stage, graph

    # Teardown
    graph.close()
    pmflow.close()
    os.unlink(db_path)
    os.unlink(pmflow_path)

def test_cognitive_stage_learn_flow(stage_stack):
    stage, graph = stage_stack
    
    # Pre-populate graph so reasoning has something to do
    graph.add_node("concept:A", "concept", "A")
    graph.add_node("concept:B", "concept", "B")
    graph.add_edge("concept:A", "concept:B", "implies", 0.9)

    # Inject a thought cycle
    # For now, grounding is empty so inferences will be empty, but it verifies the pipeline doesn't crash
    stage.learn("some input payload")
    
    assert hasattr(stage, "last_thought")
    assert stage.last_thought["perception"] is not None
    # We expect empty grounding/inference because we mocked a dumb encoder 
    # and haven't implemented the vector index in the stage yet.
    assert stage.last_thought["grounding"] == [] 
