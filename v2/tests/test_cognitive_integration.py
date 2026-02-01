import pytest
import os
import tempfile
import torch
from v2.lilith_v2.app import V2Runtime
from v2.lilith_v2.cognitive_stage import CognitiveStage
from v2.lilith_v2.mcp_router import MCPDescriptor, EndpointType

def test_cognitive_stage_wiring():
    # Define a config that requests a cognitive stage
    config = {
        "bindings": [
            {
                "node_id": "trunk.mind",
                "stage": "cognitive",
                "modalities": ["text"],
                "store_type": "sqlite"
            }
        ],
        "pmflow": {
             "state_path": "/tmp/test_pmflow.sqlite",
             "event_path": "/tmp/test_events.sqlite"
        }
    }

    # Boot the runtime
    runtime = V2Runtime.from_config(config)
    
    # Assert wiring
    stage = runtime.stages.get("trunk.mind")
    assert isinstance(stage, CognitiveStage)
    assert stage.id == "trunk.mind"
    assert stage.graph is not None
    assert stage.pmflow is not None

    # Test dispatch
    mock_ctx = {"tenant": "test_tenant", "modality": "text"}
    desc = MCPDescriptor(name="test.tool", type=EndpointType.ACTION, cost=0.0)
    
    # This should route to trunk.mind and trigger stage.learn()
    # We won't see side effects easily here unless we mock more, 
    # but ensuring no crash during route_and_dispatch is a good smoke test.
    decision = runtime.route_and_dispatch(desc, mock_ctx, payload="Hello World")
    
    # Check if the port for trunk.mind exists in decision
    assert "trunk.mind" in decision.ports
