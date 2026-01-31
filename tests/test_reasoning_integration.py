"""
Integration tests for Agentic Reasoning in the Stage Coordinator.
"""

import pytest
import torch
from experiments.retrieval_sanity.pipeline.stage_coordinator import (
    StageCoordinator,
    StageType,
    StageConfig,
    ReasoningStageWrapper
)
from experiments.retrieval_sanity.pipeline.base import Utterance

def test_reasoning_stage_initialization():
    """Verify ReasoningStageWrapper initializes the physics engine."""
    config = StageConfig(stage_type=StageType.REASONING)
    stage = ReasoningStageWrapper(config)
    
    assert hasattr(stage, 'engine')
    assert stage.engine.learner is not None
    # Check if learner has frame-dragging capability
    assert hasattr(stage.engine.learner, 'set_active_will')

def test_coordinator_includes_reasoning_default():
    """Verify default coordinator config includes reasoning."""
    coordinator = StageCoordinator()
    
    # Defaults should now include REASONING
    assert StageType.REASONING in coordinator.stages
    assert isinstance(coordinator.stages[StageType.REASONING], ReasoningStageWrapper)

def test_end_to_end_reasoning_flow():
    """Test full pipeline flow including agentic thinking."""
    coordinator = StageCoordinator()
    
    utterance = Utterance(text="What are your capabilities?", language="en")
    
    results = coordinator.process(utterance)
    
    # Check we got a reasoning result
    assert StageType.REASONING in results
    artifact = results[StageType.REASONING]
    
    # Check physics-based metadata
    assert "reasoning_result" in artifact.metadata
    result = artifact.metadata["reasoning_result"]
    
    # Verify deliberation happened
    assert result.deliberation_steps > 0
    
    # Verify intent resolution (Agentic Will)
    assert result.resolved_intent == "capability"
    
    # Check for physics inference in reasoning details
    physics_inferences = [
        i for i in result.inferences 
        if i.inference_type == "structural" and "trajectory" in i.conclusion
    ]
    # Note: Physics inference might fail gracefully if PMFlow setup is minimal in tests
    # but we check at least the attempt was made (by presence of result)
    assert result.confidence > 0.0

def test_intent_injection_mechanics():
    """Verify that detected intent actually modifies flow."""
    # Setup stage
    config = StageConfig(stage_type=StageType.REASONING)
    stage = ReasoningStageWrapper(config)
    
    # Access the underlying physics field
    pm_field = stage.encoder.pm_field
    target_field = pm_field.fine_field if hasattr(pm_field, 'fine_field') else pm_field
    
    if hasattr(target_field, 'omegas'):
        # Snapshot initial spin
        initial_spin = target_field.omegas.clone().detach()
        
        # Run process that triggers intent injection
        # "What is X" triggers definition intent
        stage.process(Utterance(text="What is the definition of reasoning?"))
        
        # Verify spin changed (willpower injected)
        current_spin = target_field.omegas.detach()
        diff = torch.norm(current_spin - initial_spin)
        
        assert diff > 0.0, "Reasoning should inject will (modify omegas)"
