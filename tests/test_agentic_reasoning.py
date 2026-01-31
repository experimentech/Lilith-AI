"""
Test suite for Agentic Reasoning Learner (Physics-Based Thinking).

Verifies:
1. Integration of PMFlow physics into the cognitive learner.
2. Frame-dragging (Intent Injection) mechanics.
3. Geodesic trajectory generation (Thinking).
4. Principle of Least Action evaluation (Efficiency = Confidence).
"""

import pytest
import torch
import numpy as np
from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.general_purpose_learner import AgenticReasoningLearner, PatternStore

class MockPatternStore:
    """Minimal pattern store for testing."""
    def add_pattern(self, *args, **kwargs):
        return "mock_id"
    
    def update_success(self, *args, **kwargs):
        pass

@pytest.fixture
def encoder():
    """Create a fresh PMFlow encoder for each test."""
    # Use small dimensions for speed
    return PMFlowEmbeddingEncoder(
        dimension=32, 
        latent_dim=16, 
        combine_mode="pm-only", 
        seed=42
    )

@pytest.fixture
def learner(encoder):
    """Create the reasoning learner."""
    store = MockPatternStore()
    return AgenticReasoningLearner(pattern_store=store, encoder=encoder)

def test_initialization(learner):
    """Test that the learner initializes correctly."""
    assert learner.layer_name == "reasoning"
    assert learner.learning_mode == "moderate"
    assert hasattr(learner.encoder.pm_field, 'fine_field') or hasattr(learner.encoder.pm_field, 'centers')

def test_intent_injection(learner):
    """Test injecting active will (modifying omegas)."""
    # Get initial spin state (omegas)
    pm_field = learner.encoder.pm_field
    target_field = pm_field.fine_field if hasattr(pm_field, 'fine_field') else pm_field
    
    if not hasattr(target_field, 'omegas'):
        pytest.skip("PMFlow version does not support omegas/flow field")
    
    initial_omegas = target_field.omegas.detach().clone()
    
    # Inject intent
    intent = "solve problem"
    learner.set_active_will(intent, strength=1.0)
    
    # Verify spin changed
    final_omegas = target_field.omegas.detach()
    diff = torch.norm(final_omegas - initial_omegas)
    
    assert diff > 0, "Intent injection should modify field spin (omegas)"

def test_thinking_process(learner):
    """Test the 'think' method produces spatial trajectories."""
    # Run a thought process
    start_concept = "problem state"
    steps = 5
    
    result = learner.think(start_concept, steps=steps)
    
    # Verify trajectory structure
    trajectory = result['trajectory']
    # Shape should be (1, Steps+1, D)
    assert len(trajectory.shape) == 3
    assert trajectory.shape[0] == 1
    assert trajectory.shape[1] == steps + 1
    
    # Verify movement metrics
    assert result['path_length'] >= 0
    assert result['displacement'] >= 0
    assert 'final_latent' in result

def test_physics_based_evaluation(learner):
    """Test that Principle of Least Action is used for evaluation."""
    # Create a synthetic outcome
    # Case 1: Efficient thought (Straight line)
    # Displacement ~ Path Length
    efficient_outcome = {
        'path_length': 1.0,
        'displacement': 1.0,
        'final_latent': None,
        'trajectory': None
    }
    
    signals_eff = learner.observe_interaction(None, efficient_outcome)
    assert signals_eff.overall_success == 1.0
    assert signals_eff.confidence >= 0.99
    
    # Case 2: Inefficient thought ( meandering )
    # Displacement < Path Length
    inefficient_outcome = {
        'path_length': 2.0,
        'displacement': 1.0,
        'final_latent': None,
        'trajectory': None
    }
    
    signals_ineff = learner.observe_interaction(None, inefficient_outcome)
    assert signals_ineff.confidence < 0.6  # Efficiency = 0.5
    
    # Case 3: Stuck thought
    # Displacement ~ 0
    stuck_outcome = {
        'path_length': 0.1,
        'displacement': 0.05,
        'final_latent': None,
        'trajectory': None
    }
    
    signals_stuck = learner.observe_interaction(None, stuck_outcome)
    assert signals_stuck.overall_success == 0.0

def test_full_agentic_flow(learner):
    """Run a full cycle: Intent -> Think -> Evaluate."""
    
    # 1. Set Will
    learner.set_active_will("reach consensus")
    
    # 2. Think
    outcome = learner.think("conflict", steps=8)
    
    # 3. Evaluate
    signals = learner.observe_interaction(None, outcome)
    
    # Check that we got valid signals
    assert signals.layer_name == "reasoning"
    assert "mental_effort" in signals.layer_signals
    assert "cognitive_efficiency" in signals.layer_signals
    
    # Physics sanity check
    # Displacement should never be greater than path length (triangle inequality)
    # Allow for tiny floating point errors
    assert outcome['displacement'] <= outcome['path_length'] + 1e-5
    assert signals.confidence <= 1.0 + 1e-5
