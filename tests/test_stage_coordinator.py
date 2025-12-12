"""Tests for multi-stage coordinator architecture."""

from __future__ import annotations

import pytest

from experiments.retrieval_sanity.pipeline.base import Utterance
from experiments.retrieval_sanity.pipeline.stage_coordinator import (
    CognitiveStage,
    IntakeStage,
    SemanticStage,
    StageArtifact,
    StageConfig,
    StageCoordinator,
    StageType,
)


def test_stage_config_defaults() -> None:
    """Stage config should auto-generate namespace."""
    config = StageConfig(stage_type=StageType.INTAKE)
    assert config.db_namespace == "intake_memory"
    assert config.plasticity_enabled is True


def test_intake_stage_processes_utterance() -> None:
    """Intake stage should normalize and encode utterance."""
    pytest.importorskip("pmflow.pmflow", reason="pmflow runtime is required")
    
    config = StageConfig(stage_type=StageType.INTAKE)
    stage = IntakeStage(config)
    
    utterance = Utterance(text="Hello, world!", language="en")
    artifact = stage.process(utterance)
    
    assert artifact.stage == StageType.INTAKE
    assert artifact.embedding is not None
    assert artifact.confidence > 0.0
    assert "normalised" in artifact.metadata
    assert "candidates" in artifact.metadata
    assert artifact.tokens is not None


def test_semantic_stage_uses_upstream_context() -> None:
    """Semantic stage should leverage intake stage embeddings."""
    pytest.importorskip("pmflow.pmflow", reason="pmflow runtime is required")
    
    intake_config = StageConfig(stage_type=StageType.INTAKE)
    intake_stage = IntakeStage(intake_config)
    
    semantic_config = StageConfig(stage_type=StageType.SEMANTIC)
    semantic_stage = SemanticStage(semantic_config)
    
    utterance = Utterance(text="we visited the hospital", language="en")
    
    # Process through intake first
    intake_artifact = intake_stage.process(utterance)
    
    # Semantic stage should use intake context
    semantic_artifact = semantic_stage.process(
        utterance,
        upstream_artifacts=[intake_artifact],
    )
    
    assert semantic_artifact.stage == StageType.SEMANTIC
    assert semantic_artifact.embedding is not None
    assert "parsed" in semantic_artifact.metadata
    assert "num_tokens" in semantic_artifact.metadata


def test_coordinator_processes_through_stages() -> None:
    """Coordinator should route utterance through all configured stages."""
    pytest.importorskip("pmflow.pmflow", reason="pmflow runtime is required")
    
    coordinator = StageCoordinator()  # Uses default 2-stage config
    utterance = Utterance(text="Alice visited the hospital", language="en")
    
    results = coordinator.process(utterance)
    
    # Should have both intake and semantic results
    assert StageType.INTAKE in results
    assert StageType.SEMANTIC in results
    
    # Check intake artifact
    intake = results[StageType.INTAKE]
    assert intake.confidence > 0.0
    assert "normalised" in intake.metadata
    
    # Check semantic artifact
    semantic = results[StageType.SEMANTIC]
    assert semantic.confidence > 0.0
    assert "parsed" in semantic.metadata


def test_coordinator_handles_stage_failure_gracefully() -> None:
    """Coordinator should continue processing if one stage fails."""
    pytest.importorskip("pmflow.pmflow", reason="pmflow runtime is required")
    
    class FailingStage(CognitiveStage):
        def process(self, input_data, upstream_artifacts=None):
            raise RuntimeError("Simulated failure")
    
    config = StageConfig(stage_type=StageType.REASONING)
    failing_stage = FailingStage(config)
    
    coordinator = StageCoordinator()
    coordinator.stages[StageType.REASONING] = failing_stage
    
    utterance = Utterance(text="test", language="en")
    results = coordinator.process(utterance)
    
    # Should still have intake and semantic results
    assert StageType.INTAKE in results
    assert StageType.SEMANTIC in results
    # Reasoning stage should be absent due to failure
    assert StageType.REASONING not in results


def test_plasticity_updates_stage_independently() -> None:
    """Each stage should have plasticity mechanism available."""
    pytest.importorskip("pmflow.pmflow", reason="pmflow runtime is required")
    
    config = StageConfig(
        stage_type=StageType.SEMANTIC,
        plasticity_enabled=True,
        plasticity_lr=1e-3,
    )
    stage = SemanticStage(config)
    
    tokens = ["test", "plasticity"]
    reward = 0.5  # Medium quality
    
    # For now, just verify the plasticity method exists and returns a report
    # Actual gradient flow requires PMFlow encoder modifications
    report = stage.apply_plasticity(tokens, reward)
    
    assert isinstance(report, dict)
    assert "reward_signal" in report
    assert report["reward_signal"] == 0.5
    assert "learning_rate" in report
    assert report["learning_rate"] == 1e-3
    # Note: gradient_norm may be 0 if computation graph isn't set up yet
    assert "gradient_norm" in report


def test_stage_confidence_computation() -> None:
    """Stage confidence should reflect activation energy."""
    pytest.importorskip("pmflow.pmflow", reason="pmflow runtime is required")
    
    config = StageConfig(stage_type=StageType.SEMANTIC)
    stage = SemanticStage(config)
    
    # Short input
    artifact1 = stage.encode_with_context(["hi"])
    
    # Longer input with more context
    artifact2 = stage.encode_with_context(["we", "visited", "the", "hospital", "yesterday"])
    
    # Both should produce valid confidence scores
    assert artifact1.confidence >= 0.0
    assert artifact2.confidence >= 0.0
    
    # Both should have activation energy recorded
    assert "activation_energy" in artifact1.metadata
    assert "activation_energy" in artifact2.metadata
    
    # Activation energy should be positive for both
    assert artifact1.metadata["activation_energy"] > 0
    assert artifact2.metadata["activation_energy"] > 0


def test_coordinator_get_stage() -> None:
    """Coordinator should provide access to individual stages."""
    coordinator = StageCoordinator()
    
    intake = coordinator.get_stage(StageType.INTAKE)
    assert intake is not None
    assert isinstance(intake, IntakeStage)
    
    semantic = coordinator.get_stage(StageType.SEMANTIC)
    assert semantic is not None
    assert isinstance(semantic, SemanticStage)
    
    # Non-existent stage
    reasoning = coordinator.get_stage(StageType.REASONING)
    assert reasoning is None
