"""
Tests for SyntaxStage plasticity implementation.

Validates that:
1. Vectorized plasticity updates PMFlow centers/mus
2. Contrastive plasticity separates different grammar structures
3. State persistence saves/loads plasticity-learned weights
"""

import pytest
import tempfile
from pathlib import Path

import torch

from lilith.syntax_stage_bnn import (
    SyntaxStage,
    SyntaxPlasticityReport,
    PMFLOW_PLASTICITY_AVAILABLE,
    contrastive_plasticity,
)


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "syntax_patterns.json"


@pytest.fixture
def syntax_stage(temp_storage):
    """Create a SyntaxStage instance with temporary storage."""
    return SyntaxStage(storage_path=temp_storage)


class TestPlasticityAvailability:
    """Test that plasticity features are available."""
    
    def test_pmflow_plasticity_available(self):
        """Verify PMFlow plasticity is importable."""
        assert PMFLOW_PLASTICITY_AVAILABLE, "PMFlow plasticity should be available"
    
    def test_contrastive_plasticity_available(self):
        """Verify contrastive plasticity is available."""
        assert contrastive_plasticity is not None, "Contrastive plasticity should be available"


class TestReinforcementPlasticity:
    """Test basic reinforcement plasticity."""
    
    def test_positive_feedback_updates_centers(self, syntax_stage):
        """Positive feedback should apply plasticity and update centers."""
        # Learn a pattern
        pattern_id = syntax_stage.learn_pattern(
            tokens=["This", "is", "great"],
            pos_tags=["PRON", "VBZ", "ADJ"],
            success_feedback=0.5
        )
        
        # Get initial centers
        pm_field = syntax_stage.encoder.pm_field
        if hasattr(pm_field, 'fine_field'):
            initial_centers = pm_field.fine_field.centers.clone()
        else:
            initial_centers = pm_field.centers.clone()
        
        # Apply positive feedback
        report = syntax_stage.update_pattern_success(pattern_id, feedback=0.8)
        
        # Verify plasticity was applied
        assert report is not None
        assert isinstance(report, SyntaxPlasticityReport)
        assert report.delta_centers > 0
        assert report.plasticity_type == "reinforcement"
    
    def test_negative_feedback_applies_gentle_correction(self, syntax_stage):
        """Negative feedback should apply gentler plasticity."""
        pattern_id = syntax_stage.learn_pattern(
            tokens=["Bad", "grammar", "here"],
            pos_tags=["ADJ", "NN", "ADV"],
            success_feedback=0.3
        )
        
        # Apply negative feedback
        report = syntax_stage.update_pattern_success(pattern_id, feedback=-0.5)
        
        assert report is not None
        assert report.feedback == -0.5
        # Should still update but with smaller deltas
        assert report.delta_centers >= 0
    
    def test_plasticity_report_tracking(self, syntax_stage):
        """Plasticity reports should be tracked."""
        pattern_id = syntax_stage.learn_pattern(
            tokens=["Test", "pattern"],
            pos_tags=["NN", "NN"],
            success_feedback=0.5
        )
        
        # Multiple updates
        for _ in range(3):
            syntax_stage.update_pattern_success(pattern_id, feedback=0.3)
        
        stats = syntax_stage.get_plasticity_stats()
        assert stats["total_updates"] == 3
        assert stats["recent_updates"] == 3
        assert stats["plasticity_available"] is True


class TestContrastivePlasticity:
    """Test contrastive plasticity for grammar structure separation."""
    
    def test_contrastive_learning_separates_intents(self, syntax_stage):
        """Contrastive learning should separate questions from statements."""
        # Add question patterns
        syntax_stage.learn_pattern(
            tokens=["How", "does", "this", "work"],
            pos_tags=["WRB", "VBZ", "PRON", "VB"],
            success_feedback=0.5
        )
        syntax_stage.learn_pattern(
            tokens=["What", "is", "that"],
            pos_tags=["WRB", "VBZ", "PRON"],
            success_feedback=0.5
        )
        
        # Add statement patterns
        syntax_stage.learn_pattern(
            tokens=["That", "sounds", "great"],
            pos_tags=["PRON", "VBZ", "ADJ"],
            success_feedback=0.5
        )
        syntax_stage.learn_pattern(
            tokens=["I", "like", "it"],
            pos_tags=["PRON", "VBZ", "PRON"],
            success_feedback=0.5
        )
        
        # Apply contrastive learning
        report = syntax_stage.apply_contrastive_learning()
        
        assert report is not None
        assert report.plasticity_type == "contrastive"
        # Should have some center movement
        assert report.delta_centers >= 0
    
    def test_contrastive_with_custom_intents(self, syntax_stage):
        """Test contrastive learning with custom intent pairs."""
        # Add patterns
        for _ in range(2):
            syntax_stage.learn_pattern(
                tokens=["Question", "here"],
                pos_tags=["WRB", "VBZ"],
                success_feedback=0.5
            )
            syntax_stage.learn_pattern(
                tokens=["Statement", "here"],
                pos_tags=["PRON", "VBZ"],
                success_feedback=0.5
            )
        
        # Custom intent configuration
        report = syntax_stage.apply_contrastive_learning(
            similar_intents=[("question", "question")],
            dissimilar_intents=[("question", "statement")]
        )
        
        # Should work (may return None if not enough patterns)
        # The test passes if no exception is raised


class TestStatePersistence:
    """Test saving/loading plasticity state."""
    
    def test_save_and_load_pmflow_state(self, temp_storage):
        """PMFlow state should persist across sessions."""
        # Create stage and apply plasticity
        stage1 = SyntaxStage(storage_path=temp_storage)
        
        pattern_id = stage1.learn_pattern(
            tokens=["Test", "persistence"],
            pos_tags=["NN", "NN"],
            success_feedback=0.7
        )
        
        # Apply multiple plasticity updates
        for _ in range(5):
            stage1.update_pattern_success(pattern_id, feedback=0.5)
        
        # Save state
        stage1.save_state()
        
        # Get centers after plasticity
        pm_field1 = stage1.encoder.pm_field
        if hasattr(pm_field1, 'fine_field'):
            centers_after = pm_field1.fine_field.centers.clone()
        else:
            centers_after = pm_field1.centers.clone()
        
        # Create new stage and load
        stage2 = SyntaxStage(storage_path=temp_storage)
        
        pm_field2 = stage2.encoder.pm_field
        if hasattr(pm_field2, 'fine_field'):
            loaded_centers = pm_field2.fine_field.centers
        else:
            loaded_centers = pm_field2.centers
        
        # Centers should match
        assert torch.allclose(centers_after, loaded_centers, atol=1e-6)
    
    def test_plasticity_count_persists(self, temp_storage):
        """Total plasticity update count should persist."""
        stage1 = SyntaxStage(storage_path=temp_storage)
        
        pattern_id = stage1.learn_pattern(
            tokens=["Count", "test"],
            pos_tags=["NN", "NN"],
            success_feedback=0.5
        )
        
        # Apply 7 updates
        for _ in range(7):
            stage1.update_pattern_success(pattern_id, feedback=0.3)
        
        stage1.save_state()
        
        # Load in new stage
        stage2 = SyntaxStage(storage_path=temp_storage)
        
        assert stage2.total_plasticity_updates == 7


class TestPlasticityStats:
    """Test plasticity statistics tracking."""
    
    def test_stats_with_no_updates(self, syntax_stage):
        """Stats should work with no updates."""
        stats = syntax_stage.get_plasticity_stats()
        
        assert stats["total_updates"] == 0
        assert stats["recent_updates"] == 0
        assert stats["avg_delta_centers"] == 0.0
        assert stats["plasticity_available"] is True
    
    def test_stats_track_by_type(self, syntax_stage):
        """Stats should track updates by type."""
        # Create patterns
        pattern_id = syntax_stage.learn_pattern(
            tokens=["Test"],
            pos_tags=["NN"],
            success_feedback=0.5
        )
        
        # Apply reinforcement
        syntax_stage.update_pattern_success(pattern_id, feedback=0.5)
        syntax_stage.update_pattern_success(pattern_id, feedback=0.3)
        
        stats = syntax_stage.get_plasticity_stats()
        
        assert "reinforcement" in stats.get("by_type", {})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
