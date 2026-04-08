"""Tests for lilith.pipeline.SymbolicPipeline encoder fallback behaviour."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lilith.embedding import HashedEmbeddingEncoder
from lilith.pipeline import SymbolicPipeline


def test_pipeline_no_pmflow_returns_hashed_encoder() -> None:
    """SymbolicPipeline(use_pmflow=False) must use HashedEmbeddingEncoder."""
    pipeline = SymbolicPipeline(use_pmflow=False)
    assert isinstance(pipeline.encoder, HashedEmbeddingEncoder)


def test_pipeline_falls_back_when_pmflow_raises_import_error() -> None:
    """SymbolicPipeline should fall back to HashedEmbeddingEncoder when
    PMFlowEmbeddingEncoder raises ImportError (e.g. pmflow not installed).

    Before the fix, only RuntimeError was caught, so ImportError / ModuleNotFoundError
    would propagate instead of triggering the HashedEmbeddingEncoder fallback.
    """
    with (
        patch("lilith.pipeline.HAS_SEMANTIC_ENCODER", False),
        patch(
            "lilith.pipeline.PMFlowEmbeddingEncoder",
            side_effect=ImportError("No module named 'pmflow'"),
        ),
    ):
        pipeline = SymbolicPipeline(use_pmflow=True)
        assert isinstance(pipeline.encoder, HashedEmbeddingEncoder), (
            "Expected HashedEmbeddingEncoder fallback when pmflow is unavailable, "
            f"got {type(pipeline.encoder).__name__}"
        )


def test_attach_state_path_works_for_hashed_encoder() -> None:
    """_attach_pmflow_state must not crash for HashedEmbeddingEncoder (no attach_state_path)."""
    pipeline = SymbolicPipeline(use_pmflow=False, pmflow_state_path=Path("/tmp/state"))
    # No exception should be raised and encoder must still be HashedEmbeddingEncoder.
    assert isinstance(pipeline.encoder, HashedEmbeddingEncoder)


def test_attach_state_path_called_on_encoder_that_supports_it() -> None:
    """_attach_pmflow_state must call attach_state_path on encoders that have it."""

    class _MockEncoder:
        """Minimal stand-in that reports attach_state_path calls."""

        attached: Path | None = None

        def attach_state_path(self, path: Path | None) -> None:
            _MockEncoder.attached = path

    state_path = Path("/tmp/mock_state")

    with patch("lilith.pipeline.SymbolicPipeline._build_encoder", return_value=_MockEncoder()):
        pipeline = SymbolicPipeline(use_pmflow=False, pmflow_state_path=state_path)

    assert _MockEncoder.attached == state_path, (
        "_attach_pmflow_state did not call attach_state_path on the encoder"
    )
