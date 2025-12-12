"""Regression tests for richer parsing and PMFlow-aware embeddings."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from experiments.retrieval_sanity.pipeline.base import Utterance
from experiments.retrieval_sanity.pipeline.embedding import (
    HashedEmbeddingEncoder,
    PMFlowEmbeddingEncoder,
)
from experiments.retrieval_sanity.pipeline.parser import parse
from experiments.retrieval_sanity.pipeline.pipeline import SymbolicPipeline


def _ensure_pmflow_on_path() -> None:
    # Prefer installed PMFlow; only fall back to workspace checkout if necessary.
    try:
        importlib.import_module("pmflow.pmflow")
        return
    except ModuleNotFoundError:
        pass

    workspace_root = Path(__file__).resolve().parents[1]
    candidate = workspace_root / "Pushing-Medium" / "src"
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def test_parse_captures_location_time_and_intent() -> None:
    parsed = parse("when can we visit the hospital next week?")
    assert parsed.intent == "when"
    assert parsed.location_index is not None
    assert parsed.role_map.get("location") == parsed.location_index
    assert parsed.modifiers.get("time_phrase") == "next week"
    assert parsed.confidence > 0.5


def test_parse_detects_negation() -> None:
    parsed = parse("we do not need another appointment")
    assert parsed.negation_indices
    assert parsed.modifiers.get("negated") == "true"


def test_pipeline_falls_back_without_pmflow(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import_module = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name in {
            "pmflow.pmflow",
            "PMFlow.pmflow",
        }:
            raise ModuleNotFoundError(name)
        return original_import_module(name, package)

    monkeypatch.setattr(
        "experiments.retrieval_sanity.pipeline.embedding.PM_MODULE",
        None,
    )
    monkeypatch.setattr(
        "experiments.retrieval_sanity.pipeline.embedding.importlib.import_module",
        fake_import,
    )

    pipeline = SymbolicPipeline(use_pmflow=True)
    assert isinstance(pipeline.encoder, HashedEmbeddingEncoder)


def test_pmflow_embedding_dimensions() -> None:
    _ensure_pmflow_on_path()
    pytest.importorskip("pmflow.pmflow")
    encoder = PMFlowEmbeddingEncoder(dimension=48, latent_dim=24)
    vector = encoder.encode(["hola", "hospital"])
    assert vector.shape == (1, 72)
    # Ensure concatenation keeps hashed component accessible.
    hashed = HashedEmbeddingEncoder(dimension=48).encode(["hola", "hospital"])
    assert pytest.approx(hashed.norm().item(), rel=1e-5) == 1.0


def test_pipeline_with_pmflow_encodes_frames() -> None:
    _ensure_pmflow_on_path()
    pmflow_module = pytest.importorskip("pmflow.pmflow")
    assert any(
        hasattr(pmflow_module, attr) for attr in ("PMField", "ParallelPMField")
    ), "pmflow module lacks field implementation"

    pipeline = SymbolicPipeline(use_pmflow=True, pmflow_kwargs={"dimension": 32, "latent_dim": 16})
    artefact = pipeline.process(Utterance(text="We need to visit the clinic tomorrow", language="en"))
    assert artefact.embedding.shape == (1, 48)
    assert artefact.frame.attributes.get("time_phrase") == "tomorrow"