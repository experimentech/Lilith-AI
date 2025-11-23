"""Tests for the batch ingestion CLI helpers."""

from __future__ import annotations

import argparse

import pytest

from experiments.retrieval_sanity.pipeline.batch_feeder import (
    BatchMetrics,
    _iter_sentences,
    run_batch,
)


def make_args(tmp_path, corpus_text: str | None = None, **overrides):
    corpus_path = tmp_path / "corpus.txt"
    if corpus_text is None:
        # Ensure any previous file is removed to simulate a missing corpus.
        if corpus_path.exists():
            corpus_path.unlink()
    else:
        corpus_path.write_text(corpus_text, encoding="utf-8")

    defaults = {
        "corpus_path": corpus_path,
        "sqlite_path": tmp_path / "vectors.db",
        "scenario": "test",
        "language": "en",
        "limit": None,
        "min_tokens": 5,
        "min_alpha_ratio": 0.5,
        "max_tokens": None,
        "chunk_sentences": True,
        "lowercase": False,
        "skip_headers": True,
        "drop_patterns": None,
        "flush_interval": 1,
        "log_every": None,
        "summary_path": None,
        "trace_path": None,
        "no_trace": True,
        "no_store": True,
        "clear": False,
        "topk": 3,
        "min_score": 0.15,
        "no_plasticity": True,
        "plasticity_threshold": 0.55,
        "plasticity_mu_lr": 5e-4,
        "plasticity_center_lr": 5e-4,
        "plasticity_state_path": None,
        "preview": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_iter_sentences_respects_limit(tmp_path):
    text = (
        "First clean sentence has plenty of tokens for testing.\n"
        "Second clean sentence also has many useful tokens for checks.\n"
        "Third clean sentence ensures the limit logic is exercised properly."
    )
    args = make_args(tmp_path, corpus_text=text, limit=2)
    sentences = list(_iter_sentences(args))

    assert len(sentences) == 2
    assert sentences[0].startswith("First clean sentence")
    assert sentences[1].startswith("Second clean sentence")


def test_run_batch_preview_outputs_sentences(tmp_path, capsys):
    text = (
        "Preview sentence number one with enough relevant words.\n"
        "Preview sentence number two also carries multiple tokens."
    )
    args = make_args(tmp_path, corpus_text=text, preview=2)

    metrics = run_batch(args)

    captured = capsys.readouterr()
    output_lines = [line.strip() for line in captured.out.splitlines() if line.strip()]

    assert isinstance(metrics, BatchMetrics)
    assert metrics.processed == 0
    assert metrics.stored == 0
    assert output_lines[:2] == [
        "Preview sentence number one with enough relevant words.",
        "Preview sentence number two also carries multiple tokens.",
    ]


def test_run_batch_missing_corpus_raises(tmp_path):
    args = make_args(tmp_path, corpus_text=None)

    with pytest.raises(FileNotFoundError):
        run_batch(args)
