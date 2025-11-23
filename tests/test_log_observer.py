from __future__ import annotations

import json
from datetime import datetime, timezone

import torch

from experiments.retrieval_sanity.log_observer import (
    DriftPolicy,
    MetricSnapshot,
    RunRecord,
    _collect_sqlite_summary,
    _infer_sqlite_path,
    evaluate_policy,
    load_records,
)
from experiments.retrieval_sanity.storage.sqlite_store import SQLiteVectorStore


def make_run(pm_retrieval: float, baseline_retrieval: float, *, ts_offset: int = 0, **overrides) -> RunRecord:
    timestamp = datetime.now(timezone.utc)
    if ts_offset:
        timestamp = timestamp.replace(microsecond=ts_offset)
    metrics = MetricSnapshot(
        baseline_retrieval=baseline_retrieval,
        pmflow_retrieval=pm_retrieval,
        baseline_class=baseline_retrieval,
        pmflow_class=pm_retrieval,
        final_loss_baseline=0.1,
        final_loss_pmflow=0.05,
    )
    config = overrides.pop("config", {})
    dataset = overrides.pop("dataset", {})
    device = overrides.pop("device", "cpu")
    return RunRecord(
        timestamp=timestamp,
        config=config,
        metrics=metrics,
        device=device,
        dataset=dataset,
        **overrides,
    )


def test_evaluate_policy_detects_negative_delta():
    runs = [
        make_run(0.90, 0.85),
        make_run(0.88, 0.86),
        make_run(0.80, 0.85),
    ]
    policy = DriftPolicy(min_retrieval_delta=0.02, max_retrieval_drop=0.1, window=3, min_runs=3)
    alerts = evaluate_policy(runs, policy)

    assert alerts, "Expected an alert when PMFlow underperforms baseline"
    assert "dropped" in alerts[0].reason or "delta" in alerts[0].reason


def test_evaluate_policy_detects_drop():
    runs = [
        make_run(0.96, 0.90),
        make_run(0.97, 0.91),
        make_run(0.85, 0.88),
    ]
    policy = DriftPolicy(min_retrieval_delta=0.0, max_retrieval_drop=0.05, window=3, min_runs=3)
    alerts = evaluate_policy(runs, policy)

    assert alerts
    assert "dropped" in alerts[0].reason


def test_alert_reason_includes_context_metadata():
    context_run = make_run(
        0.82,
        0.90,
        config_label="smoke",
        config={"data_mode": "synthetic"},
        provenance={"runner": "cli", "device": "cuda:0", "git": {"commit": "deadbeefcafebabe"}},
        device="cuda:0",
        dataset={"support_samples": 120, "query_samples": 80},
        config_overrides={"epochs": 2, "lr": 1e-3},
    )
    runs = [
        make_run(0.96, 0.90),
        make_run(0.95, 0.91),
        context_run,
    ]
    policy = DriftPolicy(min_retrieval_delta=0.0, max_retrieval_drop=0.05, window=3, min_runs=3)

    alerts = evaluate_policy(runs, policy)

    assert alerts
    reason = alerts[0].reason
    assert "context:" in reason
    assert "config=smoke" in reason
    assert "git=deadbee" in reason
    assert "device=cuda:0" in reason


def test_load_records_parses_json(tmp_path):
    log_path = tmp_path / "retrieval_runs.log"
    payload = {
        "timestamp": "2025-10-07T12:00:00.000000Z",
        "config": {"foo": "bar"},
        "metrics": {
            "baseline": {"retrieval_accuracy": 0.8, "class_accuracy": 0.75, "final_loss": 0.2},
            "pmflow": {"retrieval_accuracy": 0.9, "class_accuracy": 0.85, "final_loss": 0.1},
        },
        "device": "cuda",
        "dataset": {"support_samples": 100},
        "provenance": {"runner": "cli", "git": {"commit": "abcdef123456"}},
        "config_label": "smoke",
        "config_path": "/tmp/overrides.json",
        "config_overrides": {"epochs": 1},
    }
    log_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    runs = load_records(log_path)

    assert len(runs) == 1
    record = runs[0]
    assert record.metrics.pmflow_retrieval == 0.9
    assert record.metrics.baseline_retrieval == 0.8
    assert record.device == "cuda"
    assert record.provenance is not None
    assert record.provenance["runner"] == "cli"
    assert record.config_label == "smoke"
    assert record.config_path is not None and record.config_path.endswith("overrides.json")
    assert record.config_overrides == {"epochs": 1}


def test_collect_sqlite_summary_groups_counts(tmp_path):
    db_path = tmp_path / "vectors.db"
    store = SQLiteVectorStore(db_path)
    store.clear()

    embeddings = torch.randn(6, 4)
    labels = torch.arange(6, dtype=torch.long)

    store.add(embeddings[:2], labels[:2], scenario="alpha:baseline")
    store.add(embeddings[2:4], labels[2:4], scenario="alpha:pmflow")
    store.add(embeddings[4:5], labels[4:5], scenario="baseline")
    store.add(embeddings[5:], labels[5:], scenario="custom-scenario")

    summary = _collect_sqlite_summary(db_path)

    assert summary is not None
    data = {entry["base"]: entry for entry in summary}
    assert data["alpha"]["baseline"] == 2
    assert data["alpha"]["pmflow"] == 2
    assert data["alpha"]["other"] == {}

    default = data["<default>"]
    assert default["baseline"] == 1
    assert default["pmflow"] == 0

    custom = data["custom-scenario"]
    assert custom["other"][None] == 1
    assert custom["total"] == 1


def test_infer_sqlite_path_prefers_recent_run(tmp_path):
    db_path = tmp_path / "vectors.db"
    runs = [
        make_run(0.9, 0.85, vector_store="memory"),
        make_run(0.91, 0.86, vector_store="sqlite", sqlite_path=str(db_path)),
    ]

    resolved = _infer_sqlite_path(None, runs)
    assert resolved == db_path

    override = tmp_path / "override.db"
    resolved_override = _infer_sqlite_path(override, runs)
    assert resolved_override == override