from __future__ import annotations

from pathlib import Path

import torch

from experiments.retrieval_sanity import run_retrieval_sanity as runner
from experiments.retrieval_sanity import run_suite
from experiments.retrieval_sanity.storage.sqlite_store import SQLiteVectorStore


def test_sqlite_vector_store_roundtrip(tmp_path):
    db_path = tmp_path / "vectors.db"
    store = SQLiteVectorStore(db_path)
    store.clear()

    embeddings = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 0, 2], dtype=torch.long)

    store.add(embeddings, labels, scenario="baseline")
    store.add(embeddings + 0.1, labels, scenario="pmflow")

    assert store.count() == 8
    assert store.count(scenario="baseline") == 4

    queries = embeddings
    values, top_labels = store.search(queries, scenario="baseline")

    assert values.shape == (4, 1)
    assert top_labels.shape == (4, 1)


def test_run_experiment_with_sqlite_store(tmp_path):
    db_path = tmp_path / "vectors.db"
    args = runner.parse_args(
        [
            "--epochs",
            "10",
            "--points-per-class",
            "60",
            "--support-ratio",
            "0.5",
            "--pm-steps",
            "2",
            "--pm-beta",
            "0.5",
            "--vector-store",
            "sqlite",
            "--sqlite-path",
            str(db_path),
            "--sqlite-scenario",
            "test-suite",
        ]
    )
    runner.set_seed(args.seed)
    results = runner.run_experiment(args)

    assert results["vector_store"] == "sqlite"
    sqlite_path = Path(str(results["sqlite_path"]))
    assert sqlite_path.exists()
    assert results["sqlite_scenario"] == "test-suite"

    store = SQLiteVectorStore(sqlite_path)
    assert store.count(scenario="test-suite:baseline") > 0
    assert store.count(scenario="test-suite:pmflow") > 0


def test_run_suite_sqlite_integration(tmp_path):
    db_path = tmp_path / "vectors.db"
    scenario_name = "synthetic-baseline"
    prefix = "pytest"
    scenario_id = f"{prefix}:{scenario_name}"

    cli_tokens = list(run_suite.SCENARIOS[scenario_name])
    cli_tokens.extend(
        [
            "--vector-store",
            "sqlite",
            "--sqlite-path",
            str(db_path),
            "--sqlite-scenario",
            scenario_id,
        ]
    )

    args, results = run_suite.run_scenario(scenario_name, cli_tokens)

    assert results["vector_store"] == "sqlite"
    assert results["sqlite_scenario"] == scenario_id

    store = SQLiteVectorStore(db_path)
    assert store.count(scenario=f"{scenario_id}:baseline") > 0
    assert store.count(scenario=f"{scenario_id}:pmflow") > 0