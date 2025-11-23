#!/usr/bin/env python3
"""Inspect SQLite-backed vector stores and run sample symbolic retrievals."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from experiments.retrieval_sanity.storage.sqlite_store import SQLiteVectorStore
from experiments.retrieval_sanity.log_observer import _collect_sqlite_summary


@dataclass
class ScenarioSnapshot:
    base: str
    baseline_count: int
    pmflow_count: int
    other_counts: dict


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe stored embeddings from the SQLite vector store.")
    parser.add_argument("--sqlite-path", type=Path, required=True, help="Path to the SQLite database.")
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario base identifier (e.g. 'soak-00:synthetic-baseline').",
    )
    parser.add_argument("--samples", type=int, default=5, help="Number of synthetic queries to execute per scenario.")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise multiplier when perturbing embeddings for queries.")
    parser.add_argument("--topk", type=int, default=3, help="Top-k neighbours to inspect for each query.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file containing the probe results.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    return parser.parse_args(list(argv) if argv is not None else None)


def load_snapshot(path: Path, scenario: str) -> ScenarioSnapshot:
    summary = _collect_sqlite_summary(path)
    if summary is None:
        raise RuntimeError(f"Failed to read SQLite database {path}")
    for entry in summary:
        if entry["base"] == scenario:
            return ScenarioSnapshot(
                base=scenario,
                baseline_count=entry["baseline"],
                pmflow_count=entry["pmflow"],
                other_counts=entry["other"],
            )
    raise ValueError(f"Scenario '{scenario}' not found in {path}")


def compute_label_distribution(labels: torch.Tensor) -> List[tuple[int, int]]:
    counts = Counter(labels.tolist())
    return sorted(counts.items())


def make_queries(source: torch.Tensor, noise: float, samples: int) -> torch.Tensor:
    if samples > source.shape[0]:
        samples = source.shape[0]
    indices = random.sample(range(source.shape[0]), samples)
    base_vectors = source[indices]
    noise_tensor = torch.randn_like(base_vectors) * noise
    return base_vectors + noise_tensor


def run_probe() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    scenario_base = args.scenario
    baseline_ns = f"{scenario_base}:baseline"
    pmflow_ns = f"{scenario_base}:pmflow"

    store = SQLiteVectorStore(args.sqlite_path)
    baseline_vectors, baseline_labels = store.fetch(scenario=baseline_ns)
    pm_vectors, pm_labels = store.fetch(scenario=pmflow_ns)

    snapshot = load_snapshot(args.sqlite_path, scenario_base)

    results = {
        "sqlite_path": str(args.sqlite_path),
        "scenario": scenario_base,
        "baseline_count": snapshot.baseline_count,
        "pmflow_count": snapshot.pmflow_count,
        "baseline_label_distribution": compute_label_distribution(baseline_labels),
        "pmflow_label_distribution": compute_label_distribution(pm_labels),
        "queries": [],
    }

    queries = make_queries(pm_vectors, args.noise, args.samples)
    for idx, query in enumerate(queries, start=1):
        values, neighbours = store.search(query.unsqueeze(0), topk=args.topk, scenario=baseline_ns)
        neighbour_labels = neighbours.squeeze(0).tolist()
        results["queries"].append(
            {
                "id": idx,
                "neighbour_labels": neighbour_labels,
                "scores": values.squeeze(0).tolist(),
            }
        )
        print(
            f"[probe] Query {idx}: neighbours={neighbour_labels} scores={['{:.3f}'.format(v) for v in values.squeeze(0).tolist()]}"
        )

    print(
        f"[probe] Baseline label distribution: {compute_label_distribution(baseline_labels)} | "
        f"PMFlow label distribution: {compute_label_distribution(pm_labels)}"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[probe] Wrote results to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_probe())
