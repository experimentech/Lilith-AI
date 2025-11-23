#!/usr/bin/env python3
"""Compare performance of in-memory and SQLite vector stores used in the retrieval sanity lab."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, cast

import sys

import torch

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from experiments.retrieval_sanity import run_retrieval_sanity as runner
from experiments.retrieval_sanity.storage.sqlite_store import SQLiteVectorStore


@dataclass
class Measurement:
    backend: str
    size: int
    add_times: Sequence[float]
    search_times: Sequence[float]

    def summary(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "size": self.size,
            "add_mean": statistics.mean(self.add_times),
            "add_std": statistics.pstdev(self.add_times) if len(self.add_times) > 1 else 0.0,
            "search_mean": statistics.mean(self.search_times),
            "search_std": statistics.pstdev(self.search_times) if len(self.search_times) > 1 else 0.0,
            "samples": len(self.add_times),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vector store backends.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1000, 5000, 20000],
        help="Dataset sizes (number of embeddings) to benchmark.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=64,
        help="Dimensionality of generated embeddings.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=256,
        help="Number of query vectors to use when measuring search latency.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repetitions per size/backend combination.",
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Similarity metric to use for both stores.",
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=Path("benchmarks/sqlite_vectors.db"),
        help="Location of the SQLite database used for benchmarking.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file to write benchmark results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def _make_tensors(size: int, dim: int, num_queries: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = torch.randn(size, dim)
    labels = torch.randint(0, max(2, dim // 2), (size,), dtype=torch.long)
    queries = torch.randn(num_queries, dim)
    return embeddings, labels, queries


def _benchmark_memory(embeddings: torch.Tensor, labels: torch.Tensor, queries: torch.Tensor, metric: str) -> tuple[float, float]:
    store = runner.VectorStore(metric=metric)
    store.clear()
    start = time.perf_counter()
    store.add(embeddings, labels)
    add_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    store.search(queries, topk=1)
    search_elapsed = time.perf_counter() - start
    return add_elapsed, search_elapsed


def _benchmark_sqlite(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    queries: torch.Tensor,
    metric: str,
    db_path: Path,
    scenario: str,
) -> tuple[float, float]:
    store = SQLiteVectorStore(db_path, metric=metric)
    store.clear(scenario=scenario)

    start = time.perf_counter()
    store.add(embeddings, labels, scenario=scenario)
    add_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    store.search(queries, topk=1, scenario=scenario)
    search_elapsed = time.perf_counter() - start

    store.clear(scenario=scenario)
    return add_elapsed, search_elapsed


def format_table(measurements: List[Measurement]) -> str:
    header = f"{'backend':10} {'size':>8} {'add_mean(ms)':>14} {'add_std(ms)':>12} {'search_mean(ms)':>16} {'search_std(ms)':>14}"
    lines = [header, "-" * len(header)]
    for m in measurements:
        summary = m.summary()
        add_mean = cast(float, summary["add_mean"])
        add_std = cast(float, summary["add_std"])
        search_mean = cast(float, summary["search_mean"])
        search_std = cast(float, summary["search_std"])
        lines.append(
            f"{summary['backend']:10} {summary['size']:8d} "
            f"{add_mean * 1000:14.3f} {add_std * 1000:12.3f} "
            f"{search_mean * 1000:16.3f} {search_std * 1000:14.3f}"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    runner.set_seed(args.seed)

    measurements: List[Measurement] = []
    db_path = args.sqlite_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    for size in args.sizes:
        add_memory: List[float] = []
        search_memory: List[float] = []
        add_sqlite: List[float] = []
        search_sqlite: List[float] = []

        for trial in range(args.repeats):
            embeddings, labels, queries = _make_tensors(size, args.dim, args.queries)

            add_elapsed, search_elapsed = _benchmark_memory(embeddings, labels, queries, args.metric)
            add_memory.append(add_elapsed)
            search_memory.append(search_elapsed)

            scenario = f"benchmark-{size}-{trial}"
            add_sql, search_sql = _benchmark_sqlite(embeddings, labels, queries, args.metric, db_path, scenario)
            add_sqlite.append(add_sql)
            search_sqlite.append(search_sql)

        measurements.append(Measurement("memory", size, add_memory, search_memory))
        measurements.append(Measurement("sqlite", size, add_sqlite, search_sqlite))

    print("\nVector store benchmark results (times in milliseconds):")
    print(format_table(measurements))

    if args.output:
        payload = {
            "generated_at": time.time(),
            "metric": args.metric,
            "dim": args.dim,
            "queries": args.queries,
            "repeats": args.repeats,
            "results": [m.summary() for m in measurements],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON results to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
