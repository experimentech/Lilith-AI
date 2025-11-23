#!/usr/bin/env python3
"""Multi-run soak harness for the retrieval sanity suite using a persistent SQLite store."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from experiments.retrieval_sanity import run_suite
from experiments.retrieval_sanity.log_observer import _collect_sqlite_summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the suite repeatedly to soak a SQLite vector store.")
    parser.add_argument("--iterations", type=int, default=5, help="Number of times to execute the suite.")
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(run_suite.SCENARIOS.keys()),
        help="Subset of scenarios to run; defaults to all presets.",
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=Path("runs/soak_vectors.db"),
        help="Database file to populate during the soak run.",
    )
    parser.add_argument(
        "--sqlite-prefix",
        type=str,
        default="soak",
        help="Prefix used when generating scenario namespaces (prefix:scenario:phase).",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("retrieval_runs.log"),
        help="Log file to append run summaries to.",
    )
    parser.add_argument(
        "--skip-log",
        action="store_true",
        help="Do not append runs to the log file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file capturing soak metadata and SQLite summary.",
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Similarity metric to enforce in the child runs.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def build_cli_tokens(
    scenario_name: str,
    sqlite_path: Path,
    scenario_id: str,
    metric: str,
) -> List[str]:
    tokens: List[str] = list(run_suite.SCENARIOS[scenario_name])
    tokens.extend(
        [
            "--vector-store",
            "sqlite",
            "--sqlite-path",
            str(sqlite_path),
            "--sqlite-scenario",
            scenario_id,
            "--metric",
            metric,
        ]
    )
    return tokens


def run_iteration(
    index: int,
    scenario_names: Sequence[str],
    sqlite_path: Path,
    prefix: str,
    metric: str,
    log_path: Path,
    skip_log: bool,
) -> List[run_suite.ScenarioResult]:
    summaries: List[run_suite.ScenarioResult] = []
    iteration_prefix = f"{prefix}-{index:02d}"

    for name in scenario_names:
        scenario_id = f"{iteration_prefix}:{name}"
        cli_tokens = build_cli_tokens(name, sqlite_path, scenario_id, metric)
        parsed_args, results = run_suite.run_scenario(name, cli_tokens)
        summaries.append(run_suite.summarise(name, results))
        if not skip_log:
            payload = run_suite.format_log_entry(name, parsed_args, results)
            run_suite.append_log(log_path, payload)
            print(
                f"[soak] Iteration {index + 1}, scenario '{name}' appended to {log_path} ({payload['timestamp']})."
            )
        else:
            print(f"[soak] Iteration {index + 1}, scenario '{name}' completed (log skipped).")

    run_suite.print_summary(summaries)
    return summaries


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    scenario_names = args.scenario or list(run_suite.SCENARIOS.keys())
    sqlite_path = args.sqlite_path
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    all_summaries: List[dict] = []

    print(
        f"[soak] Starting soak run for {args.iterations} iterations using {sqlite_path} with prefix '{args.sqlite_prefix}'."
    )

    start = time.time()
    for idx in range(args.iterations):
        summaries = run_iteration(
            idx,
            scenario_names,
            sqlite_path,
            args.sqlite_prefix,
            args.metric,
            args.log_path,
            args.skip_log,
        )
        all_summaries.extend(summary.__dict__ for summary in summaries)

    elapsed = time.time() - start
    summary = _collect_sqlite_summary(sqlite_path)

    print(f"[soak] Completed soak in {elapsed:.2f}s. SQLite summary:")
    if summary is None:
        print(f"  - Failed to read {sqlite_path}")
    elif not summary:
        print(f"  - Database {sqlite_path} is empty.")
    else:
        for entry in summary:
            other = entry["other"]
            extras = (
                ", other=" + ", ".join(f"{suffix or '<none>'}={count}" for suffix, count in sorted(other.items()))
                if other
                else ""
            )
            print(
                f"  - {entry['base']}: baseline={entry['baseline']}, pmflow={entry['pmflow']}, "
                f"total={entry['total']}{extras}"
            )

    if args.output:
        payload = {
            "generated_at": time.time(),
            "iterations": args.iterations,
            "scenarios": scenario_names,
            "sqlite_path": str(sqlite_path),
            "sqlite_prefix": args.sqlite_prefix,
            "metric": args.metric,
            "log_path": str(args.log_path),
            "skip_log": args.skip_log,
            "elapsed_seconds": elapsed,
            "summaries": all_summaries,
            "sqlite_summary": summary,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[soak] Wrote soak report to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
