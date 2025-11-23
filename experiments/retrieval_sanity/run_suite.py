#!/usr/bin/env python3
"""Run a batch of retrieval sanity scenarios and record provenance-rich results."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from experiments.retrieval_sanity import run_retrieval_sanity as runner
from experiments.retrieval_sanity.log_observer import DriftPolicy, evaluate_policy, load_records

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_LOG_PATH = Path.cwd() / "retrieval_runs.log"
DEFAULT_DB_PATH = Path.cwd() / "retrieval_vectors.db"
ISO8601 = "%Y-%m-%dT%H:%M:%S.%fZ"


SCENARIOS: Dict[str, Sequence[str]] = {
    "synthetic-baseline": ("--config", str(CONFIG_DIR / "synthetic_baseline.json")),
    "synthetic-hard": ("--config", str(CONFIG_DIR / "synthetic_hard.json")),
    "relational-sample": ("--config", str(CONFIG_DIR / "relational_sample.json")),
}


@dataclass
class ScenarioResult:
    name: str
    baseline_retrieval: float
    pmflow_retrieval: float
    baseline_class: float
    pmflow_class: float
    delta: float
    config_label: str | None


def parse_suite_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a suite of retrieval sanity scenarios.")
    parser.add_argument(
        "--scenario",
        "-s",
        action="append",
        choices=sorted(SCENARIOS.keys()),
        help="Scenario name to execute. Specify multiple times or omit to run all presets.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="JSONL file to append run records to (for the observer).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write a JSON summary of the executed scenarios.",
    )
    parser.add_argument(
        "--skip-log",
        action="store_true",
        help="Run scenarios but do not append structured entries to the log file.",
    )
    parser.add_argument(
        "--evaluate-policy",
        action="store_true",
        help="Evaluate the drift policy against the log after the suite completes.",
    )
    parser.add_argument(
        "--policy-window",
        type=int,
        default=5,
        help="Window size to use when evaluating the drift policy.",
    )
    parser.add_argument(
        "--policy-min-delta",
        type=float,
        default=0.0,
        help="Minimum acceptable PMFlow vs baseline retrieval delta for policy evaluation.",
    )
    parser.add_argument(
        "--policy-max-drop",
        type=float,
        default=0.05,
        help="Maximum allowed drop from recent PMFlow best accuracy for policy evaluation.",
    )
    parser.add_argument(
        "--vector-store",
        choices=["memory", "sqlite"],
        default="memory",
        help="Backend passed through to run_retrieval_sanity.",
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Database file used when running scenarios with the SQLite vector store.",
    )
    parser.add_argument(
        "--sqlite-prefix",
        type=str,
        help="Optional prefix for SQLite scenarios; final scenario becomes '<prefix>:<name>'.",
    )
    return parser.parse_args()


def run_scenario(name: str, cli_tokens: Sequence[str]) -> tuple[argparse.Namespace, Dict[str, float]]:
    args = runner.parse_args(list(cli_tokens))
    runner.set_seed(args.seed)
    results = runner.run_experiment(args)
    return args, results


def summarise(name: str, results: Dict[str, float]) -> ScenarioResult:
    baseline_retrieval = float(results["baseline_retrieval_accuracy"])
    pmflow_retrieval = float(results["pmflow_retrieval_accuracy"])
    baseline_class = float(results["baseline_class_accuracy"])
    pmflow_class = float(results["pmflow_class_accuracy"])
    delta = pmflow_retrieval - baseline_retrieval
    config_label = results.get("config_label") or results.get("data_mode")
    return ScenarioResult(
        name=name,
        baseline_retrieval=baseline_retrieval,
        pmflow_retrieval=pmflow_retrieval,
        baseline_class=baseline_class,
        pmflow_class=pmflow_class,
        delta=delta,
        config_label=str(config_label) if config_label is not None else None,
    )


def format_log_entry(name: str, args: argparse.Namespace, results: Dict[str, float]) -> Dict[str, object]:
    timestamp = datetime.now(timezone.utc).strftime(ISO8601)
    metrics = {
        "baseline": {
            "retrieval_accuracy": results.get("baseline_retrieval_accuracy"),
            "class_accuracy": results.get("baseline_class_accuracy"),
            "final_loss": None,
        },
        "pmflow": {
            "retrieval_accuracy": results.get("pmflow_retrieval_accuracy"),
            "class_accuracy": results.get("pmflow_class_accuracy"),
            "final_loss": None,
        },
    }
    dataset = {
        "support_samples": results.get("support_samples"),
        "query_samples": results.get("query_samples"),
        "data_mode": results.get("data_mode"),
    }
    payload: Dict[str, object] = {
        "timestamp": timestamp,
        "scenario": name,
        "config": _sanitize(vars(args)),
        "metrics": metrics,
        "device": results.get("device"),
        "dataset": dataset,
        "provenance": results.get("provenance"),
        "config_label": results.get("config_label"),
        "config_path": results.get("config_path"),
        "config_overrides": results.get("config_overrides"),
    }
    vector_store = results.get("vector_store")
    if vector_store:
        payload["vector_store"] = vector_store
    if results.get("sqlite_path"):
        payload["sqlite_path"] = results.get("sqlite_path")
    if results.get("sqlite_scenario"):
        payload["sqlite_scenario"] = results.get("sqlite_scenario")
    delta = results.get("pmflow_retrieval_accuracy")
    baseline = results.get("baseline_retrieval_accuracy")
    if delta is not None and baseline is not None:
        payload["retrieval_delta"] = float(delta) - float(baseline)
    return payload


def append_log(log_path: Path, payload: Dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, default=_json_default)
        handle.write("\n")


def _sanitize(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _sanitize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(item) for item in value]
    return value


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def print_summary(summaries: Iterable[ScenarioResult]) -> None:
    summaries = list(summaries)
    if not summaries:
        print("No scenarios executed.")
        return

    header = f"{'scenario':20} {'baseline':>10} {'pmflow':>10} {'delta':>10}"
    print("\n=== Retrieval summary ===")
    print(header)
    print("-" * len(header))
    for item in summaries:
        print(
            f"{item.name:20} "
            f"{item.baseline_retrieval:>10.4f} "
            f"{item.pmflow_retrieval:>10.4f} "
            f"{item.delta:>+10.4f}"
        )
    print("-" * len(header))


def evaluate_policy_if_requested(args: argparse.Namespace) -> None:
    if not args.evaluate_policy:
        return
    records = load_records(args.log_path)
    if not records:
        print("[suite] No log records available for policy evaluation.")
        return
    policy = DriftPolicy(
        min_retrieval_delta=args.policy_min_delta,
        max_retrieval_drop=args.policy_max_drop,
        window=args.policy_window,
        min_runs=max(3, min(args.policy_window, len(records))),
    )
    alerts = evaluate_policy(records, policy)
    if not alerts:
        last = records[-1]
        delta = last.metrics.retrieval_delta()
        print(
            "[suite] Policy evaluation produced no alerts. "
            f"Latest delta: {delta if delta is not None else 'N/A'}"
        )
        return
    print("[suite] Policy alerts detected:")
    for alert in alerts:
        print(f"  - {alert.reason}")


def main() -> int:
    args = parse_suite_args()
    scenario_names = args.scenario or list(SCENARIOS.keys())

    summaries: List[ScenarioResult] = []
    for name in scenario_names:
        cli_tokens = list(SCENARIOS[name])
        cli_tokens.extend(["--vector-store", args.vector_store])
        if args.vector_store == "sqlite":
            cli_tokens.extend(["--sqlite-path", str(args.sqlite_path)])
            scenario_id = f"{args.sqlite_prefix}:{name}" if args.sqlite_prefix else name
            cli_tokens.extend(["--sqlite-scenario", scenario_id])
        print(f"\n[suite] Running scenario '{name}' ...")
        parsed_args, results = run_scenario(name, cli_tokens)
        summaries.append(summarise(name, results))
        if not args.skip_log:
            payload = format_log_entry(name, parsed_args, results)
            append_log(args.log_path, payload)
            print(f"[suite] Appended run to {args.log_path} ({payload['timestamp']}).")

    print_summary(summaries)

    if args.output:
        data = {
            "generated_at": datetime.now(timezone.utc).strftime(ISO8601),
            "log_path": str(args.log_path),
            "scenarios": [summary.__dict__ for summary in summaries],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[suite] Wrote summary to {args.output}.")

    evaluate_policy_if_requested(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
