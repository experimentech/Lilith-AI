#!/usr/bin/env python3
"""Lightweight observer for `retrieval_runs.log`.

This module provides scaffolding for an automated watcher that can surface
potential regression or drift signals from the JSONL log emitted by the
`pmflow_retrieval_lab` notebook and the `run_retrieval_sanity.py` script.

It focuses on three responsibilities:

1. Parsing log entries into structured records ready for downstream analysis.
2. Computing rolling aggregates to detect sudden drops or trends.
3. Emitting alert suggestions that future automation can route to CI, Slack,
   or scheduling queues that trigger fine-tuning jobs.

The current implementation is intentionally conservative—it only prints alert
recommendations. Hook the exposed functions into your orchestration layer to
wire side effects (retraining, notifications, etc.).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

ISO8601 = "%Y-%m-%dT%H:%M:%S.%fZ"


# ---------------------------------------------------------------------------
# Structured representations
# ---------------------------------------------------------------------------


@dataclass
class MetricSnapshot:
    """Compact representation of key metrics for a single run."""

    baseline_retrieval: Optional[float]
    pmflow_retrieval: Optional[float]
    baseline_class: Optional[float]
    pmflow_class: Optional[float]
    final_loss_baseline: Optional[float]
    final_loss_pmflow: Optional[float]

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "MetricSnapshot":
        baseline = payload.get("baseline", {})
        pmflow = payload.get("pmflow", {})
        return cls(
            baseline_retrieval=_safe_float(baseline.get("retrieval_accuracy")),
            pmflow_retrieval=_safe_float(pmflow.get("retrieval_accuracy")),
            baseline_class=_safe_float(baseline.get("class_accuracy")),
            pmflow_class=_safe_float(pmflow.get("class_accuracy")),
            final_loss_baseline=_safe_float(baseline.get("final_loss")),
            final_loss_pmflow=_safe_float(pmflow.get("final_loss")),
        )

    def retrieval_delta(self) -> Optional[float]:
        if self.baseline_retrieval is None or self.pmflow_retrieval is None:
            return None
        return self.pmflow_retrieval - self.baseline_retrieval


@dataclass
class RunRecord:
    """Full record of a retrieval sanity experiment run."""

    timestamp: datetime
    config: Dict[str, Any]
    metrics: MetricSnapshot
    device: Optional[str]
    dataset: Dict[str, Any]
    provenance: Optional[Dict[str, Any]] = None
    config_label: Optional[str] = None
    config_path: Optional[str] = None
    config_overrides: Optional[Dict[str, Any]] = None
    vector_store: Optional[str] = None
    sqlite_path: Optional[str] = None
    sqlite_scenario: Optional[str] = None

    @classmethod
    def from_json_line(cls, line: str) -> "RunRecord":
        payload = json.loads(line)
        timestamp_str = payload.get("timestamp")
        timestamp = _parse_timestamp(timestamp_str)
        metrics = MetricSnapshot.from_payload(payload.get("metrics", {}))
        raw_config = payload.get("config")
        config = raw_config if isinstance(raw_config, dict) else {}
        raw_dataset = payload.get("dataset")
        dataset = raw_dataset if isinstance(raw_dataset, dict) else {}
        device = payload.get("device")
        provenance = payload.get("provenance")
        config_label = payload.get("config_label") or config.get("label")
        config_path = payload.get("config_path")
        config_overrides = payload.get("config_overrides")
        vector_store = payload.get("vector_store")
        sqlite_path = payload.get("sqlite_path")
        sqlite_scenario = payload.get("sqlite_scenario")

        return cls(
            timestamp=timestamp,
            config=config,
            metrics=metrics,
            device=device,
            dataset=dataset,
            provenance=provenance if isinstance(provenance, dict) else None,
            config_label=str(config_label) if config_label is not None else None,
            config_path=str(config_path) if config_path is not None else None,
            config_overrides=config_overrides if isinstance(config_overrides, dict) else None,
            vector_store=str(vector_store) if vector_store is not None else None,
            sqlite_path=str(sqlite_path) if sqlite_path is not None else None,
            sqlite_scenario=str(sqlite_scenario) if sqlite_scenario is not None else None,
        )


@dataclass
class DriftPolicy:
    """Thresholds that generate alert recommendations."""

    min_retrieval_delta: float = 0.0
    max_retrieval_drop: float = 0.05
    window: int = 5
    min_runs: int = 3


@dataclass
class Alert:
    """Alert describing potential issues discovered in the log."""

    reason: str
    recent_runs: List[RunRecord]

    def __str__(self) -> str:  # pragma: no cover - formatting helper
        ts = ", ".join(r.timestamp.isoformat() for r in self.recent_runs)
        return f"{self.reason} | recent runs: {ts}"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_records(log_path: Path) -> List[RunRecord]:
    """Load all JSONL log entries into structured run records."""

    if not log_path.exists():
        return []

    runs: List[RunRecord] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                run = RunRecord.from_json_line(text)
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[observer] Skipping malformed line: {exc}", file=sys.stderr)
                continue
            runs.append(run)
    return runs


def iter_tail(path: Path, *, interval: float = 2.0) -> Iterator[str]:
    """Yield appended lines as they are written to the file."""

    with path.open("r", encoding="utf-8") as fh:
        fh.seek(0, 2)  # Seek to end
        while True:
            line = fh.readline()
            if not line:
                time.sleep(interval)
                continue
            yield line


def evaluate_policy(runs: List[RunRecord], policy: DriftPolicy) -> List[Alert]:
    """Evaluate recent runs against the drift policy."""

    if len(runs) < policy.min_runs:
        return []

    alerts: List[Alert] = []
    recent = runs[-policy.window :]

    # Check for PMFlow under-performing vs baseline.
    negative_deltas = [r.metrics.retrieval_delta() for r in recent]
    negative_deltas = [d for d in negative_deltas if d is not None]
    if negative_deltas and min(negative_deltas) < policy.min_retrieval_delta:
        reason = (
            "PMFlow retrieval delta dropped below minimum allowed "
            f"({min(negative_deltas):.4f} < {policy.min_retrieval_delta:.4f})."
        )
        alerts.append(Alert(reason=_reason_with_context(reason, recent), recent_runs=recent))

    # Check for absolute retrieval drops.
    pmflow_vals = [r.metrics.pmflow_retrieval for r in recent if r.metrics.pmflow_retrieval is not None]
    if pmflow_vals:
        current = pmflow_vals[-1]
        baseline_val = max(pmflow_vals)
        if baseline_val - current > policy.max_retrieval_drop:
            reason = (
                "PMFlow retrieval accuracy dropped more than "
                f"{policy.max_retrieval_drop:.4f} from recent best "
                f"({baseline_val:.4f} → {current:.4f})."
            )
            alerts.append(Alert(reason=_reason_with_context(reason, recent), recent_runs=recent))

    return alerts


# ---------------------------------------------------------------------------
# CLI utilities
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor retrieval_runs.log for drift signals.")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path.cwd() / "retrieval_runs.log",
        help="Path to the JSONL log file produced by the notebook or CLI.",
    )
    parser.add_argument("--window", type=int, default=5, help="Number of recent runs to inspect.")
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum acceptable PMFlow vs baseline retrieval delta.",
    )
    parser.add_argument(
        "--max-drop",
        type=float,
        default=0.05,
        help="Maximum allowed drop from the recent PMFlow best accuracy.",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Follow the log file for new entries instead of evaluating once.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval (seconds) when --follow is enabled.",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=3,
        help="Minimum number of runs before alerts are considered.",
    )
    parser.add_argument(
        "--sqlite-stats",
        action="store_true",
        help="When set, report embedding counts from a SQLite vector store after evaluation.",
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        help="Optional override for the SQLite database path when reporting stats.",
    )
    arg_list = list(argv) if argv is not None else None
    return parser.parse_args(arg_list)


def run_once(args: argparse.Namespace) -> List[Alert]:
    runs = load_records(args.log_path)
    policy = DriftPolicy(
        min_retrieval_delta=args.min_delta,
        max_retrieval_drop=args.max_drop,
        window=args.window,
        min_runs=args.min_runs,
    )
    alerts = evaluate_policy(runs, policy)
    for alert in alerts:
        print(f"[observer] {alert}")
    if not alerts:
        print(
            "[observer] No alerts. Last PMFlow retrieval: "
            f"{runs[-1].metrics.pmflow_retrieval if runs else 'N/A'}"
        )
    if args.sqlite_stats:
        _report_sqlite_stats(args.sqlite_path, runs)
    return alerts


def run_follow(args: argparse.Namespace) -> None:
    """Continuously monitor the log file for new runs."""

    print(f"[observer] Watching {args.log_path} (Ctrl+C to stop)…")
    existing_runs = load_records(args.log_path)
    policy = DriftPolicy(
        min_retrieval_delta=args.min_delta,
        max_retrieval_drop=args.max_drop,
        window=args.window,
        min_runs=args.min_runs,
    )
    if existing_runs:
        evaluate_policy(existing_runs, policy)

    try:
        for line in iter_tail(args.log_path, interval=args.interval):
            try:
                existing_runs.append(RunRecord.from_json_line(line))
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[observer] Failed to parse new line: {exc}", file=sys.stderr)
                continue
            alerts = evaluate_policy(existing_runs, policy)
            for alert in alerts:
                print(f"[observer] {alert}")
            if args.sqlite_stats:
                _report_sqlite_stats(args.sqlite_path, existing_runs)
    except KeyboardInterrupt:  # pragma: no cover - CLI convenience
        print("\n[observer] Stopped watching.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.strptime(value, ISO8601)
    except ValueError:
        # Support truncated millisecond formats
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)


def _reason_with_context(reason: str, runs: Sequence[RunRecord]) -> str:
    if not runs:
        return reason

    latest = runs[-1]
    context_parts: List[str] = []

    label = latest.config_label or latest.config.get("label")
    if label:
        context_parts.append(f"config={label}")

    if latest.config_path:
        context_parts.append(f"config_path={Path(latest.config_path).name}")

    if latest.config_overrides:
        override_keys = ",".join(sorted(str(key) for key in latest.config_overrides.keys()))
        if override_keys:
            context_parts.append(f"overrides={override_keys}")

    device = latest.device
    provenance = latest.provenance if isinstance(latest.provenance, dict) else {}
    if not device:
        device = provenance.get("device")
    if device:
        context_parts.append(f"device={device}")

    git_info = provenance.get("git") if isinstance(provenance, dict) else None
    if isinstance(git_info, dict):
        commit = git_info.get("commit")
        if commit:
            context_parts.append(f"git={commit[:7]}")

    runner = provenance.get("runner") if isinstance(provenance, dict) else None
    if runner:
        context_parts.append(f"runner={runner}")

    dataset = latest.dataset if isinstance(latest.dataset, dict) else {}
    data_mode = dataset.get("data_mode") or latest.config.get("data_mode")
    if data_mode:
        context_parts.append(f"data_mode={data_mode}")
    support = dataset.get("support_samples")
    query = dataset.get("query_samples")
    if support is not None and query is not None:
        context_parts.append(f"samples={support}/{query}")

    if not context_parts:
        return reason

    return f"{reason} [context: {', '.join(context_parts)}]"


def _report_sqlite_stats(cli_path: Optional[Path], runs: Sequence[RunRecord]) -> None:
    path = _infer_sqlite_path(cli_path, runs)
    if not path:
        print("[observer] SQLite stats requested but no database path was found.")
        return
    summary = _collect_sqlite_summary(path)
    if summary is None:
        print(f"[observer] SQLite stats unavailable; failed to read {path}.")
        return
    if not summary:
        print(f"[observer] SQLite stats: database {path} contains no embeddings.")
        return
    print(f"[observer] SQLite stats from {path}:")
    for entry in summary:
        extras = ""
        if entry["other"]:
            formatted = ", ".join(f"{suffix or '<none>'}={count}" for suffix, count in sorted(entry["other"].items()))
            extras = f", other={formatted}"
        print(
            f"    {entry['base']}: baseline={entry['baseline']}, pmflow={entry['pmflow']}, "
            f"total={entry['total']}{extras}"
        )


def _infer_sqlite_path(cli_path: Optional[Path], runs: Sequence[RunRecord]) -> Optional[Path]:
    if cli_path:
        return cli_path
    for record in reversed(runs):
        if record.vector_store != "sqlite":
            continue
        if record.sqlite_path:
            return Path(record.sqlite_path)
    return None


def _collect_sqlite_summary(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return []
    try:
        with sqlite3.connect(path) as conn:
            rows = conn.execute(
                "SELECT scenario, COUNT(*) FROM embeddings GROUP BY scenario ORDER BY scenario"
            ).fetchall()
    except sqlite3.Error:
        return None

    aggregates: Dict[str, Dict[str, Any]] = {}
    for scenario, count in rows:
        base, suffix = _split_scenario(scenario)
        entry = aggregates.setdefault(
            base,
            {"baseline": 0, "pmflow": 0, "other": {}},
        )
        if suffix == "baseline":
            entry["baseline"] = count
        elif suffix == "pmflow":
            entry["pmflow"] = count
        else:
            entry["other"][suffix] = count

    summary: List[Dict[str, Any]] = []
    for base, data in sorted(aggregates.items()):
        total = data["baseline"] + data["pmflow"] + sum(data["other"].values())
        summary.append(
            {
                "base": base,
                "baseline": data["baseline"],
                "pmflow": data["pmflow"],
                "other": data["other"],
                "total": total,
            }
        )
    return summary


def _split_scenario(value: Any) -> tuple[str, Optional[str]]:
    if value is None:
        return "<none>", None
    name = str(value)
    if name.endswith(":baseline"):
        return name[: -len(":baseline")], "baseline"
    if name.endswith(":pmflow"):
        return name[: -len(":pmflow")], "pmflow"
    if name == "baseline" or name == "pmflow":
        return "<default>", name
    return name, None


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.follow:
        run_follow(args)
        return 0
    run_once(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
