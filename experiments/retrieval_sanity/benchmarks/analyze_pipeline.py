#!/usr/bin/env python3
"""Aggregate retrieval logs and SQLite metadata to assess pipeline stability."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, cast

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from experiments.retrieval_sanity.log_observer import _collect_sqlite_summary


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise PMFlow vs baseline metrics and SQLite counts.")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("retrieval_runs.log"),
        help="JSONL log produced by run_suite / run_retrieval_sanity.",
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        help="SQLite database to cross-check. Defaults to the most common path in the log.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="Filter to one or more scenario names (matches the 'scenario' field in the log).",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        help="Filter by SQLite scenario prefix (e.g. 'soak-01').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file capturing the aggregate results.",
    )
    parser.add_argument(
        "--frames-path",
        action="append",
        type=Path,
        help="Optional JSON dumps produced by the symbol pipeline smoke tests.",
    )
    parser.add_argument(
        "--trace-path",
        action="append",
        type=Path,
        help="Optional JSONL trace files produced by the conversation responder.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_log(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    entries: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in log: {exc}")
    return entries


def filter_entries(entries: List[Dict[str, object]], scenario_filters: Optional[List[str]], prefix_filters: Optional[List[str]]) -> List[Dict[str, object]]:
    filtered = []
    for entry in entries:
        scenario = str(entry.get("scenario", ""))
        sqlite_scenario = str(entry.get("sqlite_scenario", "")) if entry.get("sqlite_scenario") else None
        if scenario_filters and scenario not in scenario_filters:
            continue
        if prefix_filters and sqlite_scenario:
            if not any(sqlite_scenario.startswith(prefix) for prefix in prefix_filters):
                continue
        filtered.append(entry)
    return filtered


def infer_sqlite_path(entries: List[Dict[str, object]]) -> Optional[Path]:
    counts: Dict[Path, int] = defaultdict(int)
    for entry in entries:
        sqlite_path = entry.get("sqlite_path")
        if isinstance(sqlite_path, str) and sqlite_path:
            counts[Path(sqlite_path)] += 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def aggregate_metrics(entries: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    sqlite_ids: Dict[str, List[str]] = defaultdict(list)
    for entry in entries:
        scenario = str(entry.get("scenario", "unknown"))
        metrics = entry.get("metrics", {})
        baseline = metrics.get("baseline", {}) if isinstance(metrics, dict) else {}
        pmflow = metrics.get("pmflow", {}) if isinstance(metrics, dict) else {}
        if baseline.get("retrieval_accuracy") is not None:
            grouped[scenario]["baseline_retrieval"].append(float(baseline["retrieval_accuracy"]))
        if pmflow.get("retrieval_accuracy") is not None:
            grouped[scenario]["pmflow_retrieval"].append(float(pmflow["retrieval_accuracy"]))
        if baseline.get("class_accuracy") is not None:
            grouped[scenario]["baseline_class"].append(float(baseline["class_accuracy"]))
        if pmflow.get("class_accuracy") is not None:
            grouped[scenario]["pmflow_class"].append(float(pmflow["class_accuracy"]))
        delta_value = entry.get("retrieval_delta")
        if isinstance(delta_value, (int, float, str)) and delta_value is not None:
            grouped[scenario]["retrieval_delta"].append(float(delta_value))
        if entry.get("sqlite_scenario"):
            sqlite_ids[scenario].append(str(entry["sqlite_scenario"]))
    summary: Dict[str, Dict[str, object]] = {}
    for scenario, metrics in grouped.items():
        entry_summary: Dict[str, object] = {
            "runs": len(metrics.get("baseline_retrieval", [])),
            "sqlite_scenarios": sqlite_ids.get(scenario, []),
        }
        for key, values in metrics.items():
            if not values:
                continue
            entry_summary[f"{key}_mean"] = statistics.mean(values)
            if len(values) > 1:
                entry_summary[f"{key}_stdev"] = statistics.pstdev(values)
        summary[scenario] = entry_summary
    return summary


def summarise_frames(paths: List[Path]) -> Dict[str, object]:
    total = 0
    confidences: List[float] = []
    languages: Dict[str, int] = defaultdict(int)
    slot_counts: Dict[str, int] = defaultdict(int)
    path_totals: Dict[str, int] = {}

    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            frames = json.load(handle)
        path_totals[str(path)] = len(frames)
        total += len(frames)
        for frame in frames:
            confidence = frame.get("confidence")
            if isinstance(confidence, (int, float)):
                confidences.append(float(confidence))
            elif isinstance(confidence, str):
                try:
                    confidences.append(float(confidence))
                except ValueError:
                    pass
            language = frame.get("language") or "unknown"
            languages[str(language)] += 1
            for slot in ("actor", "action", "target"):
                if frame.get(slot):
                    slot_counts[slot] += 1

    mean_conf = statistics.mean(confidences) if confidences else None
    return {
        "total_frames": total,
        "mean_confidence": mean_conf,
        "languages": dict(sorted(languages.items(), key=lambda kv: kv[0])),
        "slot_support": dict(sorted(slot_counts.items(), key=lambda kv: kv[0])),
        "path_totals": path_totals,
    }


def summarise_traces(paths: List[Path]) -> Dict[str, object]:
    total = 0
    recall_scores: List[float] = []
    first_neighbour_scores: List[float] = []
    per_scenario: Dict[str, int] = defaultdict(int)
    languages: Dict[str, int] = defaultdict(int)
    path_totals: Dict[str, int] = {}

    for path in paths:
        if not path.exists():
            continue
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    continue
                count += 1
                total += 1
                scenario = str(record.get("scenario", "unknown"))
                per_scenario[scenario] += 1
                language = str(record.get("language", "unknown"))
                languages[language] += 1
                recall_score = record.get("recall_score")
                if isinstance(recall_score, (int, float)):
                    recall_scores.append(float(recall_score))
                nearest = record.get("nearest")
                if isinstance(nearest, list) and nearest:
                    top = nearest[0]
                    score = top.get("score") if isinstance(top, dict) else None
                    if isinstance(score, (int, float)):
                        first_neighbour_scores.append(float(score))
        path_totals[str(path)] = count

    recall_rate = len(recall_scores) / total if total else None
    mean_recall = statistics.mean(recall_scores) if recall_scores else None
    mean_top_neighbour = statistics.mean(first_neighbour_scores) if first_neighbour_scores else None

    return {
        "total_turns": total,
        "recall_rate": recall_rate,
        "mean_recall_score": mean_recall,
        "mean_top_neighbour_score": mean_top_neighbour,
        "scenarios": dict(sorted(per_scenario.items(), key=lambda kv: kv[0])),
        "languages": dict(sorted(languages.items(), key=lambda kv: kv[0])),
        "path_totals": path_totals,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    entries = load_log(args.log_path)
    entries = filter_entries(entries, args.scenario, args.prefix)
    if not entries:
        print("[analysis] No log entries matched the filters.")
        return 0

    sqlite_path = args.sqlite_path or infer_sqlite_path(entries)
    sqlite_summary = _collect_sqlite_summary(sqlite_path) if sqlite_path else None

    metric_summary = aggregate_metrics(entries)

    print("[analysis] Run summary:")
    for scenario, data in metric_summary.items():
        runs = data.get("runs", 0)
        baseline = data.get("baseline_retrieval_mean")
        pmflow = data.get("pmflow_retrieval_mean")
        delta = data.get("retrieval_delta_mean")
        baseline_text = f"{baseline:.4f}" if baseline is not None else "N/A"
        pmflow_text = f"{pmflow:.4f}" if pmflow is not None else "N/A"
        delta_text = f"{delta:.4f}" if delta is not None else "N/A"
        print(
            f"  - {scenario}: runs={runs}, baseline={baseline_text}, pmflow={pmflow_text}, delta={delta_text}"
        )

    if sqlite_path:
        print(f"\n[analysis] SQLite summary ({sqlite_path}):")
        if sqlite_summary is None:
            print("  - Failed to read database")
        elif not sqlite_summary:
            print("  - Database is empty")
        else:
            for entry in sqlite_summary:
                other = entry["other"]
                extras = (
                    ", other=" + ", ".join(f"{suffix or '<none>'}={count}" for suffix, count in sorted(other.items()))
                    if other
                    else ""
                )
                print(
                    f"  - {entry['base']}: baseline={entry['baseline']}, pmflow={entry['pmflow']}, total={entry['total']}{extras}"
                )

    frame_summary = None
    if args.frames_path:
        frame_summary = summarise_frames(args.frames_path)
        print("\n[analysis] Symbolic frames summary:")
        print(f"  Total frames: {frame_summary['total_frames']}")
        mean_conf = frame_summary.get("mean_confidence")
        if mean_conf is not None:
            print(f"  Mean confidence: {mean_conf:.3f}")
        languages = cast(Dict[str, int], frame_summary.get("languages") or {})
        if languages:
            print("  Languages:")
            for lang, count in languages.items():
                print(f"    - {lang}: {count}")
        slot_support = cast(Dict[str, int], frame_summary.get("slot_support") or {})
        if slot_support:
            print("  Slot coverage:")
            for slot, count in slot_support.items():
                print(f"    - {slot}: {count}")
        path_totals = cast(Dict[str, int], frame_summary.get("path_totals") or {})
        if path_totals:
            print("  Paths:")
            for path, count in path_totals.items():
                print(f"    - {path}: {count}")

    trace_summary = None
    if args.trace_path:
        trace_summary = summarise_traces(args.trace_path)
        print("\n[analysis] Responder trace summary:")
        print(f"  Total turns: {trace_summary['total_turns']}")
        if trace_summary["recall_rate"] is not None:
            print(f"  Recall rate: {trace_summary['recall_rate']:.3f}")
        if trace_summary["mean_recall_score"] is not None:
            print(f"  Mean recall score: {trace_summary['mean_recall_score']:.3f}")
        if trace_summary["mean_top_neighbour_score"] is not None:
            print(f"  Mean top neighbour score: {trace_summary['mean_top_neighbour_score']:.3f}")
        scenarios = cast(Dict[str, int], trace_summary.get("scenarios") or {})
        if scenarios:
            print("  Scenarios:")
            for scenario, count in scenarios.items():
                print(f"    - {scenario}: {count}")
        languages = cast(Dict[str, int], trace_summary.get("languages") or {})
        if languages:
            print("  Languages:")
            for lang, count in languages.items():
                print(f"    - {lang}: {count}")
        path_totals = cast(Dict[str, int], trace_summary.get("path_totals") or {})
        if path_totals:
            print("  Paths:")
            for path, count in path_totals.items():
                print(f"    - {path}: {count}")

    if args.output:
        payload = {
            "log_path": str(args.log_path),
            "sqlite_path": str(sqlite_path) if sqlite_path else None,
            "filters": {
                "scenarios": args.scenario,
                "prefixes": args.prefix,
            },
            "metrics": metric_summary,
            "sqlite_summary": sqlite_summary,
            "frame_summary": frame_summary,
            "trace_summary": trace_summary,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n[analysis] Wrote JSON report to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
