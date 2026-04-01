#!/usr/bin/env python3
"""Run full LM adaptation cycle: export episodes, retrain PMFlow LM, print config snippet."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return (result.stdout or "").strip()


def _parse_keyvals(output: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _build_pmflow_lm_config(checkpoint: str, vocab: str, metadata_path: str) -> Dict[str, object]:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    model_kwargs = dict(metadata.get("model_kwargs") or {})
    model_kwargs.pop("vocab_size", None)
    return {
        "enabled": True,
        "checkpoint_path": checkpoint,
        "vocab_path": vocab,
        "device": "cpu",
        "model_kwargs": model_kwargs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run export + retrain adaptation loop for PMFlow LM")

    parser.add_argument("--graph-db", required=True, help="Path to Lilith graph SQLite DB")
    parser.add_argument("--base-checkpoint", required=True, help="Path to base PMFlow LM checkpoint")
    parser.add_argument("--base-vocab", required=True, help="Path to base PMFlow LM vocab JSON")
    parser.add_argument("--output-root", required=True, help="Directory for versioned adapted artifacts")
    parser.add_argument(
        "--model-kwargs-json",
        required=True,
        help="JSON string for PMFlowLanguageModel kwargs excluding vocab_size",
    )

    parser.add_argument("--dataset-out", default="", help="Optional explicit path for exported JSONL")
    parser.add_argument(
        "--dataset-status",
        default="accepted",
        choices=["accepted", "rejected", "pending_feedback", ""],
        help="Episode status filter for export",
    )
    parser.add_argument("--dataset-limit", type=int, default=5000)

    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-feedback", type=float, default=0.1)

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    py = sys.executable

    dataset_path = args.dataset_out or tempfile.mktemp(prefix="lm_adapt_", suffix=".jsonl")

    export_script = repo_root / "scripts" / "export_lm_adaptation_dataset.py"
    retrain_script = repo_root / "scripts" / "retrain_pmflow_lm_from_adaptation.py"

    export_cmd = [
        py,
        str(export_script),
        "--graph-db",
        args.graph_db,
        "--output",
        dataset_path,
        "--status",
        args.dataset_status,
        "--limit",
        str(args.dataset_limit),
    ]
    export_out = _run(export_cmd)
    export_kv = _parse_keyvals(export_out)

    wrote = int(export_kv.get("wrote_records", "0") or "0")
    if wrote <= 0:
        print("wrote_records=0")
        print(f"dataset={dataset_path}")
        print("message=No adaptation records matched filters; skipping retrain")
        return

    retrain_cmd = [
        py,
        str(retrain_script),
        "--dataset",
        dataset_path,
        "--base-checkpoint",
        args.base_checkpoint,
        "--base-vocab",
        args.base_vocab,
        "--output-root",
        args.output_root,
        "--model-kwargs-json",
        args.model_kwargs_json,
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--steps-per-epoch",
        str(args.steps_per_epoch),
        "--batch-size",
        str(args.batch_size),
        "--seq-len",
        str(args.seq_len),
        "--learning-rate",
        str(args.learning_rate),
        "--min-feedback",
        str(args.min_feedback),
        "--status-filter",
        args.dataset_status,
    ]

    retrain_out = _run(retrain_cmd)
    retrain_kv = _parse_keyvals(retrain_out)

    checkpoint = retrain_kv.get("checkpoint", "")
    vocab = retrain_kv.get("vocab", "")
    metadata = retrain_kv.get("metadata", "")

    print(f"dataset={dataset_path}")
    print(f"wrote_records={wrote}")
    print(f"records_used={retrain_kv.get('records_used', '')}")
    print(f"checkpoint={checkpoint}")
    print(f"vocab={vocab}")
    print(f"metadata={metadata}")

    if checkpoint and vocab and metadata:
        cfg = _build_pmflow_lm_config(checkpoint=checkpoint, vocab=vocab, metadata_path=metadata)
        print("pmflow_lm_config_json=")
        print(json.dumps(cfg, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
