#!/usr/bin/env python3
"""Fine-tune PMFlow LM from exported adaptation JSONL and emit versioned artifacts."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import torch

try:
    from pmflow.lm.pmflow_lm import PMFlowLanguageModel  # type: ignore[import-not-found]
    from pmflow.lm.train import PMFlowLMTrainer  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    from PMFlow_upstream.pmflow.lm.pmflow_lm import PMFlowLanguageModel
    from PMFlow_upstream.pmflow.lm.train import PMFlowLMTrainer


@dataclass
class AdaptationRecord:
    prompt: str
    target: str
    status: str
    feedback_score: float


def _load_vocab(vocab_path: str) -> Dict[str, int]:
    obj = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    if "token_to_id" in obj and isinstance(obj["token_to_id"], dict):
        return {str(k): int(v) for k, v in obj["token_to_id"].items()}
    if isinstance(obj, dict):
        return {str(k): int(v) for k, v in obj.items()}
    raise ValueError("Unsupported vocab format")


def _parse_records(dataset_path: str, min_feedback: float, status_filter: str) -> List[AdaptationRecord]:
    records: List[AdaptationRecord] = []
    with Path(dataset_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            status = str(obj.get("status", ""))
            if status_filter and status != status_filter:
                continue

            feedback = obj.get("feedback") or {}
            score = float(feedback.get("score", 0.0))
            if score < min_feedback:
                continue

            prompt = str(obj.get("prompt") or "").strip()
            target = str(obj.get("target") or "").strip()
            if not prompt or not target:
                continue

            records.append(
                AdaptationRecord(
                    prompt=prompt,
                    target=target,
                    status=status,
                    feedback_score=score,
                )
            )
    return records


def _tokenize(text: str) -> List[str]:
    import re

    token_re = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)
    return [t.lower() for t in token_re.findall(text)]


def _records_to_ids(records: List[AdaptationRecord], token_to_id: Dict[str, int]) -> List[int]:
    unk = token_to_id.get("<unk>", 0)
    ids: List[int] = []
    for r in records:
        seq = f"{r.prompt} {r.target}"
        ids.extend([token_to_id.get(tok, unk) for tok in _tokenize(seq)])
    return ids


def _save_versioned_artifacts(
    out_root: str,
    checkpoint_state: Dict[str, Any],
    token_to_id: Dict[str, int],
    metadata: Dict[str, Any],
) -> Tuple[str, str, str]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(out_root) / f"pmflow_lm_adapt_{ts}"
    out_dir.mkdir(parents=True, exist_ok=False)

    ckpt_path = out_dir / "pmflow_lm_checkpoint.pt"
    vocab_path = out_dir / "pmflow_lm_vocab.json"
    meta_path = out_dir / "metadata.json"

    torch.save(checkpoint_state, ckpt_path)
    vocab_path.write_text(json.dumps({"token_to_id": token_to_id}), encoding="utf-8")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")

    return str(ckpt_path), str(vocab_path), str(meta_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune PMFlow LM from Lilith adaptation dataset")
    parser.add_argument("--dataset", required=True, help="Path to adaptation JSONL dataset")
    parser.add_argument("--base-checkpoint", required=True, help="Path to base PMFlow LM checkpoint")
    parser.add_argument("--base-vocab", required=True, help="Path to base PMFlow LM vocab JSON")
    parser.add_argument("--output-root", required=True, help="Directory to write versioned adapted artifacts")
    parser.add_argument(
        "--model-kwargs-json",
        required=True,
        help="JSON string for PMFlowLanguageModel kwargs excluding vocab_size",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-feedback", type=float, default=0.1)
    parser.add_argument(
        "--status-filter",
        default="accepted",
        choices=["accepted", "rejected", "pending_feedback", ""],
        help="Filter records by status (empty string means all)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    token_to_id = _load_vocab(args.base_vocab)
    records = _parse_records(args.dataset, min_feedback=args.min_feedback, status_filter=args.status_filter)
    if not records:
        raise SystemExit("No records matched filters; nothing to train")

    token_ids = _records_to_ids(records, token_to_id)
    if len(token_ids) <= args.seq_len + 1:
        raise SystemExit("Not enough tokenized data for requested sequence length")

    model_kwargs = json.loads(args.model_kwargs_json)
    model_kwargs["vocab_size"] = len(token_to_id)

    model = PMFlowLanguageModel(**model_kwargs)
    base_checkpoint = torch.load(args.base_checkpoint, map_location="cpu")
    state_dict = base_checkpoint.get("model_state_dict", base_checkpoint)
    model.load_state_dict(state_dict, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    trainer = PMFlowLMTrainer(model, optimizer, device="cpu")

    losses: List[float] = []
    perplexities: List[float] = []

    for _epoch in range(args.epochs):
        for _step in range(args.steps_per_epoch):
            batch = []
            max_start = len(token_ids) - args.seq_len - 1
            for _ in range(args.batch_size):
                start = random.randint(0, max_start)
                batch.append(token_ids[start:start + args.seq_len])
            batch_t = torch.tensor(batch, dtype=torch.long)
            metrics = trainer.train_step(batch_t)
            losses.append(metrics["loss"])
            perplexities.append(metrics["perplexity"])

    metadata = {
        "kind": "pmflow_lm_adaptation_checkpoint",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_checkpoint": args.base_checkpoint,
        "base_vocab": args.base_vocab,
        "dataset": args.dataset,
        "status_filter": args.status_filter,
        "min_feedback": args.min_feedback,
        "record_count": len(records),
        "token_count": len(token_ids),
        "seed": args.seed,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "learning_rate": args.learning_rate,
        "model_kwargs": model_kwargs,
        "train_loss_mean": mean(losses),
        "train_perplexity_mean": mean(perplexities),
        "rollback": {
            "checkpoint": args.base_checkpoint,
            "vocab": args.base_vocab,
        },
    }

    checkpoint_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": trainer.global_step,
        "total_tokens": trainer.total_tokens,
        "epoch": args.epochs,
    }

    ckpt_path, vocab_path, meta_path = _save_versioned_artifacts(
        out_root=args.output_root,
        checkpoint_state=checkpoint_state,
        token_to_id=token_to_id,
        metadata=metadata,
    )

    print(f"records_used={len(records)}")
    print(f"tokens_used={len(token_ids)}")
    print(f"train_loss_mean={metadata['train_loss_mean']:.6f}")
    print(f"train_perplexity_mean={metadata['train_perplexity_mean']:.6f}")
    print(f"checkpoint={ckpt_path}")
    print(f"vocab={vocab_path}")
    print(f"metadata={meta_path}")


if __name__ == "__main__":
    main()
