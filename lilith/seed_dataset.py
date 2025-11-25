"""Seed a SQLite scenario with deterministic utterances for plasticity experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import torch

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[2]
import sys

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from experiments.retrieval_sanity.pipeline import SymbolicPipeline, SymbolicStore, Utterance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed the pipeline store from a JSON or JSONL dataset.")
    parser.add_argument(
        "dataset_path",
        type=Path,
        nargs="?",
        default=Path("experiments/retrieval_sanity/datasets/interactive_seed.jsonl"),
        help="Path to a JSONL (one utterance per line) or JSON list of utterances.",
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=Path("runs/pipeline_interactive_vectors.db"),
        help="Destination SQLite database for embeddings.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="interactive",
        help="Scenario namespace stored inside SQLite and frame dumps.",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Override language for all utterances (defaults to per-entry language or 'unknown').",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove existing embeddings and frame dumps for the scenario before seeding.",
    )
    parser.add_argument(
        "--pmflow-state-path",
        type=Path,
        help="Optional PMFlow state file passed to the encoder for consistent plasticity warm-starts.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Verify round-trip retrieval after seeding by querying the stored embeddings.",
    )
    return parser.parse_args()


def load_utterances(path: Path, *, default_language: str | None) -> List[Utterance]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    entries: List[dict]
    if path.suffix == ".jsonl":
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "utterances" in payload:
            entries = list(payload["utterances"])
        elif isinstance(payload, list):
            entries = list(payload)
        else:
            raise ValueError("Dataset file must be a list or an object with an 'utterances' list.")

    utterances: List[Utterance] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict) or "text" not in entry:
            raise ValueError(f"Entry {idx} missing required 'text' field: {entry!r}")
        text = str(entry["text"])
        language = default_language or str(entry.get("language", "unknown"))
        metadata = entry.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Entry {idx} has non-dict metadata: {metadata!r}")
        utterances.append(Utterance(text=text, language=language, metadata=metadata))

    if not utterances:
        raise ValueError(f"Dataset {path} produced no utterances.")
    return utterances


def batch_process(pipeline: SymbolicPipeline, utterances: Iterable[Utterance]):
    return pipeline.run(list(utterances))


def main() -> None:
    args = parse_args()
    utterances = load_utterances(args.dataset_path, default_language=args.language)

    pipeline = SymbolicPipeline(pmflow_state_path=args.pmflow_state_path)
    artefacts = batch_process(pipeline, utterances)

    store = SymbolicStore(args.sqlite_path, scenario=args.scenario)
    store.vector_store.path.parent.mkdir(parents=True, exist_ok=True)

    if args.clear:
        store.vector_store.clear(scenario=args.scenario)
        if store.frames_path.exists():
            store.frames_path.unlink()
        existing_count = 0
    else:
        existing_count = store.vector_store.count(scenario=args.scenario)

    store.persist(artefacts)

    print(f"[seed] Stored {len(artefacts)} artefacts under scenario '{args.scenario}'.")
    print(f"[seed] SQLite path : {store.vector_store.path}")
    print(f"[seed] Frames path : {store.frames_path}")

    if args.validate:
        queries = torch.cat([artefact.embedding for artefact in artefacts], dim=0)
        scores, labels = store.vector_store.search(queries, topk=1, scenario=args.scenario)
        expected = torch.arange(existing_count, existing_count + labels.shape[0], dtype=torch.long)
        matched = (labels.squeeze(1).cpu() == expected).float().mean().item()
        print(f"[seed] Retrieval self-check accuracy: {matched:.3f}")
        print(f"[seed] Sample scores: {[round(float(s), 3) for s in scores.squeeze(1).tolist()]}")


if __name__ == "__main__":
    main()
