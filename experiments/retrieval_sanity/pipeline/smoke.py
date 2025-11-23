"""Run a minimal language-to-symbol pipeline smoke test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import torch

from experiments.retrieval_sanity.pipeline import (
    SymbolicPipeline,
    SymbolicStore,
    TemplateDecoder,
    Utterance,
)


SAMPLES = [
    Utterance(text="I went to the hospitol yesterday.", language="en"),
    Utterance(text="Ella fui al hospital para ver a su madre.", language="es"),
    Utterance(text="We realy need another appointment soon!", language="en"),
]


def main() -> None:
    pipeline = SymbolicPipeline()
    artefacts = pipeline.run(SAMPLES)
    decoder = TemplateDecoder()

    sqlite_path = Path("runs/pipeline_smoke_vectors.db")
    store = SymbolicStore(sqlite_path, scenario="pipeline-smoke")
    store.persist(artefacts)

    # Use the stored embeddings as queries to verify round-trip retrieval.
    queries = torch.cat([artifact.embedding for artifact in artefacts], dim=0)
    scores, labels = store.vector_store.search(queries, topk=1, scenario="pipeline-smoke")

    frame_dump_path = Path("runs") / "pipeline-smoke_frames.json"
    frames = json.loads(frame_dump_path.read_text(encoding="utf-8"))

    print("=== Pipeline smoke summary ===")
    for idx, (artifact, label, score) in enumerate(zip(artefacts, labels.squeeze(1), scores.squeeze(1))):
        frame = frames[label]
        print(f"Sample {idx}: text='{artifact.utterance.text}'")
        print(f"  Normalised : {artifact.normalised_text}")
        print(f"  Frame      : actor={frame.get('actor')}, action={frame.get('action')}, target={frame.get('target')}")
        print(f"  Decoder    : {decoder.generate(artifact.frame)}")
        print(f"  Retrieval  : label={int(label)}, score={float(score):.4f}")
        print()

    print(f"Frames saved to: {frame_dump_path}")
    print(f"SQLite path   : {sqlite_path}")


if __name__ == "__main__":
    main()
