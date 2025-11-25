"""Interactive CLI for the language-to-symbol pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from experiments.retrieval_sanity.pipeline import SymbolicPipeline  # noqa: E402
from experiments.retrieval_sanity.pipeline import SymbolicStore
from experiments.retrieval_sanity.pipeline import TemplateDecoder
from experiments.retrieval_sanity.pipeline import Utterance
from experiments.retrieval_sanity.pipeline import ConversationResponder
from experiments.retrieval_sanity.pipeline import TraceLogger
from experiments.retrieval_sanity.pipeline import PMFlowEmbeddingEncoder
from experiments.retrieval_sanity.pipeline import PlasticityController
from experiments.retrieval_sanity.pipeline import ConversationState


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive symbol pipeline shell")
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=Path("runs/pipeline_interactive_vectors.db"),
        help="SQLite database for storing/retrieving embeddings.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="interactive",
        help="Scenario namespace used for SQLite storage.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="unknown",
        help="Default language hint attached to each utterance.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of neighbours to display per query.",
    )
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Skip persisting new frames/embeddings; useful for dry runs.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=5,
        help="Maximum number of alternate spellings to display.",
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        help="Optional path to append JSONL trace records (defaults to runs/<scenario>_trace.jsonl).",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable trace logging even if --trace-path is provided.",
    )
    parser.add_argument(
        "--no-plasticity",
        action="store_true",
        help="Disable PMFlow plasticity updates even if PMFlow is active.",
    )
    parser.add_argument(
        "--plasticity-threshold",
        type=float,
        default=0.55,
        help="Recall score threshold below which PMFlow plasticity triggers.",
    )
    parser.add_argument(
        "--plasticity-mu-lr",
        type=float,
        default=5e-4,
        help="Learning rate for PMFlow mu updates during plasticity.",
    )
    parser.add_argument(
        "--plasticity-center-lr",
        type=float,
        default=5e-4,
        help="Learning rate for PMFlow center updates during plasticity.",
    )
    parser.add_argument(
        "--plasticity-state-path",
        type=Path,
        help="Optional path to persist PMFlow plasticity state (defaults to runs/<scenario>_pmflow_state.pt).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def run_shell(args: argparse.Namespace) -> None:
    state_path = args.plasticity_state_path or Path("runs") / f"{args.scenario}_pmflow_state.pt"
    pipeline = SymbolicPipeline(pmflow_state_path=state_path)
    decoder = TemplateDecoder()
    store = SymbolicStore(args.sqlite_path, scenario=args.scenario)
    trace_logger = None
    if not args.no_trace:
        trace_path = args.trace_path or Path("runs") / f"{args.scenario}_trace.jsonl"
        trace_logger = TraceLogger(trace_path, scenario=args.scenario)

    plasticity_controller = None
    if not args.no_plasticity and isinstance(pipeline.encoder, PMFlowEmbeddingEncoder):
        try:
            plasticity_controller = PlasticityController(
                pipeline.encoder,
                threshold=args.plasticity_threshold,
                mu_lr=args.plasticity_mu_lr,
                center_lr=args.plasticity_center_lr,
            )
        except RuntimeError as exc:
            print(f"[plasticity] disabled: {exc}")

    conversation_state = ConversationState(pipeline.encoder if isinstance(pipeline.encoder, PMFlowEmbeddingEncoder) else None)

    responder = ConversationResponder(
        store,
        decoder,
        topk=args.topk,
        trace_logger=trace_logger,
        plasticity_controller=plasticity_controller,
        conversation_state=conversation_state,
    )

    print("Type sentences to process them through the pipeline.")
    print("Commands: :quit (exit), :show (print stored frame count). Empty line repeats prompt.")
    if trace_logger:
        print(f"[trace] logging to {trace_logger.path}")

    while True:
        try:
            raw = input("→ ").strip()
        except EOFError:
            print()
            break

        if raw == "":
            continue
        if raw == ":quit":
            break
        if raw == ":show":
            frames = store.load_frames()
            print(f"[store] Frames stored: {len(frames)}")
            continue

        utterance = Utterance(text=raw, language=args.language)
        artefact = pipeline.process(utterance)

        response = responder.reply(artefact)

        if not args.no_store:
            store.persist([artefact])

        print("\n=== Artifact ===")
        print(f"text        : {artefact.utterance.text}")
        print(f"normalised  : {artefact.normalised_text}")
        print(f"confidence  : {artefact.confidence:.3f}")
        candidates = artefact.candidates[: args.candidates]
        if candidates:
            print(f"candidates  : {', '.join(candidates)}")
        frame = artefact.frame
        print(
            "frame       : actor={actor}, action={action}, target={target}".format(
                actor=frame.actor or "-",
                action=frame.action or "-",
                target=frame.target or "-",
            )
        )
        print(f"decoder     : {decoder.generate(frame)}")
        print(f"response    : {response.text}")

        frames = store.load_frames()
        label_list = response.nearest_labels
        score_list = response.nearest_scores
        if not label_list and store.vector_store.count(scenario=args.scenario) == 0:
            print("[retrieval] store is empty; add examples first.")
        elif label_list:
            print("\n=== Nearest frames (including this entry) ===")
            for rank, (lbl, score) in enumerate(zip(label_list, score_list), start=1):
                if lbl < len(frames):
                    neighbour = frames[lbl]
                    summary = (
                        f"actor={neighbour.get('actor')}, action={neighbour.get('action')}, "
                        f"target={neighbour.get('target')}"
                    )
                else:
                    summary = "<missing frame metadata>"
                print(f"#{rank}: label={lbl}, score={score:.4f}, {summary}")
        else:
            try:
                scores, labels = store.vector_store.search(
                    artefact.embedding, topk=args.topk, scenario=args.scenario
                )
            except RuntimeError:
                print("[retrieval] store is empty; add examples first.")
            else:
                label_list = labels.squeeze(0).tolist()
                score_list = scores.squeeze(0).tolist()
                print("\n=== Nearest frames (including this entry) ===")
                for rank, (lbl, score) in enumerate(zip(label_list, score_list), start=1):
                    if lbl < len(frames):
                        neighbour = frames[lbl]
                        summary = (
                            f"actor={neighbour.get('actor')}, action={neighbour.get('action')}, "
                            f"target={neighbour.get('target')}"
                        )
                    else:
                        summary = "<missing frame metadata>"
                    print(f"#{rank}: label={lbl}, score={score:.4f}, {summary}")
        if response.plasticity:
            print("[plasticity] update")
            for key, value in sorted(response.plasticity.items()):
                print(f"    {key}: {value:.6f}")
        print()

    print("[exit] bye")


def main() -> None:
    args = parse_args()
    run_shell(args)


if __name__ == "__main__":
    main()
