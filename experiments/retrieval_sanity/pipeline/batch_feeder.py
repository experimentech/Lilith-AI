"""Automate feeding a text corpus through the language-to-symbol pipeline."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from experiments.retrieval_sanity.pipeline import (  # noqa: E402
    ConversationResponder,
    PipelineArtifact,
    PMFlowEmbeddingEncoder,
    PlasticityController,
    SymbolicPipeline,
    SymbolicStore,
    TemplateDecoder,
    TraceLogger,
    Utterance,
    ConversationState,
)
from experiments.retrieval_sanity.pipeline.text_cleaner import clean_lines  # noqa: E402


@dataclass
class BatchMetrics:
    """Aggregated metrics for a batch ingestion run."""

    processed: int = 0
    stored: int = 0
    recall_scores: list[float] = field(default_factory=list)
    recall_failures: int = 0
    plasticity_events: int = 0
    plasticity_delta_centers: float = 0.0
    plasticity_delta_mus: float = 0.0
    durations: list[float] = field(default_factory=list)

    def record_turn(self, *, duration: float, recall_score: float | None, plasticity_payload: dict | None) -> None:
        self.processed += 1
        self.durations.append(duration)
        if recall_score is None:
            self.recall_failures += 1
        else:
            self.recall_scores.append(float(recall_score))
        if plasticity_payload:
            self.plasticity_events += 1
            self.plasticity_delta_centers += float(plasticity_payload.get("delta_centers", 0.0))
            self.plasticity_delta_mus += float(plasticity_payload.get("delta_mus", 0.0))

    def record_persisted(self, count: int) -> None:
        self.stored += count

    def as_dict(self) -> dict[str, float | int | None]:
        summary: dict[str, float | int | None] = {
            "processed": self.processed,
            "stored": self.stored,
            "recall_success": len(self.recall_scores),
            "recall_failures": self.recall_failures,
            "plasticity_events": self.plasticity_events,
        }
        if self.recall_scores:
            summary.update(
                {
                    "recall_mean": round(statistics.mean(self.recall_scores), 4),
                    "recall_median": round(statistics.median(self.recall_scores), 4),
                    "recall_min": round(min(self.recall_scores), 4),
                    "recall_max": round(max(self.recall_scores), 4),
                }
            )
        else:
            summary.update({"recall_mean": None, "recall_median": None, "recall_min": None, "recall_max": None})
        if self.durations:
            summary.update(
                {
                    "latency_mean_sec": round(statistics.mean(self.durations), 4),
                    "latency_median_sec": round(statistics.median(self.durations), 4),
                }
            )
        else:
            summary.update({"latency_mean_sec": None, "latency_median_sec": None})
        if self.plasticity_events:
            summary.update(
                {
                    "plasticity_delta_centers_total": round(self.plasticity_delta_centers, 6),
                    "plasticity_delta_mus_total": round(self.plasticity_delta_mus, 6),
                    "plasticity_delta_centers_mean": round(
                        self.plasticity_delta_centers / self.plasticity_events, 6
                    ),
                    "plasticity_delta_mus_mean": round(self.plasticity_delta_mus / self.plasticity_events, 6),
                }
            )
        else:
            summary.update(
                {
                    "plasticity_delta_centers_total": 0.0,
                    "plasticity_delta_mus_total": 0.0,
                    "plasticity_delta_centers_mean": 0.0,
                    "plasticity_delta_mus_mean": 0.0,
                }
            )
        return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-feed a text corpus into the symbol pipeline.")
    parser.add_argument("corpus_path", type=Path, help="Plain text corpus to ingest (e.g., a Project Gutenberg dump).")
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=Path("runs/pipeline_interactive_vectors.db"),
        help="SQLite database for storing embeddings.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="batch",
        help="Scenario namespace for the SQLite store and trace outputs.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="unknown",
        help="Language code attached to each utterance (defaults to 'unknown').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on the number of cleaned sentences processed from the corpus.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=5,
        help="Minimum token count required for a cleaned sentence to be considered.",
    )
    parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.5,
        help="Reject sentences whose alphabetic character ratio falls below this threshold.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Truncate sentences above this token count (after cleaning).",
    )
    parser.add_argument(
        "--no-sentence-chunk",
        action="store_false",
        dest="chunk_sentences",
        help="Treat each cleaned line as a single utterance instead of sentence-chunking it.",
    )
    parser.set_defaults(chunk_sentences=True)
    parser.add_argument("--lowercase", action="store_true", help="Lowercase outputs from the cleaner.")
    parser.add_argument(
        "--no-skip-headers",
        action="store_false",
        dest="skip_headers",
        help="Preserve Project Gutenberg header/footer text instead of skipping it.",
    )
    parser.set_defaults(skip_headers=True)
    parser.add_argument(
        "--drop-pattern",
        action="append",
        dest="drop_patterns",
        help="Additional substrings (case-insensitive) that cause lines to be skipped.",
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=1,
        help="Persist artefacts to SQLite after this many processed sentences.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print a progress heartbeat every N processed sentences.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        help="Optional JSON file to write the final metrics summary.",
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        help="Optional JSONL trace path (defaults to runs/<scenario>_trace.jsonl).",
    )
    parser.add_argument("--no-trace", action="store_true", help="Disable trace logging even if a path is provided.")
    parser.add_argument("--no-store", action="store_true", help="Skip persisting new artefacts to SQLite.")
    parser.add_argument("--clear", action="store_true", help="Remove existing embeddings/frames before ingestion.")
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of neighbours to request during retrieval for progress reporting.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.15,
        help="Minimum retrieval score considered a successful recall in the responder.",
    )
    parser.add_argument("--no-plasticity", action="store_true", help="Disable PMFlow plasticity updates.")
    parser.add_argument(
        "--plasticity-threshold",
        type=float,
        default=0.55,
        help="Recall threshold below which plasticity triggers (when enabled).",
    )
    parser.add_argument(
        "--plasticity-mu-lr",
        type=float,
        default=5e-4,
        help="PMFlow plasticity learning rate for the μ parameters.",
    )
    parser.add_argument(
        "--plasticity-center-lr",
        type=float,
        default=5e-4,
        help="PMFlow plasticity learning rate for the latent field centres.",
    )
    parser.add_argument(
        "--plasticity-state-path",
        type=Path,
        help="Optional PMFlow state checkpoint (defaults to runs/<scenario>_pmflow_state.pt).",
    )
    parser.add_argument(
        "--preview",
        type=int,
        help="Print the first N cleaned sentences without running the pipeline, then exit.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _iter_sentences(args: argparse.Namespace) -> Iterator[str]:
    cleaner_kwargs = {
        "chunk_sentences": args.chunk_sentences,
        "min_tokens": args.min_tokens,
        "min_alpha_ratio": args.min_alpha_ratio,
        "drop_patterns": args.drop_patterns,
        "skip_headers": args.skip_headers,
        "lowercase": args.lowercase,
        "max_tokens": args.max_tokens,
    }
    iterator = clean_lines(args.corpus_path, **cleaner_kwargs)
    if args.limit is not None:
        iterator = itertools.islice(iterator, args.limit)
    return iterator


def _preview_sentences(args: argparse.Namespace) -> None:
    iterator = _iter_sentences(args)
    count = 0
    for sentence in iterator:
        print(sentence)
        count += 1
        if args.preview and count >= args.preview:
            break
    if count == 0:
        print("[preview] No sentences produced by the cleaner.")


def run_batch(args: argparse.Namespace) -> BatchMetrics:
    if args.flush_interval is not None and args.flush_interval < 1:
        raise ValueError("--flush-interval must be >= 1")
    if args.log_every is not None and args.log_every < 1:
        raise ValueError("--log-every must be >= 1")

    if args.preview:
        _preview_sentences(args)
        return BatchMetrics()

    if not args.corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {args.corpus_path}")

    iterator = _iter_sentences(args)
    iterator = iter(iterator)
    try:
        first_sentence = next(iterator)
    except StopIteration as exc:
        raise ValueError("Cleaning produced no sentences. Adjust the filters or corpus path.") from exc

    sentences = itertools.chain([first_sentence], iterator)

    state_path = args.plasticity_state_path or Path("runs") / f"{args.scenario}_pmflow_state.pt"
    pipeline = SymbolicPipeline(pmflow_state_path=state_path)
    decoder = TemplateDecoder()
    store = SymbolicStore(args.sqlite_path, scenario=args.scenario)
    store.vector_store.path.parent.mkdir(parents=True, exist_ok=True)

    if args.clear:
        store.vector_store.clear(scenario=args.scenario)
        if store.frames_path.exists():
            store.frames_path.unlink()

    trace_logger: TraceLogger | None = None
    if not args.no_trace:
        trace_path = args.trace_path or Path("runs") / f"{args.scenario}_trace.jsonl"
        trace_logger = TraceLogger(trace_path, scenario=args.scenario)

    plasticity_controller: PlasticityController | None = None
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
        min_score=args.min_score,
        topk=args.topk,
        trace_logger=trace_logger,
        plasticity_controller=plasticity_controller,
        conversation_state=conversation_state,
    )

    metrics = BatchMetrics()
    pending: list[PipelineArtifact] = []

    print("[batch] Prepared sentences. Starting ingestion…")
    start_time = time.time()
    for idx, sentence in enumerate(sentences, start=1):
        utterance = Utterance(text=sentence, language=args.language)
        turn_start = time.time()
        artefact = pipeline.process(utterance)
        response = responder.reply(artefact)
        turn_end = time.time()
        metrics.record_turn(
            duration=turn_end - turn_start,
            recall_score=response.recall_score,
            plasticity_payload=response.plasticity,
        )

        if not args.no_store:
            pending.append(artefact)
            if len(pending) >= max(args.flush_interval, 1):
                store.persist(pending)
                metrics.record_persisted(len(pending))
                pending.clear()

        if args.log_every and idx % args.log_every == 0:
            if metrics.recall_scores:
                avg_recall = statistics.mean(metrics.recall_scores)
            else:
                avg_recall = 0.0
            print(
                f"[batch] processed={idx} stored={metrics.stored} "
                f"recall successes={len(metrics.recall_scores)} failures={metrics.recall_failures} "
                f"avg_recall={avg_recall:.3f}"
            )

    if pending:
        store.persist(pending)
        metrics.record_persisted(len(pending))
        pending.clear()

    elapsed = time.time() - start_time
    print(
        f"[batch] Complete. Processed {metrics.processed} sentences in {elapsed:.2f}s. "
        f"Stored {metrics.stored} artefacts."
    )

    summary = metrics.as_dict()
    if metrics.recall_scores:
        print(
            f"[batch] Recall mean={summary['recall_mean']} min={summary['recall_min']} max={summary['recall_max']} "
            f"failures={summary['recall_failures']}"
        )
    else:
        print(f"[batch] No successful recalls (store may have started empty). Failures={summary['recall_failures']}")

    if metrics.plasticity_events:
        print(
            f"[batch] Plasticity events={metrics.plasticity_events} "
            f"Δcenters_total={summary['plasticity_delta_centers_total']} "
            f"Δmus_total={summary['plasticity_delta_mus_total']}"
        )

    if args.summary_path:
        payload = {
            "summary": summary,
            "scenario": args.scenario,
            "sqlite_path": str(args.sqlite_path),
            "corpus_path": str(args.corpus_path),
        }
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[batch] Summary written to {args.summary_path}")

    return metrics


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run_batch(args)


if __name__ == "__main__":
    main()
