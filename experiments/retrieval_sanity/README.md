# PMFlow Retrieval Sanity Lab

Interactive workspace for contrasting a vanilla linear embedding network with a PMFlow-enhanced embedding network from the Pushing-Medium Biological Neural Network stack. The lab supports both a CLI workflow and a notebook-driven exploration mode with logging and plotting utilities so you can iterate quickly on hyperparameters, datasets, and staged cognition ideas.

> **Primary goal:** validate that the PMFlow latent field improves nearest-neighbour retrieval quality before committing to richer neuro-symbolic integrations.

## Contents

- `pmflow_retrieval_lab.ipynb` – The interactive notebook used for rapid experiments, metric logging, plot exports, and design documentation.
- `run_retrieval_sanity.py` – Scriptable entry point that mirrors the notebook flow; useful for CI or batch experimentation.
- `run_suite.py` – Batch runner that executes canned scenarios, logs results, and evaluates drift policies.
- `log_observer.py` – Scaffold for a drift observer that watches `retrieval_runs.log` and raises alert suggestions.
- `provenance.py` – Helper for capturing git/environment metadata to accompany each logged run.
- `config_utils.py` – Loader utilities for reusable override files that parameterise experiments.
- `configs/` – Sample JSON override files for baseline, hard synthetic, and relational runs.
- `relational_export.json` – Deterministic mock dataset representing a relational export with class labels and 2-D coordinates.
- `README.md` – (This file) Reference for running the lab, understanding outputs, and planning next steps.

## Prerequisites

- Python 3.9+
- PyTorch (CUDA-enabled builds unlock GPU support but are optional)
- NumPy, Matplotlib, and `pmflow-bnn`

Install the Python dependencies into your preferred virtual environment:

```bash
pip install torch numpy matplotlib
pip install git+https://github.com/experimentech/Pushing-Medium.git#subdirectory=programs/demos/machine_learning/nn_lib_v2
```

If you maintain a local checkout of `pmflow_bnn`, set `PMFLOW_LIB_ROOT` to that directory before running the lab.

## Quick start (CLI)

Run the scripted experiment from the repository root:

```bash
python experiments/retrieval_sanity/run_retrieval_sanity.py --verbose
```

Key flags:

- `--data-mode` – `synthetic` (default) or `relational`
- `--relational-path` – Override the sample export location
- `--metric` – `cosine` (default) or `euclidean`
- `--vector-store` – `memory` (default) or `sqlite` for persistent symbol storage
- `--sqlite-path` – Location of the SQLite database when `--vector-store sqlite`
- `--sqlite-scenario` – Optional namespace prefix for stored embeddings
- `--epochs`, `--latent-dim`, `--n-centers`, `--pm-steps`, `--pm-beta` – Hyperparameters shared with the notebook
- `--cpu` – Force CPU execution even when CUDA is available

The script prints a JSON summary containing per-model classification accuracy, retrieval accuracy, and final loss. A higher PMFlow retrieval accuracy indicates a successful boost over the baseline.

## Quick start (Notebook)

1. Open `experiments/retrieval_sanity/pmflow_retrieval_lab.ipynb` in VS Code or Jupyter.
2. Execute the setup cell that ensures `pmflow_bnn` is importable (installs from PyPI or falls back to a local checkout).
3. Run the retrieval cell to train both models and produce the structured `results` dictionary.
4. Call `run_and_log()` to append runs to the in-memory `history` and to `retrieval_runs.log` in the working directory.
5. Use `plot_runs(history)` and `plot_training_curves()` to generate comparison plots. Figures auto-save to `retrieval_plots/` with timestamped filenames.

The notebook also contains the **PMFlow Staged Cognition Infrastructure Guidelines & Roadmap** markdown, which captures long-term design principles for the neuro-symbolic stack.

## Logging and artifacts

- **Run log:** `retrieval_runs.log` (JSONL) – each entry records timestamps, resolved config, compacted metrics, dataset info, and device context.
- **Plot exports:** `retrieval_plots/*.png` – saved automatically whenever the plotting helpers run.
- **Metrics in memory:** `history` list – accessible within the notebook for ad-hoc analysis.
- **Observer scaffold:** `log_observer.py` – preliminary watcher that can evaluate rolling retrieval metrics and print alert recommendations.
- **Provenance helper:** `provenance.py` – collects git commit, branch, dirty status, Python/Torch versions, and runner metadata for repeatability.
- **Config overrides:** `configs/*.json` – reusable parameter sets that can be loaded via `config_utils.load_overrides`.

These artifacts make it easy to replay or share interesting runs, perform offline analysis, and seed automated observers that watch for drift.

### Using the observer scaffold

Run the observer once-off to evaluate recent runs:

```bash
python experiments/retrieval_sanity/log_observer.py --log-path retrieval_runs.log --window 5 --max-drop 0.03
```

Follow the log continuously (hit <kbd>Ctrl</kbd>+<kbd>C</kbd> to stop):

```bash
python experiments/retrieval_sanity/log_observer.py --follow
```

Enable SQLite summaries to report how many embeddings each scenario retains:

```bash
python experiments/retrieval_sanity/log_observer.py --sqlite-stats --sqlite-path runs/sqlite_vectors.db
```

The observer currently prints alert suggestions when:

- PMFlow underperforms the baseline by more than the configured `--min-delta`.
- PMFlow retrieval accuracy drops more than `--max-drop` compared with the recent best.

Each alert now auto-includes provenance context such as the config label, override keys, device, runner, and Git commit snippet so you can immediately trace which run tripped the threshold. To enrich the log, make sure you launch runs through the CLI/notebook helpers that emit provenance payloads.

When `--sqlite-stats` is enabled and a database path is available (either via `--sqlite-path` or inferred from the most recent log entries), the observer prints a compact summary such as `nightly:synthetic-baseline: baseline=120, pmflow=120, total=240`, helping you spot drifts where a scenario stops ingesting embeddings.

### Benchmarking vector stores

Curious about the trade-off between the in-memory and SQLite backends? Run the benchmark harness:

```bash
python experiments/retrieval_sanity/benchmarks/vector_store_benchmark.py \
	--output experiments/retrieval_sanity/benchmarks/vector_store_benchmark_results.json
```

The default run compares add/search timings (cosine metric, 64-dim embeddings, 256 queries, 3 repeats) across 1k, 5k, and 20k items. On a modest CPU, recent measurements showed:

| Backend | Embeddings | Add mean (ms) | Search mean (ms) |
| --- | --- | ---: | ---: |
| memory | 1k | 0.04 | 10.86 |
| sqlite | 1k | 142.60 | 10.07 |
| memory | 5k | 0.03 | 22.30 |
| sqlite | 5k | 378.21 | 45.05 |
| memory | 20k | 0.03 | 31.39 |
| sqlite | 20k | 883.41 | 202.50 |

Interpretation: searching stays within an order-of-magnitude even at 20k vectors, but inserts are two to four orders slower in SQLite because each write hits disk. Keep using the in-memory store when you only need transient embeddings or rapid iteration; switch to SQLite when permanence and observability outweigh ingest latency.

### Soaking the suite with SQLite persistence

To stress the end-to-end pipeline with persistent storage, run the soak harness. It executes the canned scenarios repeatedly, applies a unique namespace prefix per iteration, and appends each run to `retrieval_runs.log`:

```bash
python experiments/retrieval_sanity/benchmarks/suite_soak.py \
	--iterations 3 \
	--sqlite-path runs/soak_vectors.db \
	--sqlite-prefix soak \
	--output experiments/retrieval_sanity/benchmarks/suite_soak_report.json
```

After the final iteration the script prints a SQLite summary mirroring `log_observer.py --sqlite-stats`, making it easy to confirm that each scenario accumulates both baseline and PMFlow embeddings without overwriting previous runs. The optional JSON report captures per-scenario metrics plus elapsed time so you can diff soak behaviour over longer experiments.

### Analysing pipeline health

Once you have a few runs logged, summarise the metrics and database counts:

```
python experiments/retrieval_sanity/benchmarks/analyze_pipeline.py \
	--log-path retrieval_runs.log \
	--sqlite-path runs/soak_vectors.db
```

Provide `--scenario` or `--prefix` to focus on specific runs, and add `--output` to snapshot a JSON report. The script cross-references the log deltas with SQLite namespace totals so you can quickly spot drift between PMFlow and the baseline or detect ingestion gaps.

### Language-to-symbol pipeline sketch

A draft design for an error-resilient language ingestion path lives in `docs/language_symbol_pipeline.md`. It outlines modular layers for noisy text normalisation, heuristic parsing, symbolic frame construction, and storage/retrieval via the existing SQLite bridge. The accompanying modules under `pipeline/` include a smoke test (`python experiments/retrieval_sanity/pipeline/smoke.py`) that ingests a few noisy utterances, emits symbolic frames with confidence scores, writes embeddings to SQLite, runs the template decoder for human-readable playback, and verifies round-trip retrieval without pulling in heavyweight NLP dependencies. Feed the resulting frame dumps into `analyze_pipeline.py --frames-path runs/pipeline-smoke_frames.json` to summarise languages, slot coverage, and confidence trends alongside the usual suite metrics. When you want to iterate manually, the interactive shell (`python experiments/retrieval_sanity/pipeline/interactive.py`) lets you type sentences, persist frames, receive memory-backed responses, and inspect nearest neighbours in real time.

Launch the shell with plasticity instrumentation enabled by default:

```bash
python experiments/retrieval_sanity/pipeline/interactive.py \
	--sqlite-path runs/pipeline_interactive_vectors.db \
	--scenario interactive \
	--plasticity-threshold 0.6 \
	--plasticity-mu-lr 5e-4 \
	--plasticity-center-lr 5e-4 \
	--plasticity-state-path runs/pipeline_interactive_pmflow.pt
```

Use `--no-plasticity` to freeze the PMFlow field or `--no-trace` if you want to skip JSONL trace emission during quick experiments.

Seed the baseline dataset for the interactive scenario before launching the shell. A curated set of coordination utterances (English primary with a few Spanish, French, and Portuguese variants) lives in `datasets/interactive_seed.jsonl`; ingest it into SQLite with:

```bash
python experiments/retrieval_sanity/pipeline/seed_dataset.py --clear --validate
```

The script wipes any previous entries (because of `--clear`), regenerates frames via the pipeline, and stores the embeddings under the `interactive` namespace. Pass `--sqlite-path` or `--scenario` to target a different database/namespace, `--pmflow-state-path` to align with a pre-trained PMFlow snapshot, and `--validate` to confirm the self-retrieval accuracy of the seeded frames. Swap in a custom JSON/JSONL file if you want to test different vocabularies or languages—the loader accepts either format as long as each entry has a `text` field.

Sample transcript (sanitised prompt/output) showing a plasticity event and persisted state:

```text
$ python experiments/retrieval_sanity/pipeline/interactive.py \
	--sqlite-path runs/pipeline_interactive_vectors.db \
	--scenario interactive \
	--plasticity-threshold 0.6 \
	--plasticity-mu-lr 5e-4 \
	--plasticity-center-lr 5e-4 \
	--plasticity-state-path runs/pipeline_interactive_pmflow.pt
[info] loaded 128 frames into namespace interactive
[assistant] ready — type an utterance or :quit
> remind me about the venue change
[plasticity] recall=0.71 Δμ=1.2e-03 Δc=8.5e-04 refreshed=4
[plasticity] encoder state saved to runs/pipeline_interactive_pmflow.pt
[assistant] We moved the venue to the waterfront studio for tomorrow's session.
```

Recent pipeline upgrades:

- The heuristic parser now records question intent, negation, and coarse slot roles (actor, action, target, location, time). These richer signals feed into symbolic frames so downstream decoders can surface context such as `time='next week'` and `intent='when'`.
- A PMFlow-backed embedding encoder (`PMFlowEmbeddingEncoder`) combines hashed bag-of-words features with a deterministic PMFlow latent field. The pipeline prefers it automatically and falls back to the original hashed encoder when `pmflow_bnn` is unavailable, keeping the workflow dependency-light.
- Decoder summaries expose the new metadata, making it easier to verify that location/time detection and negation cues survive the full ingest → retrieval → response loop.
- Each conversational turn can now be traced to JSONL via the interactive shell (`--trace-path`) or programmatic `TraceLogger`. The analyzer consumes these traces to report recall rates and neighbour confidence trends, laying the groundwork for plasticity experiments.
- PMFlow plasticity updates can now run inline during interactive sessions. Tune activation with `--plasticity-threshold`, `--plasticity-mu-lr`, and `--plasticity-center-lr`, or disable the controller via `--no-plasticity`. When plasticity fires, trace records include a `plasticity` payload summarising the recall score and Δparameters so you can audit adaptation episodes.
- Plasticity hooks now re-encode the affected scenario immediately after a PMFlow update. Stored frames keep the token sequence alongside the symbolic summary so the refresh logic can rebuild embeddings in SQLite. The trace payload also reports `refreshed`, indicating how many embeddings were recomputed for the turn.
- The PMFlow encoder now saves and reloads its state automatically after every plasticity step. By default the interactive shell drops a snapshot to `runs/<scenario>_pmflow.pt`; override the location with `--plasticity-state-path` when you want multiple experiments to share or isolate PMFlow memory.

Representative redacted trace fragment showing the persisted state metadata:

```json
{
	"turn": 3,
	"speaker": "assistant",
	"text": "We moved the venue to the waterfront studio for tomorrow's session.",
	"plasticity": {
		"scenario": "interactive",
		"recall": 0.71,
		"refreshed": 4,
		"state_path": "runs/pipeline_interactive_pmflow.pt",
		"delta_mu_norm": 0.0012,
		"delta_center_norm": 0.00085
	}
}
```

### Probing stored symbols

Use the symbolic probe to sample stored embeddings and verify that persisted representations still deliver sensible neighbours once training is finished:

```bash
python experiments/retrieval_sanity/benchmarks/symbolic_probe.py \
	--sqlite-path runs/soak_vectors.db \
	--scenario soak-02:synthetic-hard \
	--samples 5 \
	--output experiments/retrieval_sanity/benchmarks/symbolic_probe_soak02.json
```

Each query perturbs a PMFlow embedding with configurable noise and retrieves top-`k` matches from the baseline namespace, printing the neighbour labels and scores. The JSON payload records label distributions and neighbour lists so you can track how symbolic retrieval evolves between soak runs.

If no thresholds are violated you'll see a heartbeat similar to:

```
[observer] No alerts. Last PMFlow retrieval: 1.0000
```

where the trailing metric reflects the newest PMFlow run observed in the log.

Future work can plug these hooks into notification systems or trigger automated fine-tuning jobs when alerts fire.

### Sample retrieval snapshot (CLI)

The latest smoke run (synthetic data, 60 epochs on CPU) produced the following JSON payload:

```
{
	"baseline_retrieval_accuracy": 1.0,
	"pmflow_retrieval_accuracy": 1.0,
	"support_samples": 144,
	"query_samples": 96,
	"pm_steps": 3,
	"n_centers": 24,
	"provenance": {
		"runner": "cli",
		"device": "cpu",
		"git": {"commit": null}
	}
}
```

With the current hyperparameters both models tie at 100% retrieval, which is expected on the easier synthetic clusters. When you dial up difficulty (more classes, lower support ratio, noisier clusters) or switch to the relational export, the PMFlow variant should maintain a higher retrieval accuracy than the baseline. Use these snapshots as breadcrumbs in the README to communicate experiment intent and to highlight when an alert should prompt deeper investigation.

#### Harder synthetic stress test

To raise the difficulty we cranked up class count, reduced the support set, and injected additional cluster noise:

```bash
python experiments/retrieval_sanity/run_retrieval_sanity.py \
	--num-classes 6 \
	--points-per-class 160 \
	--support-ratio 0.4 \
	--cluster-std 0.9 \
	--epochs 120 \
	--n-centers 48 \
	--pm-steps 6 \
	--pm-beta 0.12 \
	--pm-clamp 4.0
```

Baseline retrieval landed at **0.858**, while PMFlow held **0.863**, keeping a small but positive margin under the noisier regime. Expect run-to-run variance at this difficulty—use multiple seeds or bootstrapped averages if you need stronger statistical guarantees. Even modest lifts like this one are enough for the observer to stay quiet now but flip to alerting if the gap collapses on future runs.

#### Relational export check-in

For a closer-to-production signal, run the relational preset:

```bash
python experiments/retrieval_sanity/run_retrieval_sanity.py \
	--config experiments/retrieval_sanity/configs/relational_sample.json
```

This configuration uses the deterministic `relational_export.json` mock and currently yields identical scores (baseline and PMFlow both at **1.0** retrieval/class accuracy). The takeaway is that the bundled export is linearly separable enough that PMFlow cannot showcase an advantage. To increase realism, consider:

- Re-exporting a noisier slice of relational data or injecting perturbations.
- Lowering the support ratio further to stress zero-shot retrieval.
- Tracking multiple random seeds and feeding the results into the observer to catch degradation once the dataset becomes less trivial.

### Batch benchmarking with `run_suite.py`

Drive the scenarios above (and future ones) via a single entry point that logs each run and computes retrieval deltas:

```bash
python experiments/retrieval_sanity/run_suite.py --evaluate-policy
```

By default the suite executes the `synthetic-baseline`, `synthetic-hard`, and `relational-sample` presets (defined in `configs/`). Specify `--scenario` repeatedly to target a subset or add new configs as the roadmap expands. Each run is appended to `retrieval_runs.log` with provenance, making the observer immediately useful. Sample summary:

```
=== Retrieval summary ===
scenario               baseline     pmflow      delta
-----------------------------------------------------
synthetic-baseline       1.0000     1.0000    +0.0000
synthetic-hard           0.8576     0.8628    +0.0052
relational-sample        1.0000     1.0000    +0.0000
-----------------------------------------------------
[suite] Policy evaluation produced no alerts. Latest delta: 0.0
```

Tune `--policy-window`, `--policy-min-delta`, and `--policy-max-drop` to mirror production thresholds; alerts will surface inline when the drift policy fires.

### Persisting symbols in SQLite

Switch the vector-store backend to SQLite to capture embeddings across runs and scenarios:

```bash
python experiments/retrieval_sanity/run_retrieval_sanity.py \
	--vector-store sqlite \
	--sqlite-path runs/sqlite_vectors.db \
	--sqlite-scenario smoke
```

The runner records the selected backend and database path in the JSON summary and automatically creates the database (including tables and indexes) if it does not exist. Each scenario gets a distinct namespace—when `--sqlite-scenario smoke`, embeddings persist under `smoke:baseline` and `smoke:pmflow`. Pass the same flags to `run_suite.py` to reuse the database across preset scenarios:

```bash
python experiments/retrieval_sanity/run_suite.py \
	--vector-store sqlite \
	--sqlite-path runs/sqlite_vectors.db \
	--sqlite-prefix nightly
```

`run_suite.py` automatically expands each preset to a unique namespace such as `nightly:synthetic-baseline:baseline` and `nightly:synthetic-baseline:pmflow`, keeping scenario data isolated while sharing the same database. You can inspect the resulting file with `sqlite3 runs/sqlite_vectors.db` to snapshot stored embeddings, label assignments, and run metadata for downstream symbolic reasoning experiments.

### Adding provenance metadata

Both the notebook and CLI now attach metadata from `provenance.collect_provenance()` to every result payload. This records:

- Git commit, branch, and dirty state when the workspace is within a git repository.
- Python/Torch versions and basic platform info.
- The runner origin (`cli` vs `notebook`) along with optional extras like CLI arguments.

Use this metadata to trace regressions back to exact code revisions or environment changes before kicking off automated mitigations.

### Loading configuration overrides

The `config_utils` helpers make it easy to reuse parameter sets across scripts and the notebook:

```python
from pathlib import Path

from experiments.retrieval_sanity.config_utils import load_overrides

overrides = load_overrides(Path("experiments/retrieval_sanity/configs/synthetic_baseline.json"))
run_and_log(overrides)
```

Each config supplies a `label` and an `overrides` mapping aligned with the CLI/notebook keyword arguments. Add your own JSON files to `configs/` to capture new experimental conditions.

### Running tests

Pytest scaffolding lives at the repository root. After installing dependencies, run:

```bash
pytest
```

The test suite currently covers provenance logging, configuration helpers, and the log observer pipeline. Extend it as new components are introduced.

## Configuration surface

Both the script and notebook share the same configuration parameters:

| Parameter | Purpose |
| --- | --- |
| `data-mode` | Choose `synthetic` clusters or load a relational export |
| `support-ratio` | Split between support and query sets |
| `cluster-std`, `points-per-class`, `num-classes` | Control synthetic dataset difficulty |
| `latent-dim`, `n-centers`, `pm-steps`, `pm-beta`, `pm-dt`, `pm-clamp` | PMFlow latent field hyperparameters |
| `metric` | Vector store similarity (`cosine` or `euclidean`) |
| `epochs`, `lr`, `weight-decay` | Training loop knobs |

Adjust these through CLI flags or notebook overrides (e.g., `run_and_log(overrides={"epochs": "600"})`).

## Validation checklist

After each experiment, confirm:

1. **Baseline vs PMFlow gap** – PMFlow retrieval accuracy should match or exceed the baseline.
2. **Training stability** – Plot loss/accuracy curves via `plot_training_curves()` to catch divergence or overfitting.
3. **Artifact integrity** – Ensure logs and plots write correctly so future automation can ingest them.

## Suggested evolution

- Wrap the CLI in a scheduled job or lightweight service to collect longer-term trend data.
- Extend the log schema with git commit hashes and environment metadata for full provenance.
- Build a streaming observer that tails `retrieval_runs.log` and triggers fine-tuning when drift thresholds fire.
- Integrate symbolic reasoning modules that can consume the retrieved facts, as outlined in the notebook roadmap.

For a deeper architectural plan, review the roadmap embedded in the notebook or port it into a project-wide documentation site as the platform matures.
