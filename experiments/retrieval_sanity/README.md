# Lilith - Neuro-Symbolic Conversational AI# PMFlow Retrieval Sanity Lab



**A pure neuro-symbolic conversational AI system that learns through interaction without LLM dependency.**Interactive workspace for contrasting a vanilla linear embedding network with a PMFlow-enhanced embedding network from the Pushing-Medium Biological Neural Network stack. The lab supports both a CLI workflow and a notebook-driven exploration mode with logging and plotting utilities so you can iterate quickly on hyperparameters, datasets, and staged cognition ideas.



Lilith demonstrates that sophisticated conversational AI can emerge from:> **Primary goal:** validate that the PMFlow latent field improves nearest-neighbour retrieval quality before committing to richer neuro-symbolic integrations.

- **Pattern learning from dialogue** (Cornell Movie Dialogs dataset)

- **Learning from teaching** (users teach new knowledge interactively)## Contents

- **External knowledge augmentation** (Wikipedia integration)

- **Multi-turn coherence** (maintains topics across conversation)- `pmflow_retrieval_lab.ipynb` – The interactive notebook used for rapid experiments, metric logging, plot exports, and design documentation.

- **Typo tolerance** (fuzzy matching for real-world input)- `run_retrieval_sanity.py` – Scriptable entry point that mirrors the notebook flow; useful for CI or batch experimentation.

- **Grammar refinement** (hybrid adaptation + error correction)- `run_suite.py` – Batch runner that executes canned scenarios, logs results, and evaluates drift policies.

- `log_observer.py` – Scaffold for a drift observer that watches `retrieval_runs.log` and raises alert suggestions.

> **Core Philosophy:** No LLM grafting - pure learned behavior through neuro-symbolic architecture.- `provenance.py` – Helper for capturing git/environment metadata to accompany each logged run.

- `config_utils.py` – Loader utilities for reusable override files that parameterise experiments.

## Quick Start- `configs/` – Sample JSON override files for baseline, hard synthetic, and relational runs.

- `relational_export.json` – Deterministic mock dataset representing a relational export with class labels and 2-D coordinates.

### Prerequisites- `README.md` – (This file) Reference for running the lab, understanding outputs, and planning next steps.



- Python 3.9+## Prerequisites

- Virtual environment (recommended)

- Python 3.9+

### Setup- PyTorch (CUDA-enabled builds unlock GPU support but are optional)

- NumPy, Matplotlib, and `pmflow-bnn`

1. **Create and activate virtual environment:**

Install the Python dependencies into your preferred virtual environment:

```bash

cd /path/to/lilith/experiments/retrieval_sanity```bash

python3 -m venv .venvpip install torch numpy matplotlib

source .venv/bin/activate  # On Windows: .venv\Scripts\activatepip install git+https://github.com/experimentech/Pushing-Medium.git#subdirectory=programs/demos/machine_learning/nn_lib_v2

``````



2. **Install dependencies:**If you maintain a local checkout of `pmflow_bnn`, set `PMFLOW_LIB_ROOT` to that directory before running the lab.



```bash## Quick start (CLI)

pip install torch numpy matplotlib requests

```Run the scripted experiment from the repository root:



**Required packages:**```bash

- `torch` - PyTorch for neural network components (BNN semantic encoder)python experiments/retrieval_sanity/run_retrieval_sanity.py --verbose

- `numpy` - Numerical operations and embeddings```

- `matplotlib` - Visualization for analysis and debugging

- `requests` - HTTP requests for Wikipedia API integrationKey flags:



**Optional:**- `--data-mode` – `synthetic` (default) or `relational`

- CUDA-enabled PyTorch for GPU acceleration- `--relational-path` – Override the sample export location

- `pmflow_bnn` - For advanced PMFlow experiments (not required for core functionality)- `--metric` – `cosine` (default) or `euclidean`

- `--vector-store` – `memory` (default) or `sqlite` for persistent symbol storage

### Running Lilith- `--sqlite-path` – Location of the SQLite database when `--vector-store sqlite`

- `--sqlite-scenario` – Optional namespace prefix for stored embeddings

**Interactive conversation mode:**- `--epochs`, `--latent-dim`, `--n-centers`, `--pm-steps`, `--pm-beta` – Hyperparameters shared with the notebook

- `--cpu` – Force CPU execution even when CUDA is available

```bash

python3 conversation_loop.pyThe script prints a JSON summary containing per-model classification accuracy, retrieval accuracy, and final loss. A higher PMFlow retrieval accuracy indicates a successful boost over the baseline.

```

## Quick start (Notebook)

Type messages to chat with Lilith. The system will:

- Retrieve relevant patterns from learned conversations1. Open `experiments/retrieval_sanity/pmflow_retrieval_lab.ipynb` in VS Code or Jupyter.

- Maintain multi-turn coherence (tracks topics)2. Execute the setup cell that ensures `pmflow_bnn` is importable (installs from PyPI or falls back to a local checkout).

- Look up Wikipedia when knowledge is missing3. Run the retrieval cell to train both models and produce the structured `results` dictionary.

- Learn from your teachings after fallback responses4. Call `run_and_log()` to append runs to the in-memory `history` and to `retrieval_runs.log` in the working directory.

- Handle typos and spelling errors gracefully5. Use `plot_runs(history)` and `plot_training_curves()` to generate comparison plots. Figures auto-save to `retrieval_plots/` with timestamped filenames.



**Example conversation:**The notebook also contains the **PMFlow Staged Cognition Infrastructure Guidelines & Roadmap** markdown, which captures long-term design principles for the neuro-symbolic stack.

```

You: Hi there!## Logging and artifacts

Lilith: Hey! How's it going?

- **Run log:** `retrieval_runs.log` (JSONL) – each entry records timestamps, resolved config, compacted metrics, dataset info, and device context.

You: Tell me about machine learning- **Plot exports:** `retrieval_plots/*.png` – saved automatically whenever the plotting helpers run.

Lilith: [Queries Wikipedia...] Machine learning is a field of artificial intelligence concerned with the development and study of statistical algorithms that can learn from data...- **Metrics in memory:** `history` list – accessible within the notebook for ad-hoc analysis.

- **Observer scaffold:** `log_observer.py` – preliminary watcher that can evaluate rolling retrieval metrics and print alert recommendations.

You: What are the main types?- **Provenance helper:** `provenance.py` – collects git commit, branch, dirty status, Python/Torch versions, and runner metadata for repeatability.

Lilith: The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning...- **Config overrides:** `configs/*.json` – reusable parameter sets that can be loaded via `config_utils.load_overrides`.



You: machien learing is cool  [note: typos!]These artifacts make it easy to replay or share interesting runs, perform offline analysis, and seed automated observers that watch for drift.

Lilith: [Fuzzy matches to "machine learning"] I'm glad you find machine learning interesting!

```### Using the observer scaffold



## Core FeaturesRun the observer once-off to evaluate recent runs:



### 1. Learning from Teaching ✅```bash

python experiments/retrieval_sanity/log_observer.py --log-path retrieval_runs.log --window 5 --max-drop 0.03

The system can learn new knowledge when users teach it:```



```Follow the log continuously (hit <kbd>Ctrl</kbd>+<kbd>C</kbd> to stop):

You: What is a Merkle tree?

Lilith: I'm not sure about that yet...```bash

python experiments/retrieval_sanity/log_observer.py --follow

You: A Merkle tree is a data structure used in cryptography for efficient verification.```

Lilith: [Learns pattern] Thanks for teaching me!

Enable SQLite summaries to report how many embeddings each scenario retains:

[Later...]

You: Tell me about Merkle trees```bash

Lilith: A Merkle tree is a data structure used in cryptography for efficient verification.python experiments/retrieval_sanity/log_observer.py --sqlite-stats --sqlite-path runs/sqlite_vectors.db

``````



**How it works:**The observer currently prints alert suggestions when:

- Detects teaching scenarios (fallback + factual statement)

- Extracts topic from "X is Y" patterns- PMFlow underperforms the baseline by more than the configured `--min-delta`.

- Stores with high confidence (0.85) and `intent='taught'`- PMFlow retrieval accuracy drops more than `--max-drop` compared with the recent best.

- Future queries retrieve the learned knowledge

Each alert now auto-includes provenance context such as the config label, override keys, device, runner, and Git commit snippet so you can immediately trace which run tripped the threshold. To enrich the log, make sure you launch runs through the CLI/notebook helpers that emit provenance payloads.

**Files:** `pipeline/pragmatic_learner.py`, `pipeline/response_learner.py`

When `--sqlite-stats` is enabled and a database path is available (either via `--sqlite-path` or inferred from the most recent log entries), the observer prints a compact summary such as `nightly:synthetic-baseline: baseline=120, pmflow=120, total=240`, helping you spot drifts where a scenario stops ingesting embeddings.

### 2. Wikipedia Knowledge Augmentation ✅

### Benchmarking vector stores

Automatically acquires knowledge from Wikipedia when needed:

Curious about the trade-off between the in-memory and SQLite backends? Run the benchmark harness:

```

You: What is quantum entanglement?```bash

Lilith: [Low confidence, queries Wikipedia...] Quantum entanglement is the phenomenon wherein the quantum state of each particle in a group cannot be described independently...python experiments/retrieval_sanity/benchmarks/vector_store_benchmark.py \

```	--output experiments/retrieval_sanity/benchmarks/vector_store_benchmark_results.json

```

**How it works:**

- Triggers on low-confidence retrieval (< 0.6)The default run compares add/search timings (cosine metric, 64-dim embeddings, 256 queries, 3 repeats) across 1k, 5k, and 20k items. On a modest CPU, recent measurements showed:

- Queries Wikipedia REST API

- Extracts concise summary (2 sentences, ~50 words)| Backend | Embeddings | Add mean (ms) | Search mean (ms) |

- Returns as response AND learns as pattern| --- | --- | ---: | ---: |

- Future queries use learned knowledge (no re-lookup)| memory | 1k | 0.04 | 10.86 |

| sqlite | 1k | 142.60 | 10.07 |

**Files:** `pipeline/knowledge_augmenter.py`, `pipeline/response_composer.py`| memory | 5k | 0.03 | 22.30 |

| sqlite | 5k | 378.21 | 45.05 |

### 3. Multi-Turn Coherence ✅| memory | 20k | 0.03 | 31.39 |

| sqlite | 20k | 883.41 | 202.50 |

Maintains conversation topics across multiple turns:

Interpretation: searching stays within an order-of-magnitude even at 20k vectors, but inserts are two to four orders slower in SQLite because each write hits disk. Keep using the in-memory store when you only need transient embeddings or rapid iteration; switch to SQLite when permanence and observability outweigh ingest latency.

```

You: Tell me about Python programming### Soaking the suite with SQLite persistence

Lilith: Python is a high-level programming language...

To stress the end-to-end pipeline with persistent storage, run the soak harness. It executes the canned scenarios repeatedly, applies a unique namespace prefix per iteration, and appends each run to `retrieval_runs.log`:

You: What are its main features?

Lilith: [Maintains "Python" topic] Python's main features include...```bash

python experiments/retrieval_sanity/benchmarks/suite_soak.py \

You: Is it good for beginners?	--iterations 3 \

Lilith: [Still tracking "Python"] Yes, Python is excellent for beginners...	--sqlite-path runs/soak_vectors.db \

```	--sqlite-prefix soak \

	--output experiments/retrieval_sanity/benchmarks/suite_soak_report.json

**How it works:**```

- Context encoder extracts key topics from conversation history

- Enriches queries: "current_input" → "topic current_input"After the final iteration the script prints a SQLite summary mirroring `log_observer.py --sqlite-stats`, making it easy to confirm that each scenario accumulates both baseline and PMFlow embeddings without overwriting previous runs. The optional JSON report captures per-scenario metrics plus elapsed time so you can diff soak behaviour over longer experiments.

- Pattern retrieval uses enriched context

- Example: "What are its main features?" → "Python What are its main features?"### Analysing pipeline health



**Files:** `pipeline/context_encoder.py`, `pipeline/response_composer.py`Once you have a few runs logged, summarise the metrics and database counts:



### 4. Typo Tolerance ✅```

python experiments/retrieval_sanity/benchmarks/analyze_pipeline.py \

Handles spelling errors and typos gracefully:	--log-path retrieval_runs.log \

	--sqlite-path runs/soak_vectors.db

``````

You: machien learing  [typos: should be "machine learning"]

Lilith: [Fuzzy matches correctly] Machine learning is...Provide `--scenario` or `--prefix` to focus on specific runs, and add `--output` to snapshot a JSON report. The script cross-references the log deltas with SQLite namespace totals so you can quickly spot drift between PMFlow and the baseline or detect ingestion gaps.



You: nueral netwrok  [typos: should be "neural network"]### Language-to-symbol pipeline sketch

Lilith: [Fuzzy matches correctly] A neural network is...

```A draft design for an error-resilient language ingestion path lives in `docs/language_symbol_pipeline.md`. It outlines modular layers for noisy text normalisation, heuristic parsing, symbolic frame construction, and storage/retrieval via the existing SQLite bridge. The accompanying modules under `pipeline/` include a smoke test (`python experiments/retrieval_sanity/pipeline/smoke.py`) that ingests a few noisy utterances, emits symbolic frames with confidence scores, writes embeddings to SQLite, runs the template decoder for human-readable playback, and verifies round-trip retrieval without pulling in heavyweight NLP dependencies. Feed the resulting frame dumps into `analyze_pipeline.py --frames-path runs/pipeline-smoke_frames.json` to summarise languages, slot coverage, and confidence trends alongside the usual suite metrics. When you want to iterate manually, the interactive shell (`python experiments/retrieval_sanity/pipeline/interactive.py`) lets you type sentences, persist frames, receive memory-backed responses, and inspect nearest neighbours in real time.



**How it works:**Launch the shell with plasticity instrumentation enabled by default:

- Fuzzy matcher with multiple similarity metrics

- Levenshtein edit distance (character-level)```bash

- Token overlap (word-level, handles word order)python experiments/retrieval_sanity/pipeline/interactive.py \

- Common variation detection	--sqlite-path runs/pipeline_interactive_vectors.db \

- Threshold: 0.65 minimum similarity	--scenario interactive \

	--plasticity-threshold 0.6 \

**Test results:**	--plasticity-mu-lr 5e-4 \

- Exact match: 1.00 score	--plasticity-center-lr 5e-4 \

- Single typo: 0.95 score  	--plasticity-state-path runs/pipeline_interactive_pmflow.pt

- Double typos: 0.72 score```



**Files:** `pipeline/fuzzy_matcher.py`, `pipeline/response_fragments.py`Use `--no-plasticity` to freeze the PMFlow field or `--no-trace` if you want to skip JSONL trace emission during quick experiments.



### 5. Grammar RefinementSeed the baseline dataset for the interactive scenario before launching the shell. A curated set of coordination utterances (English primary with a few Spanish, French, and Portuguese variants) lives in `datasets/interactive_seed.jsonl`; ingest it into SQLite with:



Hybrid approach combining pattern adaptation and grammatical correction:```bash

python experiments/retrieval_sanity/pipeline/seed_dataset.py --clear --validate

``````

Before: "discuss think about"

After: "think about and discuss"The script wipes any previous entries (because of `--clear`), regenerates frames via the pipeline, and stores the embeddings under the `interactive` namespace. Pass `--sqlite-path` or `--scenario` to target a different database/namespace, `--pmflow-state-path` to align with a pre-trained PMFlow snapshot, and `--validate` to confirm the self-retrieval accuracy of the seeded frames. Swap in a custom JSON/JSONL file if you want to test different vocabularies or languages—the loader accepts either format as long as each entry has a `text` field.

```

Sample transcript (sanitised prompt/output) showing a plasticity event and persisted state:

**How it works:**

- Pattern-based adaptation (learns from context)```text

- Grammar post-processing (fixes errors)$ python experiments/retrieval_sanity/pipeline/interactive.py \

- Maintains conversational quality (8.2/10)	--sqlite-path runs/pipeline_interactive_vectors.db \

	--scenario interactive \

**Files:** `pipeline/syntax_stage_bnn.py`	--plasticity-threshold 0.6 \

	--plasticity-mu-lr 5e-4 \

## Architecture	--plasticity-center-lr 5e-4 \

	--plasticity-state-path runs/pipeline_interactive_pmflow.pt

### Pipeline Flow[info] loaded 128 frames into namespace interactive

[assistant] ready — type an utterance or :quit

```> remind me about the venue change

User Input[plasticity] recall=0.71 Δμ=1.2e-03 Δc=8.5e-04 refreshed=4

    ↓[plasticity] encoder state saved to runs/pipeline_interactive_pmflow.pt

[Context Encoder] → Enriches with conversation history[assistant] We moved the venue to the waterfront studio for tomorrow's session.

    ↓```

[Semantic Stage] → BNN embedding

    ↓Recent pipeline upgrades:

[Pattern Retrieval] → Fuzzy-tolerant matching

    ↓- The heuristic parser now records question intent, negation, and coarse slot roles (actor, action, target, location, time). These richer signals feed into symbolic frames so downstream decoders can surface context such as `time='next week'` and `intent='when'`.

Low confidence? → [Wikipedia Lookup] → Learn & respond- A PMFlow-backed embedding encoder (`PMFlowEmbeddingEncoder`) combines hashed bag-of-words features with a deterministic PMFlow latent field. The pipeline prefers it automatically and falls back to the original hashed encoder when `pmflow_bnn` is unavailable, keeping the workflow dependency-light.

    ↓- Decoder summaries expose the new metadata, making it easier to verify that location/time detection and negation cues survive the full ingest → retrieval → response loop.

[Response Composer] → Grammar refinement- Each conversational turn can now be traced to JSONL via the interactive shell (`--trace-path`) or programmatic `TraceLogger`. The analyzer consumes these traces to report recall rates and neighbour confidence trends, laying the groundwork for plasticity experiments.

    ↓- PMFlow plasticity updates can now run inline during interactive sessions. Tune activation with `--plasticity-threshold`, `--plasticity-mu-lr`, and `--plasticity-center-lr`, or disable the controller via `--no-plasticity`. When plasticity fires, trace records include a `plasticity` payload summarising the recall score and Δparameters so you can audit adaptation episodes.

Bot Response- Plasticity hooks now re-encode the affected scenario immediately after a PMFlow update. Stored frames keep the token sequence alongside the symbolic summary so the refresh logic can rebuild embeddings in SQLite. The trace payload also reports `refreshed`, indicating how many embeddings were recomputed for the turn.

    ↓- The PMFlow encoder now saves and reloads its state automatically after every plasticity step. By default the interactive shell drops a snapshot to `runs/<scenario>_pmflow.pt`; override the location with `--plasticity-state-path` when you want multiple experiments to share or isolate PMFlow memory.

[Pragmatic Learner] → Learn from success/teaching

```Representative redacted trace fragment showing the persisted state metadata:



### Key Components```json

{

**`pipeline/`** - Core neuro-symbolic processing	"turn": 3,

- `semantic_stage.py` - BNN-based semantic encoding	"speaker": "assistant",

- `response_fragments.py` - Pattern storage and retrieval	"text": "We moved the venue to the waterfront studio for tomorrow's session.",

- `response_composer.py` - Response assembly with coherence	"plasticity": {

- `context_encoder.py` - Multi-turn context tracking		"scenario": "interactive",

- `pragmatic_learner.py` - Learning from teaching mechanism		"recall": 0.71,

- `knowledge_augmenter.py` - Wikipedia integration		"refreshed": 4,

- `fuzzy_matcher.py` - Typo-tolerant matching		"state_path": "runs/pipeline_interactive_pmflow.pt",

- `syntax_stage_bnn.py` - Grammar refinement		"delta_mu_norm": 0.0012,

		"delta_center_norm": 0.00085

**`conversation_loop.py`** - Main interactive interface	}

}

**`test_*.py`** - Comprehensive test suite```



## Testing### Probing stored symbols



### Run All TestsUse the symbolic probe to sample stored embeddings and verify that persisted representations still deliver sensible neighbours once training is finished:



```bash```bash

# Learning from teachingpython experiments/retrieval_sanity/benchmarks/symbolic_probe.py \

python3 test_teach_then_converse.py	--sqlite-path runs/soak_vectors.db \

	--scenario soak-02:synthetic-hard \

# Wikipedia knowledge augmentation	--samples 5 \

python3 test_knowledge_augmentation.py	--output experiments/retrieval_sanity/benchmarks/symbolic_probe_soak02.json

```

# Typo tolerance

python3 test_typo_integration.pyEach query perturbs a PMFlow embedding with configurable noise and retrieves top-`k` matches from the baseline namespace, printing the neighbour labels and scores. The JSON payload records label distributions and neighbour lists so you can track how symbolic retrieval evolves between soak runs.



# Multi-turn coherenceIf no thresholds are violated you'll see a heartbeat similar to:

python3 test_multi_turn_coherence.py

```

# Clean teaching scenarios[observer] No alerts. Last PMFlow retrieval: 1.0000

python3 test_clean_teaching.py```

```

where the trailing metric reflects the newest PMFlow run observed in the log.

### Quality Assessment

Future work can plug these hooks into notification systems or trigger automated fine-tuning jobs when alerts fire.

```bash

python3 test_conversation_quality.py### Sample retrieval snapshot (CLI)

```

The latest smoke run (synthetic data, 60 epochs on CPU) produced the following JSON payload:

Current quality: **8.2/10** (maintains conversational naturalness while adding learning capabilities)

```

## Configuration{

	"baseline_retrieval_accuracy": 1.0,

### Pattern Store	"pmflow_retrieval_accuracy": 1.0,

	"support_samples": 144,

Patterns are stored in `response_patterns.json` (default). To reset:	"query_samples": 96,

	"pm_steps": 3,

```bash	"n_centers": 24,

rm response_patterns.json	"provenance": {

# Will bootstrap with Cornell Movie Dialogs on next run		"runner": "cli",

```		"device": "cpu",

		"git": {"commit": null}

### Enable/Disable Features	}

}

In `conversation_loop.py`:```



```pythonWith the current hyperparameters both models tie at 100% retrieval, which is expected on the easier synthetic clusters. When you dial up difficulty (more classes, lower support ratio, noisier clusters) or switch to the relational export, the PMFlow variant should maintain a higher retrieval accuracy than the baseline. Use these snapshots as breadcrumbs in the README to communicate experiment intent and to highlight when an alert should prompt deeper investigation.

# Knowledge augmentation

composer = ResponseComposer(#### Harder synthetic stress test

    fragment_store=fragments,

    conversation_state=state,To raise the difficulty we cranked up class count, reduced the support set, and injected additional cluster noise:

    enable_knowledge_augmentation=True  # Toggle here

)```bash

python experiments/retrieval_sanity/run_retrieval_sanity.py \

# Fuzzy matching	--num-classes 6 \

fragments = ResponseFragmentStore(	--points-per-class 160 \

    semantic_encoder,	--support-ratio 0.4 \

    enable_fuzzy_matching=True  # Toggle here	--cluster-std 0.9 \

)	--epochs 120 \

	--n-centers 48 \

# Grammar refinement	--pm-steps 6 \

composer = ResponseComposer(	--pm-beta 0.12 \

    use_grammar=True  # Toggle here	--pm-clamp 4.0

)```

```

Baseline retrieval landed at **0.858**, while PMFlow held **0.863**, keeping a small but positive margin under the noisier regime. Expect run-to-run variance at this difficulty—use multiple seeds or bootstrapped averages if you need stronger statistical guarantees. Even modest lifts like this one are enough for the observer to stay quiet now but flip to alerting if the gap collapses on future runs.

### Confidence Thresholds

#### Relational export check-in

In `pipeline/pragmatic_learner.py`:

For a closer-to-production signal, run the relational preset:

```python

# Teaching confidence boost```bash

initial_confidence = 0.85  # High confidence for taught knowledgepython experiments/retrieval_sanity/run_retrieval_sanity.py \

	--config experiments/retrieval_sanity/configs/relational_sample.json

# Adaptive thresholds```

threshold = 0.65 if usage_count < 5 else 0.80  # New vs established patterns

```This configuration uses the deterministic `relational_export.json` mock and currently yields identical scores (baseline and PMFlow both at **1.0** retrieval/class accuracy). The takeaway is that the bundled export is linearly separable enough that PMFlow cannot showcase an advantage. To increase realism, consider:



## Project Structure- Re-exporting a noisier slice of relational data or injecting perturbations.

- Lowering the support ratio further to stress zero-shot retrieval.

```- Tracking multiple random seeds and feeding the results into the observer to catch degradation once the dataset becomes less trivial.

experiments/retrieval_sanity/

├── README.md                           # This file### Batch benchmarking with `run_suite.py`

├── conversation_loop.py                # Main interactive interface

├── pipeline/                           # Core neuro-symbolic componentsDrive the scenarios above (and future ones) via a single entry point that logs each run and computes retrieval deltas:

│   ├── semantic_stage.py              # BNN semantic encoding

│   ├── response_fragments.py          # Pattern storage/retrieval```bash

│   ├── response_composer.py           # Response assemblypython experiments/retrieval_sanity/run_suite.py --evaluate-policy

│   ├── context_encoder.py             # Multi-turn context```

│   ├── pragmatic_learner.py           # Learning from teaching

│   ├── knowledge_augmenter.py         # Wikipedia integrationBy default the suite executes the `synthetic-baseline`, `synthetic-hard`, and `relational-sample` presets (defined in `configs/`). Specify `--scenario` repeatedly to target a subset or add new configs as the roadmap expands. Each run is appended to `retrieval_runs.log` with provenance, making the observer immediately useful. Sample summary:

│   ├── fuzzy_matcher.py               # Typo tolerance

│   └── syntax_stage_bnn.py            # Grammar refinement```

├── test_teach_then_converse.py        # Learning validation=== Retrieval summary ===

├── test_knowledge_augmentation.py     # Wikipedia integration testscenario               baseline     pmflow      delta

├── test_typo_integration.py           # Typo tolerance test-----------------------------------------------------

├── test_multi_turn_coherence.py       # Coherence validationsynthetic-baseline       1.0000     1.0000    +0.0000

├── test_conversation_quality.py       # Quality assessmentsynthetic-hard           0.8576     0.8628    +0.0052

├── docs/                              # Documentationrelational-sample        1.0000     1.0000    +0.0000

│   ├── multi_modal_architecture.md    # Multi-modal vision-----------------------------------------------------

│   ├── knowledge_augmentation.md      # Wikipedia integration docs[suite] Policy evaluation produced no alerts. Latest delta: 0.0

│   └── session_summary_nov25.md       # Recent session summary```

└── response_patterns.json             # Learned pattern store

```Tune `--policy-window`, `--policy-min-delta`, and `--policy-max-drop` to mirror production thresholds; alerts will surface inline when the drift policy fires.



## Development### Persisting symbols in SQLite



### Adding New Knowledge SourcesSwitch the vector-store backend to SQLite to capture embeddings across runs and scenarios:



Extend `pipeline/knowledge_augmenter.py`:```bash

python experiments/retrieval_sanity/run_retrieval_sanity.py \

```python	--vector-store sqlite \

class KnowledgeAugmenter:	--sqlite-path runs/sqlite_vectors.db \

    def lookup(self, query):	--sqlite-scenario smoke

        # Try Wikipedia```

        wiki_result = self.wikipedia.lookup(query)

        if wiki_result:The runner records the selected backend and database path in the JSON summary and automatically creates the database (including tables and indexes) if it does not exist. Each scenario gets a distinct namespace—when `--sqlite-scenario smoke`, embeddings persist under `smoke:baseline` and `smoke:pmflow`. Pass the same flags to `run_suite.py` to reuse the database across preset scenarios:

            return wiki_result

        ```bash

        # Add new source herepython experiments/retrieval_sanity/run_suite.py \

        # wolframalpha_result = self.wolfram.lookup(query)	--vector-store sqlite \

        # if wolframalpha_result:	--sqlite-path runs/sqlite_vectors.db \

        #     return wolframalpha_result	--sqlite-prefix nightly

        ```

        return None

````run_suite.py` automatically expands each preset to a unique namespace such as `nightly:synthetic-baseline:baseline` and `nightly:synthetic-baseline:pmflow`, keeping scenario data isolated while sharing the same database. You can inspect the resulting file with `sqlite3 runs/sqlite_vectors.db` to snapshot stored embeddings, label assignments, and run metadata for downstream symbolic reasoning experiments.



### Adding New Learning Mechanisms### Adding provenance metadata



1. Create pattern extractor in `pipeline/pragmatic_learner.py`Both the notebook and CLI now attach metadata from `provenance.collect_provenance()` to every result payload. This records:

2. Define success signals in `pipeline/response_learner.py`

3. Add tests in `test_*.py`- Git commit, branch, and dirty state when the workspace is within a git repository.

- Python/Torch versions and basic platform info.

### Debugging- The runner origin (`cli` vs `notebook`) along with optional extras like CLI arguments.



Enable verbose logging:Use this metadata to trace regressions back to exact code revisions or environment changes before kicking off automated mitigations.



```python### Loading configuration overrides

# In conversation_loop.py

import loggingThe `config_utils` helpers make it easy to reuse parameter sets across scripts and the notebook:

logging.basicConfig(level=logging.DEBUG)

``````python

from pathlib import Path

View pattern retrieval process:

from experiments.retrieval_sanity.config_utils import load_overrides

```python

# Prints matched patterns and scoresoverrides = load_overrides(Path("experiments/retrieval_sanity/configs/synthetic_baseline.json"))

patterns = fragments.retrieve_patterns(query, topk=5)run_and_log(overrides)

for pattern, score in patterns:```

    print(f"Pattern: {pattern.trigger_context} | Score: {score:.2f}")

```Each config supplies a `label` and an `overrides` mapping aligned with the CLI/notebook keyword arguments. Add your own JSON files to `configs/` to capture new experimental conditions.



## Performance### Running tests



### Current MetricsPytest scaffolding lives at the repository root. After installing dependencies, run:



- **Quality**: 8.2/10 conversational naturalness```bash

- **Pattern Store**: 1,265-1,296 patterns (Cornell base)pytest

- **Teaching Success**: ~70% (some spurious matches)```

- **Wikipedia Success**: ~70% query success rate

- **Typo Tolerance**: 95% single typo, 72% double typoThe test suite currently covers provenance logging, configuration helpers, and the log observer pipeline. Extend it as new components are introduced.

- **Multi-turn Coherence**: 100% (when knowledge exists)

## Configuration surface

### Optimization Opportunities

Both the script and notebook share the same configuration parameters:

1. **Pattern consolidation** - Merge similar patterns

2. **Teaching reliability** - Fix spurious matches| Parameter | Purpose |

3. **Dataset expansion** - Add knowledge-based conversations| --- | --- |

4. **Caching** - Cache Wikipedia lookups| `data-mode` | Choose `synthetic` clusters or load a relational export |

5. **Indexing** - Optimize pattern retrieval for large stores| `support-ratio` | Split between support and query sets |

| `cluster-std`, `points-per-class`, `num-classes` | Control synthetic dataset difficulty |

## Roadmap| `latent-dim`, `n-centers`, `pm-steps`, `pm-beta`, `pm-dt`, `pm-clamp` | PMFlow latent field hyperparameters |

| `metric` | Vector store similarity (`cosine` or `euclidean`) |

### Completed ✅| `epochs`, `lr`, `weight-decay` | Training loop knobs |

- [x] Learning from teaching mechanism

- [x] Multi-turn coherenceAdjust these through CLI flags or notebook overrides (e.g., `run_and_log(overrides={"epochs": "600"})`).

- [x] Wikipedia knowledge augmentation

- [x] Typo tolerance (fuzzy matching)## Validation checklist

- [x] Grammar refinement

- [x] Automatic success detectionAfter each experiment, confirm:



### In Progress 🔄1. **Baseline vs PMFlow gap** – PMFlow retrieval accuracy should match or exceed the baseline.

- [ ] Teaching reliability improvements2. **Training stability** – Plot loss/accuracy curves via `plot_training_curves()` to catch divergence or overfitting.

- [ ] Pattern consolidation and pruning3. **Artifact integrity** – Ensure logs and plots write correctly so future automation can ingest them.



### Planned 📋## Suggested evolution

- [ ] Multi-modal inputs (vision, audio)

- [ ] Additional knowledge sources (Wolfram Alpha, Stack Overflow)- Wrap the CLI in a scheduled job or lightweight service to collect longer-term trend data.

- [ ] Fact verification and cross-referencing- Extend the log schema with git commit hashes and environment metadata for full provenance.

- [ ] Long-term memory consolidation- Build a streaming observer that tails `retrieval_runs.log` and triggers fine-tuning when drift thresholds fire.

- [ ] Emotional intelligence (sentiment, empathy)- Integrate symbolic reasoning modules that can consume the retrieved facts, as outlined in the notebook roadmap.

- [ ] Multi-language support

For a deeper architectural plan, review the roadmap embedded in the notebook or port it into a project-wide documentation site as the platform matures.

## Contributing

This is a research project exploring neuro-symbolic approaches to conversational AI. Contributions welcome!

### Areas of Interest

1. **Better learning mechanisms** - Improve pattern extraction
2. **Knowledge integration** - New external sources
3. **Evaluation metrics** - Better quality assessment
4. **Efficiency** - Faster retrieval, lower memory
5. **Robustness** - Handle edge cases, errors

## Citation

If you use Lilith in your research:

```
@software{lilith2025,
  title={Lilith: A Neuro-Symbolic Conversational AI},
  author={[Author]},
  year={2025},
  url={https://github.com/[repo]/lilith}
}
```

## License

[To be determined]

## Acknowledgments

- Cornell Movie Dialogs Corpus for base conversation patterns
- Wikipedia for knowledge augmentation
- PyTorch community
- PMFlow BNN architecture inspiration

---

**Status**: Active development (November 2025)

For more details, see `docs/` directory or run the test suite.
