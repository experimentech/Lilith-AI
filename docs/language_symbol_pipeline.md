# Language-to-Symbol Pipeline Sketch

This document captures a lightweight blueprint for building an error-resilient ingestion path that normalises raw text, extracts symbolic structure, persists those symbols alongside PMFlow embeddings, and reuses them for reasoning or generation. The intent is to keep the design modular and swappable so each component can be upgraded independently as richer models come online.

## Layered overview

1. **Intake and noise normalisation**
   - Accept messy, multilingual text (typos, casing drift, code switching).
   - Apply character-level cleanups (whitespace collapse, punctuation spacing, trivial typo maps).
   - Generate alternate spellings using heuristic edits to keep downstream options open.

2. **Syntactic and morphological analysis**
   - Tokenise text into coarse units (words, subwords).
   - Assign lightweight part-of-speech categories via heuristics or compact taggers.
   - Produce a simple dependency-like structure to indicate likely subjects, verbs, and objects.

3. **Symbolic envelope**
   - Collapse analysed tokens into schema-free tuples: `(actor, action, target, modifiers)`.
   - Retain language metadata and provenance for auditability.
   - Represent modifiers as key–value pairs to support incremental enrichment later.

4. **Embedding + storage bridge**
   - Convert the cleaned sentence into an embedding (PMFlow when training loops exist, hashed bag-of-words for smoke tests).
   - Persist both the embedding and the symbolic tuple into SQLite under a scenario namespace.
   - Mirror the tuple in JSON form for easy inspection and for linking with external knowledge stores.

5. **Retrieval and generation**
   - Query the vector store with noisy text; retrieve nearest embeddings and their associated symbolic tuples.
   - Use the tuples to guide a response template or a downstream generative model.
   - Feed failures (missing tuples, low confidence) back into the intake layer for iterative hardening.

## Design principles

- **Modularity first:** each layer exposes a narrow interface so we can swap in stronger models without rewriting the pipeline.
- **Language agnostic:** rely on heuristics and universal representations when full morphosyntactic coverage is unavailable.
- **Error resilience:** maintain alternate hypotheses instead of discarding noisy input outright.
- **Observability:** log intermediate artefacts so regression tracking and debugging stay simple.
- **Low dependency footprint:** default implementations use the Python standard library and NumPy/PyTorch that are already present in the project.

## Smoke-test scope

The first implementation aims to be quick-to-run proof-of-concept code, not a production-quality multilingual parser. The smoke test should:

- Accept a list of short utterances in mixed languages with deliberate typos.
- Produce normalised tokens, heuristic POS tags, and a best-guess symbolic frame for each utterance.
- Generate a simple embedding (hashed bag-of-words) and write it into the existing SQLite vector store.
- Retrieve the embedding via the vector store and surface the stored symbolic frame to prove round-tripping works.
- Render each frame through the template decoder to demonstrate symbol-to-language playback.

The runnable prototype lives under `pipeline/`. Execute the smoke script from the repository root to verify end-to-end behaviour:

```bash
python experiments/retrieval_sanity/pipeline/smoke.py
```

The script writes embeddings to `runs/pipeline_smoke_vectors.db`, dumps the symbolic frames to `runs/pipeline-smoke_frames.json`, and prints a compact summary that confirms normalisation, frame extraction, decoding, and retrieval are wired together.

### Interactive shell

For quick experiments, launch the interactive CLI:

```bash
python experiments/retrieval_sanity/pipeline/interactive.py --sqlite-path runs/pipeline_interactive_vectors.db
```

Type sentences (use `:quit` to exit). Each input is normalised, parsed, converted into a symbolic frame, optionally persisted, and queried against the SQLite store. The shell prints the decoder summary, generates a conversational response that references the closest stored memory, and lists the top neighbours so you can iterate on the ingestion heuristics without rerunning the smoke script.

### Module interfaces

- `pipeline/intake.py` – `NoiseNormalizer.normalise()` returns cleaned text; `generate_candidates()` offers alternate spellings.
- `pipeline/parser.py` – `parse()` yields `ParsedSentence` with heuristic POS tags and a confidence score indicating recognition quality.
- `pipeline/symbolic.py` – `build_frame()` converts parses into `SymbolicFrame` objects carrying modifiers, slot presence attributes, and confidence values.
- `pipeline/embedding.py` – `HashedEmbeddingEncoder.encode(tokens)` returns a unit-normalised vector ready for SQLite storage.
- `pipeline/decoder.py` – `TemplateDecoder.generate(frame)` renders a human-readable summary that includes language hints and confidence.
- `pipeline/responder.py` – `ConversationResponder.reply(artefact)` crafts a response by retrieving similar frames and feeding them through the template decoder.
- `pipeline/storage_bridge.py` – `SymbolicStore.persist(artefacts)` batches embeddings into SQLite and mirrors frames to JSON for analysis.
- `pipeline/pipeline.py` – `SymbolicPipeline.run()` orchestrates the full chain and produces `PipelineArtifact` bundles for inspection/testing.

Future iterations can replace each layer with more sophisticated models (e.g., actual UD parsers, PMFlow encoders, neural decoders) while keeping the interfaces intact.
