# Lilith Project: Scope and Goals

**Last Updated:** 23 November 2025

## Project Overview

Lilith is a **neuro-symbolic AI architecture** designed to provide capable reasoning and learning without requiring massive computational resources. The system inverts the traditional "giant LLM" paradigm by using databases as the primary knowledge store and treating neural components as adaptive retrieval and reasoning aids.

## Core Architectural Principles

### 1. Database-First Knowledge Storage
- **Relational database** as the canonical, human-editable memory layer for symbols, relations, and provenance
- Facts and knowledge stored in structured DB (target: Postgres + pgvector) rather than encoded in neural weights
- Drastically reduces memory and compute requirements compared to large parameter-count models
- Enables direct human inspection, editing, and validation of the knowledge base

### 2. Symbolic Decomposition (Ideographic Processing)
- Break complex concepts into symbolic primitives (ideographs) that are source-agnostic
- Symbolic frames capture: `(actor, action, target, modifiers)` structure
- Enables cross-modal reasoning and integration of heterogeneous data sources
- Language-agnostic representation allows multilingual and multi-format input/output
- Compositional reasoning: combine and recombine symbolic primitives

### 3. Staged Pipeline Processing
- Mirror biological cognition with successive specialized transformations:
  1. **Intake & Normalization**: Clean noisy input, generate candidate interpretations
  2. **Parsing**: Extract linguistic structure (POS tags, dependencies)
  3. **Symbolization**: Convert to symbolic frames
  4. **Embedding**: Generate adaptive PMFlow embeddings
  5. **Retrieval**: Query vector store for relevant context
  6. **Generation**: Compose responses from retrieved symbols + LLM orchestration
- Each stage can be optimized, upgraded, or replaced independently
- Modular design enables incremental improvement without full system rewrites

### 4. Adaptive Embeddings via PMFlow
- **PMFlow (Pushing Medium Flow)**: Physics-inspired neural dynamics with gravitational flow centers
- Provides adaptive, plastic embeddings that improve through online learning
- BNN (Bayesian Neural Network) plasticity enables continuous refinement without full retraining
- Temporal parallelism and efficient computation keep resource requirements modest
- Embeddings capture semantic similarity for retrieval and reasoning

### 5. Online Learning and Adaptation
- System learns continuously from interactions (online learning)
- Reward shaping guides BNN plasticity to improve retrieval quality
- No need for expensive periodic retraining cycles
- Automated constraint checking prevents knowledge base corruption
- Provenance tracking enables debugging and rollback of learned behaviors

### 6. Thin LLM Orchestration Layer
- LLM used for **planning and composition**, not knowledge storage
- Prompts constructed from validated symbolic context retrieved from DB
- Constraint-aware decoding ensures language layer respects symbolic ground truth
- Reduces hallucination by grounding generation in explicit facts
- LLM is swappable; system doesn't depend on any specific model

### 7. Modest Hardware Requirements
- Designed to run on consumer-grade hardware
- Database handles knowledge scale; neural components stay lightweight
- Retrieval + symbolic reasoning replaces massive parameter counts
- GPU optional but beneficial for PMFlow acceleration
- Horizontal scaling via distributed DB rather than vertical scaling of model size

## Current State: Experimental Phase

### Retrieval Sanity Pipeline (`experiments/retrieval_sanity/`)

**Primary Goal**: Validate that PMFlow embeddings outperform baseline embeddings for nearest-neighbor retrieval before committing to full neuro-symbolic architecture.

**Components Developed**:
1. **Symbolic Language Pipeline**:
   - Intake and noise normalization
   - Heuristic linguistic parser
   - Symbolic frame builder
   - Template-based decoder
   - Conversation responder

2. **Embedding Layer**:
   - Hashed baseline encoder (deterministic)
   - PMFlow embedding encoder (adaptive with plasticity)
   - Component access for plasticity training
   - State persistence (save/load)

3. **Storage Bridge**:
   - SQLite vector store (prototype; will migrate to Postgres + pgvector)
   - Symbolic frame persistence (JSON + DB)
   - Scenario namespacing for isolation
   - Batch operations for efficiency

4. **Conversation State**:
   - Working memory via PMFlow activation patterns
   - Topic tracking and continuity detection
   - Novelty metric for attention guidance
   - Response planning using retrieved context

5. **Observability**:
   - Provenance tracking (git, environment, config)
   - Run logging (JSONL format)
   - Drift detection and alerting
   - Automated constraint monitoring

### Test Coverage
- 12 passing tests covering pipeline, embedding, storage, conversation
- Benchmark suite for retrieval quality
- Integration tests for end-to-end flows
- Plasticity controller validation

## Roadmap to Production

### Phase 1: Complete Prototype Validation (Current)
- ✅ Symbolic pipeline implementation
- ✅ PMFlow embedding integration
- ✅ SQLite storage bridge
- ⚠️ Conversation state (fixing topic continuity)
- ⚠️ Plasticity reward shaping (in progress)

### Phase 2: Production Infrastructure (6-8 weeks)
- Migrate to Postgres + pgvector
- Implement relational schema for pilot domain (physical units)
- Build ETL pipelines for data ingestion
- Add constraint checking and validation
- Deploy reward ledger with dashboards

### Phase 3: LLM Integration
- Planner harness for query decomposition
- Prompt construction from symbolic context
- Constraint-aware decoding
- Response validation against KB
- Fallback handling for low-confidence scenarios

### Phase 4: Multi-Domain Expansion
- Domain-specific specialized learners
- Cross-domain reasoning capabilities
- Distributed PMFlow centers per domain
- Coordinated planner for complex queries
- Schema evolution tooling

## Key Design Decisions

### Why Database-First?
- **Scalability**: DB scales horizontally; neural models scale vertically (expensive)
- **Transparency**: Humans can inspect and edit knowledge directly
- **Reliability**: ACID guarantees, backup/recovery, proven technology
- **Speed**: Modern DBs + vector indexes are extremely fast for retrieval
- **Correctness**: Explicit facts > implicit neural encodings for verifiable reasoning

### Why PMFlow Over Standard Embeddings?
- **Plasticity**: Online learning without full retraining
- **Principled dynamics**: Physics-based rather than purely statistical
- **Efficiency**: Temporal parallelism enables fast computation
- **Biological plausibility**: Mimics neural tissue dynamics
- **Proven results**: Outperforms baselines on MNIST and retrieval benchmarks

### Why Symbolic Frames?
- **Compositionality**: Combine primitives to represent novel concepts
- **Abstraction**: Language-agnostic, modality-agnostic
- **Reasoning**: Enable logical inference over structured representations
- **Debugging**: Easier to understand than dense vectors
- **Human-in-loop**: Enables editing, validation, constraint authoring

## Success Criteria

### Prototype Phase
- ✅ PMFlow retrieval accuracy > baseline by ≥5% on hard synthetic clusters
- ✅ Symbolic pipeline processes multilingual, noisy text correctly
- ✅ Round-trip: text → symbols → embedding → retrieval → generation
- ⚠️ Conversation state tracks topics across multi-turn dialogue
- ⚠️ Plasticity improves retrieval quality under reward signals

### Production Phase
- Postgres KB handles ≥1M facts with <100ms query latency
- Retrieval accuracy ≥90% on domain-specific test sets
- Constraint violations <1% with automated monitoring
- System runs on single GPU workstation (RTX 3090 or equivalent)
- Online learning improves metrics by ≥10% over baseline within 1000 interactions

### Long-Term Vision
- Multi-domain reasoning across scientific, linguistic, and common-sense knowledge
- Human-editable knowledge base with GUI tooling
- Community-contributed domain modules and schemas
- Open-source reference implementation
- Academic publications demonstrating neuro-symbolic advantages

## Non-Goals

- **Not building**: Another massive LLM to compete with GPT/Claude/etc.
- **Not aiming for**: Best-in-class performance on standard LLM benchmarks
- **Not prioritizing**: Deployment as API service (focus is architecture research)
- **Not requiring**: Datacenter-scale infrastructure
- **Not targeting**: Real-time latency <10ms (acceptable to be slower but smarter)

## Related Work and Differentiation

### Compared to Standard RAG
- **Lilith**: Symbolic reasoning layer between retrieval and generation
- **Standard RAG**: Vector store bolted onto frozen LLM
- **Advantage**: Compositional reasoning, online learning, constraint enforcement

### Compared to Knowledge Graphs + LLM
- **Lilith**: Adaptive embeddings with plasticity, unified symbolic substrate
- **KG + LLM**: Static schema, separate reasoning and generation stacks
- **Advantage**: Online adaptation, physics-grounded embeddings

### Compared to Neuro-Symbolic Systems (e.g., Scallop, DeepProbLog)
- **Lilith**: Production-oriented architecture with DB persistence
- **Existing**: Research frameworks without operational tooling
- **Advantage**: Observability, provenance, human-editable KB

### Compared to Pushing-Medium (Parent Library)
- **Pushing-Medium**: Physics simulations + PMFlow/BNN ML components (library/toolkit)
- **Lilith**: Complete AI system architecture using PMFlow as embedding layer (application)
- **Relationship**: Lilith depends on Pushing-Medium for PMFlow/BNN primitives

## Repository Structure

```
lilith/
├── experiments/
│   └── retrieval_sanity/       # Current prototype implementation
│       ├── pipeline/           # Symbolic language processing
│       ├── benchmarks/         # Retrieval quality tests
│       ├── configs/            # Experiment configurations
│       └── docs/               # Design documentation
├── tests/                      # Test suite
├── Pushing-Medium/             # Dependency (PMFlow/BNN library)
├── concept_assessment_summary.md  # Architecture concept doc
├── PROJECT_SCOPE.md            # This file
└── pyproject.toml              # Python package config
```

## Key Resources

- **Concept Assessment**: `concept_assessment_summary.md` - Original architectural vision
- **Retrieval Lab README**: `experiments/retrieval_sanity/README.md` - How to run experiments
- **Language Pipeline**: `experiments/retrieval_sanity/docs/language_symbol_pipeline.md` - Pipeline design
- **Pushing-Medium Repo**: https://github.com/experimentech/Pushing-Medium - Dependency library

## Team and Contributions

- **Current Status**: Single-developer research project
- **Open Source**: TBD (likely after prototype validation)
- **Contributions**: Not yet accepting external contributions
- **Contact**: Via repository owner

## License

TBD - likely permissive open-source after prototype phase

---

**Summary**: Lilith aims to demonstrate that **symbolic reasoning + adaptive embeddings + database-first architecture** can produce capable AI systems without requiring massive computational resources, while enabling human oversight, online learning, and compositional reasoning that pure neural approaches struggle with.
