# Lilith Project: Scope and Goals

**Last Updated:** 10 December 2024

## Project Overview

Lilith is a **neuro-symbolic conversational AI** that demonstrates sophisticated reasoning and learning without requiring massive language models or computational resources. The system uses databases as primary knowledge storage and treats biological neural networks (BioNN) as adaptive retrieval and reasoning components.

## Core Architectural Principles

### 1. Database-First Knowledge Storage
- Relational/document databases as canonical, human-editable memory (SQLite currently, PostgreSQL planned)
- Facts and knowledge stored in structured DB rather than encoded in neural weights
- Reduces memory and compute requirements compared to large parameter-count models
- Enables direct human inspection, editing, and validation of knowledge

### 2. BioNN + Database Architecture
- Each cognitive layer pairs a Biological Neural Network (BioNN) with a database
- BioNN learns "how to look things up" (pattern recognition)
- Database stores "what to look up" (facts, patterns, rules)
- **Open book exam metaphor**: BioNN develops indexing skills, database holds the content

### 3. Staged Pipeline Processing
Lilith mirrors biological cognition with specialized layers:

1. **Intake Layer**: Noise normalization, typo correction
   - BioNN: Recognizes character/token patterns  
   - Database: Stores normalization rules
   
2. **Semantic Layer**: Concept understanding
   - BioNN: Word embeddings via PMFlow encoder
   - Database: Concept taxonomy and relationships

3. **Syntax Layer**: Grammatical structure
   - BioNN: POS patterns and phrase structures
   - Database: Grammatical templates

4. **Pragmatic/Response Layer**: Dialogue and composition
   - BioNN: Intent recognition and semantic encoding
   - Database: Conversation patterns and responses

Each layer operates independently and can be optimized or replaced modularly.

### 4. PMFlow Biological Neural Networks
- **PMFlow (Pushing Medium Flow)**: Physics-inspired neural dynamics with gravitational flow centers
- Provides adaptive, plastic embeddings that improve through online learning
- BioNN plasticity enables continuous refinement without full retraining
- Temporal parallelism keeps resource requirements modest
- Embeddings capture semantic similarity for retrieval and reasoning

### 5. Online Learning and Adaptation
- System learns continuously from conversations (no periodic retraining cycles)
- Success-based learning: tracks which patterns work for which queries
- Hybrid retrieval: BioNN semantic similarity + keyword matching + success scores
- User teaching: "X is Y" statements are learned immediately
- Adaptive confidence thresholds based on pattern usage

### 6. Modal Routing
- Automatic classification of query types: MATH, CODE, LINGUISTIC
- Math queries routed to symbolic computation (SymPy) - no database pollution
- Linguistic queries use learned patterns and knowledge sources
- Protects learning database from computational queries

### 7. Modest Hardware Requirements
- Designed for consumer-grade hardware (tested on single CPU/GPU)
- Database handles knowledge scale; neural components stay lightweight
- Retrieval + symbolic reasoning replaces massive parameter counts
- GPU optional but beneficial for PMFlow acceleration

## Current Implementation Status

### Core Features ✅

**Modal Routing**:
- Automatic MATH vs LINGUISTIC classification
- Symbolic computation using SymPy
- Database protection from math queries

**Learning Capabilities**:
- 100% recall of taught knowledge
- Success-based pattern reinforcement
- Fuzzy matching for typo tolerance
- Multi-turn conversation coherence

**Knowledge Sources**:
- Wikipedia (general knowledge)
- Wiktionary (word definitions)
- WordNet (synonyms/antonyms - offline)
- Free Dictionary (pronunciations)
- Smart source selection per query type

**Multi-User Support**:
- Isolated user databases with shared base knowledge
- SQLite backend with ACID guarantees
- Thread-safe concurrent access
- Teacher mode for base knowledge updates

**BioNN Integration**:
- PMFlow embedding encoder (64-384 dimensions)
- Contrastive learning for semantic similarity
- Intent clustering for faster retrieval
- Syntax stage for grammatical patterns
- Online plasticity (experimental)

### Architecture Components

**Core Package** (`lilith/`):
- `embedding.py` - PMFlow-based semantic encoder
- `response_composer.py` - Main orchestrator  
- `modal_classifier.py` - Query type classification
- `math_backend.py` - Symbolic computation
- `conversation_state.py` - Context management
- `knowledge_augmenter.py` - External knowledge integration
- `syntax_stage_bnn.py` - Grammar processing
- `bnn_intent_classifier.py` - Neural intent detection
- `multi_tenant_store.py` - User isolation

**Storage**:
- SQLite for current deployment
- Conversation patterns database
- User-specific databases
- Success tracking tables
- Planned migration to PostgreSQL + pgvector

### Test Coverage
- 40+ tests covering all major components
- Integration tests for end-to-end flows
- Multi-tenant and concurrency tests
- Math backend validation
- Knowledge source integration tests

## Roadmap

### Completed ✅
- BioNN + Database architecture across all layers
- Modal routing with math backend
- Multi-user architecture with SQLite
- Success-based learning
- Multiple knowledge sources
- BioNN intent clustering
- Contrastive learning for embeddings

### In Progress 🔄
- PMFlow plasticity fine-tuning
- Long-term conversation memory
- Cross-layer coordination improvements

### Planned 📋
- PostgreSQL + pgvector migration
- Multi-modal inputs (vision, audio)
- Additional knowledge sources
- Fact verification and cross-referencing
- Emotional intelligence (sentiment, empathy)
- Multi-language support beyond English

## Key Design Decisions

### Why Database-First?
- **Scalability**: DB scales horizontally; neural models scale vertically
- **Transparency**: Humans can inspect and edit knowledge directly
- **Reliability**: ACID guarantees, proven technology
- **Speed**: Modern DBs + indexes are fast for retrieval
- **Correctness**: Explicit facts > implicit neural encodings

### Why BioNN Over Standard Embeddings?
- **Plasticity**: Online learning without full retraining
- **Principled dynamics**: Physics-based rather than purely statistical
- **Efficiency**: Temporal parallelism enables fast computation
- **Biological plausibility**: Mimics neural tissue dynamics
- **Proven**: Outperforms baselines on benchmarks

### Why Modal Routing?
- **Specialization**: Different query types need different processing
- **Database protection**: Computational queries don't pollute learning
- **Efficiency**: Route to appropriate backend immediately
- **Accuracy**: Symbolic math > neural approximation

## Success Metrics

### Current Performance
- **Learning**: 100% recall of taught knowledge
- **Typo tolerance**: 95% single typo, 72% double typo  
- **Multi-turn coherence**: Tracks topics across conversation
- **Math accuracy**: 100% on supported operations (symbolic)
- **Knowledge sources**: Multiple fallback options
- **Concurrent users**: Tested with 10+ simultaneous users

### Production Goals
- Support 1000+ concurrent users
- <100ms query latency (90th percentile)
- PostgreSQL KB with 1M+ facts
- 90%+ retrieval accuracy on domain tests
- Online learning improves metrics 10%+ over 1000 interactions

## Non-Goals

- **Not building**: Another massive LLM to compete with GPT/Claude
- **Not aiming for**: Best performance on standard LLM benchmarks  
- **Not prioritizing**: Real-time latency <10ms
- **Not requiring**: Datacenter-scale infrastructure
- **Not targeting**: API service deployment (focus is architecture research)

## Related Work

### Compared to Standard RAG
- **Lilith**: Multi-layer BioNN processing, online learning, modal routing
- **RAG**: Vector store + frozen LLM
- **Advantage**: Compositional reasoning, continuous adaptation

### Compared to Knowledge Graphs + LLM
- **Lilith**: Adaptive embeddings, unified symbolic substrate
- **KG + LLM**: Static schema, separate stacks
- **Advantage**: Online adaptation, physics-grounded embeddings

### Compared to Pushing-Medium
- **Pushing-Medium**: Physics simulations + PMFlow/BioNN library  
- **Lilith**: Complete AI system using PMFlow
- **Relationship**: Lilith depends on Pushing-Medium for BioNN primitives

## Repository Structure

```
lilith/
├── lilith/                  # Core package
│   ├── embedding.py
│   ├── response_composer.py
│   ├── modal_classifier.py
│   └── ...
├── tests/                   # Test suite
├── docs/                    # Documentation
│   ├── PROJECT_SCOPE.md    # This file
│   ├── ARCHITECTURE_VERIFICATION.md
│   └── ...
├── data/                    # Databases and patterns
├── tools/                   # Utilities
├── pmflow_bnn_enhanced/     # Enhanced BioNN library
├── Pushing-Medium/          # Dependency (PMFlow library)
└── experiments/             # Research (archived)
```

## Key Documentation

- **Architecture**: `ARCHITECTURE_VERIFICATION.md` - BioNN + Database design
- **User Guide**: `USER_GUIDE.md` - How to use Lilith
- **Dependencies**: `DEPENDENCIES.md` - Installation guide
- **Storage**: `SQLITE_MIGRATION.md` - Database backend details
- **Training**: `TRAINING_GUIDE.md` - How to train Lilith
- **Pushing-Medium**: https://github.com/experimentech/Pushing-Medium

## License

TBD - likely permissive open-source

---

**Summary**: Lilith demonstrates that **layered BioNN + Database architecture** with **modal routing** and **online learning** can produce capable AI systems without massive computational resources, while enabling human oversight, continuous adaptation, and specialized processing that pure neural approaches struggle with.
