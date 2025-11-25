# Lilith - Conversational AI with Modal Routing

A conversational AI system with modal routing architecture, supporting mathematical computation, linguistic learning, and knowledge augmentation.

## Directory Structure

```
lilith/
├── lilith/              # Core Lilith package
│   ├── embedding.py           # PMFlow-based embedding encoder
│   ├── response_composer.py   # Main conversation orchestrator
│   ├── modal_classifier.py    # Query type classification
│   ├── math_backend.py        # Symbolic math computation
│   ├── conversation_state.py  # Conversation context management
│   └── ...                    # Other core modules
│
├── tests/               # All test files
│   ├── test_conversation.py   # Main conversation tests
│   ├── test_math_backend.py   # Math backend tests
│   └── ...                    # Other tests
│
├── docs/                # Documentation
│   ├── PROJECT_SCOPE.md
│   ├── PMFLOW_INTEGRATION.md
│   └── ...
│
├── data/                # Data files
│   ├── datasets/              # Training datasets
│   ├── *.json                 # Pattern files
│   ├── *.db                   # Database files
│   └── *.pt                   # Model weights
│
├── tools/               # Utilities
│   └── archive_project.sh     # Project archiver
│
├── experiments/         # Research and experiments
│   └── retrieval_sanity/      # Original research directory
│
├── pmflow_bnn_enhanced/ # Enhanced PMFlow-BNN library
└── pmflow-package/      # Standalone PMFlow package (separate repo)
```

## Quick Start

### Run Conversation Test

```bash
source .venv/bin/activate
python tests/test_conversation.py
```

### Features

- ✅ **Modal Routing**: Automatic classification of MATH vs LINGUISTIC queries
- ✅ **Math Backend**: Exact symbolic computation using SymPy
- ✅ **Database Protection**: Math queries don't pollute learning database
- ✅ **Linguistic Learning**: Learn from conversations with 100% recall
- ✅ **External Knowledge**: Wikipedia integration for unknown topics
- ✅ **BNN Intent Clustering**: Neural clustering for better understanding
- ✅ **Fuzzy Matching**: Typo tolerance in query matching
- ✅ **Multi-Tenant Architecture**: Isolated user databases with shared base knowledge
- ✅ **SQLite Backend**: Thread-safe concurrent access with ACID guarantees
- ✅ **Scalable Storage**: Handles concurrent users without data corruption

## Architecture

Lilith uses a modal routing architecture that classifies queries into different modalities:
- **MATH**: Symbolic computation (addition, multiplication, exponents, equations)
- **CODE**: Programming-related queries (not yet implemented)
- **LINGUISTIC**: Natural language conversation and learning

## Development

The main working code is in `lilith/` with comprehensive tests in `tests/`.
Research experiments remain in `experiments/` for reference.

### Multi-Tenant Support

Lilith supports multiple concurrent users with isolated storage:

```bash
# Run as teacher (writes to base knowledge)
python lilith_cli.py --mode teacher

# Run as user (isolated personal storage)
python lilith_cli.py --mode user --user alice
```

See [docs/SQLITE_MIGRATION.md](docs/SQLITE_MIGRATION.md) for details on the SQLite backend and concurrency support.

### Testing

```bash
# Run all compatible tests
pytest tests/test_multi_tenant.py tests/test_concurrency.py -v

# Test concurrent access
python tools/demo_concurrent_access.py
```

## Related Projects

- [PMFlow](https://github.com/experimentech/PMFlow) - Standalone PMFlow encoder package
