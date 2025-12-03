# Data Directory Structure

This directory contains Lilith's training data and runtime storage.

## Directory Layout

```
data/
├── seed/                    # ✅ TRACKED - Baseline training data
│   ├── qa_bootstrap.txt     # Core Q&A pairs
│   ├── grammar_bootstrap.txt # Grammar patterns
│   ├── sample_training_data.json
│   ├── pragmatic_templates.json
│   └── base_patterns.json   # Seed response patterns
│
├── generated/               # ❌ GITIGNORED - Generated from training
│   ├── squad_training.txt   # Downloaded datasets
│   ├── eli5_training.txt
│   └── *.db                 # Trained databases
│
├── base/                    # Runtime base knowledge
│   ├── patterns.json        # ✅ Tracked (seed copy)
│   └── *.db                 # ❌ Gitignored (runtime)
│
├── users/                   # ❌ GITIGNORED - Per-user data
│   └── {user_id}/
│
├── servers/                 # Per-server data
│   └── {server_id}/
│       ├── settings.json    # ✅ Tracked (config)
│       └── *.db             # ❌ Gitignored (learned)
│
└── datasets/                # Downloaded training datasets
    ├── sources.txt          # ✅ Tracked (attribution)
    └── *.txt, *.json        # ❌ Gitignored (large files)
```

## What Gets Tracked in Git

✅ **Tracked (seed/baseline data):**
- `seed/` - All baseline training files
- `base/patterns.json` - Seed response patterns
- `servers/*/settings.json` - Server configurations
- `datasets/sources.txt` - Dataset attribution
- `*.md` files - Documentation

❌ **Not tracked (generated/runtime):**
- `*.db` files - SQLite databases
- `*.pt` files - PyTorch model weights
- `users/` - User-specific learned data
- `generated/` - All generated training data
- `datasets/*.txt`, `datasets/*.json` - Downloaded datasets

## Regenerating Data

If you clone fresh, you can regenerate everything:

```bash
# 1. Download open datasets
pip install datasets
python scripts/download_datasets.py all --output-dir data/generated/

# 2. Bootstrap from seed data
python scripts/bootstrap_qa.py

# 3. Train from downloaded datasets
python scripts/bootstrap_qa.py --qa-file data/generated/squad_training.txt
```

## Adding New Seed Data

To add permanent baseline data:

1. Add to `data/seed/` directory
2. Update `data/seed/qa_bootstrap.txt` or create new file
3. Commit to git

## Data Flow

```
seed/ (tracked)
    ↓
bootstrap_qa.py / train_from_document.py
    ↓
generated/ + base/*.db (gitignored)
    ↓
Runtime learning → users/*.db, servers/*.db
```
