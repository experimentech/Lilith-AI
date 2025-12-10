# Lilith Setup Guide

Complete guide to installing, configuring, and seeding Lilith with knowledge.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/experimentech/lilith.git
cd lilith

# Run the setup script (recommended)
./install.sh

# Or manual setup:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 2. Basic Testing

```bash
# Activate environment
source .venv/bin/activate

# Run CLI
python lilith_cli.py

# Run tests
pytest tests/
```

## Seeding Lilith with Knowledge

Lilith starts with minimal knowledge. Use these scripts to populate it with useful information:

### Option 1: Bootstrap with Q&A Pairs

The fastest way to give Lilith domain knowledge:

```bash
# Create a Q&A file (data/qa_bootstrap.txt)
cat > data/qa_bootstrap.txt <<'QAFILE'
# Basic Facts
Q: What is Python?
A: Python is a high-level programming language known for readability and versatility.

Q: What is machine learning?
A: Machine learning is a subset of AI where systems learn from data without explicit programming.

# Your Domain
Q: What is our refund policy?
A: We offer full refunds within 30 days with the original receipt.

Q: What are our business hours?
A: Monday-Friday 9am-5pm EST, closed weekends.
QAFILE

# Bootstrap the knowledge
python scripts/bootstrap_qa.py --qa-file data/qa_bootstrap.txt

# Or bootstrap to a specific user/server
python scripts/bootstrap_qa.py --qa-file data/qa_bootstrap.txt --user-id alice
```

**Format**: Simple Q:/A: pairs, one per line. Comments (#) and blank lines ignored.

### Option 2: Use SQuAD Dataset

Convert the Stanford Question Answering Dataset:

```bash
# Download SQuAD (if you have it)
# https://rajpurkar.github.io/SQuAD-explorer/

# Convert to Lilith format
python scripts/convert_squad.py \
    --input squad/train-v2.0.json \
    --output data/generated/squad_training.txt \
    --limit 1000 \
    --shuffle

# Bootstrap from converted file
python scripts/bootstrap_qa.py --qa-file data/generated/squad_training.txt
```

**What it does**: Extracts question-answer pairs from SQuAD JSON and converts to Q:/A: format.

### Option 3: Train from Documents

Ingest unstructured documents and extract knowledge:

```bash
# Single document
python scripts/train_from_document.py --file docs/manual.txt

# Directory of documents
python scripts/train_from_document.py --dir knowledge_base/

# From a URL
python scripts/train_from_document.py --url "https://example.com/faq"

# For Discord server
python scripts/train_from_document.py \
    --file company_wiki.txt \
    --server-id 123456789012345678
```

**Supported formats**: .txt, .md, .html, .pdf (requires pypdf), .docx (requires python-docx)

**How it works**: 
1. Loads and parses document
2. Chunks into meaningful segments
3. Extracts key concepts and facts
4. Creates Q&A pairs automatically
5. Stores in pattern database

### Option 4: Seed from Conversation Logs

Have existing chat logs? Convert them to training data:

```bash
# Create log file (data/conversation_seed.txt)
cat > data/conversation_seed.txt <<'LOGFILE'
user: hello
assistant: Hi! How can I help you today?

user: what is rust?
assistant: Rust is a systems programming language focused on safety and performance.

user: thanks
assistant: You're welcome! Let me know if you need anything else.
LOGFILE

# Seed from logs
python tools/import_conversations.py \
    --file data/conversation_seed.txt \
    --format simple
```

**Formats supported**:
- `simple`: user:/assistant: pairs
- `json`: JSON array of message objects
- `csv`: Columns for speaker, message, timestamp

### Option 5: Interactive Teaching Mode

Teach Lilith directly through conversation:

```bash
# Start in teacher mode
python lilith_cli.py --mode teacher

# Then use /teach command
> /teach
Question: What is TypeScript?
Answer: TypeScript is a typed superset of JavaScript that compiles to plain JavaScript.

# Or just correct wrong answers
> What is Go?
Lilith: I'm not sure about that yet...
> Go is a statically typed programming language developed at Google.
Lilith: Thanks for teaching me!
```

**Best for**: Domain-specific knowledge, corrections, custom terminology

## Advanced Setup

### Multi-User Configuration

For multiple users or Discord server deployment:

```bash
# Create base knowledge (teacher mode)
python lilith_cli.py --mode teacher
# Teach core knowledge here

# Run for specific user
python lilith_cli.py --mode user --user alice
python lilith_cli.py --mode user --user bob

# Discord bot (multi-user by default)
python discord_bot.py
```

**Storage structure**:
```
data/
├── base/                          # Shared base knowledge (teacher-managed)
│   └── response_patterns.db
├── users/                         # User-specific knowledge
│   ├── alice/
│   │   └── response_patterns.db
│   └── bob/
│       └── response_patterns.db
└── servers/                       # Discord server-specific
    └── 123456789012345678/
        └── knowledge.db
```

### Contrastive Learning (Advanced)

Train semantic similarity for better retrieval:

```bash
# Train on semantic relationships
python tools/train_contrastive.py \
    --pairs-file data/semantic_pairs.txt \
    --epochs 100 \
    --save-model data/models/semantic.pt

# Semantic pairs file format:
cat > data/semantic_pairs.txt <<'PAIRS'
# Format: positive_1 ||| positive_2 ||| negative
machine learning ||| ML ||| cooking
artificial intelligence ||| AI ||| biology
neural network ||| deep learning ||| gardening
PAIRS
```

**When to use**: Improves retrieval quality for large knowledge bases (1000+ patterns)

### Custom Knowledge Sources

Add your own knowledge sources:

```python
# In lilith/knowledge_sources.py
class CustomAPI(KnowledgeSource):
    def lookup(self, query: str) -> Optional[str]:
        # Your API integration here
        result = your_api.search(query)
        return result.summary if result else None

# Register in knowledge_augmenter.py
augmenter.add_source(CustomAPI())
```

## Verification

### Check Pattern Count

```bash
# Python
python -c "
from lilith.multi_tenant_store import MultiTenantFragmentStore
from lilith.user_auth import UserIdentity, AuthMode
from lilith.embedding import PMFlowEmbeddingEncoder

encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
user = UserIdentity('test', AuthMode.SIMPLE, 'Test User')
store = MultiTenantFragmentStore(encoder, user)
stats = store.user_store.get_stats()
print(f\"Patterns: {stats['total_patterns']}\")
"
```

### Test Retrieval

```bash
# Test a query
python -c "
from lilith.session import LilithSession, SessionConfig

config = SessionConfig(user_id='test', mode='user')
session = LilithSession(config)
response = session.process_message('What is Python?')
print(response)
"
```

### View Database

```bash
# SQLite CLI
sqlite3 data/base/response_patterns.db

# Inside SQLite:
.tables
SELECT COUNT(*) FROM patterns;
SELECT trigger_context, response_text FROM patterns LIMIT 5;
.quit
```

## Common Workflows

### Initial Setup for Production

1. Install and verify:
```bash
./install.sh
pytest tests/
```

2. Seed base knowledge (teacher mode):
```bash
python scripts/bootstrap_qa.py --qa-file data/domain_knowledge.txt
```

3. Test retrieval:
```bash
python lilith_cli.py --mode teacher
> What is [your domain topic]?
```

4. Deploy (CLI or Discord):
```bash
python lilith_cli.py --mode user --user production
# or
python discord_bot.py
```

### Adding Domain Knowledge

1. Create Q&A file for your domain
2. Bootstrap: `python scripts/bootstrap_qa.py --qa-file domain.txt`
3. Test and refine
4. Optional: Train contrastive learning for better retrieval

### Updating Knowledge

```bash
# Add new patterns
python scripts/bootstrap_qa.py --qa-file new_facts.txt

# Or update interactively
python lilith_cli.py --mode teacher
> /teach
Question: [new question]
Answer: [new answer]
```

## Troubleshooting

### "No patterns found"

**Solution**: Seed knowledge first using bootstrap_qa.py or train_from_document.py

### "Low confidence responses"

**Solution**: Add more training data or use contrastive learning

### "Pattern database locked"

**Solution**: Close other Lilith instances, wait 30 seconds, retry

### Import errors

**Solution**: Ensure virtual environment is activated:
```bash
source .venv/bin/activate
```

## Next Steps

- [User Guide](USER_GUIDE.md) - Learn about feedback and data management
- [Discord Bot Setup](DISCORD_BOT_SETUP.md) - Deploy to Discord
- [Training Guide](TRAINING_GUIDE.md) - Advanced training techniques
- [Architecture](ARCHITECTURE_VERIFICATION.md) - Understand the design

## Quick Reference

| Task | Command |
|------|---------|
| Install | `./install.sh` |
| Run CLI | `python lilith_cli.py` |
| Run tests | `pytest` |
| Seed Q&A | `python scripts/bootstrap_qa.py --qa-file <file>` |
| Train from docs | `python scripts/train_from_document.py --file <file>` |
| Convert SQuAD | `python scripts/convert_squad.py --input <file>` |
| Teacher mode | `python lilith_cli.py --mode teacher` |
| Discord bot | `python discord_bot.py` |
| Check stats | `python tools/manage_users.py stats` |
