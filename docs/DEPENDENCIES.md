# Lilith AI - Dependencies Documentation

## Core Dependencies

### Required Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **numpy** | ≥1.24.0 | Numerical computations, array operations |
| **torch** | ≥2.0.0 | Neural network backend for PMFlow |
| **sympy** | ≥1.12 | Symbolic mathematics computation |
| **requests** | ≥2.31.0 | HTTP requests for external knowledge APIs |
| **rapidfuzz** | ≥3.0.0 | Fast fuzzy string matching for typo tolerance |

### Optional but Recommended

| Library | Version | Purpose |
|---------|---------|---------|
| **nltk** | ≥3.8.1 | Natural Language Toolkit for WordNet (offline synonyms/antonyms) |
| **discord.py** | ≥2.3.0 | Discord bot integration |
| **python-dotenv** | ≥1.0.0 | Environment variable management |

### Built-in (No Installation Required)

- **sqlite3** - Database backend (included with Python)
- **re** - Regular expressions
- **json** - JSON parsing
- **pathlib** - File path handling

---

## External Knowledge Sources

Lilith integrates multiple external knowledge sources for comprehensive information retrieval:

### 1. Wikipedia 🌐
- **Type**: Online API
- **Purpose**: General knowledge, concepts, people, places, events
- **API**: `https://en.wikipedia.org/api/rest_v1/page/summary/{title}`
- **Authentication**: None required
- **Rate Limits**: Generous (no API key needed)
- **Confidence**: 0.75
- **Best for**: 
  - "What is machine learning?"
  - "Who was Ada Lovelace?"
  - "Tell me about Python programming"

### 2. Wiktionary 📘
- **Type**: Online API
- **Purpose**: Word definitions, etymology, parts of speech
- **API**: `https://en.wiktionary.org/api/rest_v1/page/definition/{word}`
- **Authentication**: None required
- **Rate Limits**: Generous
- **Confidence**: 0.85
- **Best for**:
  - "What does ephemeral mean?"
  - "Define recalcitrant"
  - "Meaning of serendipity"

### 3. WordNet 📖 (via NLTK)
- **Type**: Offline corpus
- **Purpose**: Synonyms, antonyms, word relationships, semantic networks
- **Library**: NLTK (Natural Language Toolkit)
- **Data Download**: Automatic on first use (or via `nltk.download('wordnet')`)
- **Authentication**: None required
- **Rate Limits**: None (offline)
- **Confidence**: 0.80
- **Best for**:
  - "What's a synonym for happy?"
  - "Antonym of good"
  - "Another word for beautiful"
  - Fast lookups without network calls

### 4. Free Dictionary API 📕
- **Type**: Online API
- **Purpose**: Definitions with phonetics, usage examples
- **API**: `https://api.dictionaryapi.dev/api/v2/entries/en/{word}`
- **Authentication**: None required
- **Rate Limits**: Fair use
- **Confidence**: 0.82
- **Best for**:
  - Definitions with pronunciation guides
  - Usage examples
  - Fallback when Wiktionary fails

---

## Smart Source Selection

Lilith automatically chooses the best knowledge source based on query type:

```python
# Query Type Detection
1. Synonym/Antonym queries → WordNet (fast, offline)
   "synonym for X", "antonym of Y", "another word for Z"

2. Word definition queries → Wiktionary → Free Dictionary → WordNet
   "what does X mean?", "define X", "meaning of X"

3. General knowledge → Wikipedia
   "what is X?", "who was X?", "tell me about X"

4. Single-word queries → Try all sources
   "ephemeral" → WordNet → Wiktionary → Free Dictionary
```

### Source Priority Logic

```
User Query: "What's a synonym for happy?"
    ↓
1. Detect: Contains "synonym" → Route to WordNet
    ↓
2. WordNet lookup → SUCCESS ✅
    ↓
3. Return: "felicitous, glad, cheerful..."

User Query: "What does ephemeral mean?"
    ↓
1. Detect: Contains "mean" → Route to definition sources
    ↓
2. Try Wiktionary → SUCCESS ✅
    ↓
3. Return: "Lasting for a short period of time."
    ↓
4. (Never tries Wikipedia - more specific source worked)
```

---

## Installation Instructions

### Option 1: Automated Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/experimentech/Lilith-AI.git
cd Lilith-AI

# Run installation script
./install.sh
```

The script will:
- Check Python version (≥3.9 required)
- Create virtual environment
- Install all dependencies
- Download WordNet data
- Optionally install Discord bot dependencies

### Option 2: Manual Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download WordNet data (for NLTK)
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# Optional: Install Discord dependencies
pip install discord.py python-dotenv
```

### Option 3: Minimal Installation (Core Only)

If you only need core functionality without external knowledge:

```bash
pip install numpy torch sympy rapidfuzz
```

This gives you:
- Local pattern learning
- Math computation
- Fuzzy matching
- No external API calls

---

## Dependency Details

### NLTK (Natural Language Toolkit)

**What it provides:**
- WordNet corpus (lexical database of English)
- Synonyms, antonyms, hypernyms, hyponyms
- Word definitions and examples
- Semantic relationships

**Data Downloads:**
```python
import nltk

# Required data packages
nltk.download('wordnet')      # WordNet corpus
nltk.download('omw-1.4')      # Open Multilingual WordNet

# Optional: Additional corpora
nltk.download('averaged_perceptron_tagger')  # POS tagging
nltk.download('brown')                       # Brown corpus
```

**Data Location:**
- Default: `~/nltk_data/`
- Can be customized with `nltk.data.path`

**Disk Space:**
- WordNet: ~10 MB
- OMW-1.4: ~5 MB

### Discord.py

**What it provides:**
- Discord API client
- Event handling for messages, reactions
- Slash commands support
- User/server management

**Setup:**
```bash
pip install discord.py python-dotenv

# Create .env file
echo "DISCORD_TOKEN=your_bot_token_here" > .env

# Run bot
python discord_bot.py
```

**Requirements:**
- Discord bot token (from Discord Developer Portal)
- Bot intents enabled: Message Content, Reactions

---

## Troubleshooting

### NLTK WordNet Not Found

```python
# Error: LookupError: Resource wordnet not found
# Solution:
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Import Error: nltk

```bash
# Error: ModuleNotFoundError: No module named 'nltk'
# Solution:
pip install nltk
```

### API Rate Limiting

If you encounter rate limits with external APIs:

1. **Wikipedia/Wiktionary**: Usually generous, but respect `User-Agent`
2. **Free Dictionary**: Fair use policy, no strict limits
3. **WordNet**: Offline, no limits! ✅

**Recommendation**: WordNet handles most vocabulary queries without network calls.

### Discord Bot Not Responding

```bash
# Check bot token
cat .env  # Should contain DISCORD_TOKEN=...

# Check intents are enabled in Discord Developer Portal:
# - Message Content Intent ✅
# - Server Members Intent (optional)

# Check bot is online
python discord_bot.py
# Should see: "🤖 Lilith Discord Bot is ready!"
```

---

## Development Dependencies

For contributors and developers:

```bash
# Install dev dependencies
pip install pytest pytest-asyncio black mypy

# Run tests
pytest tests/ -v

# Format code
black lilith/ tests/

# Type checking
mypy lilith/
```

---

## Version Compatibility

| Python | Status | Notes |
|--------|--------|-------|
| 3.9 | ✅ Supported | Minimum version |
| 3.10 | ✅ Supported | Recommended |
| 3.11 | ✅ Supported | Recommended |
| 3.12 | ✅ Supported | Latest |
| 3.13 | ✅ Supported | Bleeding edge |
| <3.9 | ❌ Not supported | Missing features |

---

## Network Requirements

### Online Features (Optional)

- **Wikipedia**: Requires internet for lookups
- **Wiktionary**: Requires internet for lookups
- **Free Dictionary**: Requires internet for lookups
- **Discord Bot**: Requires internet for Discord connection

### Offline Features (Always Available)

- **WordNet**: Fully offline after initial data download
- **Local Pattern Learning**: No internet required
- **Math Computation**: No internet required
- **Fuzzy Matching**: No internet required
- **SQLite Storage**: No internet required

**Lilith works offline** if you:
1. Pre-download WordNet data
2. Don't use Discord bot
3. Accept reduced knowledge coverage (no Wikipedia/Wiktionary)

---

## License Notes

### Third-Party Licenses

- **NLTK**: Apache 2.0
- **WordNet**: WordNet License (free for research and commercial use)
- **Wikipedia API**: CC BY-SA (attribution required for content)
- **Wiktionary API**: CC BY-SA (attribution required)
- **Free Dictionary API**: Public domain
- **discord.py**: MIT License

All dependencies are free and compatible with open-source projects.

---

## Performance Considerations

### Memory Usage

| Component | RAM Usage |
|-----------|-----------|
| Core Lilith | ~200 MB |
| PyTorch | ~300 MB |
| NLTK WordNet | ~50 MB |
| Pattern Database | ~10-100 MB (grows with learning) |
| **Total** | **~600 MB minimum** |

### Disk Space

| Component | Disk Space |
|-----------|------------|
| Virtual Environment | ~500 MB |
| NLTK Data | ~15 MB |
| Pattern Database | ~10-100 MB (grows) |
| **Total** | **~550+ MB** |

### Network Bandwidth

- WordNet: 0 KB/s (offline)
- Wikipedia: ~50-200 KB per query
- Wiktionary: ~20-50 KB per query
- Free Dictionary: ~10-30 KB per query

**Tip**: Use WordNet for vocabulary queries to minimize network usage!

---

## Updates and Maintenance

### Keeping Dependencies Updated

```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade nltk

# Check for outdated packages
pip list --outdated
```

### WordNet Updates

WordNet is relatively stable, but you can check for updates:

```python
import nltk
nltk.download('wordnet')  # Re-downloads if newer version available
```

---

## Support

For dependency issues:
- Check this document first
- Review [Troubleshooting](#troubleshooting) section
- Open an issue on GitHub with error logs
- Include Python version and OS information
