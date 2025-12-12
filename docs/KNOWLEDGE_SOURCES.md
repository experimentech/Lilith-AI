# Knowledge Sources Quick Reference

## When Each Source is Used

```
┌─────────────────────────────────────────────────────────────────┐
│                    Query Type Detection                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌────────────────────────────────────────┐
        │  Contains: synonym, antonym,            │
        │  "another word", "opposite"             │
        └────────────────────────────────────────┘
                              ↓
                         📖 WordNet (offline)
                         Confidence: 0.80
                              ↓
        ┌────────────────────────────────────────┐
        │  Definition queries (single-word):      │
        │  - "What is X?" / "What are X?"         │
        │  - "What does X mean?"                  │
        │  - "Define X" / "Meaning of X"          │
        └────────────────────────────────────────┘
                              ↓
                    📘 Wiktionary (online)
                    Confidence: 0.85
                              ↓ (fallback)
                    📕 Free Dictionary
                    Confidence: 0.82
                              ↓ (fallback)
                    📖 WordNet
                              ↓
        ┌────────────────────────────────────────┐
        │  General knowledge queries             │
        │  Multi-word topics                     │
        │  - "Tell me about X"                   │
        │  - "Who is/was X?"                     │
        └────────────────────────────────────────┘
                              ↓
                    🌐 Wikipedia (online)
                    Confidence: 0.75
```

## Topic Extraction

The system uses **BNN-based TopicExtractor** to identify topics from queries:

1. **Learned Topics**: If the topic was taught before, BNN similarity matching finds it
2. **Unknown Topics**: Falls back to regex pattern extraction

```python
# TopicExtractor in action
"Tell me about dogs" → "dogs" (if learned) 
"What does ephemeral mean?" → "ephemeral"
"Do you know about elephants?" → "elephants"
```

## Example Queries

### WordNet 📖 (Offline, Fast)
```
✅ "What's a synonym for happy?"
   → "felicitous, glad, cheerful"

✅ "Antonym of good"
   → "evil, bad, ill"

✅ "Another word for beautiful"
   → "lovely, gorgeous, stunning"
```

### Wiktionary 📘 (Definitions)
```
✅ "What does ephemeral mean?"
   → "Lasting for a short period of time."

✅ "Define recalcitrant"
   → "Marked by stubborn unwillingness to obey."

✅ "Meaning of serendipity"
   → "The occurrence of fortunate events by chance."
```

### Free Dictionary 📕 (Definitions + Examples)
```
✅ "What does ameliorate mean?"
   → "To make better, improve. Example: 'efforts to ameliorate social problems'"

✅ "Define verbose"
   → "Using more words than needed. Example: 'a verbose explanation'"
```

### Wikipedia 🌐 (Concepts)
```
✅ "What is machine learning?"
   → "ML is a field of study in AI concerned with..."

✅ "Who was Ada Lovelace?"
   → "English mathematician, first computer programmer..."

✅ "Tell me about Python"
   → "Python is a high-level programming language..."
```

## Performance Comparison

| Source | Speed | Network | Coverage |
|--------|-------|---------|----------|
| WordNet | ⚡ Instant | ❌ Offline | 155K words |
| Wiktionary | 🌐 1-2s | ✅ Online | 6M+ entries |
| Free Dict | 🌐 1-2s | ✅ Online | 150K+ words |
| Wikipedia | 🌐 2-3s | ✅ Online | 6M+ articles |

## Best Practices

### Use WordNet for:
- ✅ Vocabulary questions (fast, offline)
- ✅ Synonyms and antonyms
- ✅ Word relationships
- ✅ When network is unreliable

### Use Wiktionary for:
- ✅ Precise definitions
- ✅ Etymology and word origins
- ✅ Multiple meanings (homonyms)
- ✅ Technical terms

### Use Free Dictionary for:
- ✅ Pronunciation guides
- ✅ Usage examples
- ✅ When Wiktionary fails
- ✅ Simpler definitions

### Use Wikipedia for:
- ✅ Concepts and ideas
- ✅ People, places, events
- ✅ Historical information
- ✅ General knowledge

## Fallback Chain

```
Query: "What's a synonym for happy?"
  → Try WordNet → SUCCESS ✅
  → (Never tries other sources)

Query: "What does X mean?"
  → Try Wiktionary → SUCCESS ✅
  → (Never tries Free Dictionary or Wikipedia)

Query: "What does X mean?" (if Wiktionary fails)
  → Try Wiktionary → FAIL ❌
  → Try Free Dictionary → SUCCESS ✅
  → (Never tries Wikipedia)

Query: "What is quantum physics?"
  → Not a definition query
  → Skip word sources
  → Try Wikipedia → SUCCESS ✅
```

## Testing Knowledge Sources

```bash
# Test all sources
python test_knowledge_sources.py

# Expected output:
# WordNet: 3 successes
# Wiktionary: 3 successes
# Free Dictionary: 0-2 successes (fallback)
# Wikipedia: 4 successes
# Total: 100% success rate
```

## Troubleshooting

### WordNet Not Working
```bash
# Download WordNet data
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### API Rate Limiting
- WordNet: Never (offline) ✅
- Wiktionary: Rare (generous limits)
- Free Dictionary: Rare (fair use)
- Wikipedia: Rare (generous limits)

### Network Timeout
- All online sources have 5-second timeout
- Failures are silent (returns None)
- System tries next source automatically

## Statistics Tracking

Get knowledge source usage stats:

```python
from lilith.knowledge_augmenter import KnowledgeAugmenter

augmenter = KnowledgeAugmenter()

# After some queries...
stats = augmenter.get_stats()

print(stats)
# {
#   'lookups': 100,
#   'successes': 95,
#   'success_rate': '95.0%',
#   'enabled': True,
#   'sources': {
#     'wordnet': 30,
#     'wiktionary': 25,
#     'free_dictionary': 10,
#     'wikipedia': 30
#   },
#   'wordnet_available': True
# }
```
