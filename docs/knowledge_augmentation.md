# Knowledge Augmentation System

**Status**: ✅ Implemented and Tested (Updated December 2024)

## Overview

The knowledge augmentation system enables Lilith to automatically acquire knowledge from external sources when it encounters queries it cannot answer confidently. This creates a self-improving knowledge base that grows based on actual user needs.

### Knowledge Sources

| Source | Type | Confidence | Best For |
|--------|------|------------|----------|
| 📖 WordNet | Offline | 0.80 | Synonyms, antonyms, word relationships |
| 📘 Wiktionary | Online | 0.85 | Word definitions, etymology |
| 📕 Free Dictionary | Online | 0.82 | Definitions with examples |
| 🌐 Wikipedia | Online | 0.75 | Concepts, people, general knowledge |

## How It Works

### 1. Automatic Triggering (Two Paths)

**Path A: Low Confidence Retrieval**
```
User Query → Pattern Retrieval → Low Confidence? → External Lookup → Learn & Respond
```

**Path B: Proactive Augmentation (NEW)**
```
User Query → Deliberation → No Relevant Concepts? → External Lookup → Learn & Respond
```

When pattern retrieval returns:
- **No patterns found**, OR
- **Best pattern confidence < 0.6**, OR
- **Deliberation finds no semantically relevant concepts**

The system automatically triggers external knowledge lookup.

### 2. Smart Source Routing

The system routes queries to the most appropriate source:

```
"What is a synonym for happy?" → 📖 WordNet
"What does ephemeral mean?"    → 📘 Wiktionary  
"Define serendipity"           → 📘 Wiktionary
"Tell me about elephants"      → 🌐 Wikipedia
"What is machine learning?"    → 🌐 Wikipedia (multi-word topic)
```

### 3. BNN-Based Topic Extraction

Uses **TopicExtractor** with BNN semantic similarity:
- If topic was previously learned → BNN similarity finds it
- If unknown topic → Falls back to regex extraction

```python
# TopicExtractor learns from declarations
session.process_message("Dogs are loyal animals")
# Now "dogs" is a learned topic

# Later queries use BNN similarity
"Tell me about dogs" → topic="dogs" (BNN match, score=0.98)
"What are cats?"     → topic="cats" (fallback extraction)
```

### 4. Automatic Learning

Responses from external sources are automatically learned:
- **Trigger**: Extracted topic (e.g., "quantum entanglement")
- **Response**: Wikipedia summary
- **Intent**: `taught` (from external source)
- **Confidence**: 0.75 (Wikipedia reliability)

Future queries about the same topic retrieve the learned pattern instead of querying Wikipedia again.

## Implementation

### Core Components

**`pipeline/knowledge_augmenter.py`**:
- `WikipediaLookup`: Wikipedia API interface
- `KnowledgeAugmenter`: Main augmentation coordinator
- Query cleaning, summary extraction, response formatting

**`pipeline/response_composer.py`**:
- Integrated into fallback methods
- Tries external lookup before "I don't know" response
- Seamlessly works with teaching mechanism

### Integration Points

Modified methods in `response_composer.py`:
- `__init__()`: Initialize knowledge augmenter
- `_fallback_response()`: Try Wikipedia before fallback
- `_fallback_response_low_confidence()`: Try Wikipedia when best pattern weak

## Test Results

```
✅ Machine learning: Found and extracted
✅ Ada Lovelace: Found and extracted
✅ Python programming: Found and extracted
✅ Quantum computing: Found and extracted
✅ Neural network: Found and extracted
✅ Recursion: Found and extracted
✅ Quantum entanglement: Found and extracted
```

**Success rate**: ~70% (depends on Wikipedia article availability)

## Example Usage

### Scenario 1: Unknown Topic
```
User: What is quantum entanglement?
System: [No patterns found, confidence < 0.6]
System: [Queries Wikipedia...]
System: Quantum entanglement is the phenomenon wherein the quantum state 
        of each particle in a group cannot be described independently...
System: [Learns pattern: "quantum entanglement" → summary]
```

### Scenario 2: Future Query
```
User: Tell me about quantum entanglement
System: [Retrieves learned pattern from first query]
System: Quantum entanglement is the phenomenon wherein...
System: [No Wikipedia lookup needed - using learned knowledge]
```

## Benefits

### 1. **Bootstrap Knowledge Base**
- Starts with Cornell Movie Dialogs (casual conversation)
- Learns technical/factual knowledge through actual use
- No manual dataset curation needed

### 2. **User-Driven Curriculum**
- System learns exactly what users ask about
- Natural knowledge priorities emerge from usage
- Efficient learning (no wasted effort on unused topics)

### 3. **Self-Improving**
- Each Wikipedia lookup becomes permanent knowledge
- Knowledge base grows with each conversation
- Quality improves through pattern consolidation

### 4. **No LLM Required**
- Pure neuro-symbolic architecture maintained
- External knowledge → pattern learning → retrieval
- Full control over knowledge sources and quality

## Configuration

Enable/disable in `response_composer.py`:
```python
composer = ResponseComposer(
    fragment_store=fragments,
    conversation_state=state,
    enable_knowledge_augmentation=True  # Toggle here
)
```

Statistics tracking:
```python
stats = composer.knowledge_augmenter.get_stats()
# Returns: lookups, successes, success_rate, enabled
```

## Future Enhancements

### Multi-Source Knowledge
- Wolfram Alpha (math/science)
- Stack Overflow (programming)
- News APIs (current events)
- ArXiv (research papers)

### Quality Improvements
- Cross-reference multiple sources
- Source confidence weighting
- Fact verification
- User correction handling

### Advanced Features
- Temporal knowledge (things that change over time)
- Domain-specific sources
- Citation tracking
- Knowledge provenance

## Architecture Alignment

This implements the **multi-modal architecture vision**:

**Input Modalities**:
- ✅ Text (user conversation)
- ✅ Web (Wikipedia API)
- 🔲 Vision (future)
- 🔲 Audio (future)
- 🔲 Sensors (future)

**Semantic Convergence**:
All modalities → Extract patterns → Shared semantic space → Learn & retrieve

**No Special Cases**:
Wikipedia knowledge learned through same mechanism as user teaching. The pattern store doesn't care about knowledge source - just stores and retrieves patterns.

## Testing

Run comprehensive tests:
```bash
cd experiments/retrieval_sanity
python3 test_knowledge_augmentation.py
```

Tests cover:
- Query cleaning (question → article title)
- Wikipedia lookup (various topics)
- Knowledge augmentation system
- Integration scenario (full pipeline)

## Dependencies

```bash
pip install requests
```

## Files Modified/Created

### New Files
- `pipeline/knowledge_augmenter.py` (250 lines)
- `test_knowledge_augmentation.py` (240 lines)

### Modified Files
- `pipeline/response_composer.py`:
  - Added knowledge augmenter initialization
  - Modified `_fallback_response()` to try Wikipedia
  - Modified `_fallback_response_low_confidence()` to try Wikipedia

## Conclusion

The knowledge augmentation system successfully demonstrates:

✅ **External knowledge integration** without LLMs  
✅ **Automatic pattern learning** from Wikipedia  
✅ **Self-improving knowledge base** through use  
✅ **Multi-modal architecture** in practice  
✅ **User-driven learning** (learns what users ask about)

This is a **key milestone** in the neuro-symbolic approach: The system can bootstrap domain knowledge naturally through conversation, not through massive pre-training.
