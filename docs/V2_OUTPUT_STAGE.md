# V2 Output Stage Architecture

## Overview

The V2 output stage provides **pattern-based response generation with learning capabilities**. This document describes the output architecture ported from V1 with V2's improved storage and multi-tenant patterns.

## Components

### 1. ResponseComposer (`v2/lilith_v2/response_composer.py`)

The core output engine that generates responses through learned pattern composition.

```python
from v2.lilith_v2.response_composer import ResponseComposer, CompositionMode
```

#### Key Features

| Feature | Description |
|---------|-------------|
| **Pattern Retrieval** | Semantic + lexical similarity matching |
| **Pattern Blending** | Creates novel responses by combining fragments |
| **Success Learning** | Adjusts pattern weights based on feedback |
| **Adaptive Thresholds** | EMA-based confidence adjustment |
| **Multiple Modes** | best_match, weighted_blend, adaptive, graph_first |

#### Composition Modes

- **`best_match`**: Returns the single highest-scoring pattern
- **`weighted_blend`**: Attempts to blend compatible patterns into novel responses
- **`adaptive`**: Automatically selects strategy based on confidence level
- **`graph_first`**: Prioritizes graph traversal results over patterns

### 2. AdaptivePolicy

Learns optimal confidence thresholds through exponential moving averages:

```python
# After each response
composer.record_outcome(success=True)  # Updates thresholds

# Threshold adjusts:
# - High success rate → lower threshold (more adventurous)
# - Low success rate → higher threshold (more conservative)
```

### 3. Pattern Blending

When two patterns are close in score and have compatible intents, they're combined:

```
Pattern 1: "Machine learning enables computers to learn from data."
Pattern 2: "It's used for recommendation systems and image recognition."

Blended: "Machine learning enables computers to learn from data, 
          and it's used for recommendation systems and image recognition."
```

**Blending Rules:**
- Secondary must be ≥60% of primary score
- Both must have confidence > 0.4
- Intents must be compatible (no "greeting" + "technical")
- Neither can end with a question mark

## Integration with CognitiveStage

The ResponseComposer is automatically integrated when configured:

```python
from v2.lilith_v2.cognitive_stage import CognitiveStage
from v2.lilith_v2.relational_store import RelationalStore

# Create response store
response_store = RelationalStore("data/users/myuser/responses.db")

# Configure cognitive stage with output features
cognitive = CognitiveStage(
    node_id="brain",
    pmflow_store=pmflow,
    graph_store=graph,
    encoder=encoder,
    config={
        # Enable ResponseComposer
        "response_store": response_store,
        "composition_mode": "adaptive",
        "enable_blending": True,
        "enable_learning": True,
    }
)
```

### Processing Flow

```
User Input
    │
    ▼
┌──────────────────────────────────────────┐
│ CognitiveStage.learn()                   │
│                                          │
│  1. Linguistic Processing                │
│  2. Feedback Detection ←──────────┐      │
│  3. Semantic Extraction           │      │
│  4. Vector Grounding              │      │
│  5. Graph Reasoning               │      │
│                                   │      │
│  6. ResponseComposer.compose() ───┼──┐   │
│       ├─ Pattern Retrieval        │  │   │
│       ├─ Blending (if enabled)    │  │   │
│       └─ Return Response          │  │   │
│                                   │  │   │
│  7. Fallback: GenerativeSystem ───┼──┤   │
│                                   │  │   │
└───────────────────────────────────┼──┼───┘
                                    │  │
                           Learning │  │ Output
                             Loop   │  │
                                    │  ▼
                              Next User Input
```

## Learning Loop

### Implicit Feedback Detection

The system automatically detects feedback from follow-up messages:

| Signal | Examples | Effect |
|--------|----------|--------|
| **Strong Positive** | "thanks", "perfect", "exactly" | +1.0 feedback |
| **Weak Positive** | "okay", "I see", "interesting" | +0.3 feedback |
| **Weak Negative** | "what?", "confused", "I don't get it" | -0.3 feedback |
| **Strong Negative** | "wrong", "incorrect", "that's not right" | -1.0 feedback |

### Feedback Propagation

```
User: "What is Python?"
Lilith: "Python is a versatile programming language."
User: "Thanks!"  ← Strong Positive detected

System:
  1. FeedbackDetector returns score=+1.0
  2. AffectiveSystem mood improves
  3. ResponseComposer.record_outcome(success=True)
  4. Pattern success_score increases by 0.2
  5. Adaptive threshold adjusts toward more risk
```

### Manual Feedback (CLI)

```bash
testuser> What can you do?
Lilith: I can learn and converse with you.

testuser> feedback+
✓ Positive feedback recorded

testuser> stats
Cognitive Stats:
  response_composer:
    responses_composed: 1
    blends_succeeded: 0
    success_feedback_received: 1
    adaptive_threshold: 0.548
```

## Seeding Response Patterns

Use the seed script to populate base patterns:

```bash
python v2/seed.py
```

This seeds:
1. **Graph nodes** for Q&A and pattern triggers
2. **ResponseComposer patterns** for pattern-based generation

### Seed File Locations

```
data/seed/
├── qa_bootstrap.txt       # Q: / A: format
├── base_patterns.json     # trigger_context, response_text, intent
└── grammar_bootstrap.txt  # Syntax templates
```

### Pattern Format (base_patterns.json)

```json
[
  {
    "trigger_context": "what can you do",
    "response_text": "I can learn from our conversations and help answer questions.",
    "intent": "capability"
  },
  {
    "trigger_context": "hello",
    "response_text": "Hello! I'm Lilith, ready to learn and chat.",
    "intent": "greeting"
  }
]
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `stats` | Show cognitive stage statistics |
| `feedback+` | Record positive feedback for last response |
| `feedback-` | Record negative feedback for last response |
| `health` | System health check |
| `exit` | Quit the interactive shell |

## API Reference

### ResponseComposer

```python
class ResponseComposer:
    def compose(
        self,
        thought_context: Dict[str, Any],  # From CognitiveStage
        user_input: str = "",
        topk: int = 5,
    ) -> ComposedResponse:
        """Generate response from patterns or graph inference."""
    
    def record_outcome(self, success: bool) -> None:
        """Record feedback for learning loop."""
    
    def add_pattern(
        self,
        trigger: str,
        response: str,
        intent: str = "learned",
        initial_score: float = 0.5,
    ) -> str:
        """Add a new response pattern."""
    
    def upvote(self, pattern_id: str, strength: float = 0.25) -> None:
        """Manual positive feedback."""
    
    def downvote(self, pattern_id: str, strength: float = 0.3) -> None:
        """Manual negative feedback."""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get composition statistics."""
```

### ComposedResponse

```python
@dataclass
class ComposedResponse:
    text: str                           # Final response text
    fragment_ids: List[str]             # Patterns used
    composition_weights: List[float]    # Contribution of each
    coherence_score: float              # Overall confidence
    primary_pattern: Optional[ResponsePattern] = None
    confidence: float = 1.0
    is_fallback: bool = False
    is_blended: bool = False           # Was this blended?
    source: str = "pattern"            # pattern, graph, blended, fallback
```

## Comparison: V1 vs V2 Output

| Feature | V1 | V2 |
|---------|-----|-----|
| Pattern storage | ResponseFragmentStore | PatternStoreAdapter + RelationalStore |
| Composition modes | 5 modes | 4 modes (simplified) |
| Pattern blending | BioNN syntax-guided | Heuristic clause blending |
| Success learning | SQLite-backed | Any Store backend |
| Adaptive thresholds | Yes | Yes |
| Feedback detection | Extensive patterns | Simplified, focused |
| Multi-tenant | No | Yes (via Store namespacing) |
| Graph integration | Separate | Unified in CognitiveStage |

## Performance Notes

- Pattern retrieval: O(n) scan with optional embedding caching
- Blending: Only attempted when confidence gap is small
- Learning: Updates are immediate, no batch processing
- Storage: SQLite via RelationalStore (thread-safe)

## Future Enhancements

1. **Embedding caching**: Pre-compute pattern embeddings at seed time
2. **Vector index**: ANN for large pattern stores
3. **Contrastive learning**: Port V1's semantic correction capability
4. **Syntax-guided blending**: Integrate with LinguisticProcessor for grammatical composition
