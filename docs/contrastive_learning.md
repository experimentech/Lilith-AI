# Contrastive Learning for Semantic Embeddings

This module enables the BNN to learn semantic relationships through contrastive training,
making the system understand that concepts like "cat" and "dog" are related (both animals)
while "cat" and "computer" are not.

## Philosophy

Lilith's architecture separates **knowledge storage** (databases) from **learned behavior** (BNN):
- **Databases** store explicit facts, relationships, patterns
- **BNN** learns to navigate and connect this knowledge

Contrastive learning trains the BNN's PMFlow field to understand semantic similarity,
enabling better:
- Pattern retrieval (find relevant responses)
- Reasoning (detect concept connections)
- Intent classification (group similar queries)

## Quick Start

```bash
# Train with core semantic pairs
python tools/train_contrastive.py

# Train with additional sources
python tools/train_contrastive.py --from-patterns --from-concepts

# Just evaluate current model
python tools/train_contrastive.py --eval-only
```

## Results

Before contrastive training:
```
"machine learning" vs "artificial intelligence": 0.099
"cat" vs "dog": -0.072  (should be positive!)
"cat" vs "computer": 0.132  (should be negative!)
```

After contrastive training:
```
"machine learning" vs "artificial intelligence": 0.496  ✓
"cat" vs "dog": 0.254  ✓
"cat" vs "computer": -0.056  ✓
```

## Integration

### With ResponseComposer

```python
from lilith.response_composer import ResponseComposer

composer = ResponseComposer(
    fragment_store=store,
    conversation_state=state,
    semantic_encoder=encoder
)

# Load pre-trained weights
composer.load_contrastive_weights("data/contrastive_learner")

# Online learning from user feedback
composer.add_semantic_correction("laptop", "computer", should_be_similar=True)
composer.add_semantic_correction("laptop", "furniture", should_be_similar=False)
```

### Standalone Usage

```python
from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.contrastive_learner import ContrastiveLearner

# Create encoder and learner
encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
learner = ContrastiveLearner(encoder, learning_rate=0.01)

# Add training pairs
learner.add_symmetric_pair("cat", "dog", "positive")
learner.add_pair("cat", "computer", "negative")
learner.generate_core_semantic_pairs()

# Train
learner.train(epochs=15)

# Query similarity
sim = learner.similarity("neural network", "deep learning")  # ~0.6

# Save/load
learner.save("data/my_model")
learner.load("data/my_model")
```

## Pair Sources

### 1. Core Semantic Pairs (Built-in)
Foundational relationships every language model should know:
- Category hierarchies: "cat" is_a "animal"
- Synonyms: "happy" ≈ "joyful"
- Opposites: "hot" ≠ "cold"

```python
learner.generate_core_semantic_pairs()
```

### 2. ConceptDatabase Relations
Load from your concept database:
```python
learner.load_from_concept_database("data/concepts.db")
```

Supported relation types:
- `is_a`, `instance_of`, `subclass_of` → positive
- `related_to`, `similar_to`, `synonym` → positive
- `opposite_of`, `antonym` → hard_negative

### 3. PatternDatabase Intents
Patterns with same intent should have similar embeddings:
```python
learner.load_from_pattern_database("data/patterns.db")
```

### 4. User Corrections (Online)
Learn from user feedback without retraining:
```python
learner.add_user_correction("laptop", "computer", should_be_similar=True)
```

### 5. Incremental Updates
Quick updates without full retraining:
```python
learner.incremental_update([
    ("quantum", "physics", "positive"),
    ("quantum", "cooking", "negative"),
], steps=5)
```

## Applying to Other Cognitive Layers

The contrastive learning approach can improve any component that uses semantic similarity:

### ReasoningStage
Better inference detection when concepts converge/diverge in latent space:
```python
# Before: "cat" and "dog" have low similarity → no inference
# After: "cat" and "dog" converge → infer "both are animals"
```

### Pattern Retrieval
More relevant pattern matching:
```python
# Query: "What is machine learning?"
# Before: Might return unrelated patterns
# After: Returns AI/ML related patterns
```

### Intent Classification
Better clustering of similar queries:
```python
# "What can you do?" and "What are your capabilities?" 
# cluster together after training
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ContrastiveLearner                        │
├─────────────────────────────────────────────────────────────┤
│  Pairs:                                                      │
│  ├── ("cat", "dog", positive)                               │
│  ├── ("cat", "computer", negative)                          │
│  └── ("happy", "sad", hard_negative)                        │
├─────────────────────────────────────────────────────────────┤
│  Training:                                                   │
│  ├── Encode pairs → PMFlow latent space                     │
│  ├── Pull positives together                                │
│  ├── Push negatives apart (with margin)                     │
│  └── Update PMFlow field centers & mus                      │
├─────────────────────────────────────────────────────────────┤
│  Result:                                                     │
│  └── PMFlow field learns semantic structure                 │
└─────────────────────────────────────────────────────────────┘
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `margin` | 0.3 | Minimum distance between positive and negative pairs |
| `temperature` | 0.07 | Softmax temperature for InfoNCE |
| `learning_rate` | 0.01 | Optimizer learning rate |
| `batch_size` | 16 | Training batch size |
| `early_stop_margin` | 0.6 | Stop if pos-neg margin exceeds this |

## Files

- `lilith/contrastive_learner.py` - Main module
- `tools/train_contrastive.py` - Training script
- `data/contrastive_learner.json` - Saved pairs and metrics
- `data/contrastive_learner.encoder.pt` - Saved PMFlow weights
