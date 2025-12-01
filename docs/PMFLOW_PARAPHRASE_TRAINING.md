# PMFlow Paraphrase Training - Future Enhancement Guide

**Status**: Optional future enhancement  
**Current Approach**: Canonicalization (93.3% quality) ✅  
**Last Updated**: December 1, 2025

---

## Executive Summary

This document outlines how to train PMFlow on paraphrase pairs for improved semantic understanding. **This is NOT currently needed** (canonicalization achieves 93.3% quality), but the infrastructure exists if future requirements demand it.

### Quick Decision Matrix

| Scenario | Recommendation |
|----------|---------------|
| Quality at 90%+ | Stick with canonicalization ✅ |
| Quality drops below 85% | Consider online learning |
| Many novel phrasings in production | Consider hybrid approach |
| Domain-specific language needed | Consider full PMFlow training |

---

## Current State (Canonicalization Approach)

### How It Works

```python
# Step 1: Canonicalize query
"Explain Python" → "What is python?"
"Give me info on Python" → "What is python?"

# Step 2: Encode canonical form
encode("What is python?") → [embedding]

# Step 3: Compare embeddings
similarity("What is python?", "What is python?") = 1.0 ✅
```

### Covered Patterns (9 rules)

1. `"Can you tell me about X?"` → `"What is X?"`
2. `"Explain X"` → `"What is X?"`
3. `"Tell me about X"` → `"What is X?"`
4. `"What do you know about X?"` → `"What is X?"`
5. `"I want to know about X"` → `"What is X?"`
6. `"Describe X"` → `"What is X?"`
7. `"Give/Get me info/information/details on/about X"` → `"What is X?"`
8. `"How do X function?"` → `"How do X work?"`
9. `"How do X operate?"` → `"How do X work?"`

### Metrics

| Category | Score | Status |
|----------|-------|--------|
| Exact Matches | 100% | ✅ |
| Fuzzy (Typos) | 66.7% | ✅ |
| Case Variations | 100% | ✅ |
| Paraphrased | 100% | ✅ |
| Related Questions | 100% | ✅ |
| **Overall** | **93.3%** | **✅** |

**Semantic Score Improvements:**
- "What is Python?" ↔ "Explain Python": 0.430 → 0.726 (+69%)
- "What is Python?" ↔ "Give me info on Python": 0.252 → 0.726 (+188%)
- "How does X work?" ↔ "How do X function?": 0.470 → 0.768 (+63%)

### Pros & Cons

**Pros:**
- ✅ Deterministic and explainable
- ✅ Works immediately (no training)
- ✅ 100% accuracy for covered patterns
- ✅ Easy to debug and extend
- ✅ Lightweight (just regex rules)

**Cons:**
- ❌ Limited to hardcoded patterns
- ❌ Doesn't generalize to novel phrasings
- ❌ Need to manually add new patterns
- ❌ Minor case sensitivity issues

---

## PMFlow Paraphrase Training (Future Option)

### Infrastructure Already Available

Lilith has **ContrastiveLearner** (`lilith/contrastive_learner.py`) that can train PMFlow on semantic pairs.

```python
from lilith.contrastive_learner import ContrastiveLearner

# Already implemented!
learner = ContrastiveLearner(encoder)

# Add paraphrase pairs
learner.add_symmetric_pair(
    "What is Python?",
    "Explain Python",
    relationship="positive",  # Similar meaning
    source="paraphrase"
)

learner.add_pair(
    "What is Python?",
    "How does Java work?",
    relationship="negative",  # Different meaning
    source="paraphrase"
)

# Train PMFlow to learn relationships
metrics = learner.train_epoch()
# → PMFlow centers/mus updated via gradient descent
```

### How Contrastive Learning Works

**Principle**: Pull similar pairs together, push different pairs apart in embedding space.

```
Before Training:
  embed("What is Python?") = [0.2, 0.5, 0.3, ...]
  embed("Explain Python")  = [0.8, 0.1, 0.7, ...]
  similarity = 0.43 (low!)

After Training:
  embed("What is Python?") = [0.5, 0.6, 0.4, ...]
  embed("Explain Python")  = [0.5, 0.6, 0.4, ...]
  similarity = 0.95 (high!)
```

**Training Process:**
1. Encode anchor and comparison queries
2. Compute cosine similarity
3. For positive pairs: Loss increases if similarity < threshold
4. For negative pairs: Loss increases if similarity > threshold
5. Backpropagate to update PMFlow field parameters
6. Repeat for multiple epochs

---

## Three Implementation Approaches

### Option 1: Manual Paraphrase Dataset

**Use Case**: Quick proof-of-concept, known patterns

```python
# Create paraphrase template pairs
PARAPHRASE_PAIRS = [
    # Positive pairs (similar meaning)
    ("What is {topic}?", "Explain {topic}"),
    ("What is {topic}?", "Tell me about {topic}"),
    ("What is {topic}?", "Give me info on {topic}"),
    ("How does {topic} work?", "How do {topic} function?"),
    
    # Instantiate with real topics
    ("What is Python?", "Explain Python"),
    ("What is machine learning?", "Tell me about machine learning"),
    
    # Negative pairs (different meaning)
    ("What is Python?", "What is Java?"),
    ("What is Python?", "How does it work?"),
]

# Train
learner = ContrastiveLearner(encoder)
for anchor, other in PARAPHRASE_PAIRS:
    label = "positive" if similar(anchor, other) else "negative"
    learner.add_pair(anchor, other, label)

for epoch in range(10):
    metrics = learner.train_epoch()
    print(f"Epoch {epoch}: Loss={metrics.loss:.3f}, "
          f"Margin={metrics.margin:.3f}")

learner.save("data/paraphrase_pmflow.pt")
```

**Effort**: Low (1-2 hours)  
**Benefit**: Moderate (handles variations of taught patterns)  
**Risk**: Low (isolated, can test before deployment)

---

### Option 2: Online Learning from Usage

**Use Case**: Adapt to real user phrasing over time

```python
class ParaphraseLogger:
    """Log query→pattern matches during normal usage."""
    
    def __init__(self):
        self.matches = []  # (query, matched_pattern, confidence)
    
    def log_match(self, query: str, pattern: str, confidence: float):
        """Called after each successful match."""
        if confidence > 0.65:  # Only log confident matches
            self.matches.append((query, pattern))
    
    def generate_training_pairs(self) -> List[Tuple[str, str, str]]:
        """Convert logged matches to contrastive pairs."""
        from collections import defaultdict
        from itertools import combinations
        import random
        
        pairs = []
        
        # Group queries by matched pattern
        pattern_groups = defaultdict(list)
        for query, pattern in self.matches:
            pattern_groups[pattern].append(query)
        
        # Positive pairs: Different queries → Same pattern
        for pattern, queries in pattern_groups.items():
            for q1, q2 in combinations(queries, 2):
                pairs.append((q1, q2, "positive"))
        
        # Negative pairs: Different queries → Different patterns
        patterns = list(pattern_groups.keys())
        for p1, p2 in combinations(patterns, 2):
            q1 = random.choice(pattern_groups[p1])
            q2 = random.choice(pattern_groups[p2])
            pairs.append((q1, q2, "negative"))
        
        return pairs


# Integration in session.py
class LilithSession:
    def __init__(self, ...):
        ...
        self.paraphrase_logger = ParaphraseLogger()
        self.query_count = 0
    
    def process_message(self, user_input: str) -> ComposedResponse:
        response = ...  # Normal processing
        
        # Log successful matches
        if response.confidence > 0.65:
            matched_pattern = response.fragment_ids[0]  # Simplification
            self.paraphrase_logger.log_match(
                user_input, 
                matched_pattern, 
                response.confidence
            )
        
        # Periodic training
        self.query_count += 1
        if self.query_count % 100 == 0:
            self._update_paraphrase_model()
        
        return response
    
    def _update_paraphrase_model(self):
        """Fine-tune PMFlow on logged usage patterns."""
        pairs = self.paraphrase_logger.generate_training_pairs()
        
        if len(pairs) < 10:
            return  # Not enough data yet
        
        # Add to contrastive learner
        for q1, q2, label in pairs:
            self.composer.contrastive_learner.add_pair(q1, q2, label)
        
        # Quick incremental update (not full training)
        self.composer.contrastive_learner.incremental_update(num_steps=10)
        
        # Save updated model
        self.composer.contrastive_learner.save(
            Path(self.config.data_path) / "paraphrase_learner.pt"
        )
        
        # Clear log to prevent memory growth
        self.paraphrase_logger.matches.clear()
```

**Effort**: Medium (4-8 hours)  
**Benefit**: High (adapts to YOUR specific usage patterns)  
**Risk**: Medium (need monitoring, could drift)

---

### Option 3: Hybrid Approach (Recommended)

**Use Case**: Best of both worlds - immediate accuracy + long-term adaptation

```python
# Phase 1: Bootstrap with manual patterns
bootstrap_pairs = [
    ("What is {X}?", "Explain {X}"),
    ("What is {X}?", "Tell me about {X}"),
    # ... all 9 canonicalization rules as templates
]

learner = ContrastiveLearner(encoder)
learner.add_pairs(bootstrap_pairs)
learner.train_epoch(num_epochs=5)
learner.save("data/bootstrap_paraphrase.pt")

# Phase 2: Deploy with online learning
session = LilithSession(...)
session.composer.load_contrastive_learner("data/bootstrap_paraphrase.pt")
session.enable_online_paraphrase_learning()

# Phase 3: Automatic adaptation
# ... system learns from usage automatically ...

# Phase 4: Periodic validation
# Check that quality hasn't regressed
run_qa_retrieval_tests()  # Should maintain 93%+
```

**Effort**: Medium-High (1-2 days initial + ongoing monitoring)  
**Benefit**: Highest (immediate + adaptive)  
**Risk**: Medium (requires validation infrastructure)

---

## Expected Impact

### Before PMFlow Training (Current)

```
Query: "Give me info on Python"
→ Canonicalize: "What is python?"
→ Encode: [0.5, 0.6, 0.4, ...]
→ Compare with "What is Python?"
→ Semantic: 0.726 (via canonicalization)
```

### After PMFlow Training

```
Query: "Give me info on Python"
→ Encode directly: [0.5, 0.6, 0.4, ...]
→ Compare with "What is Python?"  
→ Semantic: 0.85 (learned by PMFlow)

Query: "Gimme details about Python" (NEW, not in rules!)
→ Encode directly: [0.52, 0.58, 0.41, ...]
→ Compare with "What is Python?"
→ Semantic: 0.72 (generalized!)
```

### Quantified Benefits

| Metric | Current | With Training | Improvement |
|--------|---------|---------------|-------------|
| **Semantic Score (known patterns)** | 0.726 | 0.85 | +17% |
| **Semantic Score (novel patterns)** | 0.0-0.3 | 0.6-0.8 | +200-600% |
| **Overall Quality** | 93.3% | 95-98% | +2-5% |
| **Novel Phrasing Coverage** | 0% | 60-80% | +60-80% |
| **False Positive Rate** | <1% | <2% | Slight increase |

### Real-World Examples

**Will Now Work (without new rules):**
- "Gimme details about Python"
- "I need information on machine learning"
- "Could you elaborate on neural networks?"
- "What's the deal with Python?"
- "Provide info regarding deep learning"

**Edge Cases (may still need rules):**
- Slang: "What's the lowdown on Python?"
- Abbreviations: "Info re: ML?"
- Non-English influenced: "Please to explain Python"

---

## When to Implement

### ✅ **Implement If:**

1. **Quality drops** below 85% in production
2. **User feedback** shows frequent misunderstandings of paraphrases
3. **Domain-specific** language not covered by general rules (medical, legal, technical jargon)
4. **Novel phrasings** appear frequently in logs (>10% of queries)
5. **International users** with non-native English patterns
6. **Competitive advantage** requires best-in-class NLU

### ❌ **Don't Implement If:**

1. Current quality is acceptable (>90%)
2. Development time better spent elsewhere
3. Canonicalization rules can handle new patterns easily
4. Low query volume (not enough training data)
5. Users are satisfied with current understanding
6. Risk of destabilizing working system outweighs benefit

---

## Implementation Checklist

If you decide to proceed:

### Phase 1: Preparation (1-2 days)

- [ ] Create paraphrase dataset (50-100 pairs)
  - [ ] Positive pairs from canonicalization rules
  - [ ] Positive pairs from usage logs (if available)
  - [ ] Negative pairs (different topics/intents)
  - [ ] Hard negatives (similar structure, different meaning)

- [ ] Set up training script
  ```bash
  python tools/train_paraphrase_pmflow.py \
    --pairs data/paraphrase_pairs.json \
    --epochs 10 \
    --output data/paraphrase_pmflow.pt
  ```

- [ ] Establish baseline metrics
  - [ ] Run current test suite (93.3% baseline)
  - [ ] Collect semantic scores for benchmark queries
  - [ ] Document current false positive/negative rates

### Phase 2: Training (2-4 hours)

- [ ] Train PMFlow on paraphrase pairs
- [ ] Monitor training metrics:
  - [ ] Loss should decrease
  - [ ] Margin (pos_sim - neg_sim) should increase
  - [ ] Positive similarity should increase toward 0.8-0.9
  - [ ] Negative similarity should decrease toward 0.1-0.3

- [ ] Validate training results
  - [ ] Check semantic scores on known pairs
  - [ ] Test novel phrasing examples
  - [ ] Ensure no catastrophic forgetting

### Phase 3: Validation (4-8 hours)

- [ ] A/B test configuration
  - [ ] Route 10% of queries to trained model
  - [ ] Route 90% to canonicalization (control)
  - [ ] Log results for comparison

- [ ] Quality metrics
  - [ ] Overall retrieval quality (maintain >90%)
  - [ ] Paraphrase handling (should improve)
  - [ ] False positive rate (should stay <2%)
  - [ ] Latency impact (should be minimal)

- [ ] User satisfaction
  - [ ] Collect feedback on understanding
  - [ ] Monitor support tickets for misunderstandings
  - [ ] Track upvotes/downvotes on responses

### Phase 4: Deployment (1 day)

- [ ] Gradual rollout
  - [ ] 10% → 25% → 50% → 100%
  - [ ] Monitor at each stage
  - [ ] Rollback plan ready

- [ ] Online learning setup (optional)
  - [ ] Enable query→pattern logging
  - [ ] Schedule periodic fine-tuning (daily/weekly)
  - [ ] Set up monitoring dashboard

- [ ] Documentation
  - [ ] Update this document with results
  - [ ] Document new patterns learned
  - [ ] Create troubleshooting guide

---

## Code Examples

### Training Script Template

```python
#!/usr/bin/env python3
"""
Train PMFlow on paraphrase pairs for improved semantic understanding.

Usage:
    python tools/train_paraphrase_pmflow.py \\
        --pairs data/paraphrase_pairs.json \\
        --epochs 10 \\
        --output data/paraphrase_pmflow.pt
"""

import json
from pathlib import Path
from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.contrastive_learner import ContrastiveLearner


def load_paraphrase_pairs(path: str):
    """Load paraphrase pairs from JSON file."""
    with open(path) as f:
        data = json.load(f)
    
    pairs = []
    for item in data:
        pairs.append((
            item['query1'],
            item['query2'],
            item['label'],  # 'positive' or 'negative'
            item.get('weight', 1.0),
        ))
    return pairs


def main(args):
    # Initialize encoder
    encoder = PMFlowEmbeddingEncoder(
        dimension=96,
        latent_dim=48,
        seed=13
    )
    
    # Load existing state if available
    state_path = Path("data/pmflow_state.pt")
    if state_path.exists():
        encoder.load_state(state_path)
        print(f"📂 Loaded existing PMFlow state from {state_path}")
    
    # Create contrastive learner
    learner = ContrastiveLearner(
        encoder,
        margin=0.3,
        temperature=0.07,
        learning_rate=1e-3
    )
    
    # Load paraphrase pairs
    pairs = load_paraphrase_pairs(args.pairs)
    print(f"📊 Loaded {len(pairs)} paraphrase pairs")
    
    for q1, q2, label, weight in pairs:
        learner.add_pair(q1, q2, label, weight=weight, source="manual")
    
    # Train
    print(f"\n🎓 Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        metrics = learner.train_epoch()
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Loss={metrics.loss:.4f}, "
              f"Margin={metrics.margin:.4f}, "
              f"Pos={metrics.positive_similarity:.4f}, "
              f"Neg={metrics.negative_similarity:.4f}")
    
    # Save trained model
    encoder.save_state(args.output)
    learner.save(args.output.replace('.pt', '_learner.pt'))
    print(f"\n✅ Saved trained model to {args.output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', required=True, help='Path to paraphrase pairs JSON')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--output', required=True, help='Output model path')
    main(parser.parse_args())
```

### Paraphrase Pairs Dataset Format

```json
{
  "pairs": [
    {
      "query1": "What is Python?",
      "query2": "Explain Python",
      "label": "positive",
      "weight": 1.0,
      "source": "canonicalization_rule"
    },
    {
      "query1": "What is Python?",
      "query2": "Tell me about Python",
      "label": "positive",
      "weight": 1.0,
      "source": "canonicalization_rule"
    },
    {
      "query1": "What is Python?",
      "query2": "What is Java?",
      "label": "negative",
      "weight": 1.0,
      "source": "different_topic"
    },
    {
      "query1": "What is machine learning?",
      "query2": "How does machine learning work?",
      "label": "negative",
      "weight": 0.8,
      "source": "different_intent"
    }
  ]
}
```

---

## Monitoring & Maintenance

### Key Metrics to Track

```python
# Quality Metrics
- overall_retrieval_quality: float  # Should stay >90%
- paraphrase_match_rate: float      # Should improve
- false_positive_rate: float        # Should stay <2%
- semantic_score_avg: float         # Should increase

# Training Metrics
- contrastive_loss: float           # Should decrease
- pos_neg_margin: float             # Should increase
- training_pairs_count: int         # Track growth
- last_training_date: datetime      # Monitor staleness

# Usage Metrics
- novel_phrasing_frequency: float   # Queries not in rules
- user_satisfaction: float          # Upvotes / total
- support_ticket_rate: float        # Should decrease
```

### Automated Validation

```python
def validate_paraphrase_model():
    """Run after each training to ensure quality."""
    
    # Test semantic scores
    test_pairs = [
        ("What is Python?", "Explain Python", 0.8),  # Should be >0.8
        ("What is Python?", "What is Java?", 0.3),   # Should be <0.3
    ]
    
    for q1, q2, expected_threshold in test_pairs:
        sim = compute_similarity(q1, q2)
        if sim < expected_threshold:
            print(f"⚠️  WARNING: {q1} vs {q2} = {sim:.3f} "
                  f"(expected {expected_threshold})")
    
    # Run full test suite
    results = run_qa_retrieval_tests()
    if results.overall_quality < 0.90:
        print(f"❌ FAIL: Quality dropped to {results.overall_quality:.1%}")
        return False
    
    print(f"✅ PASS: Quality maintained at {results.overall_quality:.1%}")
    return True
```

---

## Rollback Plan

If training causes issues:

### Quick Rollback

```python
# In session.py or relevant configuration
USE_TRAINED_PMFLOW = False  # Toggle to disable

if USE_TRAINED_PMFLOW:
    encoder.load_state("data/paraphrase_pmflow.pt")
else:
    encoder.load_state("data/baseline_pmflow.pt")  # Original
```

### Gradual Rollback

1. Reduce traffic to trained model: 100% → 50% → 25% → 0%
2. Monitor quality at each step
3. Document which queries improved vs. regressed
4. Retrain with better pairs if patterns identified

---

## Success Criteria

Consider training **successful** if:

- ✅ Overall quality maintained or improved (≥93.3%)
- ✅ Novel phrasing coverage improved (≥60%)
- ✅ Semantic scores for paraphrases increased (≥0.8)
- ✅ False positive rate stayed low (≤2%)
- ✅ User satisfaction improved or maintained
- ✅ No increase in support tickets
- ✅ Latency impact minimal (<10ms)

Consider training **unsuccessful** if:

- ❌ Overall quality dropped (<90%)
- ❌ False positives increased significantly (>5%)
- ❌ User confusion increased
- ❌ Support tickets increased
- ❌ Performance degraded significantly (>50ms)

---

## References

### Code Files

- `lilith/contrastive_learner.py` - Contrastive learning implementation
- `lilith/embedding.py` - PMFlowEmbeddingEncoder
- `lilith/response_fragments_sqlite.py` - Query canonicalization (lines 421-493)
- `test_qa_retrieval.py` - Quality validation tests

### Related Documentation

- `RETRIEVAL_ARCHITECTURE_ANALYSIS.md` - Hybrid retrieval design
- `HYBRID_RETRIEVAL_IMPLEMENTATION.md` - Implementation details
- `PARAPHRASE_FIX_SUMMARY.md` - Canonicalization approach results

### Papers & Resources

- InfoNCE: A Simple Framework for Contrastive Learning
- BERT Paraphrase Detection
- Sentence-BERT for Semantic Similarity
- PMFlow: Persistent Memory Flow Networks

---

## Conclusion

**Current Status**: Canonicalization approach is **production-ready** at 93.3% quality.

**PMFlow Training**: Available as a **future enhancement** if needed. Infrastructure exists, implementation is straightforward, but **benefits don't currently justify complexity**.

**Recommendation**: 
1. **Continue with canonicalization** for now ✅
2. **Monitor** for novel phrasing patterns in production
3. **Implement training** only if quality drops or specific needs arise
4. **Start with Option 3 (Hybrid)** if you do implement

**Key Insight**: The 9 canonicalization rules handle 100% of tested paraphrases. PMFlow training would add flexibility for edge cases, but the complexity-to-benefit ratio doesn't favor it at 93.3% quality. Save this enhancement for when it's truly needed.

---

**Last Updated**: December 1, 2025  
**Next Review**: When quality drops below 85% or user feedback indicates need  
**Owner**: Reference this document before implementing paraphrase training
