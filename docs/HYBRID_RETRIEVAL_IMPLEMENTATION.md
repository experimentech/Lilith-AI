# Hybrid Retrieval Implementation Plan

**Date:** 1 December 2025  
**Related:** See `RETRIEVAL_ARCHITECTURE_ANALYSIS.md` for full context

## Overview

This document provides concrete implementation steps for fixing the retrieval mechanism to support both semantic queries and factual Q&A through hybrid scoring.

---

## Phase 1: Immediate Implementation

### 1.1 Add Exact Match Helper Function

**Location:** `lilith/response_fragments_sqlite.py`  
**Add before:** `retrieve_patterns()` method

```python
def _compute_exact_match_score(self, query: str, trigger: str) -> float:
    """
    Compute exact match score for query vs trigger context.
    
    Returns:
        1.0 for perfect match
        0.95 for normalized match (case/punctuation insensitive)
        0.0 for no match
    """
    # Perfect exact match
    if query == trigger:
        return 1.0
    
    # Case-insensitive match
    if query.lower() == trigger.lower():
        return 0.98
    
    # Normalized match (remove punctuation, extra spaces)
    import string
    translator = str.maketrans('', '', string.punctuation)
    
    query_norm = ' '.join(query.translate(translator).split()).lower()
    trigger_norm = ' '.join(trigger.translate(translator).split()).lower()
    
    if query_norm == trigger_norm:
        return 0.95
    
    return 0.0
```

### 1.2 Add Token Overlap Helper Function

```python
def _compute_token_overlap_score(self, query: str, trigger: str) -> float:
    """
    Compute token overlap score (keyword matching).
    
    Returns:
        0.0-1.0 based on Jaccard similarity of tokens
    """
    query_tokens = set(query.lower().split())
    trigger_tokens = set(trigger.lower().split())
    
    if not query_tokens or not trigger_tokens:
        return 0.0
    
    intersection = query_tokens & trigger_tokens
    union = query_tokens | trigger_tokens
    
    jaccard = len(intersection) / len(union)
    
    # Also compute coverage (what % of query tokens appear in trigger)
    coverage = len(intersection) / len(query_tokens)
    
    # Return weighted average favoring coverage
    return 0.4 * jaccard + 0.6 * coverage
```

### 1.3 Enhance Fuzzy Matching (if not present)

**Check if fuzzy_matcher exists, if not add:**

```python
def __init__(self, ...):
    # Existing initialization...
    
    # Add fuzzy matching capability
    try:
        from rapidfuzz import fuzz
        self.fuzz = fuzz
        self.fuzzy_available = True
    except ImportError:
        try:
            from fuzzywuzzy import fuzz
            self.fuzz = fuzz
            self.fuzzy_available = True
        except ImportError:
            self.fuzz = None
            self.fuzzy_available = False
            print("⚠️  Fuzzy matching not available (install rapidfuzz or fuzzywuzzy)")

def _compute_fuzzy_match_score(self, query: str, trigger: str) -> float:
    """
    Compute fuzzy string similarity.
    
    Returns:
        0.0-1.0 based on fuzzy ratio
    """
    if not self.fuzzy_available:
        return 0.0
    
    # Use token sort ratio for word order independence
    ratio = self.fuzz.token_sort_ratio(query.lower(), trigger.lower())
    return ratio / 100.0
```

### 1.4 Modify retrieve_patterns() for Hybrid Scoring

**Replace the scoring section (around lines 467-480):**

```python
def retrieve_patterns(
    self,
    context: str,
    topk: int = 5,
    min_score: float = 0.0
) -> List[Tuple[ResponsePattern, float]]:
    """
    Retrieve best matching patterns for context using HYBRID scoring.
    
    Scoring methods:
    1. Exact match (1.0 confidence for perfect matches)
    2. Fuzzy matching (0.75-1.0 for similar text)
    3. Token overlap (0.0-1.0 for keyword matching)
    4. Semantic similarity (0.0-1.0 for concept matching)
    
    Final score: max(exact, fuzzy, token_overlap, semantic)
    
    Args:
        context: Query context
        topk: Number of results
        min_score: Minimum success score threshold
    
    Returns:
        List of (pattern, confidence_score) tuples
    """
    conn = self._get_connection()
    
    # Get all patterns above min_score
    cursor = conn.execute("""
        SELECT * FROM response_patterns 
        WHERE success_score >= ?
        ORDER BY success_score DESC
    """, (min_score,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return []
    
    # Convert to ResponsePattern objects
    patterns = []
    for row in rows:
        embedding_cache = None
        if row['embedding_cache']:
            embedding_cache = json.loads(row['embedding_cache'])
        
        pattern = ResponsePattern(
            fragment_id=row['fragment_id'],
            trigger_context=row['trigger_context'],
            response_text=row['response_text'],
            success_score=row['success_score'],
            intent=row['intent'],
            usage_count=row['usage_count'],
            embedding_cache=embedding_cache
        )
        patterns.append(pattern)
    
    # Encode query context for semantic matching
    try:
        tokens = context.split()
        query_embedding = self.encoder.encode(tokens)
        if hasattr(query_embedding, 'numpy'):
            query_embedding = query_embedding.numpy()
        query_embedding = np.array(query_embedding).flatten()
        semantic_available = True
    except Exception as e:
        print(f"  ⚠️  Encoding error for '{context}': {e}")
        query_embedding = None
        semantic_available = False
    
    # HYBRID SCORING: Compute all matching scores
    scored_patterns = []
    
    for pattern in patterns:
        scores = {
            'exact': 0.0,
            'fuzzy': 0.0,
            'token_overlap': 0.0,
            'semantic': 0.0
        }
        
        # 1. Exact match score
        scores['exact'] = self._compute_exact_match_score(
            context, 
            pattern.trigger_context
        )
        
        # 2. Fuzzy match score
        if self.fuzzy_available:
            scores['fuzzy'] = self._compute_fuzzy_match_score(
                context,
                pattern.trigger_context
            )
        
        # 3. Token overlap score
        scores['token_overlap'] = self._compute_token_overlap_score(
            context,
            pattern.trigger_context
        )
        
        # 4. Semantic similarity score
        if semantic_available and query_embedding is not None:
            # Get or compute pattern embedding
            if pattern.embedding_cache:
                pattern_embedding = np.array(pattern.embedding_cache).flatten()
            else:
                try:
                    pattern_tokens = pattern.trigger_context.split()
                    pattern_embedding = self.encoder.encode(pattern_tokens)
                    if hasattr(pattern_embedding, 'numpy'):
                        pattern_embedding = pattern_embedding.numpy()
                    pattern_embedding = np.array(pattern_embedding).flatten()
                except Exception:
                    pattern_embedding = None
            
            if pattern_embedding is not None:
                # Compute cosine similarity
                dot_product = np.dot(query_embedding, pattern_embedding)
                norm_query = np.linalg.norm(query_embedding)
                norm_pattern = np.linalg.norm(pattern_embedding)
                
                if norm_query > 0 and norm_pattern > 0:
                    scores['semantic'] = dot_product / (norm_query * norm_pattern)
                    scores['semantic'] = max(0.0, scores['semantic'])  # Clamp to [0,1]
        
        # COMBINE SCORES: Use maximum (best matching method wins)
        final_score = max(scores.values())
        
        # Optional: Log scoring breakdown for debugging
        if final_score > 0.7:
            print(f"  🎯 Match: {pattern.trigger_context[:50]}...")
            print(f"     Exact: {scores['exact']:.3f} | Fuzzy: {scores['fuzzy']:.3f} | "
                  f"Tokens: {scores['token_overlap']:.3f} | Semantic: {scores['semantic']:.3f}")
            print(f"     → Final: {final_score:.3f}")
        
        scored_patterns.append((pattern, final_score))
    
    # Sort by final score (descending) and return top-k
    scored_patterns.sort(key=lambda x: x[1], reverse=True)
    return scored_patterns[:topk]
```

### 1.5 Configuration Support

**Add to SessionConfig (if not present):**

```python
@dataclass
class SessionConfig:
    # ... existing fields ...
    
    # Retrieval scoring thresholds
    retrieval_exact_threshold: float = 0.95
    retrieval_fuzzy_threshold: float = 0.75
    retrieval_token_threshold: float = 0.70
    retrieval_semantic_threshold: float = 0.60
    
    # Scoring combination method
    retrieval_scoring_method: str = "max"  # "max", "weighted_average", "cascade"
```

---

## Phase 2: Testing & Validation

### 2.1 Create Test File

**Create:** `tests/test_hybrid_retrieval.py`

```python
"""
Test hybrid retrieval mechanism.

Validates that exact, fuzzy, and semantic matching all work correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lilith.session import LilithSession, SessionConfig


def test_exact_match():
    """Test that exact matches get high confidence."""
    config = SessionConfig()
    session = LilithSession("test_user", config=config)
    
    # Teach a pattern
    session.teach("What is Python?", "Python is a high-level programming language.")
    
    # Query with exact match
    response = session.process_message("What is Python?")
    
    print(f"Exact match test:")
    print(f"  Query: What is Python?")
    print(f"  Confidence: {response.confidence:.3f}")
    print(f"  Is fallback: {response.is_fallback}")
    print(f"  Response: {response.text[:50]}...")
    
    assert response.confidence >= 0.95, f"Expected >=0.95, got {response.confidence}"
    assert not response.is_fallback, "Should not be fallback for exact match"
    print("  ✅ PASSED\n")


def test_fuzzy_match():
    """Test that similar questions get good confidence."""
    config = SessionConfig()
    session = LilithSession("test_user", config=config)
    
    # Teach a pattern
    session.teach(
        "What is machine learning?",
        "Machine learning is a subset of AI that learns from data."
    )
    
    # Query with slight variation
    response = session.process_message("What's machine learning?")
    
    print(f"Fuzzy match test:")
    print(f"  Taught: What is machine learning?")
    print(f"  Query: What's machine learning?")
    print(f"  Confidence: {response.confidence:.3f}")
    print(f"  Is fallback: {response.is_fallback}")
    
    assert response.confidence >= 0.75, f"Expected >=0.75, got {response.confidence}"
    print("  ✅ PASSED\n")


def test_semantic_match():
    """Test that semantically similar queries work."""
    config = SessionConfig()
    session = LilithSession("test_user", config=config)
    
    # Teach a pattern
    session.teach(
        "How does AI work?",
        "AI works by processing data through algorithms to find patterns."
    )
    
    # Query with different wording but same meaning
    response = session.process_message("Explain artificial intelligence")
    
    print(f"Semantic match test:")
    print(f"  Taught: How does AI work?")
    print(f"  Query: Explain artificial intelligence")
    print(f"  Confidence: {response.confidence:.3f}")
    print(f"  Is fallback: {response.is_fallback}")
    
    # Semantic matching may be lower confidence
    assert response.confidence >= 0.40, f"Expected >=0.40, got {response.confidence}"
    print("  ✅ PASSED\n")


def test_token_overlap():
    """Test that keyword overlap provides good matching."""
    config = SessionConfig()
    session = LilithSession("test_user", config=config)
    
    # Teach a pattern
    session.teach(
        "neural networks deep learning architecture",
        "Neural networks use layered architectures for deep learning."
    )
    
    # Query with keyword overlap
    response = session.process_message("deep learning neural networks")
    
    print(f"Token overlap test:")
    print(f"  Taught: neural networks deep learning architecture")
    print(f"  Query: deep learning neural networks")
    print(f"  Confidence: {response.confidence:.3f}")
    
    assert response.confidence >= 0.60, f"Expected >=0.60, got {response.confidence}"
    print("  ✅ PASSED\n")


def test_no_match():
    """Test that unrelated queries have low confidence."""
    config = SessionConfig()
    session = LilithSession("test_user", config=config)
    
    # Teach a pattern
    session.teach("What is Python?", "Python is a programming language.")
    
    # Query something completely unrelated
    response = session.process_message("Tell me about quantum physics")
    
    print(f"No match test:")
    print(f"  Taught: What is Python?")
    print(f"  Query: Tell me about quantum physics")
    print(f"  Confidence: {response.confidence:.3f}")
    print(f"  Is fallback: {response.is_fallback}")
    
    # Should be fallback or very low confidence
    assert response.is_fallback or response.confidence < 0.5
    print("  ✅ PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("HYBRID RETRIEVAL TESTS")
    print("=" * 60 + "\n")
    
    test_exact_match()
    test_fuzzy_match()
    test_semantic_match()
    test_token_overlap()
    test_no_match()
    
    print("=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)
```

### 2.2 Update bootstrap_qa.py Testing

**Enhance the test section:**

```python
def test_retrieval_quality(session, test_queries):
    """Test retrieval quality with detailed metrics."""
    
    results = {
        'high_confidence': 0,  # >0.75
        'medium_confidence': 0,  # 0.5-0.75
        'low_confidence': 0,  # <0.5
        'fallbacks': 0
    }
    
    print("\n" + "=" * 60)
    print("RETRIEVAL QUALITY TEST")
    print("=" * 60)
    
    for query in test_queries:
        response = session.process_message(query)
        
        # Categorize
        if response.is_fallback:
            results['fallbacks'] += 1
            category = '❌ FALLBACK'
        elif response.confidence >= 0.75:
            results['high_confidence'] += 1
            category = '✅ HIGH'
        elif response.confidence >= 0.5:
            results['medium_confidence'] += 1
            category = '⚠️  MEDIUM'
        else:
            results['low_confidence'] += 1
            category = '❌ LOW'
        
        print(f"\n{category} | {response.confidence:.3f} | {query}")
        print(f"  → {response.text[:60]}...")
    
    # Summary
    total = len(test_queries)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"High confidence (>0.75):  {results['high_confidence']:2d}/{total} "
          f"({results['high_confidence']/total*100:.1f}%)")
    print(f"Medium confidence (0.5+):  {results['medium_confidence']:2d}/{total} "
          f"({results['medium_confidence']/total*100:.1f}%)")
    print(f"Low confidence (<0.5):     {results['low_confidence']:2d}/{total} "
          f"({results['low_confidence']/total*100:.1f}%)")
    print(f"Fallbacks:                 {results['fallbacks']:2d}/{total} "
          f"({results['fallbacks']/total*100:.1f}%)")
    print("=" * 60)
    
    # Success criteria: >70% should be high or medium confidence
    success_rate = (results['high_confidence'] + results['medium_confidence']) / total
    if success_rate >= 0.7:
        print(f"✅ SUCCESS: {success_rate*100:.1f}% retrieval quality")
    else:
        print(f"⚠️  NEEDS IMPROVEMENT: {success_rate*100:.1f}% retrieval quality (target: 70%)")
    
    return results
```

---

## Phase 3: Optimization & Refinement

### 3.1 Add Caching for Exact Matches

```python
def __init__(self, ...):
    # ... existing init ...
    self._exact_match_cache = {}  # {query_normalized: pattern_id}

def _get_cached_exact_match(self, query: str) -> Optional[str]:
    """Check cache for exact match pattern ID."""
    import string
    translator = str.maketrans('', '', string.punctuation)
    query_norm = ' '.join(query.translate(translator).split()).lower()
    return self._exact_match_cache.get(query_norm)

def _cache_exact_match(self, query: str, pattern_id: str):
    """Cache exact match for fast lookup."""
    import string
    translator = str.maketrans('', '', string.punctuation)
    query_norm = ' '.join(query.translate(translator).split()).lower()
    self._exact_match_cache[query_norm] = pattern_id
```

### 3.2 Weighted Scoring Option

```python
def _compute_weighted_score(self, scores: dict, weights: dict = None) -> float:
    """
    Compute weighted combination of scores.
    
    Default weights favor exact > fuzzy > tokens > semantic
    """
    if weights is None:
        weights = {
            'exact': 0.4,
            'fuzzy': 0.3,
            'token_overlap': 0.2,
            'semantic': 0.1
        }
    
    total = sum(scores[key] * weights.get(key, 0.0) for key in scores)
    return total
```

### 3.3 Cascade Scoring (Try Each Method in Order)

```python
def _compute_cascade_score(self, scores: dict) -> float:
    """
    Cascade scoring: Use best method, fall back if low.
    
    Order: exact → fuzzy → token_overlap → semantic
    """
    # Exact match: If high, use immediately
    if scores['exact'] >= 0.95:
        return scores['exact']
    
    # Fuzzy match: If good, use it
    if scores['fuzzy'] >= 0.75:
        return scores['fuzzy']
    
    # Token overlap: If decent, use it
    if scores['token_overlap'] >= 0.70:
        return scores['token_overlap']
    
    # Semantic: Last resort
    return scores['semantic']
```

---

## Expected Results

### Before Implementation

```
Query: "What is Python?" (exact match to taught pattern)
Scores:
  Exact: N/A (not implemented)
  Fuzzy: N/A (threshold too high)
  Tokens: N/A (not implemented)
  Semantic: 0.182
  → Final: 0.182 (FALLBACK)
```

### After Implementation

```
Query: "What is Python?" (exact match to taught pattern)
Scores:
  Exact: 1.000 ✅
  Fuzzy: 1.000
  Tokens: 0.857
  Semantic: 0.182
  → Final: 1.000 (HIGH CONFIDENCE)
```

```
Query: "What's Python?" (fuzzy match)
Scores:
  Exact: 0.000
  Fuzzy: 0.889 ✅
  Tokens: 0.667
  Semantic: 0.165
  → Final: 0.889 (HIGH CONFIDENCE)
```

```
Query: "Explain Python programming" (semantic match)
Scores:
  Exact: 0.000
  Fuzzy: 0.421
  Tokens: 0.750 ✅
  Semantic: 0.523
  → Final: 0.750 (MEDIUM CONFIDENCE)
```

---

## Dependencies

### Required Python Packages

```bash
# For fuzzy matching (choose one):
pip install rapidfuzz  # Recommended (faster)
# OR
pip install fuzzywuzzy python-Levenshtein  # Alternative

# Already installed:
# - numpy (for embeddings)
# - sqlite3 (standard library)
```

### Import Changes

```python
# Add to response_fragments_sqlite.py
import string  # For punctuation removal
from typing import Optional  # For type hints

# Conditional import for fuzzy matching
try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz
        FUZZY_AVAILABLE = True
    except ImportError:
        FUZZY_AVAILABLE = False
```

---

## Rollback Plan

If hybrid retrieval causes issues:

1. **Keep both methods:** Add flag `use_hybrid_retrieval: bool = True` in config
2. **Gradual rollout:** Default to semantic, opt-in to hybrid
3. **A/B testing:** Track metrics for both methods
4. **Revert:** Simple config change to disable hybrid scoring

```python
# In retrieve_patterns():
if self.use_hybrid_retrieval:
    # New hybrid scoring
    final_score = max(scores.values())
else:
    # Original semantic-only scoring
    final_score = scores['semantic']
```

---

## Success Criteria

✅ **Phase 1 Complete When:**
- Exact matches: >95% confidence
- Fuzzy matches: >75% confidence  
- Q&A bootstrap: >70% retrieval success
- All tests in `test_hybrid_retrieval.py` pass

✅ **Phase 2 Complete When:**
- Token overlap improves medium-confidence matches
- Caching speeds up repeat queries
- Documentation updated

✅ **Phase 3 Complete When:**
- Weighted scoring option available
- Cascade scoring option available
- Configuration flexible and tested

---

## Next Steps

1. **Install dependencies** (rapidfuzz or fuzzywuzzy)
2. **Implement Phase 1** helper functions
3. **Modify retrieve_patterns()** with hybrid scoring
4. **Create test file** and run tests
5. **Test with Q&A bootstrap** - measure improvement
6. **Commit and document** changes

---

## Notes

- This is a **refinement**, not a redesign
- Semantic similarity still works - we're just adding alternatives
- The syntax stage remains unchanged (composition, not retrieval)
- Backward compatible - existing patterns still work
- Future: Could train better embeddings specifically for questions

---

## Questions During Implementation

**Q: Should we remove the old fuzzy_matcher if it exists?**  
A: No - merge it with new implementation, keep best of both

**Q: What if semantic similarity is negative (opposite concepts)?**  
A: Clamp to 0.0 (done in code above)

**Q: Should we log all scoring breakdowns?**  
A: Only for high scores (>0.7) to avoid spam, make configurable

**Q: How to handle multi-sentence queries?**  
A: Token overlap and fuzzy matching handle this naturally

**Q: Should we cache embeddings?**  
A: Already done via `embedding_cache` field, verify it's used

---

Ready to implement! 🚀
