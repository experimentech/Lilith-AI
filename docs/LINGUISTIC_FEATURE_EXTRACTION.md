# Linguistic Feature Extraction - Implementation Plan

## Overview

Lilith has sophisticated linguistic infrastructure that is currently **not connected** to the Wikipedia auto-learning feature. This document outlines what exists, what's missing, and the implementation plan.

## Current State

### ✅ Already Implemented

#### 1. Semantic Relationship Storage (`production_concept_store.py`)

**Features:**
- Semantic concepts with structured relations
- Relation types: `is_type_of`, `has_property`, `used_for`
- SQLite database persistence
- PMFlow-enhanced retrieval (query expansion, hierarchical, attention)
- Usage tracking and consolidation

**Example:**
```python
concept = SemanticConcept(
    concept_id="rust_lang",
    term="Rust",
    properties=["memory safe", "systems programming", "compiled"],
    relations=[
        Relation("is_type_of", "programming language", 0.95),
        Relation("used_for", "systems programming", 0.90)
    ],
    source="wikipedia",
    confidence=0.85
)
```

#### 2. Entity Recognition (`concept_taxonomy.py`)

**Features:**
- Hierarchical concept relationships (parent/child)
- Property-based classification
- Related concept linking
- Default taxonomy for common domains

**Example:**
```python
taxonomy.add_concept(
    name="Python",
    parents={"programming_language"},
    properties={"interpreted", "high_level", "general_purpose"},
    related={"data_science", "web_development"}
)
```

#### 3. Intent Clustering (`bnn_intent_classifier.py`)

**Features:**
- Semantic intent detection via BNN embeddings
- Automatic pattern clustering by intent
- No keyword matching required
- Centroid-based similarity

**Example:**
```python
cluster = IntentCluster(
    intent_label="programming_query",
    centroid=embedding_vector,
    pattern_ids=["pattern_1", "pattern_2", "pattern_3"],
    representative_text="what is Python programming",
    confidence=0.92
)
```

### ❌ Missing: Integration with Wikipedia Auto-Learning

**Current Wikipedia Flow:**
1. User asks: "what is Rust programming language?"
2. System queries Wikipedia API
3. Response: "Rust is a general-purpose programming language..."
4. User upvotes with `/+`
5. **System stores as plain text pattern** ⚠️

**What's NOT Happening:**
- ❌ No entity extraction from Wikipedia text
- ❌ No semantic relationship parsing
- ❌ No concept store updates
- ❌ No taxonomy enrichment
- ❌ No vocabulary tracking

**Result:** Rich Wikipedia knowledge is stored as opaque text, not structured semantic knowledge.

## Implementation Phases

### Phase A: Semantic Relationship Extraction (PRIORITY)

**Goal:** Extract key facts from Wikipedia responses and store as structured concepts.

**Target Patterns:**
```
"X is a Y" → Relation(is_type_of, Y)
"X is a Y that Z" → Relation(is_type_of, Y) + properties=[Z]
"X is used for Y" → Relation(used_for, Y)
"X has/includes Y" → Relation(has_property, Y)
"X was created/developed by Y" → Relation(created_by, Y)
```

**Example Extraction:**

Input: *"Rust is a general-purpose programming language emphasizing performance, type safety, and concurrency."*

Output:
```python
Concept(
    term="Rust",
    relations=[
        Relation("is_type_of", "programming language", 0.95),
        Relation("is_type_of", "general-purpose language", 0.90)
    ],
    properties=[
        "performance",
        "type safety", 
        "concurrency"
    ],
    source="wikipedia"
)
```

**Implementation:**
1. Create `SemanticExtractor` class in new file `semantic_extractor.py`
2. Pattern matching for common fact structures
3. Entity and property extraction using regex + heuristics
4. Integration hook in `lilith_cli.py` upvote handler
5. Store in `ProductionConceptStore` alongside pattern learning

**Files to Modify:**
- Create: `lilith/semantic_extractor.py`
- Modify: `lilith_cli.py` (upvote handler)
- Modify: `lilith/multi_tenant_store.py` (add concept store integration)

### Phase B: Entity Recognition Enhancement

**Goal:** Tag entities with types and add to taxonomy.

**Entity Types:**
- Programming languages (Python, Rust, Java)
- Concepts (machine learning, quantum computing)
- People (creators, authors)
- Organizations (companies, institutions)
- Tools/Technologies

**Example:**
```python
# From: "Python is a high-level programming language created by Guido van Rossum"
entities = [
    Entity("Python", type="programming_language"),
    Entity("Guido van Rossum", type="person", role="creator")
]
```

**Implementation:**
1. Extend `concept_taxonomy.py` with entity type system
2. Named entity extraction (capitalization, context clues)
3. Auto-add to taxonomy with inferred relationships
4. Link related entities (creator → language, language → domain)

### Phase C: Vocabulary Expansion

**Goal:** Track new words and build semantic vocabulary index.

**Features:**
- Extract novel terms from Wikipedia
- Track word frequencies
- Build co-occurrence matrix
- Identify technical vocabulary vs common words

**Example:**
```python
vocabulary = {
    "borrow_checker": {
        "frequency": 3,
        "contexts": ["memory safety", "Rust language"],
        "related_terms": ["ownership", "lifetime", "reference"]
    }
}
```

**Implementation:**
1. Create `VocabularyTracker` class
2. Tokenization and normalization
3. Stop word filtering
4. Context tracking (n-gram windows)
5. Integration with BNN embeddings

### Phase D: Syntactic Pattern Learning

**Goal:** Learn sentence structures from Wikipedia for better generation.

**Patterns to Learn:**
```
"X is a Y that Z" → template for definition responses
"X was developed in Y by Z" → template for history
"X is used for Y" → template for purpose/function
```

**Example:**
```python
template = SyntacticTemplate(
    pattern="[SUBJECT] is a [TYPE] that [PROPERTY]",
    examples=[
        "Rust is a language that emphasizes safety",
        "Python is a language that focuses on readability"
    ],
    confidence=0.88
)
```

**Implementation:**
1. Extend `syntax_stage_bnn.py` with pattern extraction
2. Template generalization from examples
3. Slot filling for compositional generation
4. Integration with `template_composer.py`

## Dependencies Between Phases

```
Phase A (Semantic Relations)
    ↓ provides concepts
Phase B (Entity Recognition)
    ↓ enriches taxonomy
Phase C (Vocabulary)
    ↓ enhances embeddings
Phase D (Syntactic Patterns)
```

**Independent Implementation:** Each phase can be implemented separately, but they work best together.

**Recommended Order:** A → B → C → D

## Success Metrics

### Phase A Success Criteria:
- ✅ Wikipedia text parsed into Concept objects
- ✅ Relations extracted with >80% accuracy
- ✅ Concepts stored in ProductionConceptStore
- ✅ Subsequent queries can retrieve by concept, not just text match

### Phase B Success Criteria:
- ✅ Entity types automatically classified
- ✅ Taxonomy grows with learned entities
- ✅ Entity relationships inferred correctly

### Phase C Success Criteria:
- ✅ Vocabulary index tracks novel terms
- ✅ Word frequencies correlate with importance
- ✅ Technical terms identified accurately

### Phase D Success Criteria:
- ✅ Common sentence patterns extracted
- ✅ Templates can generate new responses
- ✅ Generated text follows learned patterns

## Integration Points

### Current Wikipedia Auto-Learning Flow:
```python
# In lilith_cli.py upvote handler
if last_pattern_id.startswith('external_'):
    # Store as pattern
    fragment_store.add_pattern(
        trigger_context=last_user_input,
        response_text=last_response_text,
        success_score=0.8,
        intent="learned_knowledge"
    )
```

### Enhanced Flow (Phase A):
```python
# In lilith_cli.py upvote handler
if last_pattern_id.startswith('external_'):
    # 1. Store as pattern (existing)
    fragment_store.add_pattern(...)
    
    # 2. Extract and store concepts (NEW)
    extractor = SemanticExtractor()
    concepts = extractor.extract_concepts(
        query=last_user_input,
        response=last_response_text
    )
    
    # 3. Add to concept store
    for concept in concepts:
        concept_store.add_concept(
            term=concept.term,
            properties=concept.properties,
            relations=concept.relations,
            source="wikipedia",
            confidence=0.85
        )
```

## File Structure

```
lilith/
├── semantic_extractor.py        # NEW: Extract concepts from text
├── vocabulary_tracker.py        # NEW: Track word usage (Phase C)
├── production_concept_store.py  # EXISTING: Modified for Wikipedia integration
├── concept_taxonomy.py          # EXISTING: Enhanced entity types (Phase B)
├── bnn_intent_classifier.py     # EXISTING: May benefit from learned patterns
└── syntax_stage_bnn.py          # EXISTING: Enhanced templates (Phase D)

lilith_cli.py                     # MODIFIED: Upvote handler calls extractor
```

## Next Steps

1. **Implement Phase A** (Semantic Relationship Extraction)
   - Create `semantic_extractor.py`
   - Implement pattern matching for common fact structures
   - Test extraction accuracy on sample Wikipedia responses
   - Integrate with upvote handler

2. **Test Integration**
   - Ask: "what is quantum computing"
   - Upvote Wikipedia response
   - Verify concept stored in database
   - Query by concept properties

3. **Iterate and Refine**
   - Improve extraction patterns based on real usage
   - Add more relation types
   - Tune confidence thresholds

4. **Phase B, C, D** (As needed)

## Open Questions

1. Should concept extraction be automatic or require upvote?
   - **Recommendation:** Require upvote (user validates quality)

2. How to handle conflicting information?
   - **Recommendation:** Track source + confidence, prefer higher confidence

3. Should we merge similar concepts?
   - **Recommendation:** Yes, use consolidation_threshold (0.85)

4. What about multi-sentence Wikipedia responses?
   - **Recommendation:** Extract from first 2-3 sentences (summary)

## Timeline Estimate

- **Phase A (Semantic Relations):** 2-3 hours
  - 1 hour: SemanticExtractor implementation
  - 1 hour: Integration with CLI and concept store
  - 30 min: Testing and refinement

- **Phase B (Entity Recognition):** 1-2 hours
- **Phase C (Vocabulary):** 1-2 hours  
- **Phase D (Syntactic Patterns):** 2-3 hours

**Total:** 6-10 hours for complete implementation
**MVP (Phase A only):** 2-3 hours
