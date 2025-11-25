# Compositional Response Architecture Design

**Version:** 1.0  
**Date:** 2024-11-25  
**Status:** Design Phase

## Problem Statement

Current response generation stores explicit response patterns (1300+ fragments), leading to:
- **Unbounded growth**: Each taught fact becomes a new pattern
- **Poor generalization**: Can't compose novel responses from learned concepts
- **Wrong abstraction layer**: Pragmatics doing storage when it should compose

### Example of Current Problem

```
Teaching Session:
User: "What is machine learning?"
Bot: [fallback]
User: "Machine learning is AI that learns from data"
→ STORES: Pattern("machine learning", "Machine learning is AI that learns from data")

User: "What is deep learning?"
Bot: [fallback]  
User: "Deep learning uses neural networks with many layers"
→ STORES: Pattern("deep learning", "Deep learning uses neural networks with many layers")

Result: 2 separate patterns when they share structure
```

## Proposed Solution: Three-Layer Composition

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INPUT                               │
│              "What is reinforcement learning?"               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: SYNTAX ANALYSIS (Already Exists)                  │
│  - Parse question structure                                  │
│  - Extract: template = "What is {concept}?"                  │
│  - Identify: intent = "definition_query"                     │
│  - Extract slots: {concept: "reinforcement learning"}        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: SEMANTIC CONCEPT RETRIEVAL (New Component)        │
│  - Encode concept: "reinforcement learning" → embedding      │
│  - Search ConceptStore for similar concepts                  │
│  - Find: {                                                   │
│      concept: "reinforcement learning",                      │
│      properties: ["trains agents", "trial and error",        │
│                   "rewards and penalties"],                  │
│      relations: [is_type_of("machine learning")]             │
│    }                                                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: TEMPLATE COMPOSITION (New Component)              │
│  - Match intent to response template                        │
│  - Template: "{concept} is {definition}. It {properties}"   │
│  - Fill slots from concept properties                       │
│  - Generate: "Reinforcement learning trains agents through  │
│              trial and error using rewards and penalties."   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  COMPOSED RESPONSE                           │
└─────────────────────────────────────────────────────────────┘
```

## Component Specifications

### 1. ConceptStore (New)

**Purpose:** Store semantic concepts with properties, not full response texts

**Data Structure:**
```python
@dataclass
class SemanticConcept:
    concept_id: str              # "concept_ml_001"
    term: str                    # "machine learning"
    embedding: np.ndarray        # BNN semantic embedding
    properties: List[str]        # ["learns from data", "uses algorithms"]
    relations: List[Relation]    # [is_type_of("AI"), has_property("adaptive")]
    examples: List[str]          # ["image recognition", "spam filtering"]
    confidence: float            # How reliable this knowledge is
    source: str                  # "taught", "wikipedia", "learned"
    usage_count: int            # Track for consolidation
    
@dataclass  
class Relation:
    relation_type: str          # "is_type_of", "has_property", "used_for"
    target: str                 # Target concept or property
    confidence: float
```

**Key Methods:**
```python
class ConceptStore:
    def add_concept(self, term: str, properties: List[str], source: str)
        """Extract and store concept from teaching"""
        
    def retrieve_similar(self, query_embedding: np.ndarray, top_k: int = 5)
        """Retrieve concepts by semantic similarity"""
        
    def get_properties(self, concept_id: str) -> List[str]
        """Get all properties of a concept"""
        
    def merge_similar_concepts(self, threshold: float = 0.90)
        """Consolidate nearly-identical concepts"""
```

**Storage Format:**
```json
{
  "concept_ml_001": {
    "term": "machine learning",
    "properties": [
      "branch of artificial intelligence",
      "enables computers to learn from data",
      "without explicit programming"
    ],
    "relations": [
      {"type": "is_type_of", "target": "artificial intelligence", "confidence": 0.95},
      {"type": "has_application", "target": "pattern recognition", "confidence": 0.85}
    ],
    "embedding": [0.234, -0.567, ...],
    "confidence": 0.90,
    "source": "taught",
    "usage_count": 5
  }
}
```

### 2. TemplateComposer (New)

**Purpose:** Generate responses by filling syntax templates with semantic concepts

**Data Structure:**
```python
@dataclass
class ResponseTemplate:
    template_id: str
    intent: str                  # "definition_query", "how_query", "why_query"
    syntax_pattern: str          # "What is {concept}?"
    response_template: str       # "{concept} is {definition}. {elaboration}"
    slots: List[str]            # ["concept", "definition", "elaboration"]
    examples: List[str]         # Example filled templates
    success_score: float
    
@dataclass
class CompositionPlan:
    """Plan for how to compose a response"""
    template: ResponseTemplate
    slot_fillers: Dict[str, str]  # {"concept": "machine learning", "definition": "..."}
    confidence: float
    fallback_needed: bool
```

**Key Methods:**
```python
class TemplateComposer:
    def match_intent(self, parsed_input: ParsedInput) -> ResponseTemplate
        """Find best template for user's intent"""
        
    def fill_template(self, template: ResponseTemplate, concept: SemanticConcept) -> str
        """Generate response by filling template slots"""
        
    def compose_response(self, user_input: str, concepts: List[SemanticConcept]) -> ComposedResponse
        """Main composition pipeline"""
```

### 3. Modified ResponseComposer (Existing - Enhanced)

**Changes:**
```python
class ResponseComposer:
    def __init__(self, ..., concept_store: ConceptStore, template_composer: TemplateComposer):
        self.concept_store = concept_store          # NEW
        self.template_composer = template_composer  # NEW
        self.fragment_store = fragment_store        # EXISTING (fallback)
        
    def compose_response(self, context: str, user_input: str) -> ComposedResponse:
        # NEW: Try compositional approach first
        if self.template_composer:
            compositional_response = self._try_compositional(user_input, context)
            if compositional_response and compositional_response.confidence > 0.7:
                return compositional_response
        
        # EXISTING: Fall back to pattern retrieval
        return self._retrieve_and_compose_patterns(context, user_input)
```

## Learning Flow Changes

### Current Learning (Storage-Based):
```
User teaches: "Machine learning is AI that learns from data"
↓
Extract pattern: (trigger="machine learning", response="Machine learning is AI...")
↓
Store in ResponseFragmentStore
↓
Result: 1 new pattern added (unbounded growth)
```

### New Learning (Compositional):
```
User teaches: "Machine learning is AI that learns from data"
↓
Parse syntax: "{concept} is {definition}"
↓
Extract concept: term="machine learning"
Extract properties: ["AI", "learns from data"]
↓
Store in ConceptStore with embedding
↓
Result: 1 concept with 2 properties (bounded growth)

Later teaching: "Machine learning uses algorithms to find patterns"
↓
Parse: "{concept} {action} {method}"
↓
Find existing concept: "machine learning"
Add property: "uses algorithms to find patterns"
↓
Result: Same concept, +1 property (consolidation!)
```

## Migration Strategy

### Phase 1: Parallel Implementation (Non-Breaking)

**Add new components alongside existing:**
- Implement `ConceptStore` 
- Implement `TemplateComposer`
- ResponseComposer tries compositional first, falls back to patterns
- Teaching goes to BOTH stores during transition

**Benefits:**
- No breaking changes
- Can compare approaches
- Gradual migration

**Metrics to Track:**
- Compositional success rate
- Pattern store growth rate (should slow)
- Response quality comparison

### Phase 2: Primary Compositional (Pattern Fallback)

**Make compositional the default:**
- New teachings only go to ConceptStore
- Pattern store becomes read-only cache
- Start pruning redundant patterns

**Consolidation Rules:**
- Concepts with >0.90 similarity → merge
- Patterns that match compositional output → mark redundant
- Remove patterns with usage_count < 2 after 100 interactions

### Phase 3: Full Integration

**Pattern store becomes transparent cache:**
- Successful compositions automatically cached
- Pattern store used only for edge cases
- ConceptStore is primary knowledge base

## Proof of Concept Scope

### Minimal PoC to Validate Approach:

**What to Build:**
1. **Simple ConceptStore** (in-memory only, no persistence)
   - Add concept with properties
   - Retrieve by embedding similarity
   - Just 2-3 test concepts

2. **Basic TemplateComposer** (1-2 templates)
   - Definition query template: "What is X?" → "{X} is {definition}"
   - How query template: "How does X work?" → "{X} works by {mechanism}"

3. **Integration Test** (no production changes)
   - Standalone script showing:
     * Teach 2 concepts
     * Ask questions
     * Compare compositional vs pattern-based responses

**Success Criteria:**
- ✓ Can store concepts with properties
- ✓ Can retrieve similar concepts by embedding
- ✓ Can compose novel response from template + concept
- ✓ Compositional response quality ≥ pattern-based
- ✓ Adding 2nd similar concept doesn't double storage

**Out of Scope for PoC:**
- Full syntax integration
- Response learning/feedback
- Pattern consolidation
- Production integration

## Technical Considerations

### 1. Concept Extraction

**Challenge:** How to parse "Machine learning is AI that learns from data" into structured concept?

**Approach:**
```python
# Simple heuristic parser (for PoC)
def extract_concept_from_teaching(text: str) -> Tuple[str, List[str]]:
    # Pattern: "{term} is {definition}"
    if " is " in text:
        parts = text.split(" is ", 1)
        term = parts[0].strip()
        definition = parts[1].strip()
        
        # Split definition into properties (simple: by punctuation/conjunctions)
        properties = split_properties(definition)
        return term, properties
```

**Future:** Use syntax stage for robust parsing

### 2. Template Selection

**Challenge:** How to pick the right template for a query?

**Approach:**
```python
# Simple intent classification (for PoC)
TEMPLATES = {
    "what_is": {
        "patterns": ["what is", "what are", "define"],
        "template": "{concept} is {definition}."
    },
    "how_does": {
        "patterns": ["how does", "how do", "explain how"],
        "template": "{concept} works by {mechanism}."
    }
}

def match_template(query: str) -> ResponseTemplate:
    query_lower = query.lower()
    for intent, config in TEMPLATES.items():
        if any(pattern in query_lower for pattern in config["patterns"]):
            return config["template"]
    return None
```

**Future:** Use BNN intent classifier

### 3. Semantic Similarity Threshold

**Challenge:** When are two concepts "the same"?

**Thresholds:**
- `>= 0.95`: Definitely same (merge immediately)
- `0.85-0.95`: Very similar (consolidate properties)
- `0.70-0.85`: Related (keep separate, track relation)
- `< 0.70`: Different concepts

### 4. Property Representation

**Challenge:** How to store "learns from data" vs "uses data to learn"?

**Approach for PoC:**
- Store as strings (simple)
- Rely on semantic embedding similarity

**Future Enhancement:**
- Normalize to canonical forms
- Extract structured relations (subject-verb-object)

## Success Metrics

### Quantitative:
1. **Storage Efficiency**
   - Current: N teachings → N patterns
   - Target: N teachings → ~N/3 concepts (consolidation)

2. **Retrieval Quality**
   - Compositional response contains expected keywords: ≥90%
   - Response grammatically correct: ≥95%

3. **Generalization**
   - Novel queries answered from taught concepts: ≥70%
   - Example: Teach "ML" + "supervised learning" → Answer "What types of ML exist?"

### Qualitative:
1. **Compositionality**
   - Can combine multiple concept properties
   - Generates novel sentences not in training

2. **Coherence**  
   - Response matches query intent
   - No nonsensical combinations

3. **Maintainability**
   - Concept store easier to inspect than pattern fragments
   - Clear separation of knowledge (concepts) vs structure (templates)

## Risks & Mitigations

### Risk 1: Template Brittleness
**Issue:** Limited templates can't handle all queries
**Mitigation:** Fall back to pattern-based for unmatched intents

### Risk 2: Concept Extraction Quality
**Issue:** Heuristic parsing may miss nuances
**Mitigation:** Start with high-confidence patterns (" is ", " has ", etc.)

### Risk 3: Worse Response Quality
**Issue:** Compositional might be less natural than stored text
**Mitigation:** A/B test, keep both approaches during migration

### Risk 4: Complexity Increase
**Issue:** More components = more maintenance
**Mitigation:** Gradual rollout, comprehensive tests, clear interfaces

## Alternative Approaches Considered

### Alternative 1: Pure Neural Composition
**Approach:** Train seq2seq model to generate responses
**Pros:** True generalization, no templates
**Cons:** Requires large dataset, harder to debug, loses explainability

### Alternative 2: Graph-Based Knowledge Store  
**Approach:** Store concepts in knowledge graph
**Pros:** Rich relational reasoning
**Cons:** Complex queries, overhead for simple lookups

### Alternative 3: Just Prune Patterns
**Approach:** Keep current architecture, aggressive deduplication
**Pros:** Minimal changes
**Cons:** Doesn't solve root problem, still unbounded growth

**Decision:** Compositional approach balances explainability, efficiency, and generalization

## Next Steps

### 1. Design Review
- [ ] Review this document
- [ ] Identify gaps or concerns
- [ ] Refine approach if needed

### 2. PoC Implementation
- [ ] Implement ConceptStore (minimal)
- [ ] Implement TemplateComposer (2 templates)
- [ ] Create standalone test script
- [ ] Run validation tests

### 3. PoC Evaluation
- [ ] Compare compositional vs pattern quality
- [ ] Measure storage efficiency
- [ ] Test generalization capability
- [ ] Decision: proceed vs revise

### 4. Production Implementation (if PoC succeeds)
- [ ] Full ConceptStore with persistence
- [ ] Integration with syntax stage
- [ ] Migration strategy for existing patterns
- [ ] Comprehensive testing

## Open Questions

1. **Q:** Should concepts store full sentences or just key phrases?
   **A:** TBD in PoC - test both approaches

2. **Q:** How to handle multi-sentence teachings?
   **A:** Extract multiple properties, link via relations

3. **Q:** What if compositional response is grammatically awkward?
   **A:** Fall back to pattern if validation fails

4. **Q:** How to preserve exact phrasings when important?
   **A:** Flag "canonical" properties that shouldn't be paraphrased

5. **Q:** When to consolidate vs keep separate?
   **A:** Tune threshold based on PoC results

---

## Appendix: Example Comparison

### Current Approach (Pattern Storage):

```
Teach: "Supervised learning uses labeled data"
Store: Pattern(trigger="supervised learning", response="Supervised learning uses labeled data")

Teach: "Unsupervised learning finds patterns in unlabeled data"  
Store: Pattern(trigger="unsupervised learning", response="Unsupervised learning finds patterns in unlabeled data")

Query: "What types of machine learning exist?"
Result: [fallback - no pattern matches]
```

### Compositional Approach:

```
Teach: "Supervised learning uses labeled data"
Store: Concept(term="supervised learning", properties=["uses labeled data"], relation=[is_type_of("machine learning")])

Teach: "Unsupervised learning finds patterns in unlabeled data"
Store: Concept(term="unsupervised learning", properties=["finds patterns in unlabeled data"], relation=[is_type_of("machine learning")])

Query: "What types of machine learning exist?"
Retrieve: Concepts where relation=is_type_of("machine learning")
Compose: "The main types of machine learning are supervised learning and unsupervised learning."
Result: [novel composition from stored concepts ✓]
```

This demonstrates the key advantage: **compositional reasoning over stored knowledge** rather than exact pattern matching.
