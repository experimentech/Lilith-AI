# Layer 4 Restructuring: Pragmatic Response Generation

**Date:** 2025-12-02  
**Status:** Implementation Phase  
**Architecture:** BNN + Database (No LLM!)

## Problem: Layer 4 Doing Two Jobs

### What Layer 4 Currently Does (WRONG):

```
ResponseComposer (Layer 4):
  1. STORAGE (wrong job) - Stores verbatim Q&A pairs
     patterns.db: "What is Python?" → "Python is a programming language"
     Result: 1,235+ patterns, unbounded growth!
  
  2. COMPOSITION (right job, but broken) - Should compose responses
     BNN generates embeddings → UNUSED for retrieval
     Keyword matching instead of semantic similarity
```

**The Issue:** Pragmatics should **compose** responses from concepts, not **store** verbatim responses!

## Solution: Separate Storage from Composition

### Layer 4 Responsibilities (FIXED):

```
┌──────────────────────────────────────────────────────────────┐
│ Layer 4: PRAGMATIC/RESPONSE                                  │
│                                                               │
│ Job: COMPOSE natural responses from semantic concepts        │
│      NOT store verbatim Q&A pairs!                           │
│                                                               │
│ Components:                                                   │
│  1. BNN: Intent classification + Semantic encoding           │
│  2. DB #1: pragmatic_templates.db (~50 templates)           │
│     → Conversational patterns (HOW to say things)            │
│  3. DB #2: concept_store.db (unbounded)                     │
│     → Factual knowledge (WHAT to say)                        │
└──────────────────────────────────────────────────────────────┘
```

## Architecture: BNN + Two Databases

### Component 1: BNN (Already Exists!)

**File:** `response_composer.py`

**What it does:**
- Intent classification: "What is X?" → "definition_query"
- Semantic encoding: Embed user query for concept retrieval
- Pattern weighting: Score templates by context

**Status:** ✅ Working, just needs to use the new databases

### Component 2: Pragmatic Templates Database (NEW!)

**File:** `pragmatic_templates.py`

**What it stores:** Conversational patterns (~50 total, grows SLOWLY)

**Categories:**
```
1. Greetings (5)
   - "Hello! {offer_help}"
   - "Hi! {continue_previous_topic}"

2. Acknowledgments (10)
   - "I see. {elaboration}"
   - "That makes sense. {related_concept}"
   - "Interesting! {follow_up_question}"

3. Definitions (15)
   - "{concept} is {primary_property}."
   - "{concept} refers to {properties}. For example, {example}"
   - "In essence, {concept} {description}."

4. Continuations (10)
   - "Building on {previous_topic}, {new_info}"
   - "As you mentioned about {topic}, {related_fact}"
   - "That relates to {concept} because {connection}."

5. Elaborations (10)
   - "For example, {examples}"
   - "This is useful for {applications}."
   - "It's related to {related_concepts}."

6. Clarifications (5)
   - "Could you clarify - are you asking about {option1} or {option2}?"
```

**Key Insight:** These are **linguistic patterns**, not knowledge!
- Grows like grammar rules (slowly)
- Similar to Layer 3 (Syntax) templates
- Size: ~50 templates (vs 1,235+ patterns before)

### Component 3: Concept Store Database (ALREADY EXISTS!)

**File:** `production_concept_store.py`

**What it stores:** Factual knowledge with properties

**Example:**
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
      {"type": "is_type_of", "target": "artificial intelligence"},
      {"type": "has_application", "target": "pattern recognition"}
    ],
    "confidence": 0.90,
    "source": "taught"
  }
}
```

**Key Insight:** This is **semantic knowledge**, not patterns!
- Grows with learning (unbounded)
- Stores facts, not how to say them
- Separate from linguistic structure

## How It Works: Three-Step Composition

### Example Query: "What is machine learning?"

#### Step 1: Intent Recognition (BNN)

```python
# BNN classifies intent
query = "What is machine learning?"
intent = bnn_classifier.classify(query)  # → "definition_query"

# Select pragmatic template category
category = intent_to_category[intent]  # → "definition"
```

#### Step 2: Concept Retrieval (BNN + Concept DB)

```python
# BNN encodes concept
concept_query = extract_concept(query)  # → "machine learning"
embedding = bnn_encoder.encode(concept_query)

# Database lookup by semantic similarity
concept = concept_store.retrieve_similar(embedding, top_k=1)
# Returns:
# {
#   "term": "machine learning",
#   "properties": ["branch of AI", "learns from data"],
#   "relations": [...]
# }
```

#### Step 3: Template Composition (Pragmatic Templates)

```python
# Find best template for category with available data
available_slots = {
    "concept": concept.term,
    "primary_property": concept.properties[0],
    "elaboration": concept.properties[1]
}

template = pragmatic_store.match_best_template("definition", available_slots)
# Returns: "{concept} is {primary_property}. {elaboration}"

# Fill template
response = template.fill(available_slots)
# → "Machine learning is a branch of AI. It learns from data."
```

## Before vs After Comparison

### BEFORE (Broken):

```
User: "What is Python?"

Layer 4:
  1. Keyword match: "python" in patterns.db
  2. Return verbatim: "Python is a programming language"
  
Database size: 1,235 patterns (one per Q&A pair taught)
Problem: Can't compose novel responses, just retrieves stored text
```

### AFTER (Fixed):

```
User: "What is Python?"

Layer 4:
  Step 1 (Intent): BNN → "definition_query"
  Step 2 (Concept): BNN + DB → concept["Python"] 
          properties: ["programming language", "high-level", "interpreted"]
  Step 3 (Compose): Template + Concept → 
          "Python is a programming language. It's known for being high-level."

Database sizes:
  - pragmatic_templates.db: 50 templates (fixed)
  - concept_store.db: N concepts (grows with learning)
  
Benefit: Can compose NOVEL responses from learned concepts!
```

## Conversational Continuity (NEW!)

With ConversationHistory (Phase 1) + Pragmatic Templates (Phase 2):

```
User: "What is Python?"
Bot:  "Python is a programming language. It's known for being high-level."
[History: stores turn]

User: "Tell me more"
Bot:  "Building on what you said about Python, it's also great for data science."
      ↑ Uses continuation template + history reference

User: "What is Python?"  [REPEAT]
Bot:  [Detects repetition via ConversationHistory]
      [Uses acknowledgment template instead]
      "I see. As I mentioned, Python is a programming language..."
```

## Implementation Plan

### Phase 2A: Create Pragmatic Templates ✅

- [x] Create `pragmatic_templates.py`
- [x] Define 50 core templates across 6 categories
- [x] Add template matching logic
- [x] Add slot filling mechanism

### Phase 2B: Wire Into ResponseComposer (NEXT)

- [ ] Add `pragmatic_templates` parameter to `ResponseComposer.__init__()`
- [ ] Add `_compose_with_templates()` method
- [ ] Integrate with existing `compose_response()` flow
- [ ] Use ConversationHistory for continuation templates

### Phase 2C: Migrate Existing Patterns (LATER)

- [ ] Extract concepts from current patterns.db
- [ ] Store concepts in concept_store.db
- [ ] Remove verbatim patterns
- [ ] Shrink database from 1,235 patterns → ~50 templates + N concepts

## File Structure

```
lilith/
├── response_composer.py          # Layer 4 BNN (existing)
├── pragmatic_templates.py         # NEW: Conversational templates DB
├── production_concept_store.py    # Existing: Semantic concepts DB
├── template_composer.py           # Existing: Template filling logic
└── conversation_history.py        # Phase 1: Turn tracking

data/
├── pragmatic_templates.json       # ~50 templates (small!)
└── users/
    └── {user_id}/
        └── concepts.db            # Per-user learned concepts
```

## Why This Is Better

### Storage Efficiency

**Before:**
```
1,235 patterns × ~50 words/pattern = 61,750 words stored
Each new Q&A pair = new pattern (unbounded growth)
```

**After:**
```
50 templates × ~10 words/template = 500 words (templates)
N concepts × ~20 words/concept = depends on learning
Total: Much smaller, grows only with NEW concepts (not new phrasings)
```

### Generalization

**Before:**
```
Taught: "What is Python?" → "Python is a programming language"
Asked:  "What is Java?"   → [fallback, no pattern match]
```

**After:**
```
Taught: concept["Python"] = {"programming language", "high-level"}
        concept["Java"] = {"programming language", "object-oriented"}

Asked:  "What is Python?" → Template + Python concept → Response
Asked:  "What is Java?"   → Template + Java concept → Response
Asked:  "What is Rust?"   → Template + Rust concept → Response (if taught)

Same template, different concepts!
```

### Novel Composition

**Before:**
```
Can only return exact stored responses
No mixing concepts or building on previous statements
```

**After:**
```
User: "What about Python?"
Bot:  [continuation template] + [Python concept] + [conversation history]
      "Building on what you mentioned about programming, Python is..."

User: "How does it compare to Java?"
Bot:  [comparison template] + [Python vs Java concepts]
      "Python is similar to Java in being high-level, but differs in syntax..."
```

## Architecture Verification

### Layer 4 in Full Pipeline

```
Layer 1: INTAKE
  Input:  Raw text
  BNN:    Typo/normalization recognition
  DB:     Correction patterns
  Output: Normalized tokens → Layer 2

Layer 2: SEMANTIC
  Input:  Tokens
  BNN:    Concept embedding
  DB:     Semantic taxonomy
  Output: Concept representations → Layer 3 & 4

Layer 3: SYNTAX  
  Input:  Tokens + concepts
  BNN:    Grammar recognition
  DB:     Syntax patterns
  Output: Parse structure → Layer 4

Layer 4: PRAGMATIC
  Input:  Query + concepts + history
  BNN:    Intent classification + semantic encoding
  DB #1:  Pragmatic templates (~50, linguistic)
  DB #2:  Concept store (N, semantic)
  Output: Composed natural response → User
```

### BNN + Database Consistency ✅

Each layer has:
1. **BNN** for recognition/encoding
2. **Database(s)** for lookup
3. **Output** to next layer

Layer 4 is unique: Has **2 databases** because it bridges structure (templates) and content (concepts).

## Next Steps

1. ✅ Create `pragmatic_templates.py` with 50 templates
2. → Wire into `ResponseComposer` 
3. → Test with ConversationHistory integration
4. → Migrate existing patterns to concepts
5. → Document performance improvements

---

**Key Takeaway:** Layer 4 is NOT getting a new BNN! It's restructuring its databases to separate linguistic patterns (templates) from semantic knowledge (concepts). This is the correct architecture for the "Open Book Exam" design.
