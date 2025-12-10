# Modal Routing Architecture

## Problem Statement

Mathematical expressions pollute the linguistic database:
- Math symbols ("+", "×", "÷", "=") don't behave like natural language
- No generalization (learning "2+2=4" doesn't help with "3+3")
- Concept confusion in semantic space
- Template mismatch (computation vs retrieval)

## Solution: Modal Classifier + Specialized Backends

### Architecture

```
User Input: "what is 2 + 2"
    │
    ▼
ModalClassifier (lightweight heuristics)
    │
    ├─────────┬──────────┬──────────┬──────────┐
    │         │          │          │          │
    ▼         ▼          ▼          ▼          ▼
Linguistic  Math     Code      Visual    Audio
Backend   Backend  Backend   Backend   Backend
    │         │          │          │          │
    │         │          │          │          │
Response  Computed  Executed  Analyzed  Processed
Fragment  Result    Code      Image     Sound
Store                │
    │         │          │          │          │
    └─────────┴──────────┴──────────┴──────────┘
                      │
                      ▼
              Unified Response
           (with modality metadata)
```

### Modal Classification Rules

**1. Mathematical Modality**
Triggers:
- Contains operators: `+`, `-`, `×`, `*`, `/`, `÷`, `^`, `=`
- Numeric patterns: `\d+\s*[+\-*/^]\s*\d+`
- Math keywords: "calculate", "compute", "solve", "evaluate"
- Function notation: `sin()`, `cos()`, `log()`, `sqrt()`

Examples:
- "what is 2 + 2" → Math backend
- "calculate the square root of 16" → Math backend
- "solve x^2 + 2x + 1 = 0" → Math backend

**2. Code Modality**
Triggers:
- Programming keywords: `function`, `class`, `def`, `return`, `if`
- Code syntax: `{...}`, `[...]`, `import`, `from`
- Language hints: "in Python", "JavaScript code", "write a function"

Examples:
- "write a function to sort an array" → Code backend
- "how do I reverse a list in Python" → Code backend (or Linguistic if asking about concept)

**3. Linguistic Modality** (Default)
Triggers:
- Natural language queries without special syntax
- Conceptual questions: "what is", "how does", "why"
- No math operators or code syntax

Examples:
- "what is machine learning" → Linguistic backend
- "how does supervised learning work" → Linguistic backend

**4. Visual Modality** (Future)
Triggers:
- Image input detected
- Keywords: "show me", "what's in this picture", "draw"

**5. Audio Modality** (Future)
Triggers:
- Audio input detected
- Keywords: "play", "sound like", "music"

### Implementation Strategy

#### Phase 1: Math Backend (Immediate)
1. **ModalClassifier**: Lightweight heuristic-based classifier
2. **MathBackend**: Symbolic computation engine
3. **Integration**: Route math queries away from linguistic database
4. **Fallback**: If math backend fails, don't learn the pattern

#### Phase 2: Code Backend
1. **CodeBackend**: Code execution sandbox (with safety!)
2. **Language detection**: Python, JavaScript, etc.
3. **Conceptual vs Executable**: "what is a function" (linguistic) vs "write a function" (code)

#### Phase 3: Multi-Modal Unification
1. **Cross-modal concepts**: "The function f(x) = x² is a parabola" (math + visual)
2. **Modal embeddings**: Separate embedding spaces per modality
3. **Unified retrieval**: Search across all modalities

---

## Math Backend Design

### Capabilities

**1. Arithmetic**
```python
"2 + 2" → 4
"15 * 7" → 105
"100 / 4" → 25.0
```

**2. Algebra**
```python
"solve x + 5 = 10" → x = 5
"expand (x + 2)^2" → x² + 4x + 4
"factor x² - 4" → (x - 2)(x + 2)
```

**3. Calculus** (using SymPy)
```python
"derivative of x^2" → 2x
"integral of sin(x)" → -cos(x)
```

**4. Matrices** (using NumPy)
```python
"multiply [[1,2],[3,4]] and [[5,6],[7,8]]" → [[19,22],[43,50]]
```

### Implementation

```python
class MathBackend:
    """Symbolic computation backend for mathematical queries"""
    
    def __init__(self):
        self.parser = MathExpressionParser()
        self.solver = SymbolicSolver()  # SymPy wrapper
    
    def can_handle(self, query: str) -> bool:
        """Check if query is mathematical"""
        # Heuristics: operators, numbers, math keywords
        return self._has_math_operators(query) or self._has_math_keywords(query)
    
    def compute(self, query: str) -> Optional[MathResult]:
        """Execute mathematical computation"""
        try:
            # Parse expression
            expr = self.parser.parse(query)
            
            # Solve/evaluate
            result = self.solver.evaluate(expr)
            
            return MathResult(
                expression=expr,
                result=result,
                steps=self.solver.get_steps(),  # Show work
                confidence=1.0  # Exact computation
            )
        except Exception as e:
            return None  # Can't compute
```

### Safety Considerations

**Math Backend is SAFE**
- Pure symbolic computation (no code execution)
- Uses SymPy (sandboxed symbolic math)
- No file system access
- No network access
- Timeout limits for complex expressions

**Code Backend NEEDS SANDBOXING**
- Docker container isolation
- Resource limits (CPU, memory, time)
- No network access
- Read-only file system
- Whitelist of safe libraries

---

## Modal Classifier Implementation

### Heuristic-Based (Phase 1)

```python
class ModalClassifier:
    """Classify input modality using heuristics"""
    
    def classify(self, query: str, context: Optional[Dict] = None) -> Modality:
        """
        Classify query modality.
        
        Returns: Modality enum (LINGUISTIC, MATH, CODE, VISUAL, AUDIO)
        """
        query_lower = query.lower()
        
        # 1. Math modality (highest priority - most specific)
        if self._is_math(query_lower):
            return Modality.MATH
        
        # 2. Code modality
        if self._is_code(query_lower):
            return Modality.CODE
        
        # 3. Visual modality (requires image input)
        if context and 'image' in context:
            return Modality.VISUAL
        
        # 4. Audio modality (requires audio input)
        if context and 'audio' in context:
            return Modality.AUDIO
        
        # 5. Default: Linguistic
        return Modality.LINGUISTIC
    
    def _is_math(self, query: str) -> bool:
        """Detect mathematical queries"""
        # Check for operators
        math_operators = ['+', '-', '*', '×', '/', '÷', '^', '=', '<', '>']
        if any(op in query for op in math_operators):
            # Verify it's not just punctuation
            if re.search(r'\d+\s*[+\-*/^×÷]\s*\d+', query):
                return True
        
        # Check for math keywords
        math_keywords = ['calculate', 'compute', 'solve', 'evaluate', 
                        'derivative', 'integral', 'factor', 'expand',
                        'simplify', 'square root', 'logarithm']
        if any(kw in query for kw in math_keywords):
            return True
        
        # Check for function notation
        if re.search(r'(sin|cos|tan|log|sqrt|abs)\s*\(', query):
            return True
        
        return False
    
    def _is_code(self, query: str) -> bool:
        """Detect code-related queries"""
        # Programming keywords
        code_keywords = ['function', 'class', 'def', 'return', 'import',
                        'variable', 'loop', 'array', 'list', 'dict']
        
        # Language hints
        lang_hints = ['python', 'javascript', 'java', 'c++', 'code',
                     'program', 'script']
        
        # Intent keywords
        intent_keywords = ['write a', 'create a', 'implement', 'code for']
        
        return (any(kw in query for kw in code_keywords) or
                any(hint in query for hint in lang_hints) or
                any(intent in query for intent in intent_keywords))
```

### BioNN-Based Classifier (Phase 2 - Future)

```python
class BNNModalClassifier:
    """Neural modal classifier using BioNN embeddings"""
    
    def __init__(self, encoder):
        self.encoder = encoder
        self.modal_prototypes = self._init_prototypes()
    
    def classify(self, query: str) -> Tuple[Modality, float]:
        """
        Classify using BioNN semantic similarity to modal prototypes.
        
        Returns: (modality, confidence)
        """
        query_emb = self.encoder.encode(query.split())
        
        best_modality = Modality.LINGUISTIC
        best_similarity = 0.0
        
        for modality, prototype_emb in self.modal_prototypes.items():
            sim = cosine_similarity(query_emb, prototype_emb)
            if sim > best_similarity:
                best_similarity = sim
                best_modality = modality
        
        return best_modality, best_similarity
    
    def _init_prototypes(self) -> Dict[Modality, np.ndarray]:
        """Initialize modal prototypes from examples"""
        examples = {
            Modality.MATH: [
                "calculate 2 plus 2",
                "solve for x",
                "derivative of x squared"
            ],
            Modality.CODE: [
                "write a function to sort",
                "implement binary search",
                "create a class for user"
            ],
            Modality.LINGUISTIC: [
                "what is machine learning",
                "how does supervised learning work",
                "explain neural networks"
            ]
        }
        
        prototypes = {}
        for modality, modal_examples in examples.items():
            embeddings = [self.encoder.encode(ex.split()) for ex in modal_examples]
            prototypes[modality] = np.mean(embeddings, axis=0)
        
        return prototypes
```

---

## Integration with ResponseComposer

### Modified Response Pipeline

```python
class ResponseComposer:
    def __init__(
        self,
        fragment_store: ResponseFragmentStore,
        conversation_state: ConversationState,
        concept_store: Optional[ProductionConceptStore] = None,
        math_backend: Optional[MathBackend] = None,  # NEW
        code_backend: Optional[CodeBackend] = None,   # NEW
        enable_modal_routing: bool = True             # NEW
    ):
        self.fragments = fragment_store
        self.state = conversation_state
        self.concept_store = concept_store
        
        # Modal backends
        self.modal_classifier = ModalClassifier()
        self.math_backend = math_backend
        self.code_backend = code_backend
        self.enable_modal_routing = enable_modal_routing
    
    def compose_response(
        self,
        context: str,
        user_input: str = "",
        topk: int = 5
    ) -> ComposedResponse:
        """Generate response with modal routing"""
        
        # 1. Classify modality
        if self.enable_modal_routing:
            modality = self.modal_classifier.classify(user_input)
            
            # 2. Route to appropriate backend
            if modality == Modality.MATH and self.math_backend:
                return self._compose_math_response(user_input)
            
            elif modality == Modality.CODE and self.code_backend:
                return self._compose_code_response(user_input)
        
        # 3. Default: Linguistic composition (existing logic)
        if self.composition_mode == "parallel" and self.concept_store:
            return self._compose_parallel(context, user_input, topk)
        
        return self._compose_from_patterns_internal(...)
    
    def _compose_math_response(self, query: str) -> ComposedResponse:
        """Generate response using math backend"""
        result = self.math_backend.compute(query)
        
        if result:
            # Format mathematical response
            response_text = self._format_math_result(result)
            
            return ComposedResponse(
                text=response_text,
                fragment_ids=[f"math_computed"],
                composition_weights=[1.0],
                coherence_score=1.0,
                confidence=result.confidence,
                modality=Modality.MATH,  # NEW field
                is_fallback=False
            )
        else:
            # Math backend couldn't compute - fallback to linguistic
            return self.compose_response(query, use_modal_routing=False)
    
    def _format_math_result(self, result: MathResult) -> str:
        """Format mathematical result as natural language"""
        if result.steps:
            # Show work
            steps_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(result.steps))
            return f"{result.expression} = {result.result}\n\nSteps:\n{steps_text}"
        else:
            # Simple result
            return f"{result.expression} = {result.result}"
```

---

## Database Protection

### Don't Learn Mathematical Patterns

```python
def record_conversation_outcome(self, success: bool):
    """Record outcome, but skip for non-linguistic modalities"""
    
    # Skip learning for math/code responses
    if self.last_response and hasattr(self.last_response, 'modality'):
        if self.last_response.modality in [Modality.MATH, Modality.CODE]:
            # Don't pollute linguistic database with symbolic patterns
            return
    
    # Normal learning for linguistic responses
    if self.last_approach == 'pattern':
        self.metrics['pattern_success'] += 1 if success else 0
    # ... rest of existing logic
```

### Separate Modal Databases (Future)

```python
# Each modality gets its own storage
class MultiModalStore:
    def __init__(self):
        self.linguistic_store = ResponseFragmentStore()  # Existing
        self.math_patterns = MathPatternStore()          # Symbolic patterns
        self.code_snippets = CodeSnippetStore()          # Code templates
        self.visual_concepts = VisualConceptStore()      # Image embeddings
```

---

## Benefits

### 1. **Database Cleanliness**
- Math/code don't pollute linguistic embeddings
- Semantic space remains coherent
- Better retrieval for natural language

### 2. **Proper Generalization**
- Math: `2+2=4` generalizes to all arithmetic via symbolic computation
- Code: Function templates work for any input
- Linguistic: Conceptual understanding of "what is X"

### 3. **Accuracy**
- Math: Exact symbolic computation (no hallucination)
- Code: Validated execution (no syntax errors)
- Linguistic: Retrieved patterns (existing quality)

### 4. **Extensibility**
- Easy to add new modalities (visual, audio)
- Each modality optimized for its domain
- Unified interface for all backends

---

## Implementation Priority

**Phase 1: Math Backend** (Immediate - Solves your concern)
1. Create `ModalClassifier` with heuristics
2. Create `MathBackend` using SymPy
3. Integrate with `ResponseComposer`
4. Add modality field to `ComposedResponse`
5. Skip learning for math responses
6. Test: arithmetic, algebra, basic calculus

**Phase 2: Code Backend** (Medium priority)
1. Create `CodeBackend` with sandboxing
2. Add code detection to `ModalClassifier`
3. Conceptual vs executable distinction
4. Test: Python snippets, explanations

**Phase 3: Multi-Modal Unification** (Future)
1. Visual + audio backends
2. Cross-modal concepts
3. Unified search across modalities
4. BioNN-based modal classification

---

## Testing Strategy

```python
def test_modal_routing():
    # Math queries don't pollute linguistic database
    composer.compose_response("what is 2 + 2")
    assert fragment_store.pattern_count == 0  # Not learned
    assert response.modality == Modality.MATH
    assert response.text == "2 + 2 = 4"
    
    # Linguistic queries work normally
    composer.compose_response("what is machine learning")
    assert fragment_store.pattern_count > 0  # Learned
    assert response.modality == Modality.LINGUISTIC
```

---

## Summary

**Your concern is valid!** Mathematical expressions would indeed pollute the linguistic database.

**Solution**: Modal routing architecture
- Classify queries by modality (math, code, linguistic)
- Route to specialized backends
- Don't learn non-linguistic patterns
- Exact computation for math (no hallucination)

**Next steps**: Shall I implement the Math Backend for Phase 1?
