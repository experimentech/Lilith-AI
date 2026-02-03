# Conversational Architecture: Missing Stages for True Dialogue

> **Status**: Design Document / TODO  
> **Created**: 2026-02-01  
> **Context**: V2 has superior learning, recall, and reasoning capabilities but lacks the architectural stages needed for natural conversation.

## The Core Problem

### LLM Approach
LLMs have a single unified mechanism: predict the next token based on all prior context. The "goal" of generating coherent, contextually appropriate continuations is implicit in training.

### Lilith's Approach
Lilith uses decoupled stages (perception → reasoning → generation) with physics-based deliberation. This creates genuine thinking but leaves a critical gap:

**There's no stage that generates *communication intent*.**

When `GenerativeSystem.compose()` has no inference or extracted knowledge, it falls back to:
```python
return "I am listening. Please provide more context."
```

This happens because nothing is asking: *"What should I say, and why?"*

## Current Pipeline (V2)

```
Input → LinguisticProcessor → CognitiveStage → GenerativeSystem → Output
                                    ↓
                            ReasoningStage (physics deliberation)
                            AffectiveSystem (mood/personality)
                            KnowledgeService (external retrieval)
```

### What Works Well
- **Learning**: Hebbian plasticity, concept grounding, relation extraction
- **Recall**: Graph traversal, semantic retrieval, knowledge augmentation
- **Reasoning**: Physics-based deliberation, concept chaining, abstraction
- **Affect**: Mood tracking, personality traits, drives (curiosity, warmth)

### What's Missing
The **bridge between having knowledge and communicating it purposefully**.

---

## Missing Stages

### 1. Discourse Management Stage (The Conductor)

*"Where are we in the conversation, and what's expected?"*

| Responsibility | Current State |
|----------------|---------------|
| Track dialogue state (opening, topic, closing) | ❌ Not tracked |
| Identify user's communicative intent | ❌ Pattern matching only |
| Determine appropriate response type | ❌ Hard-coded fallbacks |
| Manage topic continuity | ❌ No cross-turn linking |
| Handle anaphora ("it", "that", "they") | ❌ No resolution |

#### Data Structure

```python
@dataclass
class DialogueState:
    phase: str  # "opening", "topic_exploration", "teaching", "closing"
    current_topic: Optional[str]
    topic_stack: List[str]  # For nested topics (user digresses, then returns)
    last_n_acts: List[DialogueAct]  # User's communicative acts
    pending_obligations: List[str]  # "answer_question", "acknowledge_teaching"
    turn_count: int
    last_user_intent: str  # "ask", "inform", "greet", "correct", "elaborate"

@dataclass  
class DialogueAct:
    act_type: str  # "question", "statement", "greeting", "feedback", "correction"
    topic: Optional[str]
    referents: List[str]  # Entities mentioned
    timestamp: float
```

#### Key Functions

```python
class DiscourseManager:
    def update_state(self, user_input: str, extracted_info: Dict) -> DialogueState:
        """Update dialogue state based on new user input."""
        
    def identify_user_intent(self, user_input: str) -> str:
        """Classify what the user is trying to do communicatively."""
        
    def resolve_anaphora(self, text: str, context: DialogueState) -> str:
        """Replace pronouns with their referents from context."""
        
    def get_pending_obligations(self) -> List[str]:
        """What must the response address? (e.g., answer a question)"""
        
    def track_topic_shift(self, new_topic: str) -> None:
        """Handle topic changes, maintaining continuity."""
```

---

### 2. Communication Planning Stage (The Intent Generator)

*"Given what I know and the conversation state, what should I communicate?"*

This is the **missing "motivation" layer**. The `AffectiveSystem` has `curiosity_drive` but nothing consumes it to generate actual communicative goals.

#### Core Functions

| Function | Description |
|----------|-------------|
| **Goal Selection** | "Should I inform, ask, acknowledge, elaborate, or redirect?" |
| **Content Selection** | "Which concepts from working memory should I express?" |
| **Stance Selection** | "What attitude should I convey?" (certainty, curiosity, empathy) |
| **Initiative Decision** | "Should I take conversational lead or yield to user?" |

#### Data Structure

```python
@dataclass
class CommunicationPlan:
    primary_goal: str  # "inform", "ask", "acknowledge", "elaborate", "clarify"
    content_concepts: List[str]  # Concept IDs to express
    content_relations: List[Tuple[str, str, str]]  # Relations to verbalize
    stance: str  # "certain", "curious", "empathetic", "uncertain", "neutral"
    follow_up_goal: Optional[str]  # Secondary goal (e.g., ask after informing)
    follow_up_topic: Optional[str]  # What to ask about
    discourse_markers: List[str]  # "Actually...", "Speaking of...", "I wonder..."
    take_initiative: bool  # Should Lilith drive the conversation?
```

#### Planning Logic

```python
class CommunicationPlanner:
    def plan_response(
        self,
        dialogue_state: DialogueState,
        working_memory: Dict[str, ActivatedConcept],
        affective_state: Dict[str, float],
        inferences: List[Inference],
        external_knowledge: List[KnowledgeFragment],
    ) -> CommunicationPlan:
        """
        Generate a communication plan based on:
        
        1. Obligation Fulfillment
           - What does the dialogue require? (answer question, acknowledge teaching)
           - This takes priority - must address user's needs first
           
        2. Content Availability  
           - What do I know that's relevant? (working memory, inferences)
           - What did I just learn? (external knowledge)
           - What connections can I share? (concept chains)
           
        3. Affective Drives
           - High curiosity → ask follow-up questions
           - High warmth → add empathetic acknowledgment
           - High confidence → express certainty
           
        4. Engagement Optimization
           - Would a follow-up question maintain engagement?
           - Should I elaborate or keep it brief?
           - Is the user teaching or learning?
        """
```

#### Drive → Goal Mapping

| Affective Drive | Communication Goal |
|-----------------|-------------------|
| `curiosity_drive > 0.7` | Add follow-up question |
| `warmth > 0.6` | Include acknowledgment/empathy |
| `confidence > 0.7` | Express certainty in statements |
| `confidence < 0.4` | Hedge, express uncertainty |
| Low engagement history | Take more initiative |

---

### 3. Compositional Realization Stage (The Composer)

*"How do I turn this plan into actual language?"*

This is fundamentally different from `GenerativeSystem` which does template filling. This stage builds language **compositionally from concept relations**.

#### Key Insight

The composition should use:
- **Concept relations** for content (the WHAT)
- **Pragmatics** for intent realization (the HOW)  
- **Affective state** for coloring (the TONE)
- **Discourse state** for coherence (the FLOW)

#### Process

```python
class CompositionalRealizer:
    def realize(self, plan: CommunicationPlan, context: Dict) -> str:
        """
        Build response from plan, not templates.
        
        Example:
        Plan: inform(concepts=["dolphin", "mammal"], stance="curious")
        
        Step 1: Retrieve relation
                dolphin --is_a→ mammal
                
        Step 2: Choose syntactic frame for relation type
                is_a → "SUBJECT is a TYPE" or "SUBJECT is a kind of TYPE"
                
        Step 3: Apply stance modifier
                curious → "Interestingly, ..." or "I find it fascinating that..."
                
        Step 4: Add follow-up from plan
                follow_up_goal="ask" → "Have you seen them in the wild?"
                
        Step 5: Apply discourse markers
                markers=["Speaking of which"] → prepend if topic-related
                
        Output: "Interestingly, dolphins are mammals. Have you seen them in the wild?"
        """
```

#### Syntactic Frames (Learned, Not Hard-coded)

```python
# These should be learned from input, stored in graph
SYNTACTIC_FRAMES = {
    "is_a": [
        "{subject} is a {object}",
        "{subject} is a type of {object}",
        "{subject}s are {object}s",
        "A {subject} is {object}",
    ],
    "has_property": [
        "{subject} is {property}",
        "{subject} has {property}",
        "{subject} are known for being {property}",
    ],
    "part_of": [
        "{subject} is part of {object}",
        "{subject} belongs to {object}",
    ],
}

STANCE_MODIFIERS = {
    "certain": ["", "Indeed, ", "Yes, "],
    "curious": ["Interestingly, ", "I find it fascinating that ", ""],
    "uncertain": ["I think ", "It seems that ", "Perhaps "],
    "empathetic": ["I understand. ", "That makes sense. ", "I see what you mean. "],
}

FOLLOW_UP_FRAMES = {
    "ask_experience": ["Have you {verb} {topic}?", "What's your experience with {topic}?"],
    "ask_elaboration": ["Can you tell me more about {topic}?", "What else about {topic}?"],
    "ask_opinion": ["What do you think about {topic}?", "How do you feel about {topic}?"],
}
```

---

### 4. Self-Monitoring Stage (The Editor)

*"Does my planned response make sense before I output it?"*

Before outputting, evaluate the draft response:

| Check | Question |
|-------|----------|
| **Coherence** | Does this fit the conversation context? |
| **Informativeness** | Am I adding value or just repeating? |
| **Appropriateness** | Is this the right tone/length for the situation? |
| **Completeness** | Did I address the user's actual need? |
| **Repetition** | Have I said this exact thing recently? |

```python
class SelfMonitor:
    def evaluate(self, draft: str, plan: CommunicationPlan, state: DialogueState) -> float:
        """Return quality score 0-1."""
        
    def should_revise(self, score: float) -> bool:
        """Below threshold → trigger revision."""
        
    def suggest_revision(self, draft: str, issues: List[str]) -> str:
        """Attempt to fix identified issues."""
```

This creates a "generate → evaluate → revise" loop internally.

---

## Complete Pipeline Architecture

```
                          ┌─────────────────────┐
                          │   AffectiveSystem   │
                          │  (drives: curiosity,│
                          │   warmth, etc.)     │
                          └──────────┬──────────┘
                                     │ modulates
                                     ▼
┌──────────────┐    ┌──────────────────────────────────────────┐
│  User Input  │───▶│       Discourse Management               │
└──────────────┘    │  • Track dialogue state                  │
                    │  • Identify user intent                  │
                    │  • Resolve anaphora                      │
                    │  • Determine pending obligations         │
                    └──────────────────┬───────────────────────┘
                                       │ DialogueState
                                       ▼
                    ┌──────────────────────────────────────────┐
                    │       CognitiveStage (existing)          │
                    │  • Perception → Embedding                │
                    │  • Grounding → Concept IDs               │
                    │  • ReasoningStage deliberation           │
                    │  • Knowledge retrieval                   │
                    └──────────────────┬───────────────────────┘
                                       │ • activated_concepts
                                       │ • inferences  
                                       │ • external_knowledge
                                       ▼
                    ┌──────────────────────────────────────────┐
                    │       Communication Planning             │◀── AffectiveSystem
                    │  • Goal selection (inform/ask/ack)       │    (curiosity, warmth,
                    │  • Content selection from memory         │     confidence)
                    │  • Stance selection                      │
                    │  • Initiative decision                   │
                    │  • Follow-up generation                  │
                    └──────────────────┬───────────────────────┘
                                       │ CommunicationPlan
                                       ▼
                    ┌──────────────────────────────────────────┐
                    │       Compositional Realization          │
                    │  • Build from concepts + relations       │
                    │  • Apply syntactic frames                │
                    │  • Apply stance modifiers                │
                    │  • Add discourse markers                 │
                    │  • Generate compositionally              │
                    └──────────────────┬───────────────────────┘
                                       │ draft_response
                                       ▼
                    ┌──────────────────────────────────────────┐
                    │           Self-Monitoring                │
                    │  • Coherence check                       │
                    │  • Informativeness check                 │
                    │  • Revision if needed                    │
                    └──────────────────┬───────────────────────┘
                                       │ final_response
                                       ▼
                              ┌──────────────┐
                              │    Output    │
                              └──────────────┘
```

---

## Implementation Priority

| Priority | Stage | Rationale | Estimated LOC |
|----------|-------|-----------|---------------|
| **1** | Discourse Management | Foundation - need to understand dialogue context first | ~200 |
| **2** | Communication Planning | Core missing piece - the "why speak" and "what goal" | ~300 |
| **3** | Compositional Realization | Replace template-assembly with genuine composition | ~250 |
| **4** | Self-Monitoring | Quality layer - can be added after core works | ~150 |

---

## Design Principles

### 1. No Baked-in Patterns
Responses should emerge from:
- Concept relations (learned from input)
- Syntactic frames (learned from exposure)
- Pragmatic rules (learned from engagement feedback)

NOT from:
- Hard-coded response templates
- Pattern → response mappings
- Static phrase banks

### 2. Motivation from Drives
The `AffectiveSystem` already has `curiosity_drive`, `warmth`, `confidence`. These should:
- Influence goal selection (high curiosity → ask questions)
- Modulate stance (high confidence → certainty)
- Affect initiative (low engagement → take more lead)

### 3. Compositional Generation
Build sentences like building Lego:
1. Select concepts to express
2. Retrieve relations between them
3. Choose syntactic frame for relation type
4. Compose with modifiers and connectors
5. Ensure grammaticality

### 4. Obligation-Driven Responses
The dialogue state creates **obligations**:
- User asked a question → must provide answer or indicate inability
- User taught something → must acknowledge learning
- User greeted → must reciprocate

These obligations are non-optional and take priority over initiative.

---

## Relation to Existing Systems

### ReasoningStage Integration
The ReasoningStage produces:
- `inferences: List[Inference]` - connected concepts
- `chains: List[ConceptChain]` - reasoning paths
- `abstractions: List[Abstraction]` - higher-order patterns

The **CommunicationPlanner** should consume these to decide:
- Which inferences are worth sharing?
- Should I explain a reasoning chain?
- Is this abstraction relevant to the user's question?

### AffectiveSystem Integration
The `AffectiveSystem` provides:
- `mood_valence`, `mood_arousal`, `confidence` - transient state
- `curiosity_drive`, `warmth` - personality drives

The **CommunicationPlanner** should use these to:
- Select communication stance
- Decide on follow-up questions (curiosity)
- Add empathetic acknowledgments (warmth)

### GenerativeSystem Refactor
The current `GenerativeSystem` should be refactored into:
1. **CompositionalRealizer** - builds language from plans
2. **GrammarLearner** - learns syntactic frames from input (already partially exists)

The template-based approach moves to a **backup** role for cold-start scenarios.

---

## Success Criteria

A successful implementation would allow conversations like:

```
User: I saw a dolphin today.
Lilith: That's wonderful! Dolphins are fascinating mammals. 
        What were they doing?

User: Just swimming around.
Lilith: They're incredible swimmers. Did you know they can reach 
        speeds of 20 mph?

User: Really? That's fast.
Lilith: Indeed! Speaking of which, have you seen other marine 
        animals up close?
```

Key behaviors demonstrated:
1. **Acknowledgment** - "That's wonderful!"
2. **Knowledge sharing** - "Dolphins are fascinating mammals"
3. **Follow-up question** - "What were they doing?" (curiosity drive)
4. **Topic continuity** - Stays on dolphins/marine theme
5. **Initiative** - Offers related information, asks new question
6. **Discourse markers** - "Speaking of which"

---

## TODO Checklist

- [ ] **Phase 1: Discourse Management**
  - [ ] Create `v2/lilith_v2/discourse_manager.py`
  - [ ] Implement `DialogueState` tracking
  - [ ] Implement user intent classification
  - [ ] Implement anaphora resolution
  - [ ] Implement obligation tracking

- [ ] **Phase 2: Communication Planning**
  - [ ] Create `v2/lilith_v2/communication_planner.py`
  - [ ] Implement goal selection logic
  - [ ] Implement content selection from working memory
  - [ ] Integrate affective drives for stance/initiative
  - [ ] Implement follow-up question generation

- [ ] **Phase 3: Compositional Realization**
  - [ ] Refactor `GenerativeSystem` → `CompositionalRealizer`
  - [ ] Implement relation → syntactic frame mapping
  - [ ] Implement stance modifier application
  - [ ] Implement discourse marker insertion
  - [ ] Ensure learned frames (not hard-coded) are preferred

- [ ] **Phase 4: Self-Monitoring**
  - [ ] Create `v2/lilith_v2/self_monitor.py`
  - [ ] Implement coherence checking
  - [ ] Implement informativeness checking
  - [ ] Implement revision loop

- [ ] **Phase 5: Integration**
  - [ ] Wire stages into `CognitiveStage` pipeline
  - [ ] Update `start_interactive.py` for testing
  - [ ] Add conversation flow tests
  - [ ] Performance benchmarking

---

## References

- V1 attempts: `lilith/response_composer.py` (pragmatic templates)
- V1 limbic: `lilith/affective_system.py` (mood tracking)
- V2 reasoning: `v2/lilith_v2/reasoning_stage.py` (physics deliberation)
- V2 pragmatics: `v2/lilith_v2/pragmatic_system.py` (basic templates)
