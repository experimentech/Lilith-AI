# System Tuning Needed

## Issues Identified

### 1. Learning from Teaching (Primary Issue)
**Problem**: System extracts patterns but can't apply them effectively

**Root Causes**:
- ❌ Extracting **fallback text** as trigger instead of **user's question**
  - Current: trigger="I'm not sure how to answer tha", response="Backpropagation is..."
  - Should be: trigger="What is backpropagation", response="Backpropagation is..."

- ❌ **Confidence threshold too high** (0.80) for newly learned patterns
  - Learned patterns score 0.56-0.78 but need 0.80 to use
  - New patterns haven't been boosted by success tracking yet

- ❌ **No special handling for teaching** 
  - Should detect when user is explaining (long, informative input)
  - Should boost freshly learned patterns temporarily

**Test Results** (3 scenarios):
- ✅ Backpropagation: LEARNED (but got lucky with 1.00 score)
- ⚠️  Iceland capital: Partial (learned but low confidence)
- ❌ Episodic memory: FAILED (learned but confidence 0.56 < 0.80)

### 2. Grammar/Syntax Stage Bypassed
**Problem**: Grammar stage exists but is never used

**Root Cause**:
```python
# Line 247 in response_composer.py
if user_input and best_score > 0.75:
    # High confidence - adapt pattern to current context
    adapted_text = self._adapt_pattern_to_context(...)
    return response  # ← BYPASSES grammar composition!

# Line 285 - Grammar only runs if adaptation didn't trigger
response = self._compose_from_patterns(...)  # Never reached!
```

**Impact**:
- Grammar-guided blend never executes
- Syntax BNN never used for composition
- weighted_blend mode behaves same as best_match

**Symptoms Observed**:
- "Do you like movies?" → "That's an interesting topic. I'd be happy to discuss **think** with you."
- Grammatical errors in adapted responses
- Pattern adaptation quality issues

### 3. Pattern Extraction Logic
**Problem**: Extracting wrong parts of conversation

**Current Behavior** (from learner):
```python
# response_learner.py ~line 250
trigger = bot_response[:100]  # ← WRONG! Using bot response as trigger
response = user_input          # ← WRONG! Using user input as response
```

**Should Be**:
```python
trigger = user_input                    # User's question
response = user_teaching_explanation    # User's explanation (next turn)
```

## Proposed Fixes

### Fix 1: Improve Pattern Extraction from Teaching

**Location**: `pipeline/response_learner.py` - `_extract_successful_pattern()`

**Changes**:
1. Detect teaching scenarios (long informative input after fallback)
2. Extract trigger from **previous user question**, not bot fallback
3. Extract response from **current user explanation**, not previous bot response
4. Add "taught" marker for freshly learned patterns

```python
def _extract_pattern_from_teaching(self, user_question, user_explanation):
    """
    Extract pattern when user teaches the system.
    
    user_question: "What is backpropagation?"
    user_explanation: "Backpropagation is an algorithm for..."
    
    Returns pattern:
      trigger_context: "What is backpropagation?"
      response_text: "Backpropagation is an algorithm for..."
      intent: "learned_knowledge"
      success_score: 0.8  # Boost for teaching
    """
```

### Fix 2: Lower Confidence Threshold for Learned Patterns

**Location**: `pipeline/response_composer.py` - Line 230

**Changes**:
1. Track which patterns are freshly learned (< 5 uses)
2. Use lower threshold (0.65) for learned patterns
3. Gradually increase threshold as pattern gets used/validated

```python
# Adaptive confidence threshold
if hasattr(pattern, 'learned_from_teaching') and pattern.usage_count < 5:
    confidence_threshold = 0.65  # Lower for new patterns
else:
    confidence_threshold = 0.80  # Standard threshold
```

### Fix 3: Grammar Integration with Adaptation

**Location**: `pipeline/response_composer.py` - Line 247-263

**Changes**:
1. Use pattern adaptation for **structure detection** only
2. Pass adapted result to grammar stage for **grammatical refinement**
3. Or: Use grammar stage first, THEN adapt

```python
# Option A: Adapt then refine with grammar
if user_input and best_score > 0.75:
    adapted_text = self._adapt_pattern_to_context(...)
    
    # Grammar refinement if available
    if self.syntax_stage:
        adapted_text = self._refine_with_grammar(adapted_text, user_input)
    
    return response

# Option B: Grammar first, then adapt
if self.syntax_stage and len(patterns) > 1:
    blended_text = self._blend_with_syntax_bnn(patterns[0], patterns[1])
    if blended_text:
        # Adapt blended result to context
        adapted_text = self._adapt_to_context_simple(blended_text, user_input)
        return response
```

### Fix 4: Teaching Detection Signals

**Location**: `pipeline/conversation_loop.py` - Process user input

**Changes**:
1. Detect teaching patterns (long input after fallback)
2. Set `is_teaching=True` flag
3. Pass to learner for special handling

```python
# Detect teaching scenario
prev_response_was_fallback = (
    hasattr(self, 'last_response') and 
    any(marker in self.last_response.text for marker in 
        ["don't know", "not sure", "don't have"])
)

is_teaching = (
    prev_response_was_fallback and
    len(user_input.split()) > 15 and  # Long explanation
    not user_input.endswith('?')      # Not a question
)

# Pass to learner
signals = self.learner.observe_interaction(
    ...,
    is_teaching=is_teaching  # ← NEW
)
```

## Priority Order

### Phase 1: Fix Learning from Teaching (Highest Impact)
1. ✅ Fix pattern extraction logic (use user question as trigger)
2. ✅ Detect teaching scenarios (long input after fallback)
3. ✅ Lower threshold for newly learned patterns (0.65 vs 0.80)
4. ✅ Temporary boost for taught patterns (0.2 bonus for first 5 uses)

**Expected Result**: 3/3 teaching scenarios should succeed

### Phase 2: Grammar Integration (Quality Improvement)
1. ✅ Make pattern adaptation use grammar refinement
2. ✅ Or switch order: grammar blend → context adaptation
3. ✅ Fix grammatical errors in adapted responses

**Expected Result**: No more "discuss think" errors, better sentence structure

### Phase 3: Success Tracking Tuning (Long-term Learning)
1. ✅ Persist success tracker to database
2. ✅ Higher boost for taught patterns (1.5x → 2.0x)
3. ✅ Faster boost accumulation for high-confidence teaching

**Expected Result**: System improves faster from teaching interactions

## Test Suite Updates

### New Tests Needed:
1. `test_learning_from_teaching.py` - Already exists, should pass 3/3 after Phase 1
2. `test_grammar_quality.py` - Test grammatical correctness
3. `test_pattern_confidence.py` - Test adaptive thresholds
4. `test_teaching_detection.py` - Verify teaching scenario detection

### Success Criteria:
- ✅ Teaching test: 3/3 scenarios learn successfully
- ✅ Grammar test: No grammatical errors in responses
- ✅ Confidence test: Learned patterns usable within 1-2 interactions
- ✅ Overall quality: 8.5+/10 (up from current 8.2)

## Implementation Estimate

**Phase 1** (Learning from Teaching): ~2-3 hours
- Pattern extraction refactor
- Teaching detection
- Confidence threshold logic

**Phase 2** (Grammar Integration): ~1-2 hours  
- Grammar refinement hook
- Adapter updates

**Phase 3** (Success Tracking): ~1 hour
- Database persistence
- Boost tuning

**Total**: ~4-6 hours of focused work

## Current State Summary

### What Works ✅
- Pattern adaptation (structure-level)
- Automatic success detection
- Success-based pattern boosting
- Hybrid BNN + keyword retrieval
- Grammar stage (when not bypassed)

### What Needs Fixing ❌
- Pattern extraction from teaching (wrong trigger/response)
- Confidence threshold (too strict for new patterns)
- Grammar bypass (adaptation short-circuits composition)
- Teaching detection (no special handling)

### What's Partially Working ⚠️
- Learning from teaching (extracts but doesn't apply: 2/3 success)
- Grammar composition (exists but bypassed)
- Freshly learned patterns (low confidence prevents use)
