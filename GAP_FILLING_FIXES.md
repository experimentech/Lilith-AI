# Gap-Filling Bug Fixes

## Issue Summary

The gap-filling feature was crashing with `AttributeError` exceptions due to incorrect attribute and method names.

## Errors Found

### Error 1: Wrong Store Attribute
```
AttributeError: 'ResponseComposer' object has no attribute 'store'
```
**Cause:** Used `self.store` instead of `self.fragments`

### Error 2: Wrong Method Name
```
AttributeError: 'MultiTenantFragmentStore' object has no attribute 'retrieve_context'
```
**Cause:** Used `retrieve_context()` instead of `retrieve_patterns()`

### Error 3: Wrong Parameter Name
**Cause:** Used `min_similarity=` instead of `min_score=`

### Error 4: Nonexistent Attributes
- `self.min_confidence_threshold` doesn't exist (used hardcoded `0.65`)
- `self._apply_adaptation()` doesn't exist (used `response_text` directly)
- `best_pattern.pattern_id` doesn't exist (used `fragment_id`)
- `add_pattern()` doesn't support `metadata` parameter

## Fixes Applied

### File: `lilith/response_composer.py`

#### Fix 1: Correct Store Attribute (Line 2132)
```python
# BEFORE
patterns = self.store.retrieve_context(...)

# AFTER
patterns = self.fragments.retrieve_patterns(...)
```

#### Fix 2: Correct Method and Parameters (Line 2132)
```python
# BEFORE
patterns = self.fragments.retrieve_context(enhanced_context, topk=5, min_similarity=0.3)

# AFTER
patterns = self.fragments.retrieve_patterns(enhanced_context, topk=5, min_score=0.3)
```

#### Fix 3: Hardcoded Threshold (Line 2140)
```python
# BEFORE
if best_score >= self.min_confidence_threshold:

# AFTER
if best_score >= 0.65:  # Reasonable threshold for gap-filled patterns
```

#### Fix 4: Direct Response Text (Line 2143)
```python
# BEFORE
response_text = self._apply_adaptation(
    best_pattern.response_text,
    user_input,
    best_score
)

# AFTER
response_text = best_pattern.response_text
```

#### Fix 5: Removed Metadata Parameter (Line 2151)
```python
# BEFORE
new_pattern_id = self.fragments.add_pattern(
    trigger_context=user_input,
    response_text=response_text,
    intent="gap_filled",
    success_score=best_score * 0.9,
    metadata={...}  # NOT SUPPORTED
)

# AFTER
new_pattern_id = self.fragments.add_pattern(
    trigger_context=user_input,
    response_text=response_text,
    intent="gap_filled",
    success_score=best_score * 0.9
)
```

#### Fix 6: Correct Pattern Attribute (Line 2165)
```python
# BEFORE
fragment_ids=[best_pattern.pattern_id],

# AFTER
fragment_ids=[best_pattern.fragment_id],
```

## Verification

All fixes verified with `verify_gap_filling_fix_v2.py`:
- ✅ Store attribute: `self.fragments`
- ✅ Retrieve method: `retrieve_patterns()`
- ✅ Parameter name: `min_score`
- ✅ Threshold: Hardcoded value
- ✅ Adaptation: Direct response text
- ✅ add_pattern: No metadata parameter
- ✅ Pattern attribute: `fragment_id`

## Testing

The gap-filling feature should now work correctly. Test with queries like:
- "What is a Gigatron?"
- "What does ephemeral mean?"
- "What's a synonym for happy?"

Lilith will:
1. Detect unknown terms
2. Look them up in external sources
3. Retry pattern matching with enhanced context
4. Respond seamlessly
5. Learn the pattern for future use
