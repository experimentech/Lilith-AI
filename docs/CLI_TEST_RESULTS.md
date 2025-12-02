# CLI Conversational Test Results

**Date:** December 2, 2025  
**Test User:** test_bugfix  
**Commit:** d76d011

## Test Results Summary

Ran automated conversational test to verify the three bugfixes. Results show **partial success** - some issues remain.

## Test 1: Wikipedia Disambiguation ❌ FAILED

**Query Sequence:**
```
User: What do you know about birds?
Lilith: Birds are a group of warm-blooded theropod dinosaurs...
       [✅ Correct - got Bird (animal)]

User: Do you like birds?
Lilith: The Liver Birds is a British sitcom...
       [❌ WRONG - got The Liver Birds (sitcom)]
```

**Expected:** Should understand from context that we're talking about birds (animals)  
**Actual:** Retrieved "The Liver Birds" sitcom

**Root Cause:**
The disambiguation scorer gives:
- "The Liver **Birds**" → exact word match "birds" → 2 points
- "**Bird** (animal)" → partial match "birds"/"bird" → 1 point

The sitcom wins because it has the exact word!

**Fix Needed:**
Need smarter context scoring:
1. Consider word stems/lemmas ("birds" and "bird" are same concept)
2. Boost recent conversation topics
3. Penalize title matches that are proper nouns (capitalized multi-word titles likely specific topics)

---

## Test 2: Pattern vs Template Selection ⚠️ UNCLEAR

**Query:**
```
User: What do you know about computers?
Pattern Match: "Do you know what a computer is?" (0.844 fuzzy confidence)
Response: "Hi there! How can I assist you today?" (greeting template)
```

**Expected:** Should use the pattern match (0.844 > 0.70 threshold)  
**Actual:** Returned greeting template

**Possible Causes:**
1. Pattern response_text might be empty or just contain greeting
2. Template confidence might be >0.844
3. Comparison logic might have a bug

**Status:** ⚠️ Needs investigation - unclear why template won over 0.844 pattern

---

## Test 3: Pronoun Resolution ⚠️ STILL MANGLING

**Query Sequence:**
```
User: What is Python?
Lilith: [fallback - disambiguation failed]

User: I like that language
Resolved: 'I like that language' → 'I like i language language'
```

**Expected:** Should resolve "that" → "python" or skip if referent invalid  
**Actual:** Mangled to "i language language"

**Root Cause:**
The validation added isn't catching this case. The referent is probably "i language" which passes:
- Length > 4: ✅
- Not all common words: ✅ (contains "language")  
- Pronoun density < 50%: ✅ (1 out of 2 words = 50%, not >50%)

**Fix Needed:**
- Change pronoun density check from `>` to `>=` (reject if ≥50%)
- OR add better topic extraction that doesn't include first-person pronouns
- OR use actual noun phrase extraction instead of naive topic summarization

---

## Additional Observations

### Python Disambiguation Also Failing

```
What is Python?
🔀 Disambiguation page detected for 'Python'
⚠️  Could not resolve disambiguation - returning None
```

The Python disambiguation also failed to resolve, which is why the pronoun test couldn't work properly. This suggests the disambiguation resolution is too strict or not finding any context matches.

### Query Repetition Detection

```
User: What do you know about computers?
🔁 Detected query repetition - varying response
```

Interesting - the system detected this as a repetition (probably similar to previous query), which affected the response. This is correct behavior but wasn't expected in the test.

---

## Fixes Still Needed

### 1. Disambiguation Scoring Algorithm

**Current:**
```python
if word in desc_lower.split():
    score += 2  # Exact match
elif word in desc_lower or any(word in token or token in word ...):
    score += 1  # Partial match
```

**Problem:** "birds" exact matches "The Liver Birds" better than "Bird"

**Proposed Fix:**
```python
# Use stemming/lemmatization
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

word_stem = stemmer.stem(word)
for token in desc_lower.split():
    token_stem = stemmer.stem(token)
    if token_stem == word_stem:
        score += 3  # Stem match (bird/birds)
    elif word == token:
        score += 2  # Exact match
    elif word in token or token in word:
        score += 1  # Partial match
```

### 2. Pronoun Density Check

**Current:**
```python
elif sum(1 for w in referent_words if w in pronouns) > len(referent_words) // 2:
    referent = None
```

**Problem:** `>` allows exactly 50%, so "i language" (1/2 = 50%) passes

**Fix:**
```python
elif sum(1 for w in referent_words if w in pronouns) >= len(referent_words) // 2:
    referent = None  # Reject if ≥50% pronouns
```

### 3. Pattern Response Text Investigation

Need to check why pattern with 0.844 confidence returned greeting instead of pattern text.

---

## Positive Results

### ✅ No Hard Crashes
All three tests completed without exceptions (after fixing the parameter passing).

### ✅ Conversation Context Is Being Passed
The logs show `💬 Conversation context: N recent turns` for each query, proving the context extraction works.

### ✅ Wikipedia Integration Works
Successfully fetched Wikipedia articles for "birds" (first query) and "computers".

### ✅ Basic Infrastructure Sound
- Session initialization ✅
- Multi-turn conversation ✅
- Pattern matching ✅
- Knowledge augmentation ✅

---

## Next Steps

1. **Fix disambiguation scoring** - Use stemming/lemmatization to match bird/birds
2. **Fix pronoun density** - Change `>` to `>=`
3. **Investigate pattern vs template** - Debug why 0.844 pattern lost to template
4. **Test again** - Verify all fixes work together

---

## Test Code

The test successfully exercised:
- 3-turn conversation flow
- Wikipedia lookups with context
- Pronoun resolution
- Pattern vs template selection
- Disambiguation resolution

This automated test can be re-run after each fix to verify progress.

---

## Conclusion

The bugfixes are **partially working**:
- ✅ Conversation context is extracted and passed correctly
- ✅ Infrastructure for all three fixes is in place
- ❌ Disambiguation scoring algorithm needs improvement
- ❌ Pronoun validation still has edge cases
- ⚠️ Pattern vs template needs investigation

The CLI testing successfully revealed remaining issues that need to be addressed.
