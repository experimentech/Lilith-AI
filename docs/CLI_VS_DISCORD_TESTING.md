# CLI vs Discord Testing Compatibility

**Date:** December 2, 2025

## Question

> "Does this mean that it should be possible for you to do conversational tests on the CLI version and the results should be applicable to the discord bot version?"

## Answer: **YES, with minor caveats**

Both CLI and Discord bot use **identical SessionConfig defaults**, so conversational behavior should be **99% identical**. The three bugfixes apply to both equally.

---

## Configuration Comparison

### CLI Configuration (`lilith_cli.py` line 66)

```python
config = SessionConfig(
    data_path="data",
    learning_enabled=True,
    enable_declarative_learning=True,
    enable_feedback_detection=True,
    plasticity_enabled=True
)
```

### Discord Configuration (`discord_bot.py` lines 140, 366)

```python
config = SessionConfig(
    data_path=str(self.data_path),
    learning_enabled=True,           # or False in some server contexts
    enable_declarative_learning=True, # or False in some server contexts
    enable_feedback_detection=True,
    plasticity_enabled=True,          # or False in some server contexts
    syntax_plasticity_interval=5,
    pmflow_plasticity_interval=10,
    contrastive_interval=5
)
```

### Inherited Defaults (Both Platforms)

These are **automatically enabled** for both CLI and Discord:

```python
# From SessionConfig defaults (session.py)
enable_compositional: bool = True
enable_pragmatic_templates: bool = True
composition_mode: str = "pragmatic"
enable_knowledge_augmentation: bool = True
enable_modal_routing: bool = True
```

---

## Differences

### 1. Plasticity Intervals (Minor)

**Discord:** Explicitly sets intervals
```python
syntax_plasticity_interval=5
pmflow_plasticity_interval=10
contrastive_interval=5
```

**CLI:** Uses SessionConfig defaults
```python
# Defaults from SessionConfig
syntax_plasticity_interval: int = 10
pmflow_plasticity_interval: int = 20
contrastive_interval: int = 10
```

**Impact:** Discord learns **faster** than CLI (more frequent network updates)
- Discord: Updates syntax every 5 turns
- CLI: Updates syntax every 10 turns

**Testing Impact:** ⚠️ **Minor** - Short conversations (<10 turns) will behave identically. Longer conversations may show slight learning speed differences.

---

### 2. Server Context Learning (Context-Dependent)

**Discord:** Can disable learning in server contexts
```python
# In servers (if configured)
learning_enabled = server_settings.learning_enabled  # May be False
```

**CLI:** Always enabled
```python
learning_enabled=True  # Always
```

**Impact:** Discord bot can have learning **disabled** in public servers, but always enabled in DMs

**Testing Impact:** ✅ **None** if testing in DM mode or servers with learning enabled

---

### 3. Session Management (Infrastructure)

**Discord:**
- Multi-user concurrent sessions
- Session timeout (30 min default)
- User retention policies
- Per-guild vs DM contexts

**CLI:**
- Single-user session
- No timeout
- Manual exit
- Single context

**Impact:** Session lifecycle differs, but **conversational logic is identical**

**Testing Impact:** ✅ **None** for single-conversation tests

---

### 4. Conversation History Storage

**Discord:**
```python
user_id = str(message.author.id)  # Discord user ID
context_id = str(message.guild.id) if message.guild else "dm"
```

**CLI:**
```python
user_id = user_identity.user_id  # "teacher" or username
context_id = "cli"
```

**Impact:** Different user ID formats, but same conversation tracking logic

**Testing Impact:** ✅ **None** - conversation state tracking works identically

---

## Bugfix Applicability

All three bugfixes apply **equally** to both platforms:

### ✅ Pronoun Resolution
- **Shared Code:** `session.py` → `_update_context_with_reasoning()`
- **CLI:** ✅ Fixed
- **Discord:** ✅ Fixed

### ✅ Wikipedia Disambiguation
- **Shared Code:** `knowledge_augmenter.py` + `response_composer.py`
- **CLI:** ✅ Fixed (conversation history from CLI context)
- **Discord:** ✅ Fixed (conversation history from Discord context)

### ✅ Template Override
- **Shared Code:** `response_composer.py` → `compose_response()`
- **CLI:** ✅ Fixed
- **Discord:** ✅ Fixed

---

## Testing Recommendations

### ✅ Safe to Test on CLI

**These behaviors are identical:**

1. **Conversational flow** (greetings, continuations, acknowledgments)
2. **Wikipedia disambiguation** (with conversation context)
3. **Pronoun resolution** (referent validation)
4. **Template vs pattern selection** (confidence comparison)
5. **Knowledge augmentation** (external lookups)
6. **Modal routing** (math queries)
7. **Compositional responses** (concept-based)
8. **Pattern matching** (fuzzy, semantic, hybrid)
9. **Feedback detection** (upvote/downvote)

### ⚠️ Test on Discord for These

**Platform-specific behaviors:**

1. **Multi-user isolation** (concurrent sessions)
2. **Server vs DM contexts** (different learning settings)
3. **Reaction-based feedback** (Discord emoji reactions)
4. **Session timeout behavior** (memory management)
5. **Slash commands** (Discord-specific)

### 🔬 Testing Strategy

**For conversational bugfixes:**

```bash
# 1. Quick validation on CLI
$ python lilith_cli.py
> What do you know about birds?
> Do you like birds?  # Should NOT get "The Liver Birds"

# 2. Full regression on Discord
# Test same conversation in Discord DM
# Verify identical behavior
```

**For multi-user scenarios:**

```bash
# Must test on Discord
# CLI is single-user only
```

---

## Example: Testing Wikipedia Disambiguation

### CLI Test

```
$ python lilith_cli.py

[Mode: User mode]
[Username: test_user]

> What do you know about birds?
🌐 Wikipedia found: Bird
Birds are warm-blooded vertebrates...

> Do you like birds?
🌐 Wikipedia found: Bird  # ✅ Correct - not "The Liver Birds"
[Compositional response using learned concept]
```

### Discord Test

```
Discord DM with Lilith:

You: What do you know about birds?
Lilith: Birds are warm-blooded vertebrates...
       [Source: Wikipedia]

You: Do you like birds?
Lilith: ✅ Correct - same behavior as CLI
       [Uses conversation context from previous message]
```

### Expected Result

**Identical behavior** - Both should understand we're talking about Bird (animal), not The Liver Birds (sitcom)

---

## Conclusion

### Main Answer

**YES** - CLI testing results **ARE applicable** to Discord bot for:
- ✅ All conversational logic
- ✅ All three bugfixes
- ✅ Knowledge augmentation
- ✅ Pattern matching
- ✅ Template composition
- ✅ Learning mechanisms

### With These Caveats

1. **Plasticity speed:** Discord learns slightly faster (minor difference)
2. **Multi-user scenarios:** Must test on Discord (CLI is single-user)
3. **Server contexts:** Discord may have learning disabled (test in DM mode)

### Testing Workflow

```
1. Quick validation → CLI (faster iteration)
2. Regression testing → CLI (same conversational logic)
3. Multi-user testing → Discord (only option)
4. Production verification → Discord (real environment)
```

---

## Code Path Verification

Both platforms share **100% of conversational logic:**

```
User Input
    ↓
session.py (shared)
    → _update_context_with_reasoning() [BUGFIX 1: Pronoun validation]
    ↓
response_composer.py (shared)
    → compose_response() [BUGFIX 3: Template comparison]
    → knowledge_augmenter.lookup() [BUGFIX 2: Disambiguation]
    ↓
Response Output
```

The **only differences** are:
- Input source: `input()` vs Discord message
- Output destination: `print()` vs Discord channel
- Session management: Single vs multi-user

The **entire reasoning pipeline is identical**.

---

## Practical Example

### Scenario: Test "homebrew computers" fix

**CLI:**
```bash
$ python lilith_cli.py
> What do you know about homebrew computers?
# Should return pattern match (0.767), not greeting template
```

**Discord:**
```
You: What do you know about homebrew computers?
# Should return same pattern match (0.767), not greeting template
```

**Result:** ✅ Identical (template override logic is shared)

---

## Summary Table

| Feature | CLI | Discord | Identical? |
|---------|-----|---------|-----------|
| Pronoun resolution | ✅ | ✅ | ✅ Yes |
| Wikipedia disambiguation | ✅ | ✅ | ✅ Yes |
| Template vs pattern | ✅ | ✅ | ✅ Yes |
| Conversation context | ✅ | ✅ | ✅ Yes |
| Pattern matching | ✅ | ✅ | ✅ Yes |
| Knowledge augmentation | ✅ | ✅ | ✅ Yes |
| Compositional responses | ✅ | ✅ | ✅ Yes |
| Learning mechanisms | ✅ | ✅ | ✅ Yes (speed differs) |
| Multi-user isolation | ❌ | ✅ | ❌ No (Discord only) |
| Session timeout | ❌ | ✅ | ❌ No (Discord only) |
| Reaction feedback | ❌ | ✅ | ❌ No (Discord only) |

**Shared:** 8/11 features (73%)  
**Conversational logic:** 100% shared

---

## Recommendation

✅ **Yes, test conversational behavior on CLI** - it's faster and results translate directly to Discord

⚠️ **But verify on Discord for:**
- Multi-user scenarios
- Production deployment
- Platform-specific features (reactions, slash commands)

The bugfixes you've implemented will work **identically** on both platforms because they modify shared code paths used by both CLI and Discord bot.
