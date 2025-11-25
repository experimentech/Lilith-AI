# User Guide: Feedback & Data Management

Quick reference for managing your Lilith knowledge base and providing feedback.

## Manual Feedback

### When to Use Each Type

| Situation | Command | Effect |
|-----------|---------|--------|
| **Excellent answer** - exactly what you wanted | `store.upvote(pattern_id)` | +20% score |
| **Good answer** - generally helpful | `store.mark_helpful(pattern_id)` | +15% score |
| **Suboptimal answer** - not quite right | `store.mark_not_helpful(pattern_id)` | -15% score |
| **Bad answer** - wrong or unhelpful | `store.downvote(pattern_id)` | -30% score |

### Python Example

```python
from lilith.multi_tenant_store import MultiTenantFragmentStore
from lilith.user_auth import UserIdentity, AuthMode
from lilith.embedding import PMFlowEmbeddingEncoder

encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
user = UserIdentity("alice", AuthMode.SIMPLE, "Alice")
store = MultiTenantFragmentStore(encoder, user)

# Get a response
patterns = store.retrieve_patterns("What is Python?", topk=1)
if patterns:
    pattern, confidence = patterns[0]
    print(f"Response: {pattern.response_text}")
    
    # If it's a great answer
    store.upvote(pattern.fragment_id)
    
    # If it's wrong
    store.downvote(pattern.fragment_id)
```

### Score Changes

```
Initial score: 0.50

After 1 upvote:     0.50 → 0.60  (+0.10)
After 2 upvotes:    0.60 → 0.68  (+0.08)
After 3 upvotes:    0.68 → 0.744 (+0.064)

After 1 downvote:   0.50 → 0.35  (-0.15)
After 2 downvotes:  0.35 → 0.245 (-0.105)
After 3 downvotes:  0.245 → 0.172 (-0.073)
```

Patterns with low scores (< 0.1) can be automatically pruned.

## Data Management

### Reset Your Data

**Start fresh with a clean slate:**

```python
# Reset with backup (recommended)
backup_path = store.reset_user_data(keep_backup=True)
# Creates: data/users/alice/response_patterns.backup.20251126_083353.db

# Reset without backup (dangerous!)
store.reset_user_data(keep_backup=False)

# Reset with seed patterns (4 basic responses)
store.reset_user_data(keep_backup=True, bootstrap=True)
```

**What gets deleted:**
- ✅ All your personal patterns
- ❌ Base knowledge (remains intact)

**Output:**
```
💾 Backup created: data/users/alice/response_patterns.backup.20251126_083353.db
🔄 Database reset complete
📊 Old patterns: 47
📊 New patterns: 0
```

### Manual Backup

```python
import shutil
from pathlib import Path
from datetime import datetime

# Your database path
db_path = Path("data/users/alice/response_patterns.db")

# Create backup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = db_path.with_suffix(f'.manual_backup.{timestamp}.db')
shutil.copy2(db_path, backup_path)

print(f"Backup created: {backup_path}")
```

### Clean Up Low-Quality Patterns

```python
# Remove patterns with score < 0.1
pruned = store.user_store.prune_low_quality_patterns(threshold=0.1)
print(f"Removed {pruned} low-quality patterns")

# Apply time-based decay (patterns not used in 30 days lose confidence)
store.user_store.apply_temporal_decay(half_life_days=30)
print("Applied temporal decay")
```

### View Statistics

```python
# Get detailed stats
stats = store.user_store.get_stats()

print(f"Total patterns: {stats['total_patterns']}")
print(f"Average score: {stats['avg_success_score']:.3f}")
print(f"Min score: {stats['min_success_score']:.3f}")
print(f"Max score: {stats['max_success_score']:.3f}")

# By intent
for intent, info in stats['by_intent'].items():
    print(f"  {intent}: {info['count']} patterns (avg: {info['avg_score']:.3f})")
```

## Error Recovery

### What Happens if Database Gets Corrupted?

Lilith **automatically recovers** corrupted databases:

1. **Detects corruption** on startup
2. **Backs up** corrupted file with timestamp
3. **Recovers** as many patterns as possible
4. **Creates** fresh database
5. **Continues running** - no crash!

**You'll see:**
```
⚠️  Database corruption detected
⚠️  Attempting to recover database
💾 Backed up corrupted database to: response_patterns.corrupted.20251126_083351.db
✅ Recovered 27 patterns
✅ Database recovered successfully
```

**Your corrupted data is never deleted** - it's always backed up first.

### If You See Corruption Warnings

1. **Don't panic** - your data is backed up
2. **Check the backup** - it's in the same directory
3. **Review recovered patterns** - most should be intact
4. **Contact support** if recovery failed

## Command Line Tools

### List All Users

```bash
python tools/manage_users.py list
```

### Delete Your Data

```bash
# With archive
python tools/manage_users.py delete alice

# Without archive (permanent!)
python tools/manage_users.py delete alice --no-archive
```

### Export to JSON

```bash
python tools/manage_users.py export alice my_patterns.json
```

### View Statistics

```bash
python tools/manage_users.py stats
```

## Permissions

### What Users Can Do

✅ Add patterns to personal database
✅ Upvote/downvote own patterns
✅ Read from base knowledge
✅ Reset own data
✅ View own statistics

❌ Modify base knowledge
❌ Modify other users' patterns
❌ Delete base patterns

### What Teachers Can Do

✅ Everything users can do, plus:
✅ Add patterns to base knowledge
✅ Upvote/downvote base patterns
✅ Modify base knowledge

❌ Cannot reset user data (must be done by user)

## Best Practices

### Regular Maintenance

**Weekly:**
- Review low-scoring patterns
- Upvote/downvote based on recent conversations
- Check statistics

**Monthly:**
- Prune patterns with score < 0.1
- Apply temporal decay
- Create manual backup

**Quarterly:**
- Export data to JSON (archival)
- Review all intents for obsolete patterns

### Feedback Guidelines

**Upvote when:**
- Answer is exactly right
- Response is particularly helpful
- You want to see more like this

**Downvote when:**
- Answer is factually wrong
- Response is confusing or unhelpful
- You never want to see this again

**Use helpful/not helpful for:**
- Routine feedback
- Minor improvements needed
- General quality assessment

### Data Hygiene

1. **Regular feedback**: Don't let patterns accumulate without scoring
2. **Prune periodically**: Remove patterns with score < 0.1
3. **Review intents**: Keep categories organized
4. **Back up before experiments**: Test with safety net
5. **Clean slate occasionally**: Fresh start can improve quality

## Troubleshooting

### "Database is locked"

**Cause**: Another process is using the database

**Solution**:
```bash
# Close other Lilith instances
# Check for stale connections
# Wait 30 seconds and retry
```

### "Pattern not found" when providing feedback

**Cause**: Pattern was deleted or is in base knowledge (user mode)

**Solution**:
- Check if pattern still exists
- Verify you're not trying to modify base patterns as a user

### "Reset failed"

**Cause**: Database file is locked or corrupted

**Solution**:
```python
# Close all connections
store = None

# Force reset
import shutil
from pathlib import Path
db_path = Path("data/users/alice/response_patterns.db")
if db_path.exists():
    shutil.move(db_path, db_path.with_suffix('.old.db'))

# Reinitialize
store = MultiTenantFragmentStore(encoder, user)
```

## Quick Reference

### Feedback
```python
store.upvote(pattern_id)           # Strong positive
store.mark_helpful(pattern_id)     # Moderate positive
store.mark_not_helpful(pattern_id) # Moderate negative
store.downvote(pattern_id)         # Strong negative
```

### Data Management
```python
store.reset_user_data()                    # Fresh start
store.user_store.prune_low_quality_patterns() # Clean up
store.user_store.apply_temporal_decay()    # Age-based decay
store.user_store.get_stats()               # View stats
```

### Maintenance
```bash
python tools/manage_users.py list          # List users
python tools/manage_users.py delete alice  # Delete user
python tools/manage_users.py export alice  # Export data
python tools/manage_users.py stats         # Statistics
```

---

For detailed technical information, see [MULTI_TENANT_FAQ.md](MULTI_TENANT_FAQ.md)
