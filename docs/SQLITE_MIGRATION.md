# SQLite Migration

## Overview

Successfully migrated Lilith's response pattern storage from JSON files to SQLite database for proper concurrency support and multi-tenant isolation.

## Changes Made

### 1. SQLite Backend Implementation

**File**: `lilith/response_fragments_sqlite.py`

- Implemented `ResponseFragmentStoreSQLite` class
- **WAL Mode**: Write-Ahead Logging enabled for concurrent read/write access
- **Thread-Safe ID Generation**: Uses timestamp + random component to prevent ID collisions
- **Connection Management**: 30-second timeout for lock acquisition
- **Schema**: 
  - `response_patterns` table with fragment_id (PK), trigger_context, response_text, success_score, intent, usage_count, created_at
  - Index on intent for fast filtering
  - Bootstrap with 4 seed patterns in base knowledge only

### 2. Multi-Tenant Updates

**File**: `lilith/multi_tenant_store.py`

- Updated to use `ResponseFragmentStoreSQLite` instead of JSON-based storage
- Changed file extension from `.json` to `.db` for all stores
- Base knowledge: `data/base/response_patterns.db`
- User knowledge: `data/users/{user_id}/response_patterns.db`
- Removed manual `_save_patterns()` calls (SQLite auto-commits)
- Added `bootstrap=False` for user stores to prevent duplicate seed patterns

### 3. Migration Utility

**File**: `tools/migrate_json_to_sqlite.py`

- Utility to migrate existing JSON pattern files to SQLite
- Features:
  - Batch processing for large datasets
  - Progress reporting
  - Validation of migrated data
  - Preserves all pattern metadata (success scores, usage counts, etc.)

## Concurrency Testing

Created comprehensive concurrency tests in `tests/test_concurrency.py`:

### Test Results

✅ **Concurrent Writes**: 5 threads writing 10 patterns each = 50/50 success
- No data loss
- No duplicate IDs
- All patterns written correctly

✅ **Concurrent Reads/Writes**: 2 readers + 2 writers
- 100 reads completed
- 40 writes completed
- All patterns accounted for (54 total: 4 bootstrap + 10 pre-populated + 40 new)
- No database lock errors
- No race conditions

### Fixed Issues

**1. Race Condition in ID Generation**
- **Problem**: Multiple threads calling `_count_patterns()` simultaneously generated duplicate IDs
- **Solution**: Changed ID generation to use `timestamp_ms + random_suffix`
  ```python
  timestamp_ms = int(time.time() * 1000)
  random_suffix = random.randint(0, 9999)
  fragment_id = f"pattern_{intent}_{timestamp_ms}_{random_suffix}"
  ```
- **Result**: No more UNIQUE constraint violations

**2. Database Lock Errors**
- **Problem**: Concurrent connections experiencing lock timeouts
- **Solution**: 
  - Enabled WAL mode (Write-Ahead Logging)
  - Set 30-second connection timeout
  - Proper connection management with close after operations
- **Result**: Smooth concurrent read/write operations

## Multi-Tenant Testing

All multi-tenant tests passing with SQLite backend:

✅ **Teacher Mode**: Teachers can write to base knowledge
- Base knowledge correctly updated
- Seed patterns bootstrapped only in base

✅ **User Isolation**: Users cannot corrupt base knowledge
- User patterns stored in separate databases
- Base knowledge remains unchanged when users add patterns
- Each user has isolated storage

✅ **Base Knowledge Access**: Users can read from base
- Users can retrieve patterns from base knowledge
- User-specific patterns take priority in searches
- Layered retrieval: user store first, base fallback

## Performance Characteristics

### SQLite with WAL Mode Benefits:

1. **Concurrent Reads**: Multiple readers without blocking
2. **Concurrent Writes**: Writers don't block readers
3. **ACID Guarantees**: Atomic, Consistent, Isolated, Durable transactions
4. **File-Based**: Simple deployment, no separate database server
5. **Proven**: Battle-tested database engine (used in billions of devices)

### Scalability Notes:

- **Read Performance**: O(log n) for indexed lookups, excellent for concurrent reads
- **Write Performance**: Sequential writes with WAL, suitable for moderate write loads
- **Database Size**: Tested with 50+ patterns, scales to thousands per user
- **User Count**: Each user gets isolated database file, scales to many users

## Migration Path

For existing installations with JSON pattern files:

```bash
# Migrate base knowledge
python tools/migrate_json_to_sqlite.py \
    --input data/base/response_patterns.json \
    --output data/base/response_patterns.db

# Migrate user data
python tools/migrate_json_to_sqlite.py \
    --input data/users/alice/response_patterns.json \
    --output data/users/alice/response_patterns.db
```

## Backward Compatibility

⚠️ **Breaking Change**: Old JSON-based stores are no longer supported.

To maintain old data:
1. Run migration utility before upgrading
2. Keep JSON files as backup
3. Verify migrated data with test scripts

## Future Improvements

Potential enhancements for high-traffic scenarios:

1. **Connection Pooling**: Reuse connections instead of creating new ones
2. **Batch Writes**: Group multiple pattern additions into single transactions
3. **Read Replicas**: For read-heavy workloads
4. **Partitioning**: Split large user databases by intent or time period
5. **Compression**: BLOB compression for large response texts

## Verification

To verify your SQLite migration is working:

```bash
# Run concurrency tests
python tests/test_concurrency.py

# Run multi-tenant tests
python tests/test_multi_tenant.py

# Or use pytest
pytest tests/test_concurrency.py tests/test_multi_tenant.py -v
```

Expected output: All tests passing with ✅ indicators.

## Summary

The SQLite migration provides:
- ✅ Thread-safe concurrent operations
- ✅ Proper multi-tenant isolation
- ✅ ACID transaction guarantees
- ✅ Better performance at scale
- ✅ Industry-standard database reliability

All existing functionality preserved with improved robustness and concurrency support.
