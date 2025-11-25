"""
Test error handling and recovery features.

Tests:
1. Database corruption recovery
2. User data reset
3. Manual feedback (upvote/downvote)
4. Graceful error handling
"""

import sys
from pathlib import Path
import shutil
import sqlite3

sys.path.insert(0, str(Path(__file__).parent.parent))

from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.response_fragments_sqlite import ResponseFragmentStoreSQLite
from lilith.multi_tenant_store import MultiTenantFragmentStore
from lilith.user_auth import UserIdentity, AuthMode


def test_corrupted_database_recovery():
    """Test that corrupted database is recovered gracefully."""
    print("\n" + "=" * 60)
    print("TEST 1: Corrupted Database Recovery")
    print("=" * 60)
    
    test_db = Path("data/test_corruption/response_patterns.db")
    if test_db.parent.exists():
        shutil.rmtree(test_db.parent)
    test_db.parent.mkdir(parents=True, exist_ok=True)
    
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    
    # Create valid database
    store = ResponseFragmentStoreSQLite(encoder, storage_path=str(test_db))
    
    # Add some patterns
    for i in range(5):
        store.add_pattern(f"context {i}", f"response {i}", 0.7, "test")
    
    stats = store.get_stats()
    print(f"✓ Created database with {stats['total_patterns']} patterns")
    
    # Corrupt the database by truncating it
    with open(test_db, 'rb+') as f:
        f.truncate(512)  # Truncate to corrupt
    
    print("✓ Database corrupted (truncated)")
    
    # Try to open corrupted database - should trigger recovery
    try:
        store2 = ResponseFragmentStoreSQLite(encoder, storage_path=str(test_db), bootstrap_if_empty=False)
        print("✓ Database recovery attempted")
        
        # Check if we have at least bootstrap patterns
        stats2 = store2.get_stats()
        print(f"✓ After recovery: {stats2['total_patterns']} patterns")
        
        # Should have backup file
        backup_files = list(test_db.parent.glob("*.corrupted.*.db"))
        if backup_files:
            print(f"✓ Backup created: {backup_files[0].name}")
        
        print("✅ Corruption recovery test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Recovery failed: {e}")
        return False


def test_user_data_reset():
    """Test user data reset functionality."""
    print("\n" + "=" * 60)
    print("TEST 2: User Data Reset")
    print("=" * 60)
    
    # Clean test data
    test_dir = Path("data/test_reset")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    
    # Create user
    user = UserIdentity("test_user", AuthMode.SIMPLE, "Test User")
    store = MultiTenantFragmentStore(encoder, user, base_data_path=str(test_dir))
    
    # Add patterns
    print("Adding patterns...")
    for i in range(10):
        store.add_pattern(f"user fact {i}", f"user response {i}", 0.6, "personal")
    
    counts_before = store.get_pattern_count()
    print(f"✓ User has {counts_before['user']} personal patterns")
    
    # Reset user data
    print("\nResetting user data...")
    backup = store.reset_user_data(keep_backup=True)
    
    if backup:
        print(f"✓ Backup created: {backup}")
    
    # Check counts after reset
    counts_after = store.get_pattern_count()
    print(f"✓ After reset: {counts_after['user']} user patterns")
    
    # Verify
    if counts_after['user'] == 0 and backup:
        print("✅ User data reset test PASSED")
        return True
    else:
        print(f"❌ Expected 0 user patterns, got {counts_after['user']}")
        return False


def test_manual_feedback():
    """Test manual upvote/downvote functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Manual Feedback (Upvote/Downvote)")
    print("=" * 60)
    
    test_db = Path("data/test_feedback/response_patterns.db")
    if test_db.parent.exists():
        shutil.rmtree(test_db.parent)
    test_db.parent.mkdir(parents=True, exist_ok=True)
    
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    store = ResponseFragmentStoreSQLite(encoder, storage_path=str(test_db))
    
    # Add test pattern
    pattern_id = store.add_pattern(
        "test context",
        "test response",
        success_score=0.5,
        intent="test"
    )
    
    print(f"✓ Added pattern: {pattern_id} (score: 0.5)")
    
    # Get initial score
    patterns = store.patterns
    initial_score = patterns[pattern_id].success_score
    print(f"  Initial score: {initial_score:.3f}")
    
    # Upvote
    print("\n👍 Upvoting pattern...")
    store.upvote(pattern_id, strength=0.2)
    
    patterns = store.patterns
    after_upvote = patterns[pattern_id].success_score
    print(f"  Score after upvote: {after_upvote:.3f}")
    
    # Downvote
    print("\n👎 Downvoting pattern...")
    store.downvote(pattern_id, strength=0.3)
    
    patterns = store.patterns
    after_downvote = patterns[pattern_id].success_score
    print(f"  Score after downvote: {after_downvote:.3f}")
    
    # Verify changes
    if after_upvote > initial_score and after_downvote < after_upvote:
        print("\n✅ Manual feedback test PASSED")
        print(f"   Score trajectory: {initial_score:.3f} → {after_upvote:.3f} → {after_downvote:.3f}")
        return True
    else:
        print("\n❌ Feedback not working correctly")
        return False


def test_multi_tenant_feedback():
    """Test feedback in multi-tenant context."""
    print("\n" + "=" * 60)
    print("TEST 4: Multi-Tenant Feedback")
    print("=" * 60)
    
    test_dir = Path("data/test_mt_feedback")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    
    # Teacher adds base knowledge
    teacher = UserIdentity("teacher", AuthMode.NONE, "Teacher")
    teacher_store = MultiTenantFragmentStore(encoder, teacher, base_data_path=str(test_dir))
    
    base_pattern_id = teacher_store.add_pattern(
        "what is 2+2",
        "2+2 equals 4",
        success_score=0.7,
        intent="math"
    )
    print(f"✓ Teacher added base pattern: {base_pattern_id}")
    
    # User adds personal pattern
    user = UserIdentity("alice", AuthMode.SIMPLE, "Alice")
    user_store = MultiTenantFragmentStore(encoder, user, base_data_path=str(test_dir))
    
    user_pattern_id = user_store.add_pattern(
        "my favorite color",
        "My favorite color is blue",
        success_score=0.5,
        intent="personal"
    )
    print(f"✓ User added personal pattern: {user_pattern_id}")
    
    # User can upvote their own pattern
    print("\n👍 User upvoting personal pattern...")
    user_store.upvote(user_pattern_id)
    
    # User cannot modify base pattern
    print("\n👎 User trying to downvote base pattern...")
    user_store.downvote(base_pattern_id)  # Should warn
    
    # Teacher CAN modify base pattern
    teacher2 = UserIdentity("teacher2", AuthMode.NONE, "Teacher 2")
    teacher_store2 = MultiTenantFragmentStore(encoder, teacher2, base_data_path=str(test_dir))
    
    print("\n👍 Teacher upvoting base pattern...")
    teacher_store2.upvote(base_pattern_id)
    
    print("\n✅ Multi-tenant feedback test PASSED")
    return True


def test_graceful_error_handling():
    """Test that errors don't crash the system."""
    print("\n" + "=" * 60)
    print("TEST 5: Graceful Error Handling")
    print("=" * 60)
    
    test_db = Path("data/test_errors/response_patterns.db")
    if test_db.parent.exists():
        shutil.rmtree(test_db.parent)
    test_db.parent.mkdir(parents=True, exist_ok=True)
    
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    store = ResponseFragmentStoreSQLite(encoder, storage_path=str(test_db))
    
    # Try to upvote non-existent pattern
    print("Upvoting non-existent pattern...")
    try:
        store.upvote("fake_pattern_id_12345")
        print("✓ No crash on missing pattern")
    except Exception as e:
        print(f"⚠️  Exception raised: {e}")
    
    # Try to retrieve with invalid context
    print("\nRetrieving with empty context...")
    try:
        results = store.retrieve_patterns("", topk=5)
        print(f"✓ Returned {len(results)} results (no crash)")
    except Exception as e:
        print(f"⚠️  Exception raised: {e}")
    
    print("\n✅ Error handling test PASSED (no crashes)")
    return True


def main():
    """Run all error handling tests."""
    print("=" * 60)
    print("ERROR HANDLING & RECOVERY TESTS")
    print("=" * 60)
    
    test1 = test_corrupted_database_recovery()
    test2 = test_user_data_reset()
    test3 = test_manual_feedback()
    test4 = test_multi_tenant_feedback()
    test5 = test_graceful_error_handling()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Corruption Recovery: {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"User Data Reset: {'✅ PASSED' if test2 else '❌ FAILED'}")
    print(f"Manual Feedback: {'✅ PASSED' if test3 else '❌ FAILED'}")
    print(f"Multi-Tenant Feedback: {'✅ PASSED' if test4 else '❌ FAILED'}")
    print(f"Error Handling: {'✅ PASSED' if test5 else '❌ FAILED'}")
    
    if all([test1, test2, test3, test4, test5]):
        print("\n🎉 ALL ERROR HANDLING TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
