"""
Test multi-tenant functionality:
1. Teacher can write to base
2. Users can't corrupt base
3. Users are isolated from each other
4. Users can access base knowledge
"""

import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.multi_tenant_store import MultiTenantFragmentStore
from lilith.user_auth import UserIdentity, AuthMode


def cleanup_test_data():
    """Clean up test data directories"""
    test_dir = Path("data/test")
    
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Recreate structure
    (test_dir / "base").mkdir(parents=True, exist_ok=True)
    (test_dir / "users").mkdir(parents=True, exist_ok=True)


def test_teacher_mode():
    """Test that teacher can write to base knowledge"""
    cleanup_test_data()
    
    print("\n" + "=" * 60)
    print("TEST 1: Teacher Mode")
    print("=" * 60)
    
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    
    teacher_identity = UserIdentity(
        user_id="teacher",
        auth_mode=AuthMode.NONE,
        display_name="Teacher"
    )
    
    store = MultiTenantFragmentStore(
        encoder,
        teacher_identity,
        base_data_path="data/test"
    )
    
    # Teacher adds pattern
    pattern_id = store.add_pattern(
        trigger_context="what is the capital of france",
        response_text="The capital of France is Paris.",
        success_score=0.9,
        intent="geography"
    )
    
    print(f"✓ Teacher added pattern: {pattern_id}")
    
    # Verify it's in base store
    assert pattern_id in store.base_store.patterns, "Pattern should be in base store"
    
    counts = store.get_pattern_count()
    print(f"✓ Base patterns: {counts['base']}")
    print(f"✓ User patterns: {counts['user']}")
    
    assert counts['base'] > 0, "Base should have patterns"
    assert counts['user'] == 0, "Teacher mode shouldn't have user patterns"
    
    print("✅ Teacher mode test PASSED")
    return True


def test_user_isolation():
    """Test that users can't corrupt base and are isolated"""
    cleanup_test_data()
    
    print("\n" + "=" * 60)
    print("TEST 2: User Isolation")
    print("=" * 60)
    
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    
    # First, teacher adds base knowledge
    teacher_identity = UserIdentity(
        user_id="teacher",
        auth_mode=AuthMode.NONE,
        display_name="Teacher"
    )
    
    teacher_store = MultiTenantFragmentStore(
        encoder,
        teacher_identity,
        base_data_path="data/test"
    )
    
    teacher_store.add_pattern(
        trigger_context="what is machine learning",
        response_text="Machine learning is a subset of AI.",
        success_score=0.9,
        intent="tech"
    )
    
    base_count_before = len(teacher_store.base_store.patterns)
    print(f"✓ Teacher added pattern. Base count: {base_count_before}")
    
    # Now user Alice learns something
    alice_identity = UserIdentity(
        user_id="alice",
        auth_mode=AuthMode.SIMPLE,
        display_name="Alice"
    )
    
    alice_store = MultiTenantFragmentStore(
        encoder,
        alice_identity,
        base_data_path="data/test"
    )
    
    alice_store.add_pattern(
        trigger_context="my favorite color",
        response_text="Your favorite color is blue.",
        success_score=0.8,
        intent="personal"
    )
    
    alice_counts = alice_store.get_pattern_count()
    print(f"✓ Alice added pattern. Alice: {alice_counts['user']}, Base: {alice_counts['base']}")
    
    # Verify base wasn't corrupted
    base_count_after = len(teacher_store.base_store.patterns)
    assert base_count_after == base_count_before, "Base count should not change"
    print(f"✓ Base unchanged: {base_count_after}")
    
    # Now user Bob learns something
    bob_identity = UserIdentity(
        user_id="bob",
        auth_mode=AuthMode.SIMPLE,
        display_name="Bob"
    )
    
    bob_store = MultiTenantFragmentStore(
        encoder,
        bob_identity,
        base_data_path="data/test"
    )
    
    bob_store.add_pattern(
        trigger_context="my favorite color",
        response_text="Your favorite color is red.",
        success_score=0.8,
        intent="personal"
    )
    
    bob_counts = bob_store.get_pattern_count()
    print(f"✓ Bob added pattern. Bob: {bob_counts['user']}, Base: {bob_counts['base']}")
    
    # Verify users are isolated
    assert alice_counts['user'] == 1, "Alice should have 1 pattern"
    assert bob_counts['user'] == 1, "Bob should have 1 pattern"
    assert alice_counts['base'] == bob_counts['base'], "Both should see same base"
    
    print("✓ Users isolated from each other")
    print("✅ User isolation test PASSED")
    return True


def test_base_knowledge_access():
    """Test that users can access base knowledge"""
    cleanup_test_data()
    
    print("\n" + "=" * 60)
    print("TEST 3: Base Knowledge Access")
    print("=" * 60)
    
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    
    # Teacher adds base knowledge
    teacher_identity = UserIdentity(
        user_id="teacher",
        auth_mode=AuthMode.NONE,
        display_name="Teacher"
    )
    
    teacher_store = MultiTenantFragmentStore(
        encoder,
        teacher_identity,
        base_data_path="data/test"
    )
    
    teacher_store.add_pattern(
        trigger_context="hello",
        response_text="Hello! How can I help you?",
        success_score=0.9,
        intent="greeting"
    )
    
    print("✓ Teacher added greeting pattern")
    
    # User should be able to retrieve it
    user_identity = UserIdentity(
        user_id="charlie",
        auth_mode=AuthMode.SIMPLE,
        display_name="Charlie"
    )
    
    user_store = MultiTenantFragmentStore(
        encoder,
        user_identity,
        base_data_path="data/test"
    )
    
    # Retrieve patterns
    results = user_store.retrieve_patterns("hello", topk=5, min_score=0.0)
    
    print(f"✓ User retrieved {len(results)} patterns")
    
    assert len(results) > 0, "User should retrieve base patterns"
    
    pattern, score = results[0]
    print(f"✓ Best match: '{pattern.response_text}' (score: {score:.2f})")
    
    assert "Hello" in pattern.response_text, "Should get greeting"
    
    print("✅ Base knowledge access test PASSED")
    return True


def main():
    print("\n" + "=" * 60)
    print("MULTI-TENANT LILITH TESTS")
    print("=" * 60)
    
    # Clean up test data
    cleanup_test_data()
    print("✓ Test data directories prepared")
    
    # Run tests
    test1 = test_teacher_mode()
    test2 = test_user_isolation()
    test3 = test_base_knowledge_access()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Teacher Mode: {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"User Isolation: {'✅ PASSED' if test2 else '❌ FAILED'}")
    print(f"Base Access: {'✅ PASSED' if test3 else '❌ FAILED'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 ALL TESTS PASSED!")
        print("\nMulti-tenant architecture verified:")
        print("  ✅ Teachers can update base knowledge")
        print("  ✅ Users cannot corrupt base")
        print("  ✅ Users are isolated from each other")
        print("  ✅ Users can access base knowledge")
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
