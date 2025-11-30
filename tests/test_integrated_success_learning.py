#!/usr/bin/env python3
"""
Test success-based learning integrated into database retrieval.

This demonstrates the full "learning to use the index" system:
1. BNN + Database hybrid retrieval (base capability)
2. Success tracking learns from outcomes (experience)
3. Future retrievals boosted/penalized based on past success
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments/retrieval_sanity'))

from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.database_fragment_store import DatabaseBackedFragmentStore


def test_integrated_learning():
    print("=" * 80)
    print("INTEGRATED SUCCESS-BASED LEARNING TEST")
    print("=" * 80)
    
    # Initialize system
    print("\n1. Initializing BNN + Database + Success Tracker...")
    encoder = PMFlowEmbeddingEncoder(
        dimension=96,
        latent_dim=64,
        combine_mode="pm-only",
        seed=13
    )
    
    store = DatabaseBackedFragmentStore(
        semantic_encoder=encoder,
        storage_path="experiments/retrieval_sanity/conversation_patterns.db"
    )
    
    print(f"  ✓ Database loaded")
    print(f"  ✓ Success tracker initialized")
    
    # Test retrieval BEFORE learning
    print("\n2. Testing retrieval BEFORE learning...")
    print("-" * 80)
    
    test_query = "hello how are you"
    patterns_before = store.retrieve_patterns_hybrid(
        test_query,
        topk=5,
        semantic_weight=0.5
    )
    
    print(f"\nQuery: '{test_query}'")
    print("Top patterns:")
    for i, (pattern, score) in enumerate(patterns_before[:3], 1):
        print(f"  {i}. {pattern.fragment_id}: {score:.3f}")
        print(f"     Response: {pattern.response_text[:60]}...")
    
    # Simulate learning from conversation outcomes
    print("\n\n3. Simulating conversation outcomes...")
    print("-" * 80)
    
    # Scenario: greetings work with greeting patterns, fail with others
    # Use ACTUAL pattern IDs from the database
    learning_scenarios = [
        ("hi", "seed_greeting_0", True),
        ("hello", "pattern_greeting_105", True),
        ("hey there", "seed_greeting_0", True),
        ("how are you", "learned_csv_0", True),  # This is the "i'm fine" response
        
        # Wrong mappings - these patterns exist but shouldn't match greetings
        ("hello", "pattern_learned_1162", False),  # The "god" pattern
        ("hi", "pattern_question_info_104", False),  # The ML info pattern
    ]
    
    print("\nRecording outcomes:")
    for query, pattern_id, success in learning_scenarios:
        store.record_conversation_outcome(query, pattern_id, success)
        result = "✓" if success else "✗"
        print(f"  {result} '{query}' + '{pattern_id}' → {'SUCCESS' if success else 'FAIL'}")
    
    # Check stats
    stats = store.get_success_stats()
    print(f"\nSuccess tracker stats:")
    print(f"  Query clusters: {stats['query_clusters']}")
    print(f"  Tracked pairs: {stats['tracked_pairs']}")
    print(f"  Total observations: {stats['total_observations']:.1f}")
    
    # Test retrieval AFTER learning
    print("\n\n4. Testing retrieval AFTER learning...")
    print("-" * 80)
    
    patterns_after = store.retrieve_patterns_hybrid(
        test_query,
        topk=5,
        semantic_weight=0.5
    )
    
    print(f"\nQuery: '{test_query}'")
    print("Top patterns (with learned boosts):")
    for i, (pattern, score) in enumerate(patterns_after[:3], 1):
        print(f"  {i}. {pattern.fragment_id}: {score:.3f}")
        print(f"     Response: {pattern.response_text[:60]}...")
    
    # Compare
    print("\n\n5. Comparing BEFORE vs AFTER learning...")
    print("=" * 80)
    
    print("\nScore changes:")
    before_dict = {p.fragment_id: score for p, score in patterns_before}
    after_dict = {p.fragment_id: score for p, score in patterns_after}
    
    all_ids = set(before_dict.keys()) | set(after_dict.keys())
    for pid in sorted(all_ids):
        before_score = before_dict.get(pid, 0.0)
        after_score = after_dict.get(pid, 0.0)
        delta = after_score - before_score
        
        if abs(delta) > 0.001:
            direction = "↑" if delta > 0 else "↓"
            print(f"  {pid}: {before_score:.3f} → {after_score:.3f} ({delta:+.3f} {direction})")
    
    # Success indicators
    print("\n✓ Expected behavior:")
    print("  • Greeting patterns should get BOOSTED (↑)")
    print("  • Other patterns should stay same or get PENALIZED (↓)")
    print("\nThis is 'learning to use the index' - the system learns from")
    print("conversation outcomes which patterns work for which queries!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_integrated_learning()
