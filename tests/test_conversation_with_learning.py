#!/usr/bin/env python3
"""
Test conversation with success-based learning enabled.

This simulates a full conversation loop where:
1. User says something
2. System retrieves and responds
3. We evaluate if it was successful
4. System learns from the outcome
5. Future responses improve
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments/retrieval_sanity'))

from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.database_fragment_store import DatabaseBackedFragmentStore
from lilith.response_composer import ResponseComposer
from lilith.conversation_state import ConversationState


def simulate_conversation_with_learning():
    """Simulate a multi-turn conversation with success-based learning."""
    
    print("=" * 80)
    print("CONVERSATION WITH SUCCESS-BASED LEARNING")
    print("=" * 80)
    
    # Initialize system
    print("\n1. Initializing system...")
    encoder = PMFlowEmbeddingEncoder(
        dimension=96,
        latent_dim=64,
        combine_mode="pm-only",
        seed=13
    )
    
    fragments = DatabaseBackedFragmentStore(
        semantic_encoder=encoder,
        storage_path="experiments/retrieval_sanity/conversation_patterns.db"
    )
    
    state = ConversationState(encoder=encoder)
    
    composer = ResponseComposer(
        fragment_store=fragments,
        conversation_state=state,
        semantic_encoder=encoder
    )
    
    print("  ✓ System ready")
    print("  ✓ Semantic retrieval: ENABLED (BNN + keywords)")
    print("  ✓ Success tracking: ENABLED (learning from outcomes)")
    
    # Simulate conversation
    print("\n2. Simulating conversation...")
    print("=" * 80)
    
    conversations = [
        # Conversation 1: Greeting exchange (should succeed)
        {
            'user': "hi there",
            'expected_success': True,
            'why': "Greeting should get greeting response"
        },
        {
            'user': "how are you doing",
            'expected_success': True,
            'why': "Follow-up greeting continues topic"
        },
        
        # Conversation 2: Topic shift to weather
        {
            'user': "what's the weather like",
            'expected_success': True,
            'why': "Weather question should get weather response"
        },
        {
            'user': "is it cold outside",
            'expected_success': True,
            'why': "Follow-up about weather continues topic"
        },
        
        # Conversation 3: Goodbye
        {
            'user': "okay I have to go",
            'expected_success': True,
            'why': "Departure should get goodbye"
        },
        {
            'user': "see you later",
            'expected_success': True,
            'why': "Goodbye confirmation"
        },
    ]
    
    turn = 0
    for conv in conversations:
        turn += 1
        user_input = conv['user']
        expected_success = conv['expected_success']
        why = conv['why']
        
        print(f"\n--- Turn {turn} ---")
        print(f"User: {user_input}")
        print(f"Expected: {'✓ Success' if expected_success else '✗ Failure'} ({why})")
        
        # Generate response
        response = composer.compose_response(
            context=user_input,
            user_input=user_input,
            use_semantic_retrieval=True,  # Use BNN + success learning
            semantic_weight=0.5
        )
        
        print(f"Bot: {response.text}")
        print(f"Pattern: {response.primary_pattern.fragment_id if response.primary_pattern else 'None'}")
        print(f"Score: {response.coherence_score:.3f}")
        
        # Record outcome (in real system, this would be based on user's next message)
        composer.record_conversation_outcome(expected_success)
        
        if expected_success:
            print("→ Recorded as SUCCESS ✓")
        else:
            print("→ Recorded as FAILURE ✗")
    
    # Check learning stats
    print("\n\n3. Learning statistics...")
    print("=" * 80)
    
    stats = fragments.get_success_stats()
    print(f"Query clusters created: {stats['query_clusters']}")
    print(f"Query→pattern pairs tracked: {stats['tracked_pairs']}")
    print(f"Total observations: {stats['total_observations']:.0f}")
    
    # Test if learning improved retrieval
    print("\n\n4. Testing learned improvements...")
    print("=" * 80)
    
    test_queries = [
        "hello",
        "how's the weather",
        "goodbye"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Get patterns with learned boosts
        patterns = fragments.retrieve_patterns_hybrid(
            query,
            topk=3,
            semantic_weight=0.5
        )
        
        print("Top patterns (with learned boosts):")
        for i, (pattern, score) in enumerate(patterns[:3], 1):
            boost = fragments.success_tracker.get_pattern_boost(query, pattern.fragment_id)
            boost_indicator = "↑" if boost > 1.0 else ("↓" if boost < 1.0 else "=")
            print(f"  {i}. {pattern.fragment_id}: {score:.3f} (boost: {boost:.2f}x {boost_indicator})")
    
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\n✓ Success-based learning is now INTEGRATED!")
    print("\nHow it works:")
    print("  1. System responds using BNN semantic + keyword retrieval")
    print("  2. Conversation outcome is recorded (success/failure)")
    print("  3. Future similar queries → boost successful patterns")
    print("  4. System continuously improves from experience")
    print("\nThis is 'learning to use the index' - the BNN provides")
    print("semantic similarity, success tracker learns what works!")


if __name__ == "__main__":
    simulate_conversation_with_learning()
