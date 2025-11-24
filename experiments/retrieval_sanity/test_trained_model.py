#!/usr/bin/env python3
"""
Test Trained Model - Demonstrate learned conversational patterns

Shows how the system responds using patterns learned from training data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from conversation_loop import ConversationLoop


def test_trained_model():
    """Test the trained model with queries from the training domain."""
    
    print("\n" + "="*70)
    print("🧪 TESTING TRAINED MODEL")
    print("="*70)
    print("\nInitializing system with trained patterns...")
    
    # Load system with trained patterns
    loop = ConversationLoop(
        history_window=10,
        composition_mode="weighted_blend",
        use_grammar=True  # Use learned syntax patterns
    )
    
    # Note: trained_patterns.json should be loaded automatically if it exists
    # Otherwise, manually specify the path in ResponseFragmentStore
    
    print("\nPattern statistics:")
    stats = loop.fragment_store.get_stats()
    print(f"  Total patterns: {stats['total_patterns']}")
    print(f"  Learned patterns: {stats['learned_patterns']}")
    
    if hasattr(loop.composer, 'syntax_stage') and loop.composer.syntax_stage:
        print(f"  Syntax patterns: {len(loop.composer.syntax_stage.patterns)}")
    
    print("\n" + "="*70)
    print("TESTING ON SIMILAR QUERIES")
    print("="*70)
    
    # Test with queries similar to training data
    test_queries = [
        "Tell me about neural networks",
        "How do neural nets learn?",
        "What's a Bayesian network?",
        "Explain attention mechanisms",
        "What are embeddings used for?",
        "How does fine-tuning work?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}/{len(test_queries)}]")
        print(f"👤 User: {query}")
        print()
        
        response = loop.process_user_input(query)
        
        print(f"🤖 Bot: {response}")
        print("-" * 70)
    
    print("\n" + "="*70)
    print("✅ Test Complete")
    print("="*70)
    print("\nObservations:")
    print("  - System should respond using learned patterns")
    print("  - Responses should be on-topic (neural networks, ML)")
    print("  - Grammar should follow learned syntactic structures")
    print("  - No LLM used - pure retrieval and composition!")
    print()


if __name__ == "__main__":
    test_trained_model()
