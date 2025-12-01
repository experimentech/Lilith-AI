#!/usr/bin/env python3
"""
Test smart gap-filling and seamless online learning.

Demonstrates how Lilith fills knowledge gaps using external sources
before falling back to "I don't know".
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.response_fragments_sqlite import ResponseFragmentStoreSQLite
from lilith.response_composer import ResponseComposer
from lilith.conversation_state import ConversationState


def test_gap_filling():
    """Test gap-filling with external knowledge sources."""
    
    print("=" * 70)
    print("Testing Smart Gap-Filling & Seamless Online Learning")
    print("=" * 70)
    print()
    
    # Setup
    import tempfile
    import os
    
    # Create a temporary database file
    fd, temp_db = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    try:
        encoder = PMFlowEmbeddingEncoder()
        store = ResponseFragmentStoreSQLite(
            semantic_encoder=encoder,  # Correct parameter name
            storage_path=temp_db,  # Use temp file instead of :memory:
            enable_fuzzy_matching=True,
            bootstrap_if_empty=False  # Don't bootstrap for clean test
        )
        
        state = ConversationState(encoder)
        
        composer = ResponseComposer(
            fragment_store=store,
            conversation_state=state,
            semantic_encoder=encoder,
            enable_knowledge_augmentation=True,  # Enable external sources
            composition_mode="best_match"
        )
    
    # Add some basic patterns
    store.add_pattern(
        trigger_context="What is Python?",
        response_text="Python is a high-level programming language.",
        intent="learned_knowledge"
    )
    
    store.add_pattern(
        trigger_context="What is a function?",
        response_text="A function is a reusable block of code.",
        intent="learned_knowledge"
    )
    
    print("📚 Base Knowledge Added:")
    print("  - What is Python?")
    print("  - What is a function?")
    print()
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Known Query (Direct Match)',
            'query': 'What is Python?',
            'expected': 'Should match existing pattern'
        },
        {
            'name': 'Unknown Word (Gap Filling)',
            'query': 'What does ephemeral mean?',
            'expected': 'Should look up in Wiktionary/WordNet and respond'
        },
        {
            'name': 'Synonym Query (Gap Filling)',
            'query': 'What is a synonym for happy?',
            'expected': 'Should use WordNet to answer'
        },
        {
            'name': 'Unknown Concept (Wikipedia)',
            'query': 'What is machine learning?',
            'expected': 'Should look up in Wikipedia and respond'
        },
        {
            'name': 'Query with Unknown Term',
            'query': 'How do you use serendipity in a sentence?',
            'expected': 'Should fill gap about "serendipity" then attempt match'
        },
        {
            'name': 'Complex Unknown Query',
            'query': 'What is neuroplasticity?',
            'expected': 'Should look up and learn the term'
        },
        {
            'name': 'Truly Unknown (Gibberish)',
            'query': 'What is a flibbertigibbet quantum parser?',
            'expected': 'Should fallback gracefully after trying sources'
        }
    ]
    
    results = {
        'direct_match': 0,
        'gap_filled': 0,
        'external_source': 0,
        'fallback': 0
    }
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'─' * 70}")
        print(f"Test {i}: {scenario['name']}")
        print(f"Query: \"{scenario['query']}\"")
        print(f"Expected: {scenario['expected']}")
        print()
        
        # Compose response
        response = composer.compose_response(
            context=scenario['query'],
            user_input=scenario['query']
        )
        
        # Analyze result
        print(f"Result:")
        print(f"  Confidence: {response.confidence:.3f}")
        print(f"  Is Fallback: {response.is_fallback}")
        print(f"  Fragment IDs: {response.fragment_ids}")
        print(f"  Response: {response.text[:150]}{'...' if len(response.text) > 150 else ''}")
        
        # Categorize result
        if not response.is_fallback and response.confidence >= 0.5:
            if 'gap_filled' in str(response.fragment_ids):
                results['gap_filled'] += 1
                print(f"  ✨ Gap filled successfully!")
            else:
                results['direct_match'] += 1
                print(f"  ✅ Direct match!")
        elif response.is_fallback and not response.is_low_confidence:
            # External source used
            results['external_source'] += 1
            print(f"  💡 External knowledge used!")
        else:
            results['fallback'] += 1
            print(f"  ⚠️  Fell back to 'I don't know'")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total queries: {len(scenarios)}")
    print()
    print(f"Results:")
    print(f"  Direct matches: {results['direct_match']}")
    print(f"  Gap-filled: {results['gap_filled']}")
    print(f"  External sources: {results['external_source']}")
    print(f"  True fallbacks: {results['fallback']}")
    print()
    
    # Calculate seamlessness score
    seamless = results['direct_match'] + results['gap_filled'] + results['external_source']
    seamless_rate = seamless / len(scenarios) * 100
    
    print(f"Seamlessness Score: {seamless}/{len(scenarios)} ({seamless_rate:.1f}%)")
    print()
    
        if seamless_rate >= 80:
            print("✅ Excellent! Most queries handled seamlessly with online learning.")
        elif seamless_rate >= 60:
            print("✓ Good! Majority of queries handled without hard fallbacks.")
        else:
            print("⚠️  Needs improvement. Many queries falling back.")
        
        # Show knowledge augmenter stats
        if composer.knowledge_augmenter:
            stats = composer.knowledge_augmenter.get_stats()
            print(f"\nKnowledge Augmenter Statistics:")
            print(f"  Total lookups: {stats['lookups']}")
            print(f"  Success rate: {stats['success_rate']}")
            print(f"  Source breakdown:")
            for source, count in stats['sources'].items():
                if count > 0:
                    print(f"    {source}: {count}")
        
        return seamless_rate
    
    finally:
        # Cleanup temporary database
        if os.path.exists(temp_db):
            os.remove(temp_db)


def test_gap_filling_example():
    """
    Demonstrate gap-filling with a practical example.
    
    Shows how a query about an unknown concept gets filled in real-time.
    """
    print("\n" + "=" * 70)
    print("Practical Example: Gap-Filling in Action")
    print("=" * 70)
    print()
    
    # Setup
    encoder = PMFlowEmbeddingEncoder()
    store = ResponseFragmentStoreSQLite(
        semantic_encoder=encoder,
        storage_path=":memory:",
        enable_fuzzy_matching=True,
        bootstrap_if_empty=False
    )
    
    state = ConversationState(encoder)
    
    composer = ResponseComposer(
        fragment_store=store,
        conversation_state=state,
        semantic_encoder=encoder,
        enable_knowledge_augmentation=True,
        composition_mode="best_match"
    )
    
    # Teach it about programming
    store.add_pattern(
        trigger_context="What is recursion?",
        response_text="Recursion is when a function calls itself.",
        intent="learned_knowledge"
    )
    
    print("Scenario: User asks about 'memoization' (unknown concept)")
    print()
    print("Query: 'What is memoization in recursion?'")
    print()
    print("Expected Process:")
    print("  1. No direct match for 'memoization'")
    print("  2. Identify 'memoization' as unknown term")
    print("  3. Look up 'memoization' in Wikipedia")
    print("  4. Learn the definition")
    print("  5. Respond with learned knowledge")
    print()
    print("Actual Process:")
    print("-" * 70)
    
    response = composer.compose_response(
        context="What is memoization in recursion?",
        user_input="What is memoization in recursion?"
    )
    
    print()
    print(f"Response: {response.text}")
    print(f"Confidence: {response.confidence:.3f}")
    print(f"Source: {response.fragment_ids}")
    
    if not response.is_low_confidence:
        print()
        print("✨ Success! Lilith filled the knowledge gap and responded seamlessly.")
        print("   The user never knew there was a gap - it was filled on-the-fly!")
    else:
        print()
        print("⚠️  Gap-filling didn't help this time.")


if __name__ == "__main__":
    try:
        # Run main test
        seamless_rate = test_gap_filling()
        
        # Run practical example
        test_gap_filling_example()
        
        # Exit based on success rate
        if seamless_rate >= 60:
            print(f"\n✅ Tests passed! ({seamless_rate:.1f}% seamlessness)")
            sys.exit(0)
        else:
            print(f"\n⚠️  Tests concerns: Only {seamless_rate:.1f}% seamless")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
