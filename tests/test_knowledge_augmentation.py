#!/usr/bin/env python3
"""
Test Wikipedia Knowledge Augmentation

Demonstrates external knowledge lookup and automatic learning.
When the system doesn't know something, it:
1. Queries Wikipedia
2. Extracts factual summary
3. Responds with knowledge
4. Learns the pattern for future use
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lilith.knowledge_augmenter import WikipediaLookup, KnowledgeAugmenter


def test_wikipedia_lookup():
    """Test basic Wikipedia article lookup."""
    print("=" * 70)
    print("TEST 1: Wikipedia Article Lookup")
    print("=" * 70)
    
    wiki = WikipediaLookup()
    
    test_queries = [
        "What is machine learning?",
        "Who is Ada Lovelace?",
        "Tell me about Python programming",
        "What is a Merkle tree?",
        "quantum computing",  # No question formatting
        "nonexistent_article_xyz123"  # Should fail gracefully
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        result = wiki.lookup(query)
        
        if result:
            print(f"   ✅ Found: {result['title']}")
            print(f"   📄 Summary: {result['extract'][:100]}...")
            print(f"   🔗 URL: {result['url']}")
            print(f"   📊 Confidence: {result['confidence']}")
        else:
            print("   ❌ No result found")


def test_knowledge_augmenter():
    """Test the knowledge augmentation system."""
    print("\n" + "=" * 70)
    print("TEST 2: Knowledge Augmentation System")
    print("=" * 70)
    
    augmenter = KnowledgeAugmenter(enabled=True)
    
    test_queries = [
        "What is neural network?",
        "Who invented Python?",
        "Explain blockchain",
        "What is recursion?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        result = augmenter.lookup(query, min_confidence=0.6)
        
        if result:
            response, confidence, source = result
            print(f"   ✅ Response: {response[:150]}...")
            print(f"   📊 Confidence: {confidence}, Source: {source}")
        else:
            print("   ❌ No knowledge found")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("STATISTICS:")
    stats = augmenter.get_stats()
    print(f"  Total lookups: {stats['lookups']}")
    print(f"  Successful: {stats['successes']}")
    print(f"  Success rate: {stats['success_rate']}")
    print(f"  Enabled: {stats['enabled']}")


def test_query_cleaning():
    """Test query cleaning for Wikipedia article titles."""
    print("\n" + "=" * 70)
    print("TEST 3: Query Cleaning")
    print("=" * 70)
    
    wiki = WikipediaLookup()
    
    test_cases = [
        ("What is machine learning?", "Machine Learning"),
        ("Tell me about Python", "Python"),
        ("Who is Ada Lovelace?", "Ada Lovelace"),
        ("quantum computing", "Quantum Computing"),
        ("how does blockchain work", "Blockchain Work"),
    ]
    
    for query, expected_pattern in test_cases:
        cleaned = wiki._clean_query(query)
        print(f"  '{query}'")
        print(f"    → '{cleaned}'")
        print(f"    Expected pattern: '{expected_pattern}'")
        print()


def test_integration_scenario():
    """
    Test realistic integration scenario:
    User asks about something system doesn't know,
    system looks it up, learns the pattern.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Integration Scenario")
    print("=" * 70)
    print("\nScenario: User asks about quantum entanglement")
    print("System has no pattern for this topic.\n")
    
    augmenter = KnowledgeAugmenter(enabled=True)
    
    user_query = "What is quantum entanglement?"
    print(f"👤 User: {user_query}")
    
    # Simulate low confidence retrieval (no pattern found)
    print("  ⚠️ Pattern retrieval: confidence < 0.6 (no good match)")
    
    # Try external lookup
    print("  🔍 Trying external knowledge lookup...")
    result = augmenter.lookup(user_query, min_confidence=0.6)
    
    if result:
        response, confidence, source = result
        print(f"\n🤖 Bot: {response}")
        print(f"\n  🌐 Source: {source}")
        print(f"  📊 Confidence: {confidence}")
        print(f"  💡 This response will be learned as:")
        print(f"     Trigger: 'quantum entanglement'")
        print(f"     Response: '{response[:60]}...'")
        print(f"     Intent: 'taught' (from external source)")
        print(f"\n  ✅ Future queries about quantum entanglement will use learned pattern")
    else:
        print("\n  ❌ No external knowledge found - standard fallback")


if __name__ == "__main__":
    print("\n🌐 KNOWLEDGE AUGMENTATION TEST SUITE")
    print("Testing Wikipedia integration for automatic knowledge acquisition\n")
    
    try:
        test_query_cleaning()
        test_wikipedia_lookup()
        test_knowledge_augmenter()
        test_integration_scenario()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETE")
        print("=" * 70)
        print("\nKnowledge augmentation enables:")
        print("  • Automatic knowledge acquisition from Wikipedia")
        print("  • Learning patterns from external sources")
        print("  • Self-improvement through use")
        print("  • No manual dataset curation needed")
        print("\nThe system learns what users actually ask about!")
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nMake sure 'requests' library is installed:")
        print("  pip install requests")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
