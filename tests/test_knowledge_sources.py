#!/usr/bin/env python3
"""
Test the new knowledge augmentation sources.

Tests WordNet, Wiktionary, and Free Dictionary integration.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lilith.knowledge_augmenter import KnowledgeAugmenter


def test_knowledge_sources():
    """Test all knowledge sources with various queries."""
    
    print("=" * 70)
    print("Testing Knowledge Augmentation Sources")
    print("=" * 70)
    
    augmenter = KnowledgeAugmenter(enabled=True)
    
    # Test cases covering different query types
    test_queries = [
        # Word definitions
        ("What does ephemeral mean?", "wiktionary/free_dictionary"),
        ("Define recalcitrant", "wiktionary/free_dictionary"),
        
        # Synonyms/antonyms (WordNet)
        ("What's a synonym for happy?", "wordnet"),
        ("Antonym of good", "wordnet"),
        ("Another word for beautiful", "wordnet"),
        
        # Single words (should try multiple sources)
        ("serendipity", "any"),
        ("python", "wikipedia"),
        
        # General knowledge (Wikipedia)
        ("What is machine learning?", "wikipedia"),
        ("Who was Ada Lovelace?", "wikipedia"),
        
        # Mixed queries
        ("What does AI mean?", "any"),
    ]
    
    results = {
        'total': 0,
        'success': 0,
        'wordnet': 0,
        'wiktionary': 0,
        'free_dictionary': 0,
        'wikipedia': 0,
        'failed': []
    }
    
    for query, expected_source in test_queries:
        print(f"\n{'─' * 70}")
        print(f"Query: {query}")
        print(f"Expected source: {expected_source}")
        print()
        
        results['total'] += 1
        
        result = augmenter.lookup(query, min_confidence=0.6)
        
        if result:
            response_text, confidence, source = result
            results['success'] += 1
            results[source] += 1
            
            print(f"✅ Found! (confidence: {confidence:.2f})")
            print(f"Source: {source}")
            print(f"Response: {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
            
            if expected_source != "any" and source not in expected_source:
                print(f"⚠️  Warning: Expected {expected_source}, got {source}")
        else:
            results['failed'].append(query)
            print(f"❌ No result found")
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total queries: {results['total']}")
    print(f"Successful: {results['success']} ({results['success']/results['total']*100:.1f}%)")
    print()
    print("By source:")
    print(f"  WordNet: {results['wordnet']}")
    print(f"  Wiktionary: {results['wiktionary']}")
    print(f"  Free Dictionary: {results['free_dictionary']}")
    print(f"  Wikipedia: {results['wikipedia']}")
    
    if results['failed']:
        print(f"\nFailed queries ({len(results['failed'])}):")
        for q in results['failed']:
            print(f"  - {q}")
    
    # Get augmenter stats
    print(f"\n{'=' * 70}")
    stats = augmenter.get_stats()
    print("Augmenter Statistics:")
    print(f"  Total lookups: {stats['lookups']}")
    print(f"  Success rate: {stats['success_rate']}")
    print(f"  WordNet available: {stats['wordnet_available']}")
    print(f"  Source breakdown: {stats['sources']}")
    
    return results


if __name__ == "__main__":
    try:
        results = test_knowledge_sources()
        
        # Exit with success if at least 70% succeeded
        success_rate = results['success'] / results['total']
        if success_rate >= 0.7:
            print(f"\n✅ Test passed! ({success_rate*100:.1f}% success rate)")
            sys.exit(0)
        else:
            print(f"\n⚠️  Test concerns: Only {success_rate*100:.1f}% success rate")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
