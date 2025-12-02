#!/usr/bin/env python3
"""
Test Wikipedia disambiguation resolution.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lilith.knowledge_augmenter import WikipediaLookup


def test_disambiguation():
    """Test that disambiguation pages are resolved correctly."""
    print("\n" + "="*60)
    print("TEST: Wikipedia Disambiguation Resolution")
    print("="*60)
    
    wiki = WikipediaLookup()
    
    # Test 1: "Python" with programming context
    print("\n  Test 1: 'What is Python programming?'")
    result = wiki.lookup("What is Python programming?")
    if result:
        print(f"    ✅ Title: {result['title']}")
        print(f"    Extract: {result['extract'][:80]}...")
        if 'programming' in result['extract'].lower():
            print(f"    ✅ Correctly resolved to programming language!")
        else:
            print(f"    ⚠️  May not have resolved correctly")
    else:
        print(f"    ❌ No result")
    
    # Test 2: "Python" without context (may get disambiguation)
    print("\n  Test 2: 'What is Python?'")
    result = wiki.lookup("What is Python?")
    if result:
        print(f"    Title: {result['title']}")
        print(f"    Extract: {result['extract'][:80]}...")
    else:
        print(f"    ❌ No result")
    
    # Test 3: Another ambiguous term with context
    print("\n  Test 3: 'What is Java programming language?'")
    result = wiki.lookup("What is Java programming language?")
    if result:
        print(f"    ✅ Title: {result['title']}")
        print(f"    Extract: {result['extract'][:80]}...")
        if 'programming' in result['extract'].lower() or 'software' in result['extract'].lower():
            print(f"    ✅ Correctly resolved to programming language!")
    else:
        print(f"    ❌ No result")


def main():
    """Run disambiguation tests."""
    print("\n🧪 Testing Wikipedia Disambiguation Resolution")
    
    test_disambiguation()
    
    print("\n" + "="*60)
    print("✅ Disambiguation Tests Complete")
    print("="*60)


if __name__ == "__main__":
    main()
