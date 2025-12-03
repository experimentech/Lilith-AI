#!/usr/bin/env python3
"""
Test Q&A retrieval quality with hybrid system.

Tests different types of queries:
- Exact matches
- Fuzzy matches (typos/variations)
- Paraphrased questions
- Related questions
"""

from lilith.session import LilithSession, SessionConfig


def test_qa_retrieval():
    """Test retrieval quality across different query types."""
    
    print("🧪 Testing Q&A Retrieval Quality")
    print("=" * 70)
    print()
    
    # Create session (bootstrap user has 116 Q&A pairs)
    config = SessionConfig(
        enable_knowledge_augmentation=False,  # Test pure retrieval
    )
    session = LilithSession("bootstrap", context_id="qa_test", config=config)
    
    # Test categories
    test_cases = {
        "Exact Matches": [
            ("What is Python?", "Python", 0.95),
            ("What is machine learning?", "machine learning", 0.95),
            ("What are neural networks?", "neural", 0.95),
        ],
        "Fuzzy Matches (Typos)": [
            ("What's Python?", "Python", 0.75),  # Contraction
            ("Whats machine learning?", "machine learning", 0.70),  # Missing apostrophe
            ("What r neural networks?", "neural", 0.65),  # Abbreviation
        ],
        "Case Variations": [
            ("what is python?", "Python", 0.95),  # Lowercase
            ("WHAT IS PYTHON?", "Python", 0.95),  # Uppercase
            ("WhAt Is PyThOn?", "Python", 0.95),  # Mixed case
        ],
        "Paraphrased": [
            ("Can you tell me about Python?", "Python", 0.60),
            ("Explain machine learning", "machine learning", 0.60),
            ("How do neural networks function?", "neural", 0.60),
        ],
        "Related Questions": [
            ("What is Python used for?", "Python", 0.70),
            ("How does machine learning work?", "machine learning", 0.90),
            ("What is a neuron in neural networks?", "neuron", 0.85),
        ],
    }
    
    total_tests = 0
    passed_tests = 0
    results = {}
    
    for category, queries in test_cases.items():
        print(f"\n📋 {category}")
        print("-" * 70)
        
        category_passed = 0
        category_total = len(queries)
        
        for query, expected_keyword, min_confidence in queries:
            response = session.process_message(query)
            
            # Check if keyword is in response
            has_keyword = expected_keyword.lower() in response.text.lower()
            
            # Check confidence threshold
            meets_threshold = response.confidence >= min_confidence
            
            # Overall pass
            passed = has_keyword and meets_threshold
            
            if passed:
                status = "✅"
                passed_tests += 1
                category_passed += 1
            else:
                status = "❌"
            
            total_tests += 1
            
            print(f"{status} Query: '{query}'")
            print(f"   Confidence: {response.confidence:.3f} (min: {min_confidence:.3f})")
            print(f"   Contains '{expected_keyword}': {has_keyword}")
            if not passed:
                print(f"   Response: {response.text[:60]}...")
        
        results[category] = (category_passed, category_total)
        print(f"\n   Category: {category_passed}/{category_total} passed")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 OVERALL RESULTS")
    print("=" * 70)
    
    for category, (passed, total) in results.items():
        pct = (passed / total * 100) if total > 0 else 0
        print(f"{category:30s}: {passed:2d}/{total:2d} ({pct:5.1f}%)")
    
    overall_pct = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print("-" * 70)
    print(f"{'TOTAL':30s}: {passed_tests:2d}/{total_tests:2d} ({overall_pct:5.1f}%)")
    print("=" * 70)
    
    # Interpretation
    print()
    if overall_pct >= 90:
        print("🎉 EXCELLENT! Hybrid retrieval working exceptionally well!")
    elif overall_pct >= 75:
        print("✅ GOOD! Retrieval quality is solid.")
    elif overall_pct >= 60:
        print("✓ ACCEPTABLE. Room for improvement.")
    else:
        print("⚠️  Needs work. Consider tuning thresholds.")
    
    print()
    print(f"Hybrid Retrieval Performance: {overall_pct:.1f}%")
    print()
    
    # Specific insights
    print("Key Insights:")
    print("  • Exact matches should be ~100% (1.0 confidence)")
    print("  • Fuzzy matches handle typos (0.75+ confidence)")
    print("  • Token overlap handles rearrangement (0.6+ confidence)")
    print("  • Semantic BNN handles paraphrasing (0.4-0.8 confidence)")
    print()
    
    return overall_pct


if __name__ == "__main__":
    test_qa_retrieval()
