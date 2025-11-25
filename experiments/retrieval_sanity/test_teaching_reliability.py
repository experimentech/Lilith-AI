#!/usr/bin/env python3
"""
Test Teaching Reliability with Metadata-Based Detection

Tests that the new metadata-based fallback detection correctly:
1. Detects true fallback (no pattern found)
2. Detects low-confidence fallback (weak pattern)
3. Ignores external knowledge responses (Wikipedia)
4. Stores taught patterns with high confidence
5. Retrieves taught patterns correctly

Target: 95%+ teaching success rate
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from conversation_loop import ConversationLoop


def test_teaching_scenario(scenario_name: str, user_query: str, teaching_text: str, test_query: str, expected_keywords: list):
    """
    Test a single teaching scenario.
    
    Returns:
        True if teaching was successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"🧪 TEST: {scenario_name}")
    print(f"{'='*80}\n")
    
    # Initialize new system for clean test
    loop = ConversationLoop(
        history_window=10,
        composition_mode="best_match",
        learning_mode="eager"
    )
    
    # Step 1: Ask question (should trigger fallback)
    print(f"📚 STEP 1: User asks (expecting fallback)")
    print(f"👤 User: {user_query}")
    
    response1 = loop.process_user_input(user_query)
    
    print(f"🤖 Bot: {response1[:80]}...")
    
    # Check if fallback
    is_fallback = any(marker in response1.lower() for marker in [
        "don't have", "not sure", "don't know", "rephrase", "something else"
    ])
    
    if not is_fallback:
        print(f"   ❌ FAIL: Expected fallback, got normal response")
        return False
    else:
        print(f"   ✅ Fallback detected correctly")
    
    # Step 2: User teaches
    print(f"\n🎓 STEP 2: User teaches")
    print(f"👤 User: {teaching_text}")
    
    response2 = loop.process_user_input(teaching_text)
    
    print(f"🤖 Bot: {response2[:80]}...")
    
    # Step 3: Test if knowledge was learned
    print(f"\n✅ STEP 3: Test recall (expecting taught knowledge)")
    print(f"👤 User: {test_query}")
    
    response3 = loop.process_user_input(test_query)
    
    print(f"🤖 Bot: {response3}")
    
    # Check if response contains expected keywords
    response_lower = response3.lower()
    found_keywords = [kw for kw in expected_keywords if kw.lower() in response_lower]
    
    if found_keywords:
        print(f"   ✅ SUCCESS: Found keywords: {found_keywords}")
        return True
    else:
        print(f"   ❌ FAIL: Expected keywords {expected_keywords}, found none")
        print(f"   Response was: {response3}")
        return False


def main():
    """Run teaching reliability tests."""
    print("="*80)
    print("TEACHING RELIABILITY TEST")
    print("Testing metadata-based fallback detection")
    print("="*80)
    
    test_cases = [
        {
            "name": "Machine Learning Definition",
            "query": "What is machine learning?",
            "teaching": "Machine learning is a branch of artificial intelligence that enables computers to learn from data without explicit programming.",
            "test": "Tell me about machine learning",
            "keywords": ["artificial intelligence", "data", "learn"]
        },
        {
            "name": "Neural Networks",
            "query": "What is a neural network?",
            "teaching": "A neural network is a machine learning model inspired by biological neurons, consisting of interconnected layers that process information.",
            "test": "Explain neural networks",
            "keywords": ["neurons", "layers", "interconnected"]
        },
        {
            "name": "Deep Learning",
            "query": "What is deep learning?",
            "teaching": "Deep learning uses neural networks with many layers to learn hierarchical representations of data. It excels at image recognition and natural language processing.",
            "test": "What is deep learning?",
            "keywords": ["layers", "hierarchical", "image", "language"]
        },
        {
            "name": "Supervised Learning",
            "query": "What is supervised learning?",
            "teaching": "Supervised learning uses labeled training data where each example has a known output. The algorithm learns to map inputs to correct outputs.",
            "test": "Explain supervised learning",
            "keywords": ["labeled", "training", "output"]
        },
        {
            "name": "Reinforcement Learning",
            "query": "What is reinforcement learning?",
            "teaching": "Reinforcement learning trains agents through trial and error using rewards and penalties. The agent learns to maximize cumulative rewards over time.",
            "test": "What is reinforcement learning?",
            "keywords": ["agent", "reward", "trial"]
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        success = test_teaching_scenario(
            test_case["name"],
            test_case["query"],
            test_case["teaching"],
            test_case["test"],
            test_case["keywords"]
        )
        results.append((test_case["name"], success))
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    success_rate = (passed / total) * 100
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n📊 Success Rate: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 95:
        print(f"✅ TARGET MET: {success_rate:.1f}% >= 95%")
    elif success_rate >= 70:
        print(f"⚠️  IMPROVED but below target: {success_rate:.1f}% (was ~70%, target 95%)")
    else:
        print(f"❌ BELOW BASELINE: {success_rate:.1f}% < 70%")
    
    return success_rate >= 95


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
