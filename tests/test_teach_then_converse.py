#!/usr/bin/env python3
"""
Test: Teach the system about machine learning, then test multi-turn coherence.

This validates:
1. Learning from teaching (stores knowledge correctly)
2. Multi-turn coherence (uses taught knowledge across turns)
3. Integration (both systems working together)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from conversation_loop import ConversationLoop


def teach_topic(loop, teachings):
    """
    Teach the system a series of facts.
    
    Args:
        loop: ConversationLoop instance
        teachings: List of (question, explanation) tuples
    """
    print("📚 TEACHING PHASE")
    print("="*80 + "\n")
    
    for question, explanation in teachings:
        print(f"👤 User: {question}")
        response1 = loop.process_user_input(question)
        print(f"🤖 Bot: {response1[:80]}...")
        
        # Check if fallback
        is_fallback = any(marker in response1.lower() for marker in [
            "don't have", "not sure", "don't know", "rephrase"
        ])
        
        if is_fallback:
            print(f"   → Bot doesn't know, teaching...")
        else:
            print(f"   → Bot already has answer")
        
        print(f"\n👤 User: {explanation[:100]}...")
        response2 = loop.process_user_input(explanation)
        print(f"🤖 Bot: {response2[:80]}...")
        print()
    
    print(f"✅ Teaching complete! Taught {len(teachings)} concepts.\n")


def test_multi_turn_conversation(
    loop,
    scenario_name: str = "Machine Learning Basics",
    turns=None,
    expected_keywords=None,
):
    """
    Test multi-turn conversation after teaching.
    
    Args:
        loop: ConversationLoop instance
        scenario_name: Name of scenario
        turns: List of user inputs
        expected_keywords: Dict mapping turn index to expected keywords in response
    """
    print(f"\n{'='*80}")
    print(f"🧪 TESTING: {scenario_name}")
    print("="*80 + "\n")
    
    if turns is None:
        turns = [
            "What is machine learning?",
            "How does supervised learning work?",
            "Thanks!",
        ]
    if expected_keywords is None:
        expected_keywords = {1: ["supervised", "labeled"]}

    results = []
    
    for i, user_input in enumerate(turns):
        print(f"[Turn {i+1}/{len(turns)}]")
        print(f"👤 User: {user_input}")
        
        response = loop.process_user_input(user_input)
        print(f"🤖 Bot: {response}")
        
        # Check expectations
        if i in expected_keywords:
            expected = expected_keywords[i]
            found = [kw for kw in expected if kw.lower() in response.lower()]
            
            if found:
                print(f"   ✅ Contains expected keywords: {', '.join(found)}")
            else:
                print(f"   ❌ Missing keywords: {', '.join(expected)}")
                print(f"      (Expected at least one of: {', '.join(expected)})")
        
        print()
        results.append(response)
    
    return results


def main():
    print("\n🧠 Test: Teach ML → Test Multi-Turn Coherence\n")
    
    # Initialize system
    loop = ConversationLoop(
        history_window=10,
        composition_mode="best_match",
        learning_mode="eager"  # Learn aggressively
    )
    
    print("\n" + "="*80)
    print("TEACH → CONVERSE TEST")
    print("="*80)
    print("\nStrategy: Teach ML concepts, then test if multi-turn works")
    print("="*80 + "\n")
    
    # Phase 1: Teach machine learning concepts
    ml_teachings = [
        (
            "What is machine learning?",
            "Machine learning is a branch of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to find patterns in data."
        ),
        (
            "What are the main types of machine learning?",
            "The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning. Each type uses different approaches to learn from data."
        ),
        (
            "What is supervised learning?",
            "Supervised learning uses labeled training data where each example has a known output. The algorithm learns to map inputs to outputs. It's commonly used for classification and regression tasks."
        ),
        (
            "What is unsupervised learning?",
            "Unsupervised learning works with unlabeled data to discover hidden patterns or structures. Common techniques include clustering and dimensionality reduction."
        ),
        (
            "What is reinforcement learning?",
            "Reinforcement learning trains agents through trial and error using rewards and penalties. The agent learns to take actions that maximize cumulative reward over time."
        ),
        (
            "What is a neural network?",
            "A neural network is a machine learning model inspired by biological neurons. It consists of layers of interconnected nodes that process information through weighted connections."
        ),
        (
            "What is deep learning?",
            "Deep learning uses neural networks with many layers to learn hierarchical representations of data. It excels at tasks like image recognition and natural language processing."
        )
    ]
    
    teach_topic(loop, ml_teachings)
    
    # Phase 2: Test multi-turn coherence with taught knowledge
    print("\n" + "="*80)
    print("MULTI-TURN COHERENCE TESTS")
    print("="*80)
    
    # Test 1: Topic continuation
    test_multi_turn_conversation(
        loop,
        "Topic Continuation (ML)",
        turns=[
            "Tell me about machine learning",
            "What are the main types?",
            "Explain supervised learning",
            "What about unsupervised learning?"
        ],
        expected_keywords={
            1: ["supervised", "unsupervised", "reinforcement", "types"],
            2: ["labeled", "classification", "regression", "supervised"],
            3: ["unlabeled", "clustering", "patterns", "unsupervised"]
        }
    )
    
    # Test 2: Reference resolution
    test_multi_turn_conversation(
        loop,
        "Reference Resolution (Pronouns)",
        turns=[
            "What is deep learning?",
            "What tasks is it good at?",
            "How is it different from regular ML?"
        ],
        expected_keywords={
            1: ["image", "recognition", "language", "layers"],
            2: ["neural", "layers", "hierarchical"]
        }
    )
    
    # Test 3: Building on previous answers
    test_multi_turn_conversation(
        loop,
        "Elaboration Chain",
        turns=[
            "What is reinforcement learning?",
            "Can you give an example?",
            "What are the key components?"
        ],
        expected_keywords={
            0: ["reward", "agent", "trial"],
            1: ["reward", "action"],
            2: ["reward", "action", "agent"]
        }
    )
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")
    
    stats = loop.learner.get_learning_stats()
    
    print(f"📊 Learning Statistics:")
    print(f"   Total interactions: {stats['interaction_count']}")
    print(f"   Average success: {stats['average_success']:.3f}")
    print(f"   Patterns learned: {len(loop.fragment_store.patterns)}")
    
    initial_count = 1265  # Baseline
    new_patterns = len(loop.fragment_store.patterns) - initial_count
    print(f"\n📚 Pattern Growth:")
    print(f"   Initial patterns: {initial_count}")
    print(f"   Current patterns: {len(loop.fragment_store.patterns)}")
    print(f"   New patterns: {new_patterns}")
    
    print(f"\n💡 Key Insights:")
    print(f"   • Teaching phase should have added {len(ml_teachings)} concepts")
    print(f"   • Multi-turn tests should use taught knowledge")
    print(f"   • Topic continuation depends on pattern retrieval with context")
    print(f"   • Success = system maintains topic and uses taught info")
    
    print("\n" + "="*80)
    
    if new_patterns >= len(ml_teachings):
        print("✅ Learning from teaching: WORKING")
    else:
        print("⚠️  Learning from teaching: May need more iterations")
    
    print("\nNext: Check if responses use taught knowledge (keywords present)")
    print("="*80)


if __name__ == "__main__":
    main()
