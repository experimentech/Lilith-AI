#!/usr/bin/env python3
"""
Test multi-turn conversation coherence.

Tests:
1. Topic continuation - staying on topic across turns
2. Reference resolution - understanding "it", "that", "the one you mentioned"
3. Context memory - remembering earlier parts of conversation
4. Topic transitions - handling natural topic changes
5. Clarification handling - dealing with follow-up questions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from conversation_loop import ConversationLoop


def test_conversation_scenario(loop, scenario_name, turns, expected_behaviors):
    """
    Test a multi-turn conversation scenario.
    
    Args:
        loop: ConversationLoop instance
        scenario_name: Name of test scenario
        turns: List of user inputs
        expected_behaviors: Dict mapping turn indices to expected behavior descriptions
    """
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*80}\n")
    
    results = []
    
    for i, user_input in enumerate(turns):
        print(f"[Turn {i+1}/{len(turns)}]")
        print(f"👤 User: {user_input}")
        
        response = loop.process_user_input(user_input)
        print(f"🤖 Bot: {response}\n")
        
        # Check if expected behavior occurred
        if i in expected_behaviors:
            expected = expected_behaviors[i]
            
            # Simple heuristic checks
            checks = {
                'maintains_topic': lambda r: len(set(r.lower().split()) & set(user_input.lower().split())) > 2,
                'not_repetitive': lambda r: r != results[-1] if results else True,
                'references_previous': lambda r: any(word in r.lower() for word in ['it', 'that', 'this', 'mentioned']),
                'gives_fallback': lambda r: any(marker in r.lower() for marker in ["don't", "not sure", "rephrase"]),
                'no_fallback': lambda r: not any(marker in r.lower() for marker in ["don't", "not sure", "rephrase"])
            }
            
            passed = []
            failed = []
            
            for check_name, check_func in checks.items():
                if check_name in expected:
                    if check_func(response):
                        passed.append(check_name)
                    else:
                        failed.append(check_name)
            
            if failed:
                print(f"   ⚠️  Expected: {expected}")
                print(f"   ❌ Failed: {', '.join(failed)}")
            else:
                print(f"   ✅ Behavior matches expectations")
        
        results.append(response)
    
    return results


def main():
    print("🧠 Testing Multi-Turn Coherence\n")
    
    # Initialize system
    loop = ConversationLoop(
        history_window=10,  # Increased for multi-turn tests
        composition_mode="best_match",
        learning_mode="moderate"
    )
    
    print("\n" + "="*80)
    print("MULTI-TURN COHERENCE TEST SUITE")
    print("="*80)
    print("\nTesting conversation memory, topic tracking, and reference resolution")
    print("="*80 + "\n")
    
    # Test 1: Topic Continuation
    test_conversation_scenario(
        loop,
        "Topic Continuation",
        turns=[
            "Hello!",
            "Tell me about machine learning",
            "What are the main types?",
            "Which one is used for classification?"
        ],
        expected_behaviors={
            2: ['maintains_topic', 'not_repetitive', 'no_fallback'],
            3: ['maintains_topic', 'not_repetitive']
        }
    )
    
    # Reset for next test
    loop.history.clear()
    loop.conversation_state = type(loop.conversation_state)(
        loop.conversation_state.encoder,
        decay=loop.conversation_state.decay,
        max_topics=loop.conversation_state.max_topics
    )
    
    # Test 2: Reference Resolution
    test_conversation_scenario(
        loop,
        "Reference Resolution (Pronouns)",
        turns=[
            "What is Python?",
            "Is it difficult to learn?",
            "What can you build with it?"
        ],
        expected_behaviors={
            1: ['maintains_topic', 'no_fallback'],
            2: ['maintains_topic', 'no_fallback']
        }
    )
    
    # Reset
    loop.history.clear()
    loop.conversation_state = type(loop.conversation_state)(
        loop.conversation_state.encoder,
        decay=loop.conversation_state.decay,
        max_topics=loop.conversation_state.max_topics
    )
    
    # Test 3: Context Memory
    test_conversation_scenario(
        loop,
        "Context Memory (Recalling Earlier Info)",
        turns=[
            "My favorite color is blue",
            "I also like hiking",
            "What's my favorite color?"
        ],
        expected_behaviors={
            2: ['no_fallback']  # Should remember "blue"
        }
    )
    
    # Reset
    loop.history.clear()
    loop.conversation_state = type(loop.conversation_state)(
        loop.conversation_state.encoder,
        decay=loop.conversation_state.decay,
        max_topics=loop.conversation_state.max_topics
    )
    
    # Test 4: Topic Transition
    test_conversation_scenario(
        loop,
        "Topic Transition (Natural Flow)",
        turns=[
            "Tell me about dogs",
            "That's interesting. What about cats?",
            "Do they make good pets?"
        ],
        expected_behaviors={
            1: ['not_repetitive'],
            2: ['maintains_topic']  # "they" should refer to cats
        }
    )
    
    # Reset
    loop.history.clear()
    loop.conversation_state = type(loop.conversation_state)(
        loop.conversation_state.encoder,
        decay=loop.conversation_state.decay,
        max_topics=loop.conversation_state.max_topics
    )
    
    # Test 5: Clarification Loop
    test_conversation_scenario(
        loop,
        "Clarification Handling",
        turns=[
            "Explain neural networks",
            "What?",
            "Can you elaborate?"
        ],
        expected_behaviors={
            1: ['maintains_topic'],  # Should stay on neural networks
            2: ['maintains_topic']   # Should provide more detail
        }
    )
    
    # Summary
    print("\n" + "="*80)
    print("MULTI-TURN COHERENCE SUMMARY")
    print("="*80 + "\n")
    
    print("🎯 Key Findings:")
    print("   • Topic Continuation: How well does it stay on topic?")
    print("   • Reference Resolution: Does it understand 'it', 'that', pronouns?")
    print("   • Context Memory: Can it recall information from earlier?")
    print("   • Topic Transitions: Does it handle topic changes smoothly?")
    print("   • Clarification: Does it handle follow-up questions?")
    
    print("\n💡 Next Steps:")
    print("   1. Analyze failure patterns in output above")
    print("   2. Improve context encoding (weight recent turns)")
    print("   3. Add pronoun/reference resolution")
    print("   4. Implement topic coherence scoring")
    print("   5. Track conversation flow patterns")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
