#!/usr/bin/env python3
"""
Test if the system can learn from being taught.

Scenario:
1. Ask about unknown topic → Fallback ("I don't know")
2. User explains/teaches the topic
3. Ask the same question again → Should have learned!

This tests:
- Pattern extraction from user explanations
- Success-based learning strengthening taught patterns
- Ability to apply newly learned knowledge
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from conversation_loop import ConversationLoop

def test_teaching_scenario(
    loop,
    topic_question: str = "What is backpropagation?",
    teaching_input: str = "Backpropagation trains neural networks by computing gradients via the chain rule and updating weights to reduce loss.",
    retry_question: str | None = None,
):
    """
    Test a teaching scenario:
    1. Ask unknown question (expect fallback)
    2. User teaches the topic
    3. Re-ask question (expect learned response)
    """
    if retry_question is None:
        retry_question = topic_question
    
    print(f"\n{'='*80}")
    print(f"TEACHING TEST: {topic_question}")
    print(f"{'='*80}\n")
    
    # Phase 1: Ask unknown question
    print("[Phase 1: Initial Question - Expect Fallback]")
    print(f"👤 You: {topic_question}")
    response1 = loop.process_user_input(topic_question)
    print(f"🤖 Bot: {response1}")
    
    is_fallback = any(marker in response1.lower() for marker in [
        "don't have", "not sure", "don't know", "rephrase", "something else"
    ])
    
    if is_fallback:
        print("   ✅ Correctly used fallback (lacks knowledge)")
    else:
        print("   ⚠️  Provided answer without knowledge")
    
    print()
    
    # Phase 2: User teaches
    print("[Phase 2: Teaching - User Provides Information]")
    print(f"👤 You: {teaching_input}")
    response2 = loop.process_user_input(teaching_input)
    print(f"🤖 Bot: {response2}")
    print("   📚 System should extract pattern from this interaction")
    print()
    
    # Phase 3: Retry same question
    print("[Phase 3: Retry Question - Test if Learned]")
    print(f"👤 You: {retry_question}")
    response3 = loop.process_user_input(retry_question)
    print(f"🤖 Bot: {response3}")
    
    is_fallback_retry = any(marker in response3.lower() for marker in [
        "don't have", "not sure", "don't know", "rephrase", "something else"
    ])
    
    # Check if it learned
    if is_fallback and not is_fallback_retry:
        print("   🌟 SUCCESS: Learned from teaching!")
        return "learned"
    elif is_fallback and is_fallback_retry:
        print("   ❌ FAILED: Still using fallback (didn't learn)")
        return "failed"
    elif not is_fallback:
        print("   ⚠️  INCONCLUSIVE: Had answer initially")
        return "inconclusive"
    
    return "unknown"

def main():
    print("🧠 Testing Learning from Teaching Scenarios\n")
    
    # Initialize system with eager learning mode (learns from most interactions)
    loop = ConversationLoop(
        history_window=10,
        composition_mode="best_match",
        learning_mode="eager"  # More aggressive learning
    )
    
    loop.verbose_learning = True
    
    print("\n" + "="*80)
    print("LEARNING FROM TEACHING TEST")
    print("="*80)
    print("\nScenario: User asks unknown question → teaches system → retries")
    print("Expected: System should learn from teaching and apply knowledge")
    print("="*80 + "\n")
    
    # Warm up with a greeting
    print("[Warmup]")
    print("👤 You: Hello!")
    response = loop.process_user_input("Hello!")
    print(f"🤖 Bot: {response}")
    print()
    
    # Test scenarios
    results = []
    
    # Scenario 1: Technical concept
    result1 = test_teaching_scenario(
        loop,
        topic_question="What is backpropagation?",
        teaching_input="Backpropagation is an algorithm for training neural networks. It calculates gradients by working backwards through the network layers, using the chain rule to compute how each weight contributed to the error.",
        retry_question="Can you explain backpropagation?"
    )
    results.append(("Backpropagation", result1))
    
    # Scenario 2: Fact-based knowledge
    result2 = test_teaching_scenario(
        loop,
        topic_question="What is the capital of Iceland?",
        teaching_input="The capital of Iceland is Reykjavik. It's the northernmost capital city in the world and is known for its geothermal energy.",
        retry_question="Tell me about Iceland's capital"
    )
    results.append(("Iceland capital", result2))
    
    # Scenario 3: Concept definition
    result3 = test_teaching_scenario(
        loop,
        topic_question="What is episodic memory?",
        teaching_input="Episodic memory is a type of long-term memory that involves conscious recollection of specific events, situations, and experiences from your life. It's like mental time travel to past events.",
        retry_question="Explain episodic memory"
    )
    results.append(("Episodic memory", result3))
    
    # Summary
    print("\n" + "="*80)
    print("TEACHING TEST SUMMARY")
    print("="*80 + "\n")
    
    learned_count = sum(1 for _, r in results if r == "learned")
    failed_count = sum(1 for _, r in results if r == "failed")
    inconclusive_count = sum(1 for _, r in results if r == "inconclusive")
    
    for topic, result in results:
        icon = "🌟" if result == "learned" else "❌" if result == "failed" else "⚠️"
        print(f"{icon} {topic}: {result.upper()}")
    
    print(f"\n📊 Results:")
    print(f"   Learned: {learned_count}/{len(results)}")
    print(f"   Failed: {failed_count}/{len(results)}")
    print(f"   Inconclusive: {inconclusive_count}/{len(results)}")
    
    # Learning stats
    print(f"\n🎓 System Learning Stats:")
    learn_stats = loop.learner.get_learning_stats()
    print(f"   Total interactions: {learn_stats['interaction_count']}")
    print(f"   Average success: {learn_stats['average_success']:.3f}")
    print(f"   Recent success: {learn_stats['recent_success']:.3f}")
    print(f"   Patterns learned: {len(loop.fragment_store.patterns)}")
    
    # Check if patterns were extracted from teaching
    print(f"\n📚 Pattern Extraction Check:")
    initial_pattern_count = 1235  # Baseline from Cornell dialogs
    new_patterns = len(loop.fragment_store.patterns) - initial_pattern_count
    print(f"   Initial patterns: {initial_pattern_count}")
    print(f"   Current patterns: {len(loop.fragment_store.patterns)}")
    print(f"   New patterns learned: {new_patterns}")
    
    if new_patterns > 0:
        print(f"   ✅ System extracted {new_patterns} new patterns from teaching!")
    else:
        print(f"   ⚠️  No new patterns extracted (learning threshold may be too strict)")
    
    print("\n" + "="*80)
    
    if learned_count > 0:
        print("✅ Learning from teaching WORKS!")
        print("   System can extract patterns from user explanations and apply them.")
    elif new_patterns > 0:
        print("⚠️  Partial success - patterns extracted but not yet applied effectively")
        print("   May need multiple exposures or stronger success signals")
    else:
        print("❌ Learning from teaching needs improvement")
        print("   Pattern extraction or success thresholds may need adjustment")
    
    print("="*80)
    
    print("\n💡 Insights:")
    print("   • Eager learning mode: More aggressive pattern extraction")
    print("   • Success detection: Tracks engagement from teaching interactions")
    print("   • Pattern extraction: Converts user explanations into retrievable patterns")
    print("   • Next step: May need to adjust learning thresholds or add teaching markers")

if __name__ == "__main__":
    main()
