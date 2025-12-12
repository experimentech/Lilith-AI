#!/usr/bin/env python3
"""
Test learning from teaching with clean, controlled questions.

Uses technical/specialized topics unlikely to appear in Cornell Movie Dialogs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from conversation_loop import ConversationLoop

def test_teaching_scenario(
    loop,
    topic_question: str = "What is gradient descent?",
    teaching_input: str = "Gradient descent is an optimization algorithm that updates parameters in the direction of negative gradients to minimize a loss function.",
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
        print(f"   (Got: '{response1}')")
    
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
    
    # Check if learned - look for key terms from teaching in response
    teaching_lower = teaching_input.lower()
    response_lower = response3.lower()
    
    # Extract key concepts from teaching
    key_concepts = []
    if "gradient descent" in topic_question.lower():
        key_concepts = ["gradient", "minimize", "loss", "descent"]
    elif "fourier transform" in topic_question.lower():
        key_concepts = ["frequency", "signal", "domain", "fourier"]
    elif "merkle tree" in topic_question.lower():
        key_concepts = ["hash", "tree", "verify", "blockchain"]
    
    has_key_concepts = any(concept in response_lower for concept in key_concepts)
    
    # Success if: started with fallback, now gives substantive answer with key concepts
    if is_fallback and not is_fallback_retry and has_key_concepts:
        print("   🌟 SUCCESS: Learned from teaching! (uses taught concepts)")
        return "learned"
    elif is_fallback and not is_fallback_retry:
        print("   ⚠️  PARTIAL: Gave answer but may not use taught concepts")
        return "partial"
    elif is_fallback and is_fallback_retry:
        print("   ❌ FAILED: Still using fallback (didn't learn)")
        return "failed"
    elif not is_fallback:
        print("   ⚠️  INCONCLUSIVE: Had answer initially (data contamination?)")
        return "inconclusive"
    
    return "unknown"

def main():
    print("🧠 Testing Learning from Teaching - Clean Dataset\n")
    
    # Initialize system with eager learning mode
    loop = ConversationLoop(
        history_window=10,
        composition_mode="best_match",
        learning_mode="eager"
    )
    
    loop.verbose_learning = True
    
    print("\n" + "="*80)
    print("LEARNING FROM TEACHING TEST - CLEAN QUESTIONS")
    print("="*80)
    print("\nUsing specialized technical topics unlikely to appear in movie dialogue")
    print("="*80 + "\n")
    
    # Warm up with a greeting
    print("[Warmup]")
    print("👤 You: Hello!")
    response = loop.process_user_input("Hello!")
    print(f"🤖 Bot: {response}")
    print()
    
    # Test scenarios - using specialized technical terms
    results = []
    
    # Scenario 1: Machine learning concept
    result1 = test_teaching_scenario(
        loop,
        topic_question="What is gradient descent?",
        teaching_input="Gradient descent is an optimization algorithm used to minimize a loss function. It iteratively adjusts parameters in the direction of steepest descent of the gradient.",
        retry_question="Explain gradient descent"
    )
    results.append(("Gradient descent", result1))
    
    # Scenario 2: Signal processing
    result2 = test_teaching_scenario(
        loop,
        topic_question="What is a Fourier transform?",
        teaching_input="A Fourier transform is a mathematical operation that converts a signal from the time domain into the frequency domain. It decomposes signals into sine and cosine waves.",
        retry_question="Tell me about Fourier transforms"
    )
    results.append(("Fourier transform", result2))
    
    # Scenario 3: Computer science
    result3 = test_teaching_scenario(
        loop,
        topic_question="What is a Merkle tree?",
        teaching_input="A Merkle tree is a data structure used in cryptography and blockchain. It allows efficient verification of data integrity by using hierarchical hashing.",
        retry_question="Explain Merkle trees"
    )
    results.append(("Merkle tree", result3))
    
    # Summary
    print("\n" + "="*80)
    print("TEACHING TEST SUMMARY")
    print("="*80 + "\n")
    
    learned_count = sum(1 for _, r in results if r == "learned")
    partial_count = sum(1 for _, r in results if r == "partial")
    failed_count = sum(1 for _, r in results if r == "failed")
    inconclusive_count = sum(1 for _, r in results if r == "inconclusive")
    
    for topic, result in results:
        if result == "learned":
            icon = "🌟"
        elif result == "partial":
            icon = "⚡"
        elif result == "failed":
            icon = "❌"
        else:
            icon = "⚠️"
        print(f"{icon} {topic}: {result.upper()}")
    
    print(f"\n📊 Results:")
    print(f"   Fully Learned: {learned_count}/{len(results)}")
    print(f"   Partial: {partial_count}/{len(results)}")
    print(f"   Failed: {failed_count}/{len(results)}")
    print(f"   Inconclusive: {inconclusive_count}/{len(results)}")
    
    # Learning stats
    print(f"\n🎓 System Learning Stats:")
    learn_stats = loop.learner.get_learning_stats()
    print(f"   Total interactions: {learn_stats['interaction_count']}")
    print(f"   Average success: {learn_stats['average_success']:.3f}")
    print(f"   Recent success: {learn_stats['recent_success']:.3f}")
    print(f"   Patterns learned: {len(loop.fragment_store.patterns)}")
    
    # Check if patterns were extracted
    initial_pattern_count = 1235  # Baseline from Cornell dialogs
    new_patterns = len(loop.fragment_store.patterns) - initial_pattern_count
    print(f"\n📚 Pattern Extraction Check:")
    print(f"   Initial patterns: {initial_pattern_count}")
    print(f"   Current patterns: {len(loop.fragment_store.patterns)}")
    print(f"   New patterns learned: {new_patterns}")
    
    if new_patterns > 0:
        print(f"   ✅ System extracted {new_patterns} new patterns from teaching!")
    else:
        print(f"   ⚠️  No new patterns extracted (learning threshold may be too strict)")
    
    print("\n" + "="*80)
    
    success_rate = (learned_count + partial_count) / len(results) if results else 0
    
    if learned_count >= 2:
        print("✅ EXCELLENT: Learning from teaching works reliably!")
        print(f"   {learned_count}/{len(results)} scenarios learned successfully")
    elif learned_count + partial_count >= 2:
        print("⚡ GOOD: Learning works but may need refinement")
        print(f"   {learned_count + partial_count}/{len(results)} scenarios showed learning")
    elif new_patterns > 0:
        print("⚠️  PARTIAL: Patterns extracted but not applied effectively")
        print("   May need tuning: confidence thresholds, pattern matching")
    else:
        print("❌ NEEDS WORK: Learning from teaching not extracting patterns")
        print("   Check: pattern extraction logic, learning thresholds")
    
    print("="*80)

if __name__ == "__main__":
    main()
