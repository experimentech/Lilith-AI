#!/usr/bin/env python3
"""
Quick test of conversation_loop.py - automated conversation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from conversation_loop import ConversationLoop

def main():
    print("Testing ConversationLoop initialization and processing...\n")
    
    # Create loop with grammar enabled (hybrid adaptation + refinement)
    loop = ConversationLoop(
        history_window=5,
        composition_mode="best_match",
        use_grammar=True  # Enable grammar refinement
    )
    
    # Test conversation with quality assessment
    test_inputs = [
        "Hello!",
        "What can you do?",
        "That's interesting",
        "Can you tell me about machine learning?",
        "How does neural network training work?",
    ]
    
    print("\n" + "="*80)
    print("AUTOMATED TEST CONVERSATION - QUALITY ASSESSMENT")
    print("="*80 + "\n")
    
    responses = []
    for i, user_input in enumerate(test_inputs, 1):
        print(f"[Turn {i}]")
        print(f"👤 You: {user_input}")
        print()
        
        response = loop.process_user_input(user_input)
        responses.append((user_input, response))
        
        print()
        print(f"🤖 Bot: {response}")
        print()
        print("-"*80)
        print()
    
    # Show final state
    print("\nFinal system state:")
    loop.display_state()
    
    # Quality assessment
    print("\n" + "="*80)
    print("QUALITY ASSESSMENT")
    print("="*80 + "\n")
    
    scores = []
    
    print("Evaluating conversation quality:\n")
    
    # Turn 1: Greeting
    if any(word in responses[0][1].lower() for word in ["hello", "hi", "assist", "help"]):
        score1 = 10
        print(f"✅ Turn 1 (Greeting): 10/10 - Appropriate greeting response")
    else:
        score1 = 3
        print(f"❌ Turn 1 (Greeting): 3/10 - Off-topic or inappropriate")
    scores.append(score1)
    
    # Turn 2: Capability question
    if any(phrase in responses[1][1].lower() for phrase in ["help", "assist", "know", "discuss", "answer"]):
        score2 = 10
        print(f"✅ Turn 2 (Capability): 10/10 - Addresses capability question")
    else:
        score2 = 2
        print(f"❌ Turn 2 (Capability): 2/10 - Random topic jump")
    scores.append(score2)
    
    # Turn 3: Acknowledgment
    if any(phrase in responses[2][1].lower() for phrase in ["glad", "more", "else", "what", "help"]):
        score3 = 10
        print(f"✅ Turn 3 (Acknowledgment): 10/10 - Continues conversation naturally")
    else:
        score3 = 2
        print(f"❌ Turn 3 (Acknowledgment): 2/10 - Ignores context")
    scores.append(score3)
    
    # Turn 4: Specific topic
    response4 = responses[3][1].lower()
    if "machine" in response4 or "learning" in response4 or any(word in response4 for word in ["help", "know", "tell", "explain"]):
        score4 = 9
        print(f"✅ Turn 4 (Topic request): 9/10 - Engages with topic or offers help")
    else:
        score4 = 2
        print(f"❌ Turn 4 (Topic request): 2/10 - Unrelated response")
    scores.append(score4)
    
    # Turn 5: Technical question
    response5 = responses[4][1].lower()
    if any(word in response5 for word in ["neural", "training", "network", "learn", "help", "know", "tell"]):
        score5 = 9
        print(f"✅ Turn 5 (Technical): 9/10 - Addresses technical topic")
    else:
        score5 = 2
        print(f"❌ Turn 5 (Technical): 2/10 - Avoids question")
    scores.append(score5)
    
    avg_score = sum(scores) / len(scores)
    
    print(f"\n{'='*80}")
    print(f"OVERALL QUALITY SCORE: {avg_score:.1f}/10")
    print(f"{'='*80}")
    
    if avg_score >= 9.0:
        print("🌟 Excellent - Natural, contextually appropriate conversation")
    elif avg_score >= 7.0:
        print("✅ Good - Mostly coherent with minor issues")
    elif avg_score >= 5.0:
        print("⚠️  Fair - Some coherence but noticeable problems")
    else:
        print("❌ Poor - Frequent topic jumps or inappropriate responses")
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    main()
