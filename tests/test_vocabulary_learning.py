#!/usr/bin/env python3
"""Test vocabulary learning from user inputs."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from conversation_loop import ConversationLoop


def main():
    print("Testing Vocabulary Learning")
    print("=" * 80)
    print()
    
    # Initialize conversation system
    loop = ConversationLoop(
        history_window=10,
        composition_mode="best_match"
    )
    
    # Get initial pattern count
    initial_count = len(loop.fragment_store.patterns)
    print(f"Initial patterns: {initial_count}")
    print()
    
    # Simulate engaging conversation where user teaches new phrases
    print("Simulating conversation where user uses interesting phrases...")
    print()
    
    test_interactions = [
        ("Hello!", None),  # Initial greeting
        ("What can you do?", None),  # Capability question
        ("I'm excited to explore neuro-symbolic learning with you!", "engagement"),  # Engaging elaboration
        ("Tell me more", None),
        ("This is absolutely fascinating and groundbreaking!", "engagement"),  # Another engaging phrase
        ("How does pattern matching work?", None),
        ("The semantic similarity approach is quite elegant and powerful!", "engagement"),  # Technical elaboration
    ]
    
    learned_patterns = []
    
    for i, (user_input, expect_learning) in enumerate(test_interactions):
        print(f"[Turn {i+1}]")
        print(f"👤 You: {user_input}")
        
        # Check pattern count before
        before_count = len(loop.fragment_store.patterns)
        
        # Process input
        response = loop.process_user_input(user_input)
        print(f"🤖 Bot: {response}")
        
        # Check if pattern was learned
        after_count = len(loop.fragment_store.patterns)
        if after_count > before_count:
            new_pattern = list(loop.fragment_store.patterns.values())[-1]
            learned_patterns.append(new_pattern)
            print(f"  ✨ NEW PATTERN LEARNED!")
            print(f"     Response: '{new_pattern.response_text}'")
            print(f"     Intent: {new_pattern.intent}")
        
        print()
    
    # Summary
    print("=" * 80)
    print("LEARNING SUMMARY")
    print("=" * 80)
    print(f"Initial patterns: {initial_count}")
    print(f"Final patterns: {len(loop.fragment_store.patterns)}")
    print(f"Patterns learned: {len(learned_patterns)}")
    print()
    
    if learned_patterns:
        print("Learned patterns:")
        for i, pattern in enumerate(learned_patterns):
            print(f"  {i+1}. '{pattern.response_text[:60]}...'")
            print(f"     Trigger: '{pattern.trigger_context[:60]}...'")
            print(f"     Success: {pattern.success_score:.2f}")
            print()
    else:
        print("⚠️  No patterns were learned. Check engagement thresholds.")
        print("   Try using longer, more enthusiastic responses.")
    
    print("✅ Test complete!")


if __name__ == "__main__":
    main()
