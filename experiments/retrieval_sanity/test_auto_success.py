#!/usr/bin/env python3
"""
Test automatic success detection - shows engagement scoring in action.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from conversation_loop import ConversationLoop

def main():
    print("Testing Automatic Success Detection...\n")
    
    # Create loop with verbose learning
    loop = ConversationLoop(
        history_window=5,
        composition_mode="best_match"
    )
    
    # Enable verbose learning output
    loop.verbose_learning = True
    
    # Test conversations with different engagement levels
    test_conversations = [
        # High engagement - user asks follow-up questions
        ("Hello!", "High engagement test"),
        ("That's interesting! Tell me more about machine learning", None),
        ("How does that work?", None),
        
        # Low engagement - short responses
        ("Hi", "Low engagement test"),
        ("ok", None),
        ("what", None),
        
        # Mixed engagement
        ("Hey there", "Mixed engagement test"),
        ("What can you do?", None),
        ("Cool, thanks", None),
    ]
    
    print("\n" + "="*80)
    print("AUTOMATIC SUCCESS DETECTION TEST")
    print("="*80 + "\n")
    
    section = None
    for i, (user_input, marker) in enumerate(test_conversations, 1):
        if marker:
            if section:
                print("\n" + "-"*80)
                print(f"Section '{section}' complete")
                print("-"*80 + "\n")
                # Reset for new section
                loop = ConversationLoop(history_window=5, composition_mode="best_match")
                loop.verbose_learning = True
            section = marker
            print(f"\n{'='*80}")
            print(f"  {section.upper()}")
            print(f"{'='*80}\n")
        
        print(f"[Turn {i if not marker else i-test_conversations[:i].count((None, None))}]")
        print(f"👤 You: {user_input}")
        
        response = loop.process_user_input(user_input)
        
        print(f"🤖 Bot: {response}")
        print()
    
    # Show final learning stats
    print("\n" + "="*80)
    print("FINAL LEARNING STATISTICS")
    print("="*80 + "\n")
    
    learn_stats = loop.learner.get_learning_stats()
    print(f"📊 Learning Summary:")
    print(f"  Total interactions: {learn_stats['interaction_count']}")
    print(f"  Average success: {learn_stats['average_success']:.3f}")
    print(f"  Recent success: {learn_stats['recent_success']:.3f}")
    print(f"  Learning trend: {learn_stats['learning_trend']:+.3f}")
    
    hist_stats = loop.history.get_stats()
    print(f"\n📜 Conversation Quality:")
    print(f"  Turns: {hist_stats['turn_count']}")
    print(f"  Average success: {hist_stats['average_success']:.3f}")
    print(f"  Has repetition: {hist_stats['has_repetition']}")
    
    print("\n✅ Test complete!")
    print("\nKey insights:")
    print("  • Longer, question-rich inputs → higher engagement scores")
    print("  • Very short inputs ('ok', 'what') → lower engagement")
    print("  • Topic continuity → positive success scores")
    print("  • Success scores feed into pattern boosting for future improvement")

if __name__ == "__main__":
    main()
