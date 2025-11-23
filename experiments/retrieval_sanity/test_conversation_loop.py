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
    
    # Create loop
    loop = ConversationLoop(
        history_window=5,
        composition_mode="best_match"
    )
    
    # Test conversation
    test_inputs = [
        "Hello!",
        "What can you do?",
        "That's interesting",
    ]
    
    print("\n" + "="*80)
    print("AUTOMATED TEST CONVERSATION")
    print("="*80 + "\n")
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"[Turn {i}]")
        print(f"👤 You: {user_input}")
        print()
        
        response = loop.process_user_input(user_input)
        
        print()
        print(f"🤖 Bot: {response}")
        print()
        print("-"*80)
        print()
    
    # Show final state
    print("\nFinal system state:")
    loop.display_state()
    
    print("✅ Test complete!")

if __name__ == "__main__":
    main()
