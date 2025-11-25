#!/usr/bin/env python3
"""Test grammar-guided composition in conversation."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from conversation_loop import ConversationLoop


def main():
    print("Testing Grammar-Guided Composition")
    print("=" * 80)
    print()
    
    # Create two systems: with and without grammar
    print("System A: Standard blending")
    loop_standard = ConversationLoop(
        history_window=5,
        composition_mode="weighted_blend",
        use_grammar=False
    )
    
    print("\nSystem B: Grammar-guided blending")
    loop_grammar = ConversationLoop(
        history_window=5,
        composition_mode="weighted_blend",
        use_grammar=True
    )
    
    print("\n" + "=" * 80)
    print("COMPARISON TEST")
    print("=" * 80)
    
    test_inputs = [
        "Hello!",
        "That's really interesting and I'd love to know more!",
        "How does it work?",
    ]
    
    for i, user_input in enumerate(test_inputs):
        print(f"\n[Turn {i+1}] User: {user_input}")
        print("-" * 80)
        
        # Get response from standard system
        response_std = loop_standard.process_user_input(user_input)
        print(f"Standard:  {response_std}")
        
        # Get response from grammar-guided system
        response_gram = loop_grammar.process_user_input(user_input)
        print(f"Grammar:   {response_gram}")
    
    print("\n" + "=" * 80)
    print("Grammar-guided system should produce smoother punctuation and flow!")
    print("✅ Test complete!")


if __name__ == "__main__":
    main()
