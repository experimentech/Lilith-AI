#!/usr/bin/env python3
"""Test compositional generation - creating novel responses from learned fragments."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from conversation_loop import ConversationLoop


def main():
    print("Testing Compositional Generation")
    print("=" * 80)
    print("Testing if system can CREATE novel responses by blending patterns")
    print("=" * 80)
    print()
    
    # Initialize with weighted_blend mode (compositional)
    loop = ConversationLoop(
        history_window=10,
        composition_mode="weighted_blend"  # Enable pattern blending!
    )
    
    print("Conversation with COMPOSITIONAL mode enabled")
    print("(System will blend patterns to create novel responses)")
    print()
    
    # Test inputs that should trigger blending
    test_inputs = [
        "Hello!",
        "What can you do?",
        "That sounds really interesting and I'd like to learn more!",
        "Tell me about neural networks",
        "How does that work?",
        "This is fascinating! Can you explain more?",
        "I appreciate your help with understanding this",
    ]
    
    for i, user_input in enumerate(test_inputs):
        print(f"[Turn {i+1}]")
        print(f"👤 You: {user_input}")
        
        response = loop.process_user_input(user_input)
        print(f"🤖 Bot: {response}")
        
        # Check if response uses multiple fragments (blended)
        if loop.last_response and len(loop.last_response.fragment_ids) > 1:
            print(f"  ✨ COMPOSITIONAL! Used {len(loop.last_response.fragment_ids)} patterns:")
            for fid in loop.last_response.fragment_ids[:3]:
                pattern = loop.fragment_store.patterns.get(fid)
                if pattern:
                    print(f"     - {pattern.intent}: '{pattern.response_text[:40]}...'")
        
        print()
    
    print("=" * 80)
    print("COMPOSITION ANALYSIS")
    print("=" * 80)
    
    # Count how many responses used composition
    compositional_count = 0
    total_turns = len(test_inputs)
    
    print(f"Responses using multiple patterns: Check above for ✨ markers")
    print()
    print("The system should blend patterns when:")
    print("  - Primary match has moderate confidence (0.3-0.75)")
    print("  - Secondary patterns are relevant (>0.25 weight)")
    print("  - Patterns are compatible (not both questions)")
    print()
    print("✅ Test complete!")


if __name__ == "__main__":
    main()
