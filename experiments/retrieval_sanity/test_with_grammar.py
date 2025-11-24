"""
Test conversation quality with grammar/syntax stage enabled.

Compare canned phrase recitation vs grammar-guided composition.
"""

from conversation_loop import ConversationLoop


def test_with_grammar():
    """Test quality with grammar enabled."""
    
    print("=" * 80)
    print("TESTING WITH GRAMMAR/SYNTAX STAGE ENABLED")
    print("=" * 80)
    print()
    
    # Initialize with grammar enabled and weighted blend mode
    conv = ConversationLoop(
        use_grammar=True,
        composition_mode="weighted_blend"
    )
    
    # Test scenarios
    test_cases = [
        "Hi, how are you?",
        "What do you think about the weather today?",
        "Do you like movies?",
        "What's your favorite movie?",
        "What are you doing this weekend?",
        "I'm going to the beach",
        "Thanks for chatting!",
        "See you later"
    ]
    
    print("\nTest Responses:")
    print("=" * 80)
    
    for user_input in test_cases:
        response = conv.process_user_input(user_input)
        print(f"User: {user_input}")
        print(f"Bot:  {response}")
        print()


if __name__ == "__main__":
    test_with_grammar()
