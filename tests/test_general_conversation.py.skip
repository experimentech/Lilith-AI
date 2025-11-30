#!/usr/bin/env python3
"""
Test General Conversation (CSV-trained model)

Tests the model trained on Conversation.csv with general dialogue.
"""

from conversation_loop import ConversationLoop


def test_general_conversation():
    """Test with general conversational topics from the training domain"""
    
    conv = ConversationLoop()
    
    print("\n" + "="*70)
    print("TEST: General Conversation (CSV Training Domain)")
    print("="*70)
    print()
    
    # Test 1: Greetings
    print("Test 1: Greeting exchange")
    print("-" * 70)
    r1 = conv.process_user_input("Hi, how are you doing?")
    print(f"User: Hi, how are you doing?")
    print(f"Bot: {r1}\n")
    
    r2 = conv.process_user_input("I'm doing well, thanks")
    print(f"User: I'm doing well, thanks")
    print(f"Bot: {r2}\n")
    
    # Test 2: School topic
    print("\nTest 2: School conversation")
    print("-" * 70)
    r1 = conv.process_user_input("Are you in school?")
    print(f"User: Are you in school?")
    print(f"Bot: {r1}\n")
    
    r2 = conv.process_user_input("What school do you go to?")
    print(f"User: What school do you go to?")
    print(f"Bot: {r2}\n")
    
    r3 = conv.process_user_input("Do you like it there?")
    print(f"User: Do you like it there?")
    print(f"Bot: {r3}\n")
    
    # Test 3: Movies topic
    print("\nTest 3: Movies conversation")
    print("-" * 70)
    r1 = conv.process_user_input("Have you been to the movies lately?")
    print(f"User: Have you been to the movies lately?")
    print(f"Bot: {r1}\n")
    
    r2 = conv.process_user_input("What movie did you see?")
    print(f"User: What movie did you see?")
    print(f"Bot: {r2}\n")
    
    r3 = conv.process_user_input("Did you like it?")
    print(f"User: Did you like it?")
    print(f"Bot: {r3}\n")
    
    # Test 4: Context-dependent questions
    print("\nTest 4: Context-dependent pronouns")
    print("-" * 70)
    r1 = conv.process_user_input("I'm going to the beach this weekend")
    print(f"User: I'm going to the beach this weekend")
    print(f"Bot: {r1}\n")
    
    r2 = conv.process_user_input("What do you think the weather will be like?")
    print(f"User: What do you think the weather will be like?")
    print(f"Bot: {r2}\n")
    
    r3 = conv.process_user_input("That sounds good")
    print(f"User: That sounds good")
    print(f"Bot: {r3}\n")
    
    # Test 5: Farewells
    print("\nTest 5: Farewell")
    print("-" * 70)
    r1 = conv.process_user_input("Thanks for chatting!")
    print(f"User: Thanks for chatting!")
    print(f"Bot: {r1}\n")
    
    r2 = conv.process_user_input("See you later")
    print(f"User: See you later")
    print(f"Bot: {r2}\n")


if __name__ == "__main__":
    test_general_conversation()
