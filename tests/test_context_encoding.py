#!/usr/bin/env python3
"""
Test Multi-Turn Context Encoding

Demonstrates how context encoding improves multi-turn conversations
by maintaining topic coherence and understanding references.
"""

from conversation_loop import ConversationLoop


def test_multi_turn_coherence():
    """Test that system maintains context across turns"""
    
    conv = ConversationLoop()
    
    print("\n" + "="*70)
    print("TEST 1: Topic Continuation")
    print("="*70)
    print("Testing if system maintains topic across turns without re-stating it\n")
    
    r1 = conv.process_user_input("What are CNNs?")
    print(f"User: What are CNNs?")
    print(f"Bot: {r1}\n")
    
    r2 = conv.process_user_input("What are they good for?")  # "they" = CNNs
    print(f"User: What are they good for?")
    print(f"Bot: {r2}\n")
    
    r3 = conv.process_user_input("Can you explain more?")  # More about CNNs
    print(f"User: Can you explain more?")
    print(f"Bot: {r3}\n")
    
    print("\n" + "="*70)
    print("TEST 2: Topic Switch and Return")
    print("="*70)
    print("Testing if system can switch topics and return to previous topic\n")
    
    conv2 = ConversationLoop()
    
    r1 = conv2.process_user_input("Tell me about neural networks")
    print(f"User: Tell me about neural networks")
    print(f"Bot: {r1}\n")
    
    r2 = conv2.process_user_input("What about backpropagation?")  # New topic
    print(f"User: What about backpropagation?")
    print(f"Bot: {r2}\n")
    
    r3 = conv2.process_user_input("Going back to neural networks, how do they learn?")
    print(f"User: Going back to neural networks, how do they learn?")
    print(f"Bot: {r3}\n")
    
    print("\n" + "="*70)
    print("TEST 3: Pronoun Resolution")
    print("="*70)
    print("Testing if system understands pronouns referring to previous topics\n")
    
    conv3 = ConversationLoop()
    
    r1 = conv3.process_user_input("What is transfer learning?")
    print(f"User: What is transfer learning?")
    print(f"Bot: {r1}\n")
    
    r2 = conv3.process_user_input("Why is it useful?")  # "it" = transfer learning
    print(f"User: Why is it useful?")
    print(f"Bot: {r2}\n")
    
    r3 = conv3.process_user_input("Where is it commonly used?")
    print(f"User: Where is it commonly used?")
    print(f"Bot: {r3}\n")
    
    print("\n" + "="*70)
    print("TEST 4: Clarification Requests")
    print("="*70)
    print("Testing if system provides clarification based on previous exchanges\n")
    
    conv4 = ConversationLoop()
    
    r1 = conv4.process_user_input("What are embeddings?")
    print(f"User: What are embeddings?")
    print(f"Bot: {r1}\n")
    
    r2 = conv4.process_user_input("I don't understand")  # Needs previous context
    print(f"User: I don't understand")
    print(f"Bot: {r2}\n")
    
    r3 = conv4.process_user_input("Can you explain differently?")
    print(f"User: Can you explain differently?")
    print(f"Bot: {r3}\n")


if __name__ == "__main__":
    test_multi_turn_coherence()
