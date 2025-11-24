from conversation_loop import ConversationLoop

conv = ConversationLoop()

tests = [
    "What's the difference between CNN and RNN?",
    "Are neural networks always deep, and do they require lots of data?",
    "How do I choose between supervised and unsupervised learning for my problem?",
    "What about purple elephants in machine learning?",
]

for test in tests:
    print(f"\n{'='*70}")
    print(f"Test: {test}")
    result = conv.process_user_input(test)
    print(f"Response: {result}")
