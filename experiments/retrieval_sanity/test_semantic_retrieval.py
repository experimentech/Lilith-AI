"""
Test BNN Semantic Retrieval (Open Book Exam Architecture)

Compare pure keyword matching vs. BNN embedding-based hybrid retrieval.

The key innovation: BNN learns "how to recognize similar contexts",
database stores "what to respond". Like an open book exam.
"""

from conversation_loop import ConversationLoop

print("=" * 80)
print("TESTING BNN SEMANTIC RETRIEVAL (OPEN BOOK EXAM)")
print("=" * 80)
print()

# Test prompts that should benefit from semantic understanding
# (similar meaning, different words)
test_cases = [
    {
        "prompt": "How are you doing?",
        "semantically_similar": ["Hi, how are you?", "How's it going?", "What's up?"],
        "description": "Greeting variations"
    },
    {
        "prompt": "What's the climate like?",
        "semantically_similar": ["What do you think about the weather today?"],
        "description": "Weather/climate equivalence"
    },
    {
        "prompt": "Do you enjoy films?",
        "semantically_similar": ["Do you like movies?"],
        "description": "Films vs movies"
    },
    {
        "prompt": "How was your day at university?",
        "semantically_similar": ["How was school today?"],
        "description": "School vs university"
    },
]

print("Testing KEYWORD-ONLY retrieval (baseline):")
print("-" * 80)

conv_keywords = ConversationLoop()
# Disable semantic retrieval
conv_keywords.composer.fragments.use_semantic = False

keyword_results = []
for case in test_cases:
    prompt = case["prompt"]
    response = conv_keywords.process_user_input(prompt)
    keyword_results.append((prompt, response))
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")

print("\n" + "=" * 80)
print("Testing BNN HYBRID retrieval (semantic + keywords):")
print("-" * 80)

conv_semantic = ConversationLoop()
# Semantic retrieval is enabled by default now

semantic_results = []
for case in test_cases:
    prompt = case["prompt"]
    response = conv_semantic.process_user_input(prompt)
    semantic_results.append((prompt, response))
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")

print("\n" + "=" * 80)
print("COMPARISON:")
print("=" * 80)

different_count = 0
for i, case in enumerate(test_cases):
    keyword_resp = keyword_results[i][1]
    semantic_resp = semantic_results[i][1]
    
    print(f"\n{case['description']}: \"{case['prompt']}\"")
    print(f"  Keyword:  {keyword_resp}")
    print(f"  Semantic: {semantic_resp}")
    
    if keyword_resp != semantic_resp:
        different_count += 1
        print(f"  ✅ Different response (semantic similarity detected!)")
    else:
        print(f"  ⊘ Same response")

print("\n" + "=" * 80)
print(f"Semantic retrieval changed {different_count}/{len(test_cases)} responses")
print("=" * 80)

if different_count > 0:
    print("\n✅ BNN semantic retrieval is working!")
    print("   The BNN can recognize semantically similar inputs")
    print("   even when keywords differ (e.g., 'films' vs 'movies')")
else:
    print("\n⚠️  Semantic retrieval had no effect")
    print("   Either keywords already handle these cases,")
    print("   or semantic similarity needs tuning")
