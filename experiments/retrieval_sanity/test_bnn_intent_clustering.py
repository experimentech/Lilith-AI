"""
Test BNN Intent Clustering

Demonstrates semantic intent classification using BNN embeddings.
Shows how patterns are clustered by intent and used for faster retrieval.
"""

from conversation_loop import ConversationLoop
import time


def test_intent_clustering():
    """Test BNN intent clustering on trained patterns."""
    
    print("=" * 80)
    print("BNN INTENT CLUSTERING TEST")
    print("=" * 80)
    print()
    
    # 1. Initialize system with BNN intent clustering enabled
    print("📦 Initializing conversation system...")
    conv = ConversationLoop(
        use_grammar=False,
        composition_mode="adaptive",
        history_window=5
    )
    print()
    
    # 2. Check if patterns are loaded
    pattern_count = len(conv.fragment_store.patterns)
    print(f"✅ Loaded {pattern_count} patterns from training")
    print()
    
    if pattern_count == 0:
        print("⚠️  No patterns found! Train the system first:")
        print("   python train_from_csv.py datasets/Conversation.csv --max-turns 300")
        return
    
    # 3. Build intent clusters
    print("🎯 Building BNN intent clusters...")
    start_time = time.time()
    clusters = conv.build_intent_clusters()
    clustering_time = time.time() - start_time
    
    if not clusters:
        print("⚠️  Intent clustering not available (no encoder)")
        return
    
    print(f"⏱️  Clustering took {clustering_time:.2f}s")
    print()
    
    # 4. Show cluster details
    print("=" * 80)
    print("INTENT CLUSTER DETAILS")
    print("=" * 80)
    
    for intent_label, cluster in sorted(clusters.items(), key=lambda x: len(x[1].pattern_ids), reverse=True):
        print(f"\n📌 Intent: {intent_label}")
        print(f"   Patterns: {len(cluster.pattern_ids)}")
        print(f"   Coherence: {cluster.confidence:.3f}")
        print(f"   Representative: \"{cluster.representative_text[:80]}...\"")
    
    print()
    print("=" * 80)
    print("INTENT CLASSIFICATION TESTS")
    print("=" * 80)
    print()
    
    # 5. Test intent classification on various inputs
    test_inputs = [
        "Hi, how are you?",
        "What's the weather like today?",
        "I'm going to the beach this weekend",
        "Do you like watching movies?",
        "Goodbye, see you later!",
        "That sounds really cool",
        "I agree with you",
        "Tell me about neural networks",
        "How does backpropagation work?",
        "What are the applications of transformers?"
    ]
    
    for user_input in test_inputs:
        # Classify intent
        intent_scores = conv.composer.intent_classifier.classify_intent(user_input, topk=3)
        
        print(f"Input: \"{user_input}\"")
        print(f"  Top intents:")
        for intent, score in intent_scores:
            print(f"    - {intent}: {score:.3f}")
        print()
    
    # 6. Compare retrieval speed with/without intent filtering
    print("=" * 80)
    print("RETRIEVAL SPEED COMPARISON")
    print("=" * 80)
    print()
    
    test_query = "What's the weather like today?"
    
    # Without intent filtering
    start_time = time.time()
    for _ in range(100):
        response = conv.composer.compose_response(
            context=test_query,
            user_input=test_query,
            topk=5,
            use_intent_filtering=False
        )
    time_without = (time.time() - start_time) / 100
    
    # With intent filtering
    start_time = time.time()
    for _ in range(100):
        response = conv.composer.compose_response(
            context=test_query,
            user_input=test_query,
            topk=5,
            use_intent_filtering=True
        )
    time_with = (time.time() - start_time) / 100
    
    speedup = time_without / time_with if time_with > 0 else 1.0
    
    print(f"Without intent filtering: {time_without*1000:.2f}ms per response")
    print(f"With intent filtering:    {time_with*1000:.2f}ms per response")
    print(f"Speedup: {speedup:.2f}x")
    print()
    
    # 7. Show response quality with intent filtering
    print("=" * 80)
    print("RESPONSE QUALITY WITH INTENT FILTERING")
    print("=" * 80)
    print()
    
    quality_tests = [
        "How are you doing?",
        "What's the weather forecast?",
        "I'm planning to watch a movie tonight",
        "See you tomorrow!",
    ]
    
    for test_input in quality_tests:
        response = conv.composer.compose_response(
            context=test_input,
            user_input=test_input,
            topk=5,
            use_intent_filtering=True
        )
        
        # Classify intent
        intent_scores = conv.composer.intent_classifier.classify_intent(test_input, topk=1)
        top_intent = intent_scores[0][0] if intent_scores else "unknown"
        
        print(f"User: {test_input}")
        print(f"  Intent: {top_intent}")
        print(f"  Bot: {response.text}")
        print()
    
    print("=" * 80)
    print("✅ BNN INTENT CLUSTERING TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_intent_clustering()
