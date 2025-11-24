"""
Test learning capability with fruits example.

Demonstrates that the system can:
1. Learn factual information from user statements
2. Recall that information when asked
"""

from conversation_loop import ConversationLoop


def test_fruit_learning():
    """Test learning about fruits."""
    
    print("=" * 80)
    print("FRUIT LEARNING TEST - Eager Learning Mode")
    print("=" * 80)
    print()
    
    # Initialize with eager learning for easy teaching
    conv = ConversationLoop(learning_mode="eager")
    
    initial_stats = conv.composer.fragments.get_stats()
    print(f"Initial patterns: {initial_stats['total_patterns']}")
    print()
    
    # === TEACHING PHASE ===
    print("📚 TEACHING PHASE:")
    print("-" * 80)
    
    fruit_facts = [
        "Fruits are sweet edible parts of plants",
        "Apples are red or green fruits",
        "Bananas are yellow curved fruits",
        "Oranges are round citrus fruits",
        "Grapes are small round fruits",
        "Strawberries are red heart shaped fruits",
    ]
    
    for fact in fruit_facts:
        print(f"\nTeacher: {fact}")
        response = conv.process_user_input(fact)
        print(f"Bot:     {response[:60]}...")
    
    # Check learning
    final_stats = conv.composer.fragments.get_stats()
    learned_count = final_stats['total_patterns'] - initial_stats['total_patterns']
    
    print()
    print("=" * 80)
    print(f"✅ Learned {learned_count} new patterns")
    print("=" * 80)
    print()
    
    # === RECALL PHASE ===
    print("🧠 RECALL PHASE:")
    print("-" * 80)
    
    questions = [
        "What are fruits?",
        "Tell me about apples",
        "What are bananas like?",
        "Describe oranges",
        "What do you know about grapes?",
        "What are strawberries?",
    ]
    
    successful_recalls = 0
    for question in questions:
        print(f"\nUser: {question}")
        response = conv.process_user_input(question)
        print(f"Bot:  {response}")
        
        # Check if response contains expected fruit name
        fruit_name = None
        for fruit in ['fruit', 'apple', 'banana', 'orange', 'grape', 'strawberr']:
            if fruit in question.lower():
                fruit_name = fruit
                break
        
        if fruit_name and fruit_name in response.lower():
            successful_recalls += 1
            print("      ✅ Recalled relevant information")
        else:
            print("      ⚠️  Did not recall (may not have learned this one)")
    
    print()
    print("=" * 80)
    print(f"RESULTS: {successful_recalls}/{len(questions)} successful recalls")
    print("=" * 80)
    print()
    
    if successful_recalls >= len(questions) * 0.5:
        print("✅ Learning system is working! Can absorb and recall facts.")
    else:
        print("⚠️  Learning system needs tuning for better recall.")
    

if __name__ == "__main__":
    test_fruit_learning()
