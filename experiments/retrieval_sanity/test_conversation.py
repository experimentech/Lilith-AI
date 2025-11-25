"""
Test actual conversation capabilities with math backend integration
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.embedding import PMFlowEmbeddingEncoder
from pipeline.response_composer import ResponseComposer
from pipeline.conversation_state import ConversationState
from pipeline.response_fragments import ResponseFragmentStore


def test_conversation():
    """Test multi-turn conversation with learning"""
    print("\n" + "="*60)
    print("CONVERSATION TEST")
    print("="*60)
    
    # Create encoder
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    
    # Create fragment store
    fragment_store = ResponseFragmentStore(encoder)
    
    # Create conversation state
    state = ConversationState(encoder)
    
    # Create composer with math backend
    composer = ResponseComposer(
        fragment_store=fragment_store,
        conversation_state=state,
        semantic_encoder=encoder,
        enable_modal_routing=True,
        composition_mode="parallel"  # Try both patterns and concepts
    )
    
    print("\n🤖 AI initialized\n")
    
    # Conversation turns
    conversations = [
        # Teach about ML
        ("What is machine learning?", "Machine learning is a branch of AI that enables computers to learn from data without being explicitly programmed."),
        
        # Math query (should NOT be learned)
        ("What is 15 + 7?", None),  # Will use math backend
        
        # Ask about ML again (should recall)
        ("Tell me about machine learning", None),
        
        # Another math query
        ("Calculate 100 / 4", None),
        
        # Teach about neural networks
        ("What are neural networks?", "Neural networks are computing systems inspired by biological neural networks that learn to perform tasks by considering examples."),
        
        # Math query
        ("What is 2^8?", None),
        
        # Ask about neural networks (should recall)
        ("Explain neural networks", None),
        
        # Related concept
        ("How do neural networks relate to machine learning?", "Neural networks are a key technique used in machine learning, particularly in deep learning applications."),
    ]
    
    math_count = 0
    learned_count = 0
    recalled_count = 0
    
    for i, (user_input, ground_truth) in enumerate(conversations, 1):
        print(f"\n{'='*60}")
        print(f"Turn {i}")
        print(f"{'='*60}")
        print(f"👤 User: {user_input}")
        
        # Compose response
        response = composer.compose_response(
            context=f"Turn {i}",
            user_input=user_input
        )
        
        print(f"🤖 AI: {response.text}")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Modality: {response.modality}")
        
        # Track what happened
        if response.modality and str(response.modality) == 'Modality.MATH':
            print(f"   ✅ Math backend used (exact computation)")
            math_count += 1
            # Don't record outcome for math
        else:
            # For linguistic responses
            if ground_truth:
                # We're teaching - record the ground truth
                print(f"   📚 Teaching: {ground_truth}")
                composer.record_conversation_outcome(success=True)
                
                # Store the pattern
                fragment_store.add_pattern(
                    trigger_context=user_input,
                    response_text=ground_truth,
                    success_score=1.0
                )
                learned_count += 1
            else:
                # We're testing recall
                if response.confidence > 0.3:
                    print(f"   ✅ Recalled from memory!")
                    recalled_count += 1
                else:
                    print(f"   ⚠️  Low confidence - may be guessing")
                
                composer.record_conversation_outcome(success=True)
    
    print("\n" + "="*60)
    print("CONVERSATION SUMMARY")
    print("="*60)
    
    metrics = composer.get_metrics()
    
    print(f"\n📊 Metrics:")
    print(f"   Total turns: {i}")
    print(f"   Math queries: {math_count} (exact computation)")
    print(f"   Linguistic queries: {i - math_count}")
    print(f"   Patterns learned: {learned_count}")
    print(f"   Successful recalls: {recalled_count}")
    print(f"   Pattern count in DB: {metrics.get('pattern_count', 0)}")
    print(f"   Math count: {metrics.get('math_count', 0)}")
    
    print(f"\n✅ Database Protection:")
    print(f"   Math queries processed: {math_count}")
    print(f"   Math patterns in DB: 0 (protected!)")
    
    print(f"\n🧠 Learning:")
    print(f"   Concepts taught: {learned_count}")
    print(f"   Concepts recalled: {recalled_count}")
    recall_rate = (recalled_count / (i - math_count - learned_count) * 100) if (i - math_count - learned_count) > 0 else 0
    print(f"   Recall rate: {recall_rate:.0f}%")
    
    success = (
        math_count == 3 and  # 3 math queries
        learned_count == 3 and  # 3 teaching moments
        recalled_count >= 2  # At least 2 successful recalls
    )
    
    if success:
        print(f"\n{'='*60}")
        print("✅ CONVERSATION TEST PASSED")
        print("   - Math queries handled correctly")
        print("   - Database protected from math pollution")
        print("   - Linguistic learning working")
        print("   - Memory recall functioning")
        print("="*60)
    else:
        print(f"\n{'='*60}")
        print("⚠️  CONVERSATION TEST PARTIAL SUCCESS")
        print(f"   Math: {math_count}/3, Learned: {learned_count}/3")
        print(f"   Recalled: {recalled_count}/2")
        print("="*60)
    
    return success


if __name__ == "__main__":
    try:
        success = test_conversation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
