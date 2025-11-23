#!/usr/bin/env python3
"""
Debug script to understand response composition.
"""

import sys
from pathlib import Path

# Add experiments/retrieval_sanity to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import (
    ResponseFragmentStore,
    ResponseComposer,
    ConversationState,
    StageCoordinator,
    StageType,
)

def main():
    print("🔍 Response Composition Debug\n")
    
    # Initialize
    print("1. Creating components...")
    coordinator = StageCoordinator()
    semantic_stage = coordinator.get_stage(StageType.SEMANTIC)
    
    store = ResponseFragmentStore(
        semantic_encoder=semantic_stage.encoder,
        storage_path="debug_patterns.json"
    )
    
    conv_state = ConversationState(
        encoder=semantic_stage.encoder,
        decay=0.75,
        max_topics=5
    )
    
    composer = ResponseComposer(
        fragment_store=store,
        conversation_state=conv_state,
        composition_mode="best_match"
    )
    
    print(f"   Loaded {len(store.patterns)} patterns\n")
    
    # Test composition
    test_inputs = [
        "Hello!",
        "What is artificial intelligence?",
        "Tell me more",
    ]
    
    print("2. Testing composition:\n")
    
    for user_input in test_inputs:
        print(f"📝 Input: '{user_input}'")
        
        # Retrieve patterns (before composition)
        raw_patterns = store.retrieve_patterns(user_input, topk=5, min_score=0.0)
        print(f"   Raw retrieval (top 3):")
        for i, (pattern, score) in enumerate(raw_patterns[:3], 1):
            print(f"     {i}. Score: {score:.4f} - '{pattern.response_text}'")
        
        # Full composition
        response = composer.compose_response(
            context=user_input,
            user_input=user_input,
            topk=5
        )
        
        print(f"\n   Final composed:")
        print(f"     Response: '{response.text}'")
        print(f"     Fragments: {len(response.fragment_ids)}")
        print(f"     Coherence: {response.coherence_score:.4f}")
        if response.primary_pattern:
            print(f"     Primary: '{response.primary_pattern.trigger_context}'")
        print()
    
    print("✅ Debug complete")

if __name__ == "__main__":
    main()
