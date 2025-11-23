#!/usr/bin/env python3
"""
Debug script to understand why patterns aren't matching.
"""

import sys
from pathlib import Path

# Add experiments/retrieval_sanity to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from minimal pipeline (no storage_bridge issues)
from pipeline import (
    ResponseFragmentStore,
    StageCoordinator,
    StageType,
)

def main():
    print("🔍 Pattern Matching Debug\n")
    
    # Initialize
    print("1. Creating StageCoordinator...")
    coordinator = StageCoordinator()
    semantic_stage = coordinator.get_stage(StageType.SEMANTIC)
    
    print("2. Loading ResponseFragmentStore...")
    store = ResponseFragmentStore(
        semantic_encoder=semantic_stage.encoder,
        storage_path="debug_patterns.json"
    )
    
    print(f"   Loaded {len(store.patterns)} patterns\n")
    
    # Test cases
    test_inputs = [
        "Hello!",
        "What is artificial intelligence?",
        "Tell me more",
        "How does it work?",
        "That's interesting!",
    ]
    
    print("3. Testing pattern retrieval:\n")
    
    for user_input in test_inputs:
        print(f"📝 Input: '{user_input}'")
        
        # Retrieve patterns
        patterns = store.retrieve_patterns(user_input, topk=5, min_score=0.0)
        
        print(f"   Retrieved {len(patterns)} patterns:")
        
        if patterns:
            for i, (pattern, score) in enumerate(patterns[:3], 1):
                print(f"     {i}. Score: {score:.4f}")
                print(f"        Trigger: '{pattern.trigger_context}'")
                print(f"        Response: '{pattern.response_text}'")
                print(f"        Success: {pattern.success_score}")
        else:
            print("     ⚠️  No patterns retrieved!")
        
        print()
    
    # Show all seed patterns
    print("\n4. All seed patterns:")
    for pattern in list(store.patterns.values())[:10]:
        print(f"   - Intent: {pattern.intent:15} Trigger: '{pattern.trigger_context}'")
    
    print("\n✅ Debug complete")

if __name__ == "__main__":
    main()
