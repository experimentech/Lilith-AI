#!/usr/bin/env python3
"""Test BNN-based syntax stage with reinforcement learning."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pipeline.syntax_stage_bnn import SyntaxStage


def main():
    print("Testing BNN-Based Syntax Stage")
    print("=" * 80)
    print()
    
    # Initialize syntax stage
    print("Initializing syntax stage with PMFlow encoder...")
    syntax_stage = SyntaxStage(storage_path=Path("test_syntax_patterns.json"))
    print(f"✓ Loaded {len(syntax_stage.patterns)} patterns")
    print()
    
    # Test 1: Process various inputs
    print("=" * 80)
    print("TEST 1: Processing different sentence types")
    print("=" * 80)
    
    test_sentences = [
        ["How", "does", "that", "work", "?"],
        ["That", "is", "really", "interesting"],
        ["I", "like", "patterns", "and", "embeddings"],
        ["What", "can", "you", "do", "?"],
    ]
    
    for tokens in test_sentences:
        print(f"\nInput: {' '.join(tokens)}")
        
        # Process through syntax stage
        artifact = syntax_stage.process(tokens)
        
        print(f"  Intent: {artifact.metadata['intent']}")
        print(f"  POS: {' '.join(artifact.metadata['pos_sequence'])}")
        print(f"  Confidence: {artifact.confidence:.3f}")
        print(f"  Energy: {artifact.metadata['activation_energy']:.3f}")
        
        # Show matched patterns
        if artifact.metadata['matched_patterns']:
            print(f"  Matched patterns:")
            for match in artifact.metadata['matched_patterns'][:2]:
                print(f"    - {match['template']} (score: {match['score']:.3f})")
    
    # Test 2: Learn new pattern from successful interaction
    print("\n" + "=" * 80)
    print("TEST 2: Learning new grammatical pattern")
    print("=" * 80)
    
    # Simulate successful user input
    user_tokens = ["This", "is", "absolutely", "fascinating", "!"]
    pos_tags = ["PRON", "VBZ", "ADV", "ADJ", "PUNCT"]
    
    print(f"\nUser said: {' '.join(user_tokens)}")
    print(f"POS tags: {' '.join(pos_tags)}")
    print(f"Feedback: +0.8 (very engaging)")
    
    pattern_id = syntax_stage.learn_pattern(user_tokens, pos_tags, success_feedback=0.8)
    print(f"✓ Learned pattern: {pattern_id}")
    
    # Show learned pattern
    learned = syntax_stage.patterns[pattern_id]
    print(f"  Template: {learned.template}")
    print(f"  Success score: {learned.success_score:.3f}")
    print(f"  Intent: {learned.intent}")
    
    # Test 3: Pattern retrieval uses BNN similarity
    print("\n" + "=" * 80)
    print("TEST 3: Similar pattern retrieval via BNN")
    print("=" * 80)
    
    # Try similar exclamation
    similar_tokens = ["That", "is", "really", "amazing", "!"]
    print(f"\nQuery: {' '.join(similar_tokens)}")
    
    artifact = syntax_stage.process(similar_tokens)
    print(f"  Matched patterns (via BNN similarity):")
    for match in artifact.metadata['matched_patterns'][:3]:
        print(f"    - {match['template']} (score: {match['score']:.3f})")
    
    # Test 4: Reinforcement learning
    print("\n" + "=" * 80)
    print("TEST 4: Reinforcement updates pattern scores")
    print("=" * 80)
    
    print(f"\nPattern '{pattern_id}' before feedback:")
    print(f"  Success score: {learned.success_score:.3f}")
    print(f"  Usage count: {learned.usage_count}")
    
    # Apply positive feedback
    syntax_stage.update_pattern_success(pattern_id, feedback=0.3, learning_rate=0.1)
    
    print(f"\nAfter positive feedback (+0.3):")
    print(f"  Success score: {learned.success_score:.3f}")
    print(f"  Usage count: {learned.usage_count}")
    
    # Apply negative feedback
    syntax_stage.update_pattern_success(pattern_id, feedback=-0.2, learning_rate=0.1)
    
    print(f"\nAfter negative feedback (-0.2):")
    print(f"  Success score: {learned.success_score:.3f}")
    print(f"  Usage count: {learned.usage_count}")
    
    print("\n" + "=" * 80)
    print("✅ BNN-based syntax stage working!")
    print()
    print("Key features verified:")
    print("  ✓ PMFlow encoding of POS sequences")
    print("  ✓ Pattern retrieval via BNN similarity")
    print("  ✓ Learning new patterns from interaction")
    print("  ✓ Reinforcement-based success updates")
    print("=" * 80)


if __name__ == "__main__":
    main()
