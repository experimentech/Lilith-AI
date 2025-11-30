#!/usr/bin/env python3
"""
Test if contrastive-trained BNN improves retrieval quality.

Compare three approaches:
1. Keyword-only (baseline: 6.7/10)
2. Untrained BNN hybrid (3.3/10 - broken)
3. Contrastive-trained BNN hybrid (target: 8+/10)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments/retrieval_sanity'))

from pathlib import Path
from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.database_fragment_store import DatabaseBackedFragmentStore
from lilith.response_composer import ResponseComposer

def test_retrieval_quality(use_semantic=False, encoder_path=None):
    """Test retrieval with different configurations."""
    
    # Initialize encoder
    encoder = PMFlowEmbeddingEncoder(
        dimension=96,
        latent_dim=64,
        combine_mode="concat",
        seed=13
    )
    
    # Load trained parameters if provided
    if encoder_path:
        print(f"Loading trained encoder from {encoder_path}...")
        encoder.load_state(Path(encoder_path))
    
    # Initialize fragment store and response composer
    fragments = DatabaseBackedFragmentStore(
        semantic_encoder=encoder,
        storage_path="experiments/retrieval_sanity/conversation_patterns.db"
    )
    
    # Create simple conversation state
    from pipeline.conversation_state import ConversationState
    from pipeline.response_fragments import ResponseFragmentStore as MemoryFragmentStore
    
    state = ConversationState()
    
    composer = ResponseComposer(
        fragment_store=fragments,
        conversation_state=state,
        semantic_encoder=encoder
    )
    
    # Test queries
    test_queries = [
        "hi",
        "hello there",
        "how are you doing",
        "what's the weather like",
        "do you like films",
        "goodbye",
        "see you later",
    ]
    
    print(f"\n{'='*80}")
    print(f"Testing {'SEMANTIC' if use_semantic else 'KEYWORD-ONLY'} Retrieval")
    if encoder_path:
        print(f"Using TRAINED encoder from {encoder_path}")
    print(f"{'='*80}\n")
    
    for query in test_queries:
        response = composer.compose_response(
            user_input=query,
            use_semantic_retrieval=use_semantic,
            semantic_weight=0.5  # Try 50/50 instead of 30/70
        )
        print(f"Q: {query}")
        print(f"A: {response}")
        print()


def main():
    print("=" * 80)
    print("RETRIEVAL QUALITY COMPARISON")
    print("=" * 80)
    
    # Test 1: Keyword-only (baseline)
    print("\n\n1. BASELINE: Keyword-only retrieval")
    print("-" * 80)
    test_retrieval_quality(use_semantic=False)
    
    # Test 2: Untrained BNN hybrid
    print("\n\n2. UNTRAINED BNN: Hybrid retrieval with untrained encoder")
    print("-" * 80)
    test_retrieval_quality(use_semantic=True, encoder_path=None)
    
    # Test 3: Contrastive-trained BNN hybrid
    print("\n\n3. TRAINED BNN: Hybrid retrieval with contrastive-trained encoder")
    print("-" * 80)
    if Path("trained_semantic_encoder.pt").exists():
        test_retrieval_quality(use_semantic=True, encoder_path="trained_semantic_encoder.pt")
    else:
        print("⚠️  trained_semantic_encoder.pt not found. Run train_contrastive_semantics.py first")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Compare the responses above:")
    print("  - Did trained BNN find more relevant patterns?")
    print("  - Do responses match context better?")
    print("  - Rate quality: 0-10 for each approach")


if __name__ == "__main__":
    main()
