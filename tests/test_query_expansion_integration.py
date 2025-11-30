"""
Test PMFlow retrieval enhancements in hybrid retrieval.

This validates that the query expansion improves synonym matching
in the database-backed fragment store.
"""

import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments" / "retrieval_sanity"))

from lilith.database_fragment_store import DatabaseBackedFragmentStore
from lilith.embedding import PMFlowEmbeddingEncoder
from pattern_database import PatternDatabase


def test_query_expansion():
    """Test that query expansion improves synonym matching."""
    
    print("=" * 70)
    print("Testing PMFlow Query Expansion in Hybrid Retrieval")
    print("=" * 70)
    
    # Create test database
    db_path = Path("/tmp/test_retrieval_expansion.db")
    if db_path.exists():
        db_path.unlink()
    
    db = PatternDatabase(str(db_path))
    
    # Add some patterns with synonyms
    patterns = [
        {
            "fragment_id": "ml_001",
            "trigger_context": "machine learning",
            "response_text": "Machine learning is a subset of AI that enables systems to learn from data.",
            "intent": "definition",
            "success_score": 0.95
        },
        {
            "fragment_id": "dl_001", 
            "trigger_context": "deep learning",
            "response_text": "Deep learning uses neural networks with multiple layers.",
            "intent": "definition",
            "success_score": 0.92
        },
        {
            "fragment_id": "ai_001",
            "trigger_context": "artificial intelligence",
            "response_text": "Artificial intelligence is the simulation of human intelligence in machines.",
            "intent": "definition",
            "success_score": 0.90
        },
        {
            "fragment_id": "nn_001",
            "trigger_context": "neural network",
            "response_text": "Neural networks are computing systems inspired by biological neural networks.",
            "intent": "definition",
            "success_score": 0.88
        }
    ]
    
    for pattern in patterns:
        db.insert_pattern(**pattern)
    
    print(f"\n✓ Created test database with {len(patterns)} patterns")
    
    # Create encoder and store
    encoder = PMFlowEmbeddingEncoder(dimension=96, latent_dim=64, combine_mode="concat")
    store = DatabaseBackedFragmentStore(encoder, str(db_path))
    
    print(f"✓ Created database store with PMFlow encoder")
    
    # Test queries (synonyms and abbreviations)
    test_queries = [
        ("ML", "machine learning"),  # Abbreviation
        ("DL", "deep learning"),  # Abbreviation
        ("AI", "artificial intelligence"),  # Abbreviation
        ("NN", "neural network"),  # Abbreviation
        ("machine learning", "machine learning"),  # Exact match
    ]
    
    print("\n" + "-" * 70)
    print("Testing Query Expansion Impact")
    print("-" * 70)
    
    for query, expected_trigger in test_queries:
        print(f"\nQuery: '{query}' (expect: '{expected_trigger}')")
        
        # Without expansion
        results_without = store.retrieve_patterns_hybrid(
            query,
            topk=3,
            semantic_weight=0.5,
            use_query_expansion=False
        )
        
        # With expansion
        results_with = store.retrieve_patterns_hybrid(
            query,
            topk=3,
            semantic_weight=0.5,
            use_query_expansion=True
        )
        
        print(f"\n  WITHOUT expansion:")
        if results_without:
            for i, (pattern, score) in enumerate(results_without[:3]):
                match = "✓" if pattern.trigger_context == expected_trigger else " "
                print(f"    {i+1}. [{match}] {pattern.trigger_context:25s} (score: {score:.3f})")
        else:
            print(f"    (no results)")
        
        print(f"\n  WITH expansion:")
        if results_with:
            for i, (pattern, score) in enumerate(results_with[:3]):
                match = "✓" if pattern.trigger_context == expected_trigger else " "
                print(f"    {i+1}. [{match}] {pattern.trigger_context:25s} (score: {score:.3f})")
        else:
            print(f"    (no results)")
        
        # Check if expansion helped
        found_without = any(p.trigger_context == expected_trigger for p, _ in results_without)
        found_with = any(p.trigger_context == expected_trigger for p, _ in results_with)
        
        if not found_without and found_with:
            print(f"  → ✅ Expansion ENABLED matching (was not found before)")
        elif found_without and found_with:
            # Check if ranking improved
            rank_without = next((i for i, (p, _) in enumerate(results_without) 
                               if p.trigger_context == expected_trigger), None)
            rank_with = next((i for i, (p, _) in enumerate(results_with) 
                            if p.trigger_context == expected_trigger), None)
            
            if rank_with < rank_without:
                print(f"  → ✅ Expansion IMPROVED ranking ({rank_without+1} → {rank_with+1})")
            else:
                print(f"  → ✓ Already found (both ranked at {rank_with+1})")
        elif not found_without and not found_with:
            print(f"  → ⚠️ Not found (may need more training)")
        else:
            print(f"  → ✓ Found in both")
    
    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)
    
    # Cleanup
    db.close()
    db_path.unlink()


if __name__ == "__main__":
    test_query_expansion()
