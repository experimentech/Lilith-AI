#!/usr/bin/env python3
"""
Train Contrastive Learner for Semantic Embeddings

This script trains the BNN's PMFlow field to understand semantic relationships:
- Pull similar concepts together in latent space
- Push dissimilar concepts apart
- Learn from databases, patterns, and user corrections

Usage:
    python train_contrastive.py                    # Train with defaults
    python train_contrastive.py --epochs 20       # Custom epochs
    python train_contrastive.py --from-patterns   # Include pattern DB pairs
    python train_contrastive.py --eval-only       # Just evaluate current model

After training, the encoder will be saved and can be loaded by ResponseComposer.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.contrastive_learner import ContrastiveLearner


def main():
    parser = argparse.ArgumentParser(description="Train contrastive semantic embeddings")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--margin", type=float, default=0.3, help="Contrastive margin")
    parser.add_argument("--from-patterns", action="store_true", help="Load pairs from pattern DB")
    parser.add_argument("--from-concepts", action="store_true", help="Load pairs from concept DB")
    parser.add_argument("--pattern-db", type=str, default="data/patterns.db", help="Pattern DB path")
    parser.add_argument("--concept-db", type=str, default="data/concepts.db", help="Concept DB path")
    parser.add_argument("--save-path", type=str, default="data/contrastive_learner", help="Save path")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--dimension", type=int, default=64, help="Encoder dimension")
    parser.add_argument("--latent-dim", type=int, default=32, help="PMFlow latent dimension")
    args = parser.parse_args()
    
    print("=" * 70)
    print("CONTRASTIVE LEARNER TRAINING")
    print("=" * 70)
    
    # Initialize encoder and learner
    print(f"\n[1/4] Initializing encoder (dim={args.dimension}, latent={args.latent_dim})...")
    encoder = PMFlowEmbeddingEncoder(dimension=args.dimension, latent_dim=args.latent_dim)
    learner = ContrastiveLearner(
        encoder, 
        margin=args.margin, 
        learning_rate=args.lr
    )
    
    # Try to load existing state
    save_path = Path(args.save_path)
    if save_path.with_suffix('.json').exists():
        print(f"  Loading existing state from {save_path}...")
        learner.load(save_path)
        print(f"  Loaded {len(learner.pairs)} pairs, {len(learner.metrics_history)} epochs trained")
    
    if args.eval_only:
        print("\n[EVAL ONLY] Skipping training, running evaluation...")
        run_evaluation(learner)
        return
    
    # Collect training pairs
    print("\n[2/4] Collecting training pairs...")
    
    # Always include core semantic pairs
    learner.generate_core_semantic_pairs()
    
    # Optional: Load from pattern database
    if args.from_patterns:
        pattern_db = Path(args.pattern_db)
        if pattern_db.exists():
            learner.load_from_pattern_database(str(pattern_db))
        else:
            print(f"  Warning: Pattern DB not found at {pattern_db}")
    
    # Optional: Load from concept database
    if args.from_concepts:
        concept_db = Path(args.concept_db)
        if concept_db.exists():
            learner.load_from_concept_database(str(concept_db))
        else:
            print(f"  Warning: Concept DB not found at {concept_db}")
    
    print(f"  Total pairs: {len(learner.pairs)}")
    
    # Count by source
    sources = {}
    for pair in learner.pairs:
        sources[pair.source] = sources.get(pair.source, 0) + 1
    for source, count in sorted(sources.items()):
        print(f"    - {source}: {count}")
    
    # Pre-training evaluation
    print("\n[3/4] Training...")
    print("  Pre-training similarity check:")
    run_quick_eval(learner, prefix="    Before: ")
    
    # Train
    metrics = learner.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stop_margin=0.6,
        verbose=True
    )
    
    print("  Post-training similarity check:")
    run_quick_eval(learner, prefix="    After:  ")
    
    # Save
    print(f"\n[4/4] Saving to {save_path}...")
    learner.save(save_path)
    print("  Done!")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(learner.get_training_summary())
    
    # Full evaluation
    print("\n" + "-" * 70)
    run_evaluation(learner)


def run_quick_eval(learner, prefix=""):
    """Quick similarity check on key pairs."""
    pairs = [
        ("cat", "dog"),
        ("cat", "computer"),
        ("happy", "sad"),
    ]
    sims = [f"{learner.similarity(a, b):+.2f}" for a, b in pairs]
    print(f"{prefix}cat↔dog={sims[0]}, cat↔computer={sims[1]}, happy↔sad={sims[2]}")


def run_evaluation(learner):
    """Run full evaluation suite."""
    print("EVALUATION RESULTS")
    print("-" * 70)
    
    # Test pairs with expected similarity
    test_pairs = [
        # Should be similar
        ("machine learning", "artificial intelligence", True),
        ("neural network", "deep learning", True),
        ("cat", "dog", True),
        ("Python", "programming", True),
        ("apple", "fruit", True),
        ("happy", "joyful", True),
        
        # Should be different
        ("cat", "computer", False),
        ("happy", "sad", False),
        ("hot", "cold", False),
        ("up", "down", False),
        ("python", "happiness", False),
    ]
    
    print("\nPair-by-pair results:")
    correct = 0
    for a, b, should_be_similar in test_pairs:
        sim = learner.similarity(a, b)
        pred_similar = sim > 0.0
        is_correct = pred_similar == should_be_similar
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        expected = "similar" if should_be_similar else "different"
        print(f"  {status} {a:25} ↔ {b:25}: {sim:+.3f} (expected {expected})")
    
    accuracy = correct / len(test_pairs)
    print(f"\nAccuracy: {correct}/{len(test_pairs)} = {accuracy:.1%}")
    
    # Aggregate metrics
    results = learner.evaluate_pairs(test_pairs)
    print(f"Avg positive similarity: {results['avg_positive_similarity']:+.3f}")
    print(f"Avg negative similarity: {results['avg_negative_similarity']:+.3f}")
    margin = results['avg_positive_similarity'] - results['avg_negative_similarity']
    print(f"Margin (pos - neg):      {margin:+.3f}")


if __name__ == "__main__":
    main()
