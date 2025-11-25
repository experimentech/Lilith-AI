#!/usr/bin/env python3
"""
Train BNN Encoder with Contrastive Learning for Synonym Consolidation

Goal: Make synonyms cluster together in embedding space so they automatically
merge in ConceptStore (similarity >0.90).

Strategy:
1. Define ML domain synonym pairs
2. Use contrastive_plasticity to pull synonyms together
3. Test similarity before/after training
4. Validate PoC storage efficiency improves

Expected Outcome:
- Before: "ML" vs "machine learning" similarity ~0.85
- After: similarity >0.90 (automatic merge)
- Storage efficiency: 1.0 → 0.5-0.6 ratio
"""

import sys
from pathlib import Path
import numpy as np

# Add project root for pmflow_bnn_enhanced
project_root = Path(__file__).parent.parent.parent.parent
pipeline_dir = Path(__file__).parent.parent / "pipeline"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(pipeline_dir))

from embedding import PMFlowEmbeddingEncoder
from pmflow_bnn_enhanced.pmflow import contrastive_plasticity


# ML Domain Synonym Pairs
SYNONYM_PAIRS = [
    # Machine Learning variations
    ("machine learning", "ML"),
    ("machine learning", "ml"),
    ("machine learning", "machine-learning"),
    ("machine learning", "ml algorithm"),
    
    # Deep Learning variations
    ("deep learning", "DL"),
    ("deep learning", "deep neural network"),
    ("deep learning", "deep neural networks"),
    ("deep learning", "deep-learning"),
    
    # Artificial Intelligence variations
    ("artificial intelligence", "AI"),
    ("artificial intelligence", "A.I."),
    ("artificial intelligence", "ai"),
    
    # Supervised Learning variations
    ("supervised learning", "supervised"),
    ("supervised learning", "supervised training"),
    ("supervised learning", "labeled training"),
    
    # Unsupervised Learning variations
    ("unsupervised learning", "unsupervised"),
    ("unsupervised learning", "unsupervised training"),
    ("unsupervised learning", "unlabeled training"),
    
    # Reinforcement Learning variations
    ("reinforcement learning", "RL"),
    ("reinforcement learning", "reinforcement"),
    ("reinforcement learning", "reward-based learning"),
    
    # Neural Network variations
    ("neural network", "neural net"),
    ("neural network", "NN"),
    ("neural network", "artificial neural network"),
    
    # Training variations
    ("training data", "training set"),
    ("training data", "training examples"),
    
    # Model variations
    ("model", "neural model"),
    ("model", "trained model"),
]

# Dissimilar pairs (should be far apart)
DISSIMILAR_PAIRS = [
    # ML concepts vs non-ML
    ("machine learning", "bicycle"),
    ("deep learning", "cooking"),
    ("neural network", "garden"),
    
    # Different ML concepts
    ("supervised learning", "unsupervised learning"),
    ("deep learning", "shallow learning"),
    ("training", "testing"),
    
    # Opposite concepts
    ("labeled data", "unlabeled data"),
    ("supervised", "unsupervised"),
]


def compute_similarity(encoder, phrase1: str, phrase2: str) -> float:
    """Compute cosine similarity between two phrases"""
    # Tokenize and encode
    tokens1 = phrase1.lower().split()
    tokens2 = phrase2.lower().split()
    
    emb1 = encoder.encode(tokens1)
    emb2 = encoder.encode(tokens2)
    
    # Convert to numpy
    if hasattr(emb1, 'cpu'):
        emb1 = emb1.cpu().detach().numpy().flatten()
        emb2 = emb2.cpu().detach().numpy().flatten()
    
    # Cosine similarity
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def evaluate_similarities(encoder, pairs, label="Pairs"):
    """Evaluate and print similarities for pairs"""
    print(f"\n{label}:")
    print("-" * 70)
    
    similarities = []
    for phrase1, phrase2 in pairs[:10]:  # Show first 10
        sim = compute_similarity(encoder, phrase1, phrase2)
        similarities.append(sim)
        print(f"  {phrase1:30s} ↔ {phrase2:30s}  {sim:.3f}")
    
    if len(pairs) > 10:
        print(f"  ... and {len(pairs) - 10} more pairs")
    
    avg_sim = np.mean([compute_similarity(encoder, p1, p2) for p1, p2 in pairs])
    print(f"\nAverage similarity: {avg_sim:.3f}")
    
    return avg_sim


def train_contrastive(encoder, similar_pairs, dissimilar_pairs, epochs=50):
    """Train encoder using contrastive learning"""
    
    print("\n" + "=" * 70)
    print("CONTRASTIVE TRAINING")
    print("=" * 70)
    
    # Note: For MultiScalePMField, we'll train on the latent space which feeds into the field
    # The contrastive learning will adjust how the PMField processes these latents
    
    # Prepare latent embeddings for training
    import torch
    
    similar_latents = []
    for phrase1, phrase2 in similar_pairs:
        tokens1 = phrase1.lower().split()
        tokens2 = phrase2.lower().split()
        
        # Get base encoder output (before PMField)
        base1 = encoder.base_encoder.encode(tokens1).to(encoder.device)
        base2 = encoder.base_encoder.encode(tokens2).to(encoder.device)
        
        # Project to latent space
        latent1 = base1 @ encoder._projection
        latent2 = base2 @ encoder._projection
        
        similar_latents.append((latent1, latent2))
    
    dissimilar_latents = []
    for phrase1, phrase2 in dissimilar_pairs:
        tokens1 = phrase1.lower().split()
        tokens2 = phrase2.lower().split()
        
        base1 = encoder.base_encoder.encode(tokens1).to(encoder.device)
        base2 = encoder.base_encoder.encode(tokens2).to(encoder.device)
        
        latent1 = base1 @ encoder._projection
        latent2 = base2 @ encoder._projection
        
        dissimilar_latents.append((latent1, latent2))
    
    print(f"\nTraining configuration:")
    print(f"  Similar pairs: {len(similar_latents)}")
    print(f"  Dissimilar pairs: {len(dissimilar_latents)}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: 1e-3 (mu), 1e-3 (centers)")
    print(f"  Margin: 1.0")
    
    # Check if we have MultiScalePMField
    if hasattr(encoder.pm_field, 'fine_field'):
        print(f"  Field type: MultiScalePMField")
        print(f"  Training: fine_field and coarse_field")
    else:
        print(f"  Field type: Standard PMField")
    
    print("\nTraining progress:")
    
    # Training loop
    for epoch in range(epochs):
        # Train on both fine and coarse fields if MultiScale
        if hasattr(encoder.pm_field, 'fine_field'):
            # Train fine field (only - avoid dimension mismatch with coarse)
            contrastive_plasticity(
                encoder.pm_field.fine_field,
                similar_pairs=similar_latents,
                dissimilar_pairs=dissimilar_latents,
                mu_lr=1e-3,
                c_lr=1e-3,
                margin=1.0
            )
        else:
            # Train standard field
            contrastive_plasticity(
                encoder.pm_field,
                similar_pairs=similar_latents,
                dissimilar_pairs=dissimilar_latents,
                mu_lr=1e-3,
                c_lr=1e-3,
                margin=1.0
            )
        
        # Progress indicator
        if (epoch + 1) % 10 == 0:
            # Sample similarity check
            sample_sim = compute_similarity(encoder, "machine learning", "ML")
            print(f"  Epoch {epoch + 1:3d}/{epochs}: Sample similarity (ML/machine learning) = {sample_sim:.3f}")
    
    print("\n✓ Training complete!")


def main():
    print("=" * 70)
    print("BNN SYNONYM CONSOLIDATION TRAINING")
    print("=" * 70)
    
    # Initialize encoder
    print("\n📦 Initializing PMFlowEmbeddingEncoder...")
    encoder = PMFlowEmbeddingEncoder(
        dimension=96,
        latent_dim=64,
        combine_mode="concat",
        seed=13
    )
    print("✓ Encoder initialized")
    
    # Evaluate BEFORE training
    print("\n" + "=" * 70)
    print("BEFORE TRAINING")
    print("=" * 70)
    
    similar_before = evaluate_similarities(encoder, SYNONYM_PAIRS, "Similar Pairs (Should be HIGH)")
    dissimilar_before = evaluate_similarities(encoder, DISSIMILAR_PAIRS, "Dissimilar Pairs (Should be LOW)")
    
    # Key test cases
    print("\n\nKey Test Cases (BEFORE):")
    print("-" * 70)
    ml_similarity_before = compute_similarity(encoder, "machine learning", "ML")
    dl_similarity_before = compute_similarity(encoder, "deep learning", "DL")
    ai_similarity_before = compute_similarity(encoder, "artificial intelligence", "AI")
    
    print(f"  'machine learning' ↔ 'ML':                    {ml_similarity_before:.3f}")
    print(f"  'deep learning' ↔ 'DL':                       {dl_similarity_before:.3f}")
    print(f"  'artificial intelligence' ↔ 'AI':             {ai_similarity_before:.3f}")
    
    if ml_similarity_before < 0.90:
        print(f"\n  ⚠️  Similarity {ml_similarity_before:.3f} < 0.90 threshold")
        print(f"      → 'ML' and 'machine learning' will NOT merge in ConceptStore")
    else:
        print(f"\n  ✓ Similarity {ml_similarity_before:.3f} ≥ 0.90 threshold")
        print(f"      → Would merge automatically")
    
    # Train
    train_contrastive(encoder, SYNONYM_PAIRS, DISSIMILAR_PAIRS, epochs=50)
    
    # Evaluate AFTER training
    print("\n" + "=" * 70)
    print("AFTER TRAINING")
    print("=" * 70)
    
    similar_after = evaluate_similarities(encoder, SYNONYM_PAIRS, "Similar Pairs (Should be HIGH)")
    dissimilar_after = evaluate_similarities(encoder, DISSIMILAR_PAIRS, "Dissimilar Pairs (Should be LOW)")
    
    # Key test cases
    print("\n\nKey Test Cases (AFTER):")
    print("-" * 70)
    ml_similarity_after = compute_similarity(encoder, "machine learning", "ML")
    dl_similarity_after = compute_similarity(encoder, "DL", "deep learning")
    ai_similarity_after = compute_similarity(encoder, "artificial intelligence", "AI")
    
    print(f"  'machine learning' ↔ 'ML':                    {ml_similarity_after:.3f}")
    print(f"  'deep learning' ↔ 'DL':                       {dl_similarity_after:.3f}")
    print(f"  'artificial intelligence' ↔ 'AI':             {ai_similarity_after:.3f}")
    
    if ml_similarity_after >= 0.90:
        print(f"\n  ✓ Similarity {ml_similarity_after:.3f} ≥ 0.90 threshold")
        print(f"      → 'ML' and 'machine learning' will now MERGE! ✓")
    else:
        print(f"\n  ⚠️  Similarity {ml_similarity_after:.3f} < 0.90 threshold")
        print(f"      → Still won't merge (needs more training)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nSimilar Pairs:")
    print(f"  Before: {similar_before:.3f}")
    print(f"  After:  {similar_after:.3f}")
    print(f"  Change: {similar_after - similar_before:+.3f}")
    
    print(f"\nDissimilar Pairs:")
    print(f"  Before: {dissimilar_before:.3f}")
    print(f"  After:  {dissimilar_after:.3f}")
    print(f"  Change: {dissimilar_after - dissimilar_before:+.3f}")
    
    print(f"\nKey Metrics:")
    print(f"  ML/machine learning:  {ml_similarity_before:.3f} → {ml_similarity_after:.3f} ({ml_similarity_after - ml_similarity_before:+.3f})")
    print(f"  DL/deep learning:     {dl_similarity_before:.3f} → {dl_similarity_after:.3f} ({dl_similarity_after - dl_similarity_before:+.3f})")
    print(f"  AI/artificial intel:  {ai_similarity_before:.3f} → {ai_similarity_after:.3f} ({ai_similarity_after - ai_similarity_before:+.3f})")
    
    # Success criteria
    success = (
        similar_after > similar_before + 0.05 and  # Similar pairs more similar
        dissimilar_after < dissimilar_before + 0.05 and  # Dissimilar pairs stay apart
        ml_similarity_after >= 0.90  # Key test case passes threshold
    )
    
    if success:
        print("\n🎉 TRAINING SUCCESS!")
        print("   Synonyms now cluster together (≥0.90 similarity)")
        print("   ConceptStore will automatically merge them")
        
        # Save trained encoder
        save_path = Path(__file__).parent / "trained_encoder.pt"
        encoder.save_state(save_path)
        print(f"\n💾 Saved trained encoder to: {save_path}")
        print("   Use this encoder in production ConceptStore!")
        
    else:
        print("\n⚠️  TRAINING INCOMPLETE")
        print("   Similarities improved but not enough for automatic merge")
        print("   Consider:")
        print("   - More training epochs (50 → 100)")
        print("   - Higher learning rate (1e-3 → 5e-3)")
        print("   - Lower consolidation threshold (0.90 → 0.85)")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
