#!/usr/bin/env python3
"""
Test if PMFlow-only mode (no base hashing) shows contrastive learning effect.

The problem: Base hashing (96dim) + PMFlow (96dim) concatenated.
PMFlow learns but gets masked by unchanging base embeddings.

Solution: Use combine_mode="pm-only" so we see ONLY the learned PMFlow part.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments/retrieval_sanity'))

import torch
import numpy as np
from pathlib import Path
from pmflow import contrastive_plasticity
from lilith.embedding import PMFlowEmbeddingEncoder


def compute_similarity(encoder, phrase1, phrase2):
    """Compute cosine similarity between two phrases."""
    tokens1 = phrase1.lower().split()
    tokens2 = phrase2.lower().split()
    
    emb1 = encoder.encode(tokens1).cpu().detach().numpy().flatten()
    emb2 = encoder.encode(tokens2).cpu().detach().numpy().flatten()
    
    # Cosine similarity
    dot = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    return dot / (norm1 * norm2 + 1e-8)


def main():
    print("=" * 80)
    print("PMFlow-ONLY Contrastive Learning Test")
    print("=" * 80)
    
    # Initialize encoder in PM-ONLY mode (no base hashing to mask the learning)
    print("\n1. Loading BNN encoder (PM-ONLY mode)...")
    encoder = PMFlowEmbeddingEncoder(
        dimension=96,
        latent_dim=64,
        combine_mode="pm-only",  # ← KEY CHANGE: No base hashing
        seed=13
    )
    print(f"  Encoder mode: {encoder.combine_mode}")
    print(f"  PM Field type: {type(encoder.pm_field).__name__}")
    
    # Define test pairs
    similar_pairs = [
        ("hello", "hi"),
        ("how are you", "how are you doing"),
        ("what's the weather", "what's the climate"),
    ]
    
    dissimilar_pairs = [
        ("hello", "goodbye"),
        ("what's the weather", "favorite movie"),
    ]
    
    # Measure initial similarities
    print("\n2. Initial semantic similarities (PM-ONLY):")
    print("-" * 80)
    
    print("SIMILAR pairs (should pull together):")
    initial_similar = []
    for p1, p2 in similar_pairs:
        sim = compute_similarity(encoder, p1, p2)
        initial_similar.append(sim)
        print(f"  '{p1}' <-> '{p2}': {sim:.3f}")
    
    print("\nDISSIMILAR pairs (should push apart):")
    initial_dissimilar = []
    for p1, p2 in dissimilar_pairs:
        sim = compute_similarity(encoder, p1, p2)
        initial_dissimilar.append(sim)
        print(f"  '{p1}' <-> '{p2}': {sim:.3f}")
    
    # Encode pairs for training
    print("\n3. Encoding pairs for training...")
    
    def encode_pair(phrase1, phrase2):
        """Encode to latent space (what PMField sees)."""
        tokens1 = phrase1.lower().split()
        tokens2 = phrase2.lower().split()
        base1 = encoder.base_encoder.encode(tokens1).to(encoder.device)
        base2 = encoder.base_encoder.encode(tokens2).to(encoder.device)
        z1 = base1 @ encoder._projection
        z2 = base2 @ encoder._projection
        return (z1, z2)
    
    similar_emb = [encode_pair(p1, p2) for p1, p2 in similar_pairs]
    dissimilar_emb = [encode_pair(p1, p2) for p1, p2 in dissimilar_pairs]
    
    print(f"  {len(similar_emb)} similar pairs encoded")
    print(f"  {len(dissimilar_emb)} dissimilar pairs encoded")
    
    # Train with contrastive plasticity
    print("\n4. Training with contrastive plasticity...")
    pm_field = encoder.pm_field.fine_field
    
    num_epochs = 200
    c_lr = 0.1  # Higher learning rate
    margin = 1.5
    
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {c_lr}")
    print(f"  Margin: {margin}")
    print("-" * 80)
    
    initial_centers = pm_field.centers.clone().detach()
    
    for epoch in range(num_epochs):
        contrastive_plasticity(
            pm_field,
            similar_pairs=similar_emb,
            dissimilar_pairs=dissimilar_emb,
            c_lr=c_lr,
            margin=margin
        )
        
        if (epoch + 1) % 20 == 0:
            param_change = torch.mean(torch.abs(pm_field.centers - initial_centers)).item()
            print(f"  Epoch {epoch + 1:3d}: Param Δ={param_change:.6f}")
    
    # Measure final similarities
    print("\n5. Final semantic similarities (PM-ONLY):")
    print("-" * 80)
    
    print("SIMILAR pairs (expect INCREASE ↑):")
    final_similar = []
    for i, (p1, p2) in enumerate(similar_pairs):
        sim = compute_similarity(encoder, p1, p2)
        final_similar.append(sim)
        delta = sim - initial_similar[i]
        direction = "↑" if delta > 0 else "↓"
        print(f"  '{p1}' <-> '{p2}': {sim:.3f} (was {initial_similar[i]:.3f}, Δ {delta:+.3f} {direction})")
    
    print("\nDISSIMILAR pairs (expect DECREASE ↓):")
    final_dissimilar = []
    for i, (p1, p2) in enumerate(dissimilar_pairs):
        sim = compute_similarity(encoder, p1, p2)
        final_dissimilar.append(sim)
        delta = sim - initial_dissimilar[i]
        direction = "↑" if delta > 0 else "↓"
        print(f"  '{p1}' <-> '{p2}': {sim:.3f} (was {initial_dissimilar[i]:.3f}, Δ {delta:+.3f} {direction})")
    
    # Results
    print("\n6. Results Summary:")
    print("=" * 80)
    
    similar_improved = sum(1 for i in range(len(similar_pairs)) if final_similar[i] > initial_similar[i])
    dissimilar_improved = sum(1 for i in range(len(dissimilar_pairs)) if final_dissimilar[i] < initial_dissimilar[i])
    
    print(f"  Similar pairs improved: {similar_improved}/{len(similar_pairs)}")
    print(f"  Dissimilar pairs improved: {dissimilar_improved}/{len(dissimilar_pairs)}")
    
    avg_similar_change = np.mean([final_similar[i] - initial_similar[i] for i in range(len(similar_pairs))])
    avg_dissimilar_change = np.mean([final_dissimilar[i] - initial_dissimilar[i] for i in range(len(dissimilar_pairs))])
    
    print(f"  Avg similar change: {avg_similar_change:+.4f}")
    print(f"  Avg dissimilar change: {avg_dissimilar_change:+.4f}")
    
    success = similar_improved == len(similar_pairs) and dissimilar_improved == len(dissimilar_pairs)
    
    if success:
        print("\n✓ SUCCESS: BNN learned semantic structure!")
        print("  Similar phrases pulled together, dissimilar pushed apart")
        print("  This proves the 'open book exam' concept works")
    elif similar_improved > 0 or dissimilar_improved > 0:
        print("\n⚠ PARTIAL: Some learning occurred but not complete")
        print("  May need more epochs or tuning")
    else:
        print("\n✗ FAILED: No learning detected")
        print("  Base embeddings may still be dominating")


if __name__ == "__main__":
    main()
