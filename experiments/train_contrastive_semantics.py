#!/usr/bin/env python3
"""
Train BNN semantic space using contrastive plasticity.

Philosophy: BNN learns semantic STRUCTURE (how concepts relate),
not just indexing. This enables effective querying and pattern adaptation.

Approach:
1. Define similar pairs (should be close in semantic space)
2. Define dissimilar pairs (should be far apart)
3. Use contrastive_plasticity to organize the space
4. Test if retrieval improves
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments/retrieval_sanity'))

import torch
import numpy as np
from pathlib import Path
from pmflow_bnn_enhanced.pmflow import contrastive_plasticity

# Import Lilith components
from lilith.database_fragment_store import DatabaseBackedFragmentStore


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


def train_semantic_space():
    """Train BNN to organize semantic space using contrastive learning."""
    
    print("=" * 80)
    print("CONTRASTIVE SEMANTIC SPACE TRAINING")
    print("=" * 80)
    
    # Initialize BNN encoder directly
    print("\n1. Loading BNN encoder...")
    from pipeline.embedding import PMFlowEmbeddingEncoder
    
    encoder = PMFlowEmbeddingEncoder(
        dimension=96,
        latent_dim=64,
        combine_mode="concat",
        seed=13
    )
    print(f"  Encoder initialized: {type(encoder).__name__}")
    print(f"  PM Field type: {type(encoder.pm_field).__name__}")
    
    # Define similar pairs (should be semantically close)
    similar_pairs = [
        # Greetings
        ("hello", "hi"),
        ("hello", "hey"),
        ("hi", "hey"),
        ("how are you", "how are you doing"),
        ("how are you", "how's it going"),
        ("how are you doing", "how's it going"),
        
        # Weather/Climate
        ("what's the weather", "what's the climate"),
        ("what's the weather like", "how's the weather"),
        ("is it cold", "is it chilly"),
        
        # Movies/Films
        ("do you like movies", "do you enjoy films"),
        ("favorite movie", "favorite film"),
        
        # Questions about name
        ("what's your name", "what are you called"),
        ("who are you", "what's your name"),
        
        # Goodbyes
        ("goodbye", "bye"),
        ("see you later", "talk to you later"),
    ]
    
    # Define dissimilar pairs (should be far apart)
    dissimilar_pairs = [
        # Greeting vs Goodbye
        ("hello", "goodbye"),
        ("hi", "bye"),
        ("how are you", "see you later"),
        
        # Weather vs Movies
        ("what's the weather", "favorite movie"),
        ("is it cold", "do you like films"),
        
        # Name vs Weather
        ("what's your name", "what's the weather"),
        ("who are you", "is it cold"),
        
        # Greetings vs Other topics
        ("hello", "what's the weather"),
        ("how are you", "favorite movie"),
        ("hi", "what's your name"),
    ]
    
    print(f"\nSimilar pairs: {len(similar_pairs)}")
    print(f"Dissimilar pairs: {len(dissimilar_pairs)}")
    
    # Measure initial similarities
    print("\n2. Initial semantic similarities:")
    print("-" * 80)
    
    sample_tests = [
        ("hello", "hi", "SIMILAR"),
        ("how are you", "how are you doing", "SIMILAR"),
        ("what's the weather", "what's the climate", "SIMILAR"),
        ("hello", "goodbye", "DISSIMILAR"),
        ("what's the weather", "favorite movie", "DISSIMILAR"),
    ]
    
    initial_sims = []
    for p1, p2, expected in sample_tests:
        sim = compute_similarity(encoder, p1, p2)
        initial_sims.append(sim)
        print(f"  '{p1}' <-> '{p2}': {sim:.3f} ({expected})")
    
    # Convert phrase pairs to embeddings
    print("\n3. Encoding phrase pairs as embeddings...")
    
    def encode_pair(phrase1, phrase2):
        """Encode a pair of phrases as embedding tensors - extract PMFlow latent part only."""
        tokens1 = phrase1.lower().split()
        tokens2 = phrase2.lower().split()
        
        # Get base embeddings and project to latent space (what feeds into pm_field)
        base_emb1 = encoder.base_encoder.encode(tokens1).to(encoder.device)
        base_emb2 = encoder.base_encoder.encode(tokens2).to(encoder.device)
        
        # Project to latent space (this is what goes into pm_field)
        z1 = base_emb1 @ encoder._projection
        z2 = base_emb2 @ encoder._projection
        
        return (z1, z2)
    
    similar_emb_pairs = [encode_pair(p1, p2) for p1, p2 in similar_pairs]
    dissimilar_emb_pairs = [encode_pair(p1, p2) for p1, p2 in dissimilar_pairs]
    
    print(f"  Encoded {len(similar_emb_pairs)} similar pairs")
    print(f"  Encoded {len(dissimilar_emb_pairs)} dissimilar pairs")
    print(f"  Latent embedding shape: {similar_emb_pairs[0][0].shape}")
    
    # Access the MultiScalePMField
    print("\n4. Accessing MultiScalePMField for training...")
    
    # The encoder has: pm_field attribute which is a MultiScalePMField
    # We need to train its fine_field sub-component
    pm_field = encoder.pm_field.fine_field
    
    print(f"  PMField type: {type(pm_field)}")
    print(f"  Centers shape: {pm_field.centers.shape}")
    print(f"  Mus shape: {pm_field.mus.shape}")
    
    # Training hyperparameters
    num_epochs = 100
    c_lr = 0.05  # Increased from 0.01 - higher learning rate
    margin = 1.5  # Margin for dissimilar pairs
    
    print(f"\n5. Training with contrastive plasticity:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {c_lr}")
    print(f"  Margin: {margin}")
    print("-" * 80)
    
    # Track parameter changes
    initial_centers = pm_field.centers.clone().detach()
    
    # Training loop
    for epoch in range(num_epochs):
        # Apply contrastive plasticity
        contrastive_plasticity(
            pm_field,
            similar_pairs=similar_emb_pairs,
            dissimilar_pairs=dissimilar_emb_pairs,
            c_lr=c_lr,
            margin=margin
        )
        
        # Report progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Measure parameter change
            param_change = torch.mean(torch.abs(pm_field.centers - initial_centers)).item()
            
            # Compute average similarity change
            current_sims = []
            for p1, p2, _ in sample_tests:
                sim = compute_similarity(encoder, p1, p2)
                current_sims.append(sim)
            
            avg_sim_change = np.mean([abs(c - i) for c, i in zip(current_sims, initial_sims)])
            print(f"  Epoch {epoch + 1:3d}: Param Δ={param_change:.6f}, Similarity Δ={avg_sim_change:.6f}")
    
    # Measure final similarities
    print("\n6. Final semantic similarities:")
    print("-" * 80)
    
    for i, (p1, p2, expected) in enumerate(sample_tests):
        sim = compute_similarity(encoder, p1, p2)
        initial = initial_sims[i]
        delta = sim - initial
        direction = "↑" if delta > 0 else "↓"
        print(f"  '{p1}' <-> '{p2}': {sim:.3f} (was {initial:.3f}, Δ {delta:+.3f} {direction}) [{expected}]")
    
    # Expected outcomes
    print("\n7. Training Results Summary:")
    print("-" * 80)
    
    similar_improved = 0
    dissimilar_improved = 0
    
    for i, (p1, p2, expected) in enumerate(sample_tests):
        sim = compute_similarity(encoder, p1, p2)
        initial = initial_sims[i]
        delta = sim - initial
        
        if expected == "SIMILAR" and delta > 0:
            similar_improved += 1
        elif expected == "DISSIMILAR" and delta < 0:
            dissimilar_improved += 1
    
    similar_total = sum(1 for _, _, e in sample_tests if e == "SIMILAR")
    dissimilar_total = sum(1 for _, _, e in sample_tests if e == "DISSIMILAR")
    
    print(f"  Similar pairs improved: {similar_improved}/{similar_total}")
    print(f"  Dissimilar pairs improved: {dissimilar_improved}/{dissimilar_total}")
    
    success = similar_improved == similar_total and dissimilar_improved == dissimilar_total
    
    if success:
        print("\n✓ SUCCESS: BNN learned semantic structure!")
        print("  - Similar phrases pulled together")
        print("  - Dissimilar phrases pushed apart")
        print("\nNext: Test if this improves retrieval quality")
    else:
        print("\n⚠ PARTIAL: BNN learning but not all pairs improved")
        print(f"  - May need more epochs or different hyperparameters")
        print(f"  - Learning rate: {c_lr}, Margin: {margin}")
    
    print("=" * 80)
    
    # Save the trained encoder using its built-in save method
    print("\n8. Saving trained encoder...")
    encoder.save_state(Path("trained_semantic_encoder.pt"))
    print("  Saved to: trained_semantic_encoder.pt")
    print("\nNow you can test retrieval with the organized semantic space!")
    
    return encoder


if __name__ == "__main__":
    train_semantic_space()
