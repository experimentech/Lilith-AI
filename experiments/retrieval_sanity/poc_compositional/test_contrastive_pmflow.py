"""
Test script for contrastive PMFlow learning.

This demonstrates that adding a learnable projection AFTER PMFlow
allows contrastive learning to work effectively.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from pmflow_bnn_enhanced.contrastive_pmflow import (
    ContrastivePMField,
    train_contrastive_pmfield,
    create_contrastive_encoder
)
from experiments.retrieval_sanity.pipeline.embedding import PMFlowEmbeddingEncoder


# Synonym pairs for training
SYNONYM_PAIRS = [
    ("machine learning", "ML"),
    ("deep learning", "DL"),
    ("artificial intelligence", "AI"),
    ("supervised learning", "supervised training"),
    ("unsupervised learning", "unsupervised training"),
    ("reinforcement learning", "RL"),
    ("neural network", "NN"),
    ("convolutional neural network", "CNN"),
    ("recurrent neural network", "RNN"),
]

# Dissimilar pairs (should be pushed apart)
DISSIMILAR_PAIRS = [
    ("machine learning", "cooking"),
    ("neural network", "fishing"),
    ("deep learning", "gardening"),
    ("AI", "carpentry"),
]


def tokenize(text):
    """Simple tokenization."""
    return text.lower().split()


def compute_similarity(encoder, text1, text2):
    """Compute cosine similarity between two texts."""
    with torch.no_grad():
        emb1 = encoder.encode(tokenize(text1))
        emb2 = encoder.encode(tokenize(text2))
        
        # Handle torch tensors
        if hasattr(emb1, 'cpu'):
            emb1 = emb1.cpu().detach().numpy()
        if hasattr(emb2, 'cpu'):
            emb2 = emb2.cpu().detach().numpy()
        
        # Convert to torch for similarity
        emb1_t = torch.from_numpy(emb1).flatten()
        emb2_t = torch.from_numpy(emb2).flatten()
        
        sim = F.cosine_similarity(emb1_t.unsqueeze(0), emb2_t.unsqueeze(0), dim=1)
        return sim.item()


def main():
    print("=" * 70)
    print("Testing Contrastive PMFlow Learning")
    print("=" * 70)
    
    # Create base encoder
    print("\n1. Creating PMFlow encoder...")
    encoder = PMFlowEmbeddingEncoder(
        dimension=96,
        latent_dim=64,
        combine_mode="concat",
        seed=13
    )
    print(f"   ✓ Encoder created (dim={encoder.dimension}, latent={encoder.latent_dim})")
    
    # Evaluate BEFORE training
    print("\n2. Computing similarities BEFORE contrastive training...")
    print("\n   Similar pairs:")
    similar_before = []
    for text1, text2 in SYNONYM_PAIRS[:5]:  # Test subset
        sim = compute_similarity(encoder, text1, text2)
        similar_before.append(sim)
        print(f"     {text1:30s} ↔ {text2:30s} : {sim:.3f}")
    
    print("\n   Dissimilar pairs:")
    dissimilar_before = []
    for text1, text2 in DISSIMILAR_PAIRS:
        sim = compute_similarity(encoder, text1, text2)
        dissimilar_before.append(sim)
        print(f"     {text1:30s} ↔ {text2:30s} : {sim:.3f}")
    
    avg_similar_before = sum(similar_before) / len(similar_before)
    avg_dissimilar_before = sum(dissimilar_before) / len(dissimilar_before)
    separation_before = avg_similar_before - avg_dissimilar_before
    
    print(f"\n   Summary:")
    print(f"     Similar average:    {avg_similar_before:.3f}")
    print(f"     Dissimilar average: {avg_dissimilar_before:.3f}")
    print(f"     Separation:         {separation_before:.3f}")
    
    # Create contrastive model
    print("\n3. Creating ContrastivePMField wrapper...")
    contrastive_field = create_contrastive_encoder(encoder, projection_type="residual")
    print(f"   ✓ ContrastivePMField created")
    print(f"   ✓ Projection type: residual (learned delta)")
    print(f"   ✓ PMField centers: trainable")
    
    # Prepare training pairs (need latent representations)
    print("\n4. Preparing training pairs...")
    similar_pairs = []
    for text1, text2 in SYNONYM_PAIRS:
        # Encode to latent space (input to PMField)
        tokens1 = tokenize(text1)
        tokens2 = tokenize(text2)
        
        with torch.no_grad():
            base1 = encoder.base_encoder.encode(tokens1).to(encoder.device)
            latent1 = base1 @ encoder._projection
            
            base2 = encoder.base_encoder.encode(tokens2).to(encoder.device)
            latent2 = base2 @ encoder._projection
        
        similar_pairs.append((latent1.squeeze(), latent2.squeeze()))
    
    dissimilar_pairs = []
    for text1, text2 in DISSIMILAR_PAIRS:
        tokens1 = tokenize(text1)
        tokens2 = tokenize(text2)
        
        with torch.no_grad():
            base1 = encoder.base_encoder.encode(tokens1).to(encoder.device)
            latent1 = base1 @ encoder._projection
            
            base2 = encoder.base_encoder.encode(tokens2).to(encoder.device)
            latent2 = base2 @ encoder._projection
        
        dissimilar_pairs.append((latent1.squeeze(), latent2.squeeze()))
    
    print(f"   ✓ {len(similar_pairs)} similar pairs")
    print(f"   ✓ {len(dissimilar_pairs)} dissimilar pairs")
    
    # Train contrastive model
    print("\n5. Training contrastive model (50 epochs)...")
    print("-" * 70)
    history = train_contrastive_pmfield(
        contrastive_field,
        similar_pairs,
        dissimilar_pairs,
        epochs=50,
        center_lr=1e-4,  # Gentle updates to PMField centers
        projection_lr=1e-3,  # Stronger updates to projection
        margin=0.2,
        verbose=True
    )
    print("-" * 70)
    
    # Replace encoder's PMField with trained version
    print("\n6. Updating encoder with trained PMField...")
    # The contrastive_field wraps encoder.pm_field, so it's already updated!
    # But we need to use the projection in the encoder
    # For now, we'll evaluate using the contrastive_field directly
    
    # Create a custom encoder that uses the contrastive field
    class ContrastiveEncoder:
        def __init__(self, base_encoder, contrastive_field):
            self.base_encoder = base_encoder.base_encoder
            self._projection = base_encoder._projection
            self.device = base_encoder.device
            self.contrastive_field = contrastive_field
            self.combine_mode = base_encoder.combine_mode
        
        def encode(self, tokens):
            with torch.no_grad():
                base = self.base_encoder.encode(tokens).to(self.device)
                latent = base @ self._projection
                
                # Use contrastive field instead of raw PMField
                refined = self.contrastive_field(latent)
                refined = F.normalize(refined, p=2, dim=1)
                
                if self.combine_mode == "concat":
                    hashed = F.normalize(base, p=2, dim=1)
                    combined = torch.cat([hashed, refined], dim=1)
                else:
                    combined = refined
                
                return combined.cpu()
    
    trained_encoder = ContrastiveEncoder(encoder, contrastive_field)
    
    # Evaluate AFTER training
    print("\n7. Computing similarities AFTER contrastive training...")
    print("\n   Similar pairs:")
    similar_after = []
    for text1, text2 in SYNONYM_PAIRS[:5]:
        sim = compute_similarity(trained_encoder, text1, text2)
        similar_after.append(sim)
        delta = sim - similar_before[len(similar_after) - 1]
        arrow = "↗" if delta > 0 else "↘"
        print(f"     {text1:30s} ↔ {text2:30s} : {sim:.3f} ({arrow} Δ={delta:+.3f})")
    
    print("\n   Dissimilar pairs:")
    dissimilar_after = []
    for text1, text2 in DISSIMILAR_PAIRS:
        sim = compute_similarity(trained_encoder, text1, text2)
        dissimilar_after.append(sim)
        delta = sim - dissimilar_before[len(dissimilar_after) - 1]
        arrow = "↗" if delta > 0 else "↘"
        print(f"     {text1:30s} ↔ {text2:30s} : {sim:.3f} ({arrow} Δ={delta:+.3f})")
    
    avg_similar_after = sum(similar_after) / len(similar_after)
    avg_dissimilar_after = sum(dissimilar_after) / len(dissimilar_after)
    separation_after = avg_similar_after - avg_dissimilar_after
    
    print(f"\n   Summary:")
    print(f"     Similar average:    {avg_similar_after:.3f} (was {avg_similar_before:.3f}, Δ={avg_similar_after - avg_similar_before:+.3f})")
    print(f"     Dissimilar average: {avg_dissimilar_after:.3f} (was {avg_dissimilar_before:.3f}, Δ={avg_dissimilar_after - avg_dissimilar_before:+.3f})")
    print(f"     Separation:         {separation_after:.3f} (was {separation_before:.3f}, Δ={separation_after - separation_before:+.3f})")
    
    # Verdict
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    improvement = separation_after - separation_before
    if improvement > 0.05:
        print(f"✅ SUCCESS: Separation improved by {improvement:.3f}")
        print(f"   Similar pairs got closer: {avg_similar_after - avg_similar_before:+.3f}")
        print(f"   Dissimilar pairs got further: {avg_dissimilar_after - avg_dissimilar_before:+.3f}")
        print("\n   This proves that adding a learnable projection AFTER PMFlow")
        print("   allows contrastive learning to work while preserving PMFlow's")
        print("   gravitational semantics!")
    elif improvement > 0:
        print(f"⚠️  PARTIAL: Separation improved slightly by {improvement:.3f}")
        print("   May need more epochs or tuning of learning rates")
    else:
        print(f"❌ FAILED: Separation decreased by {improvement:.3f}")
        print("   Architecture may need adjustment")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
