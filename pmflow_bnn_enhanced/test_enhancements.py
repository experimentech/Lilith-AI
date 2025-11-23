#!/usr/bin/env python3
"""Quick test of v0.3.0 enhanced features."""

import torch
from pmflow import (
    ParallelPMField,
    MultiScalePMField,
    AttentionGatedPMField,
    EnergyBasedPMField,
    contrastive_plasticity,
    batch_plasticity_update,
    hybrid_similarity,
)


def test_multiscale():
    """Test MultiScalePMField."""
    print("Testing MultiScalePMField...")
    ms_field = MultiScalePMField(d_latent=16, n_centers_fine=32, n_centers_coarse=8)
    
    z = torch.randn(4, 16)
    fine, coarse, combined = ms_field(z)
    
    assert fine.shape == (4, 16), f"Fine shape mismatch: {fine.shape}"
    assert coarse.shape == (4, 8), f"Coarse shape mismatch: {coarse.shape}"
    assert combined.shape == (4, 24), f"Combined shape mismatch: {combined.shape}"
    print("  ✓ MultiScalePMField works correctly")


def test_attention_gated():
    """Test AttentionGatedPMField."""
    print("\nTesting AttentionGatedPMField...")
    gated_field = AttentionGatedPMField(d_latent=16, n_centers=32, attention_mode='gradient')
    
    z = torch.randn(4, 16)
    context = torch.randn(4, 16)
    
    # Without context
    output1, attention1 = gated_field(z)
    assert output1.shape == (4, 16), f"Output shape mismatch: {output1.shape}"
    assert attention1.shape == (4, 1), f"Attention shape mismatch: {attention1.shape}"
    
    # With context
    output2, attention2 = gated_field(z, context=context)
    assert output2.shape == (4, 16), f"Output shape mismatch: {output2.shape}"
    assert attention2.shape == (4, 1), f"Attention shape mismatch: {attention2.shape}"
    
    print("  ✓ AttentionGatedPMField works correctly")


def test_energy_based():
    """Test EnergyBasedPMField."""
    print("\nTesting EnergyBasedPMField...")
    energy_field = EnergyBasedPMField(d_latent=16, n_centers=32)
    
    z1 = torch.randn(4, 16)
    z2 = torch.randn(4, 16)
    
    # Compute energy
    energy = energy_field.compute_energy(z1)
    assert energy.shape == (4,), f"Energy shape mismatch: {energy.shape}"
    
    # Energy similarity
    sim = energy_field.energy_similarity(z1, z2)
    assert sim.shape == (4,), f"Similarity shape mismatch: {sim.shape}"
    assert (sim >= 0).all() and (sim <= 1).all(), "Similarity not in [0,1]"
    
    print("  ✓ EnergyBasedPMField works correctly")


def test_contrastive_plasticity():
    """Test contrastive_plasticity."""
    print("\nTesting contrastive_plasticity...")
    pmfield = ParallelPMField(d_latent=16, n_centers=32)
    
    # Create similar pairs
    similar_pairs = [
        (torch.randn(1, 16), torch.randn(1, 16)),
        (torch.randn(1, 16), torch.randn(1, 16)),
    ]
    
    # Create dissimilar pairs
    dissimilar_pairs = [
        (torch.randn(1, 16), torch.randn(1, 16)),
        (torch.randn(1, 16), torch.randn(1, 16)),
    ]
    
    # Update
    centers_before = pmfield.centers.clone()
    contrastive_plasticity(pmfield, similar_pairs, dissimilar_pairs, mu_lr=1e-3, c_lr=1e-3)
    centers_after = pmfield.centers
    
    # Check that centers changed
    assert not torch.allclose(centers_before, centers_after), "Centers didn't update"
    print("  ✓ contrastive_plasticity works correctly")


def test_batch_plasticity():
    """Test batch_plasticity_update."""
    print("\nTesting batch_plasticity_update...")
    pmfield = ParallelPMField(d_latent=16, n_centers=32)
    
    examples = [torch.randn(1, 16) for _ in range(10)]
    
    centers_before = pmfield.centers.clone()
    batch_plasticity_update(pmfield, examples, mu_lr=5e-4, c_lr=5e-4, batch_size=4)
    centers_after = pmfield.centers
    
    # Check that centers changed
    assert not torch.allclose(centers_before, centers_after), "Centers didn't update"
    print("  ✓ batch_plasticity_update works correctly")


def test_hybrid_similarity():
    """Test hybrid_similarity."""
    print("\nTesting hybrid_similarity...")
    energy_field = EnergyBasedPMField(d_latent=16, n_centers=32)
    
    query = torch.randn(4, 16)
    doc = torch.randn(4, 16)
    
    sim = hybrid_similarity(query, doc, energy_field, cosine_weight=0.7, energy_weight=0.3)
    
    assert sim.shape == (4,), f"Similarity shape mismatch: {sim.shape}"
    print("  ✓ hybrid_similarity works correctly")


if __name__ == "__main__":
    print("=" * 70)
    print("PMFlow BNN Enhanced v0.3.0 - Feature Tests")
    print("=" * 70)
    
    test_multiscale()
    test_attention_gated()
    test_energy_based()
    test_contrastive_plasticity()
    test_batch_plasticity()
    test_hybrid_similarity()
    
    print("\n" + "=" * 70)
    print("✅ All enhanced features working correctly!")
    print("=" * 70)
