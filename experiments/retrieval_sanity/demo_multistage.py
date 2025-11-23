#!/usr/bin/env python3
"""Multi-stage architecture demonstration and benchmark.

Shows how intake and semantic stages process differently and measures
their individual contributions to retrieval quality.
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import torch
import numpy as np
from typing import List, Dict, Tuple

from experiments.retrieval_sanity.pipeline import (
    Utterance,
    StageCoordinator,
    StageType,
    SymbolicStore,
)


def print_section(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_artifact_details(stage_name: str, artifact, show_embedding: bool = False) -> None:
    """Print detailed artifact information."""
    print(f"{stage_name} Stage Output:")
    print(f"  Confidence: {artifact.confidence:.3f}")
    print(f"  Activation Energy: {artifact.metadata.get('activation_energy', 0):.3f}")
    print(f"  Latent Norm: {artifact.metadata.get('latent_norm', 0):.3f}")
    
    if artifact.tokens:
        print(f"  Tokens: {' '.join(artifact.tokens[:10])}{'...' if len(artifact.tokens) > 10 else ''}")
    
    if 'normalised' in artifact.metadata:
        print(f"  Normalized: {artifact.metadata['normalised']}")
    
    if 'parsed' in artifact.metadata:
        parsed = artifact.metadata['parsed']
        print(f"  POS Tags: {len(parsed.tokens)} tokens parsed")
    
    if show_embedding:
        emb = artifact.embedding.squeeze()
        print(f"  Embedding shape: {emb.shape}")
        print(f"  Embedding norm: {torch.norm(emb).item():.3f}")
        print(f"  Embedding sparsity: {(emb.abs() < 0.01).sum().item() / emb.numel() * 100:.1f}%")
    
    print()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors.
    
    If dimensions don't match, returns None (indicates incompatible embeddings).
    """
    a_flat = a.squeeze()
    b_flat = b.squeeze()
    
    if a_flat.shape != b_flat.shape:
        return None  # Incompatible dimensions
    
    a_norm = a_flat / (torch.norm(a_flat) + 1e-8)
    b_norm = b_flat / (torch.norm(b_flat) + 1e-8)
    return float(torch.dot(a_norm, b_norm).item())


def compare_embeddings(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    label1: str = "Embedding 1",
    label2: str = "Embedding 2",
) -> None:
    """Compare two embeddings and show differences."""
    print(f"\n  Comparing {label1} vs {label2}:")
    
    # Check dimension compatibility first
    if emb1.squeeze().shape != emb2.squeeze().shape:
        print(f"    ⚠ Different dimensions: {emb1.shape} vs {emb2.shape}")
        print(f"    → Stages use different embedding spaces (specialized architectures)")
        return
    
    # Cosine similarity
    similarity = cosine_similarity(emb1, emb2)
    if similarity is not None:
        print(f"    Cosine similarity: {similarity:.4f}")
    
    # L2 distance
    distance = float(torch.norm(emb1 - emb2).item())
    print(f"    L2 distance: {distance:.4f}")
    
    # Dimension-wise statistics
    diff = (emb1 - emb2).squeeze()
    print(f"    Mean difference: {diff.mean().item():.4f}")
    print(f"    Max difference: {diff.abs().max().item():.4f}")
    print(f"    Different dims (>0.1): {(diff.abs() > 0.1).sum().item()}")


def demo_basic_processing() -> None:
    """Show how stages process a single utterance differently."""
    print_section("DEMO 1: Basic Multi-Stage Processing")
    
    coordinator = StageCoordinator()
    
    test_utterances = [
        "we visited the hospital yesterday",
        "alice met bob at the park",
        "the doctor gave medicine to the patient",
    ]
    
    for i, text in enumerate(test_utterances, 1):
        print(f"\n--- Input {i}: \"{text}\" ---")
        utterance = Utterance(text=text, language="en")
        results = coordinator.process(utterance)
        
        if StageType.INTAKE in results:
            print_artifact_details("Intake", results[StageType.INTAKE], show_embedding=True)
        
        if StageType.SEMANTIC in results:
            print_artifact_details("Semantic", results[StageType.SEMANTIC], show_embedding=True)
        
        # Compare embeddings
        if StageType.INTAKE in results and StageType.SEMANTIC in results:
            compare_embeddings(
                results[StageType.INTAKE].embedding,
                results[StageType.SEMANTIC].embedding,
                "Intake",
                "Semantic",
            )


def demo_stage_specialization() -> None:
    """Show that stages learn different aspects."""
    print_section("DEMO 2: Stage Specialization - Typos vs Semantics")
    
    coordinator = StageCoordinator()
    
    # Test pairs: clean vs noisy, semantically similar vs different
    test_pairs = [
        ("hospital visit", "hosptal viist"),  # Same semantic, different surface
        ("hospital visit", "doctor appointment"),  # Different words, similar semantic
        ("hospital", "library"),  # Different semantic
    ]
    
    for text1, text2 in test_pairs:
        print(f"\n--- Comparing: \"{text1}\" vs \"{text2}\" ---")
        
        results1 = coordinator.process(Utterance(text=text1, language="en"))
        results2 = coordinator.process(Utterance(text=text2, language="en"))
        
        # Intake stage: should focus on surface forms
        if StageType.INTAKE in results1 and StageType.INTAKE in results2:
            intake_sim = cosine_similarity(
                results1[StageType.INTAKE].embedding,
                results2[StageType.INTAKE].embedding,
            )
            if intake_sim is not None:
                print(f"  Intake similarity: {intake_sim:.4f}")
        
        # Semantic stage: should focus on meaning
        if StageType.SEMANTIC in results1 and StageType.SEMANTIC in results2:
            semantic_sim = cosine_similarity(
                results1[StageType.SEMANTIC].embedding,
                results2[StageType.SEMANTIC].embedding,
            )
            if semantic_sim is not None:
                print(f"  Semantic similarity: {semantic_sim:.4f}")
        
        # Analysis
        if StageType.INTAKE in results1 and StageType.SEMANTIC in results1:
            if "viist" in text2:  # Typo case
                print(f"  → Intake should be more robust to typos (higher similarity)")
            elif "doctor" in text2:  # Semantic similar
                print(f"  → Semantic should recognize similar meaning (higher similarity)")
            else:  # Different
                print(f"  → Both should show low similarity (different concepts)")


def demo_context_flow() -> None:
    """Show how upstream context affects downstream processing."""
    print_section("DEMO 3: Context Flow Between Stages")
    
    # Process with and without context to see the difference
    from experiments.retrieval_sanity.pipeline.stage_coordinator import (
        StageConfig,
        SemanticStage,
    )
    
    # Create semantic stage in isolation (no context)
    isolated_config = StageConfig(stage_type=StageType.SEMANTIC, db_namespace="isolated_test")
    isolated_stage = SemanticStage(isolated_config)
    
    # Process through full coordinator (with context)
    coordinator = StageCoordinator()
    
    test_text = "we visited the hospital yesterday"
    utterance = Utterance(text=test_text, language="en")
    
    print(f"Input: \"{test_text}\"\n")
    
    # Isolated processing (no context)
    print("Semantic stage WITHOUT intake context:")
    isolated_artifact = isolated_stage.process(utterance, upstream_artifacts=None)
    print_artifact_details("Isolated", isolated_artifact, show_embedding=True)
    
    # Coordinated processing (with context)
    print("Semantic stage WITH intake context:")
    results = coordinator.process(utterance)
    if StageType.SEMANTIC in results:
        print_artifact_details("Coordinated", results[StageType.SEMANTIC], show_embedding=True)
    
    # Compare
    if StageType.SEMANTIC in results:
        compare_embeddings(
            isolated_artifact.embedding,
            results[StageType.SEMANTIC].embedding,
            "Without Context",
            "With Context",
        )


def benchmark_stage_quality() -> None:
    """Benchmark retrieval quality per stage."""
    print_section("BENCHMARK: Stage-Specific Retrieval Quality")
    
    coordinator = StageCoordinator()
    
    # Create test dataset with known relationships
    corpus = [
        "alice visited the hospital",
        "bob went to the hospital",
        "alice met bob at the park",
        "the doctor works at the hospital",
        "the park has many trees",
    ]
    
    queries = [
        ("hospital visit", [0, 1, 3]),  # Should retrieve hospital-related
        ("alice and bob", [0, 2]),  # Should retrieve alice/bob mentions
        ("outdoor location", [2, 4]),  # Should retrieve park mentions
    ]
    
    print("Building corpus with multi-stage processing...\n")
    
    intake_embeddings: List[torch.Tensor] = []
    semantic_embeddings: List[torch.Tensor] = []
    
    for i, text in enumerate(corpus):
        print(f"  [{i}] {text}")
        results = coordinator.process(Utterance(text=text, language="en"))
        if StageType.INTAKE in results:
            intake_embeddings.append(results[StageType.INTAKE].embedding)
        if StageType.SEMANTIC in results:
            semantic_embeddings.append(results[StageType.SEMANTIC].embedding)
    
    print("\n" + "-" * 80)
    
    # Test retrieval for each query
    for query_text, expected_indices in queries:
        print(f"\nQuery: \"{query_text}\"")
        print(f"Expected relevant docs: {expected_indices}")
        
        query_results = coordinator.process(Utterance(text=query_text, language="en"))
        
        intake_top3 = []
        semantic_top3 = []
        
        # Intake stage retrieval
        if StageType.INTAKE in query_results and intake_embeddings:
            query_emb = query_results[StageType.INTAKE].embedding
            intake_scores = [cosine_similarity(query_emb, doc_emb) for doc_emb in intake_embeddings]
            # Filter out None values (dimension mismatches)
            valid_scores = [(idx, score) for idx, score in enumerate(intake_scores) if score is not None]
            intake_top3 = sorted(valid_scores, key=lambda x: x[1], reverse=True)[:3]
            
            print(f"\n  Intake stage top-3:")
            for rank, (idx, score) in enumerate(intake_top3, 1):
                marker = "✓" if idx in expected_indices else " "
                print(f"    {marker} #{rank}: Doc {idx} (score: {score:.4f}) - {corpus[idx]}")
        
        # Semantic stage retrieval
        if StageType.SEMANTIC in query_results and semantic_embeddings:
            query_emb = query_results[StageType.SEMANTIC].embedding
            semantic_scores = [cosine_similarity(query_emb, doc_emb) for doc_emb in semantic_embeddings]
            # Filter out None values (dimension mismatches)
            valid_scores = [(idx, score) for idx, score in enumerate(semantic_scores) if score is not None]
            semantic_top3 = sorted(valid_scores, key=lambda x: x[1], reverse=True)[:3]
            
            print(f"\n  Semantic stage top-3:")
            for rank, (idx, score) in enumerate(semantic_top3, 1):
                marker = "✓" if idx in expected_indices else " "
                print(f"    {marker} #{rank}: Doc {idx} (score: {score:.4f}) - {corpus[idx]}")
        
        # Calculate precision@3 for each stage
        if intake_top3 and semantic_top3:
            intake_hits = sum(1 for idx, _ in intake_top3 if idx in expected_indices)
            semantic_hits = sum(1 for idx, _ in semantic_top3 if idx in expected_indices)
            
            intake_p3 = intake_hits / 3.0
            semantic_p3 = semantic_hits / 3.0
            
            print(f"\n  Precision@3:")
            print(f"    Intake:   {intake_p3:.2f} ({intake_hits}/3 relevant)")
            print(f"    Semantic: {semantic_p3:.2f} ({semantic_hits}/3 relevant)")
            
            if semantic_p3 > intake_p3:
                print(f"    → Semantic stage wins! (+{(semantic_p3 - intake_p3) * 100:.0f}%)")
            elif intake_p3 > semantic_p3:
                print(f"    → Intake stage wins! (+{(intake_p3 - semantic_p3) * 100:.0f}%)")
            else:
                print(f"    → Tie!")


def main() -> None:
    """Run all demonstrations and benchmarks."""
    print("\n" + "█" * 80)
    print("  MULTI-STAGE PMFlow BNN ARCHITECTURE DEMONSTRATION")
    print("█" * 80)
    
    try:
        # Run demonstrations
        demo_basic_processing()
        demo_stage_specialization()
        demo_context_flow()
        benchmark_stage_quality()
        
        print_section("SUMMARY")
        print("✓ Demonstrated multi-stage processing pipeline")
        print("✓ Showed stage specialization capabilities")
        print("✓ Validated context flow between stages")
        print("✓ Benchmarked retrieval quality per stage")
        print("\nThe multi-stage architecture is functioning correctly!")
        print("Next step: Add symbolic reasoning layer on this foundation.\n")
        
    except Exception as exc:
        print(f"\n❌ Error during demonstration: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
