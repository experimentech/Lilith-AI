#!/usr/bin/env python3
"""Visualize stage embeddings to understand specialization.

Uses PCA to project embeddings into 2D space and show how
intake vs semantic stages cluster inputs differently.
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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import List, Tuple

from experiments.retrieval_sanity.pipeline import (
    Utterance,
    StageCoordinator,
    StageType,
)


def visualize_stage_embeddings() -> None:
    """Visualize how different stages embed the same inputs."""
    
    # Test corpus with clear semantic groupings
    corpus = [
        # Medical/hospital cluster
        "alice visited the hospital",
        "bob went to the hospital", 
        "the doctor works at the hospital",
        "the patient received treatment",
        
        # Park/outdoor cluster
        "alice met bob at the park",
        "the park has many trees",
        "children play in the park",
        "the outdoor garden is beautiful",
        
        # Generic/other
        "the library has books",
        "students study in the classroom",
    ]
    
    labels = [
        "hospital", "hospital", "hospital", "medical",
        "park", "park", "park", "outdoor",
        "library", "classroom",
    ]
    
    coordinator = StageCoordinator()
    
    intake_embeddings = []
    semantic_embeddings = []
    
    print("Processing corpus through multi-stage pipeline...\n")
    
    for text in corpus:
        results = coordinator.process(Utterance(text=text, language="en"))
        
        if StageType.INTAKE in results:
            intake_embeddings.append(results[StageType.INTAKE].embedding.squeeze().numpy())
        
        if StageType.SEMANTIC in results:
            semantic_embeddings.append(results[StageType.SEMANTIC].embedding.squeeze().numpy())
    
    # Convert to numpy arrays
    intake_matrix = np.array(intake_embeddings)
    semantic_matrix = np.array(semantic_embeddings)
    
    print(f"Intake embeddings: {intake_matrix.shape}")
    print(f"Semantic embeddings: {semantic_matrix.shape}\n")
    
    # Apply PCA to reduce to 2D
    print("Applying PCA to reduce to 2D...")
    pca_intake = PCA(n_components=2)
    pca_semantic = PCA(n_components=2)
    
    intake_2d = pca_intake.fit_transform(intake_matrix)
    semantic_2d = pca_semantic.fit_transform(semantic_matrix)
    
    print(f"Intake variance explained: {pca_intake.explained_variance_ratio_.sum():.2%}")
    print(f"Semantic variance explained: {pca_semantic.explained_variance_ratio_.sum():.2%}\n")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color map for labels
    colors = {
        "hospital": "red",
        "medical": "pink",
        "park": "green",
        "outdoor": "lightgreen",
        "library": "blue",
        "classroom": "purple",
    }
    
    # Plot intake stage
    for i, (x, y) in enumerate(intake_2d):
        color = colors.get(labels[i], "gray")
        ax1.scatter(x, y, c=color, s=100, alpha=0.6, edgecolors='black')
        ax1.annotate(f"{i}", (x, y), fontsize=8, ha='center', va='center')
    
    ax1.set_title("Intake Stage Embeddings (PCA 2D)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.grid(alpha=0.3)
    
    # Plot semantic stage
    for i, (x, y) in enumerate(semantic_2d):
        color = colors.get(labels[i], "gray")
        ax2.scatter(x, y, c=color, s=100, alpha=0.6, edgecolors='black')
        ax2.annotate(f"{i}", (x, y), fontsize=8, ha='center', va='center')
    
    ax2.set_title("Semantic Stage Embeddings (PCA 2D)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.grid(alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=label) for label, color in colors.items()]
    fig.legend(handles=legend_elements, loc='upper center', ncol=6, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    output_path = THIS_DIR / "stage_embeddings_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    
    # Also print corpus with indices for reference
    print("\n" + "=" * 80)
    print("CORPUS REFERENCE:")
    print("=" * 80)
    for i, (text, label) in enumerate(zip(corpus, labels)):
        print(f"[{i}] ({label:10s}) {text}")
    print("\n")


def analyze_clustering_quality() -> None:
    """Compute clustering metrics to quantify stage specialization."""
    from sklearn.metrics import silhouette_score
    
    corpus = [
        "alice visited the hospital",
        "bob went to the hospital", 
        "the doctor works at the hospital",
        "the patient received treatment",
        "alice met bob at the park",
        "the park has many trees",
        "children play in the park",
        "the outdoor garden is beautiful",
        "the library has books",
        "students study in the classroom",
    ]
    
    # Ground truth labels (0=medical, 1=park, 2=other)
    true_labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    
    coordinator = StageCoordinator()
    
    intake_embeddings = []
    semantic_embeddings = []
    
    for text in corpus:
        results = coordinator.process(Utterance(text=text, language="en"))
        
        if StageType.INTAKE in results:
            intake_embeddings.append(results[StageType.INTAKE].embedding.squeeze().numpy())
        
        if StageType.SEMANTIC in results:
            semantic_embeddings.append(results[StageType.SEMANTIC].embedding.squeeze().numpy())
    
    intake_matrix = np.array(intake_embeddings)
    semantic_matrix = np.array(semantic_embeddings)
    
    # Compute silhouette scores (how well clusters are separated)
    intake_score = silhouette_score(intake_matrix, true_labels)
    semantic_score = silhouette_score(semantic_matrix, true_labels)
    
    print("=" * 80)
    print("CLUSTERING QUALITY METRICS")
    print("=" * 80)
    print(f"\nSilhouette Score (higher = better clustering):")
    print(f"  Intake stage:   {intake_score:.4f}")
    print(f"  Semantic stage: {semantic_score:.4f}")
    
    if semantic_score > intake_score:
        improvement = ((semantic_score - intake_score) / abs(intake_score)) * 100
        print(f"\n  → Semantic stage clusters {improvement:.1f}% better!")
    elif intake_score > semantic_score:
        improvement = ((intake_score - semantic_score) / abs(semantic_score)) * 100
        print(f"\n  → Intake stage clusters {improvement:.1f}% better!")
    else:
        print(f"\n  → Equal clustering quality")
    
    print("\nNote: Score range is [-1, 1]")
    print("  +1 = Perfect clustering")
    print("   0 = Overlapping clusters")
    print("  -1 = Wrong clusters\n")


def main() -> None:
    """Run visualization and analysis."""
    try:
        visualize_stage_embeddings()
        analyze_clustering_quality()
        
        print("=" * 80)
        print("✓ Visualization complete!")
        print("  Check 'stage_embeddings_visualization.png' to see how stages")
        print("  cluster the same inputs in different embedding spaces.")
        print("=" * 80 + "\n")
        
    except ImportError as exc:
        print(f"❌ Missing dependency: {exc}")
        print("\nInstall visualization dependencies:")
        print("  pip install matplotlib scikit-learn")
        sys.exit(1)
    except Exception as exc:
        print(f"❌ Error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
