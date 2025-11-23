"""Contrastive training for semantic stage using PMFlow Enhanced.

Trains the semantic stage PMFlow field using:
- Contrastive plasticity (cluster similar, separate dissimilar)
- Batch plasticity updates (10-100x faster)
- Concept taxonomy for pair generation
- MultiScalePMField for hierarchical learning
"""

import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import silhouette_score
from typing import List, Tuple
import importlib.util

# Load modules directly from file paths to avoid package import issues
parent_dir = Path(__file__).parent

def load_module_from_path(module_name, file_path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load required modules
stage_coord = load_module_from_path("stage_coordinator", parent_dir / "pipeline" / "stage_coordinator.py")
concept_tax = load_module_from_path("concept_taxonomy", parent_dir / "pipeline" / "concept_taxonomy.py")
contrast_learn = load_module_from_path("contrastive_learning", parent_dir / "pipeline" / "contrastive_learning.py")
training_corpus = load_module_from_path("training_corpus", parent_dir / "training_corpus.py")

StageCoordinator = stage_coord.StageCoordinator
StageType = stage_coord.StageType
ConceptTaxonomy = concept_tax.ConceptTaxonomy
ContrastiveLearner = contrast_learn.ContrastiveLearner


def evaluate_clustering(coordinator: StageCoordinator, corpus: List[str], labels: dict) -> dict:
    """Evaluate clustering quality using silhouette score."""
    
    stage = coordinator.get_stage(StageType.SEMANTIC)
    if stage is None:
        return {"error": "Semantic stage not found"}
    
    # Get embeddings for all documents
    embeddings = []
    label_list = []
    
    for idx, text in enumerate(corpus):
        artifact = stage.process(text, upstream_artifacts=None)
        embeddings.append(artifact.embedding.squeeze().numpy())
        label_list.append(labels.get(idx, "unknown"))
    
    embeddings_array = np.array(embeddings)
    
    # Compute silhouette score
    if len(set(label_list)) > 1:
        silhouette = silhouette_score(embeddings_array, label_list)
    else:
        silhouette = 0.0
    
    return {
        "silhouette": silhouette,
        "n_samples": len(embeddings),
        "n_categories": len(set(label_list)),
        "embedding_dim": embeddings_array.shape[1],
    }


def evaluate_retrieval(coordinator: StageCoordinator, corpus: List[str], test_queries: List[Tuple], k: int = 3) -> dict:
    """Evaluate retrieval quality using test queries."""
    
    stage = coordinator.get_stage(StageType.SEMANTIC)
    if stage is None:
        return {"error": "Semantic stage not found"}
    
    # Get corpus embeddings
    corpus_embeddings = []
    for text in corpus:
        artifact = stage.process(text, upstream_artifacts=None)
        corpus_embeddings.append(artifact.embedding)
    
    # Evaluate each query
    precisions = []
    
    for query_text, category, relevant_docs in test_queries:
        # Get query embedding
        query_artifact = stage.process(query_text, upstream_artifacts=None)
        query_emb = query_artifact.embedding
        
        # Compute similarities
        similarities = []
        for doc_emb in corpus_embeddings:
            # Cosine similarity
            query_norm = torch.nn.functional.normalize(query_emb.squeeze(), dim=0)
            doc_norm = torch.nn.functional.normalize(doc_emb.squeeze(), dim=0)
            sim = float(torch.dot(query_norm, doc_norm).item())
            similarities.append(sim)
        
        # Get top-k
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        # Compute precision@k
        relevant_retrieved = sum(1 for idx in top_k_indices if idx in relevant_docs)
        precision = relevant_retrieved / k
        precisions.append(precision)
    
    return {
        "mean_precision_at_k": float(np.mean(precisions)),
        "std_precision_at_k": float(np.std(precisions)),
        "n_queries": len(test_queries),
        "k": k,
    }


def train_epoch(
    coordinator: StageCoordinator,
    corpus: List[str],
    taxonomy: ConceptTaxonomy,
    learner: ContrastiveLearner,
    mu_lr: float = 5e-4,
    c_lr: float = 5e-4,
    batch_size: int = 32,
    max_pairs: int = 200,
) -> dict:
    """Train one epoch of contrastive learning."""
    
    stage = coordinator.get_stage(StageType.SEMANTIC)
    if stage is None:
        return {"error": "Semantic stage not found"}
    
    # Generate contrastive pairs
    print(f"  Generating contrastive pairs (max={max_pairs})...")
    pairs = learner.generate_pairs(corpus, max_pairs=max_pairs)
    print(f"  Generated {len(pairs)} pairs")
    
    # Train using PMFlow Enhanced plasticity
    print(f"  Training with contrastive plasticity...")
    metrics = learner.train_with_pmflow_plasticity(
        pairs=pairs,
        encoder=stage.encoder,
        pm_field=stage.encoder.pm_field,
        mu_lr=mu_lr,
        c_lr=c_lr,
        batch_size=batch_size,
    )
    
    return metrics


def main():
    """Main training loop."""
    
    print("=" * 80)
    print("CONTRASTIVE TRAINING FOR SEMANTIC STAGE")
    print("=" * 80)
    
    # Load corpus
    corpus = training_corpus.get_corpus()
    labels = training_corpus.get_concept_labels()
    test_queries = training_corpus.get_test_queries()
    
    print(f"\nCorpus: {len(corpus)} documents across {len(set(labels.values()))} categories")
    print(f"Test queries: {len(test_queries)}")
    
    # Initialize coordinator with semantic stage
    print("\nInitializing stage coordinator...")
    coordinator = StageCoordinator()
    
    # Initialize taxonomy and learner
    print("Initializing concept taxonomy...")
    taxonomy = ConceptTaxonomy()
    learner = ContrastiveLearner(taxonomy, margin=1.0)
    
    # Baseline evaluation
    print("\n" + "=" * 80)
    print("BASELINE EVALUATION (before training)")
    print("=" * 80)
    
    print("\nClustering quality:")
    clustering_metrics = evaluate_clustering(coordinator, corpus, labels)
    print(f"  Silhouette score: {clustering_metrics['silhouette']:.4f}")
    print(f"  Samples: {clustering_metrics['n_samples']}")
    print(f"  Categories: {clustering_metrics['n_categories']}")
    
    print("\nRetrieval quality:")
    retrieval_metrics = evaluate_retrieval(coordinator, corpus, test_queries, k=3)
    print(f"  Mean P@3: {retrieval_metrics['mean_precision_at_k']:.4f}")
    print(f"  Std P@3: {retrieval_metrics['std_precision_at_k']:.4f}")
    
    # Training hyperparameters
    n_epochs = 5
    mu_lr = 3e-4
    c_lr = 3e-4
    batch_size = 32
    max_pairs = 200
    
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    print(f"Epochs: {n_epochs}")
    print(f"Learning rates: mu={mu_lr:.1e}, c={c_lr:.1e}")
    print(f"Batch size: {batch_size}")
    print(f"Max pairs per epoch: {max_pairs}")
    
    # Training loop
    best_silhouette = clustering_metrics['silhouette']
    
    for epoch in range(n_epochs):
        print(f"\n--- Epoch {epoch + 1}/{n_epochs} ---")
        
        # Train one epoch
        train_metrics = train_epoch(
            coordinator=coordinator,
            corpus=corpus,
            taxonomy=taxonomy,
            learner=learner,
            mu_lr=mu_lr,
            c_lr=c_lr,
            batch_size=batch_size,
            max_pairs=max_pairs,
        )
        
        print(f"  Similar pairs: {train_metrics.get('n_similar', 0)}")
        print(f"  Dissimilar pairs: {train_metrics.get('n_dissimilar', 0)}")
        
        # Evaluate
        print("  Evaluating...")
        clustering_metrics = evaluate_clustering(coordinator, corpus, labels)
        retrieval_metrics = evaluate_retrieval(coordinator, corpus, test_queries, k=3)
        
        print(f"  Silhouette: {clustering_metrics['silhouette']:.4f}")
        print(f"  P@3: {retrieval_metrics['mean_precision_at_k']:.4f}")
        
        # Save if improved
        if clustering_metrics['silhouette'] > best_silhouette:
            best_silhouette = clustering_metrics['silhouette']
            print(f"  ✓ New best silhouette! Saving checkpoint...")
            coordinator.save_all_states()
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION (after training)")
    print("=" * 80)
    
    print("\nClustering quality:")
    print(f"  Silhouette score: {clustering_metrics['silhouette']:.4f}")
    
    print("\nRetrieval quality:")
    print(f"  Mean P@3: {retrieval_metrics['mean_precision_at_k']:.4f} ± {retrieval_metrics['std_precision_at_k']:.4f}")
    
    # Query-by-query results
    print("\nPer-query results:")
    stage = coordinator.get_stage(StageType.SEMANTIC)
    corpus_embeddings = [stage.process(text, upstream_artifacts=None).embedding for text in corpus]
    
    for query_text, category, relevant_docs in test_queries[:6]:  # Show first 6
        query_artifact = stage.process(query_text, upstream_artifacts=None)
        query_emb = query_artifact.embedding
        
        similarities = []
        for doc_emb in corpus_embeddings:
            query_norm = torch.nn.functional.normalize(query_emb.squeeze(), dim=0)
            doc_norm = torch.nn.functional.normalize(doc_emb.squeeze(), dim=0)
            sim = float(torch.dot(query_norm, doc_norm).item())
            similarities.append(sim)
        
        top_3 = np.argsort(similarities)[::-1][:3]
        relevant_retrieved = sum(1 for idx in top_3 if idx in relevant_docs)
        
        print(f"  {query_text:25} P@3={relevant_retrieved}/3  top=[{top_3[0]}, {top_3[1]}, {top_3[2]}]")
    
    print("\n" + "=" * 80)
    print("Training complete! PMFlow states saved.")
    print("=" * 80)


if __name__ == "__main__":
    main()
