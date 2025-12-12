"""Simple contrastive training script using PMFlow Enhanced directly.

Trains PMFlow fields using contrastive learning without complex pipeline dependencies.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import silhouette_score
from pathlib import Path
from typing import List, Tuple

# Import PMFlow Enhanced and base encoder
from pmflow import (
    PMField,
    MultiScalePMField,
    vectorized_pm_plasticity,
    contrastive_plasticity,
    batch_plasticity_update,
    hybrid_similarity,
)

# Import training corpus
import sys
sys.path.insert(0, str(Path(__file__).parent))
from training_corpus import get_corpus, get_concept_labels, get_test_queries


class SimpleSemanticEncoder:
    """Simplified semantic encoder using MultiScalePMField."""
    
    def __init__(self, vocab_size=1000, embedding_dim=64, seed=42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.seed = seed
        
        # Word to index mapping
        self.word_to_idx = {}
        self.next_idx = 0
        
        # Random projection matrix (deterministic)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.projection = torch.randn(vocab_size, embedding_dim) * 0.1
        
        # PMFlow field (MultiScale for hierarchical concepts)
        self.pm_field = MultiScalePMField(
            d_latent=embedding_dim,
            n_centers_fine=128,
            n_centers_coarse=32,
            steps_fine=5,
            steps_coarse=3,
            dt=0.15,
            beta=1.2,
            clamp=3.0,
        )
        self.pm_field.eval()
    
    def _get_word_index(self, word: str) -> int:
        """Get or create index for word."""
        if word not in self.word_to_idx:
            if self.next_idx >= self.vocab_size:
                # Hash to existing vocab if full
                return hash(word) % self.vocab_size
            self.word_to_idx[word] = self.next_idx
            self.next_idx += 1
        return self.word_to_idx[word]
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to embedding."""
        with torch.no_grad():
            # Tokenize
            words = text.lower().split()
            
            # Get word embeddings
            indices = [self._get_word_index(w) for w in words]
            word_vecs = self.projection[indices]
            
            # Average pooling
            doc_vec = torch.mean(word_vecs, dim=0, keepdim=True)
            
            # PMFlow transformation
            fine_emb, coarse_emb, combined = self.pm_field(doc_vec)
            
            # Return combined multi-scale embedding
            return combined.squeeze()


def generate_contrastive_pairs(corpus: List[str], labels: dict, max_pairs: int = 200):
    """Generate contrastive pairs based on category labels."""
    similar_pairs = []
    dissimilar_pairs = []
    
    n = len(corpus)
    pairs_generated = 0
    
    # Generate similar pairs (same category)
    for i in range(n):
        if pairs_generated >= max_pairs // 2:
            break
        for j in range(i + 1, n):
            if labels.get(i) == labels.get(j):
                similar_pairs.append((i, j))
                pairs_generated += 1
                if pairs_generated >= max_pairs // 2:
                    break
    
    # Generate dissimilar pairs (different categories)
    pairs_generated = 0
    for i in range(n):
        if pairs_generated >= max_pairs // 2:
            break
        for j in range(i + 1, n):
            if labels.get(i) != labels.get(j):
                dissimilar_pairs.append((i, j))
                pairs_generated += 1
                if pairs_generated >= max_pairs // 2:
                    break
    
    return similar_pairs, dissimilar_pairs


def evaluate_clustering(encoder: SimpleSemanticEncoder, corpus: List[str], labels: dict) -> float:
    """Compute silhouette score for clustering quality."""
    embeddings = []
    label_list = []
    
    for idx, text in enumerate(corpus):
        emb = encoder.encode(text)
        embeddings.append(emb.numpy())
        label_list.append(labels.get(idx, "unknown"))
    
    embeddings_array = np.array(embeddings)
    
    if len(set(label_list)) > 1:
        return silhouette_score(embeddings_array, label_list)
    return 0.0


def evaluate_retrieval(encoder: SimpleSemanticEncoder, corpus: List[str], test_queries: List[Tuple], k: int = 3) -> float:
    """Compute mean precision@k for retrieval."""
    # Get corpus embeddings
    corpus_embeddings = [encoder.encode(text) for text in corpus]
    
    precisions = []
    
    for query_text, category, relevant_docs in test_queries:
        query_emb = encoder.encode(query_text)
        
        # Compute cosine similarities
        similarities = []
        for doc_emb in corpus_embeddings:
            sim = F.cosine_similarity(query_emb.unsqueeze(0), doc_emb.unsqueeze(0))
            similarities.append(float(sim.item()))
        
        # Get top-k
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        # Compute precision
        relevant_retrieved = sum(1 for idx in top_k_indices if idx in relevant_docs)
        precisions.append(relevant_retrieved / k)
    
    return float(np.mean(precisions))


def train_epoch(
    encoder: SimpleSemanticEncoder,
    corpus: List[str],
    labels: dict,
    mu_lr: float = 3e-4,
    c_lr: float = 3e-4,
    max_pairs: int = 200,
):
    """Train one epoch using contrastive plasticity."""
    
    # Snapshot centers before training
    centers_before = encoder.pm_field.fine_field.centers.clone()
    
    # Generate pairs
    similar_pairs_idx, dissimilar_pairs_idx = generate_contrastive_pairs(corpus, labels, max_pairs)
    
    print(f"  Generated {len(similar_pairs_idx)} similar + {len(dissimilar_pairs_idx)} dissimilar pairs")
    
    # Convert to embeddings - use the FINE field output, not combined
    similar_pairs_emb = []
    for i, j in similar_pairs_idx:
        # Get the fine embedding (first output from MultiScalePMField)
        with torch.no_grad():
            words_i = corpus[i].lower().split()
            words_j = corpus[j].lower().split()
            
            indices_i = [encoder._get_word_index(w) for w in words_i]
            indices_j = [encoder._get_word_index(w) for w in words_j]
            
            word_vecs_i = encoder.projection[indices_i]
            word_vecs_j = encoder.projection[indices_j]
            
            doc_vec_i = torch.mean(word_vecs_i, dim=0, keepdim=True)
            doc_vec_j = torch.mean(word_vecs_j, dim=0, keepdim=True)
            
            fine_emb_i, _, _ = encoder.pm_field(doc_vec_i)
            fine_emb_j, _, _ = encoder.pm_field(doc_vec_j)
            
            similar_pairs_emb.append((fine_emb_i, fine_emb_j))
    
    dissimilar_pairs_emb = []
    for i, j in dissimilar_pairs_idx:
        with torch.no_grad():
            words_i = corpus[i].lower().split()
            words_j = corpus[j].lower().split()
            
            indices_i = [encoder._get_word_index(w) for w in words_i]
            indices_j = [encoder._get_word_index(w) for w in words_j]
            
            word_vecs_i = encoder.projection[indices_i]
            word_vecs_j = encoder.projection[indices_j]
            
            doc_vec_i = torch.mean(word_vecs_i, dim=0, keepdim=True)
            doc_vec_j = torch.mean(word_vecs_j, dim=0, keepdim=True)
            
            fine_emb_i, _, _ = encoder.pm_field(doc_vec_i)
            fine_emb_j, _, _ = encoder.pm_field(doc_vec_j)
            
            dissimilar_pairs_emb.append((fine_emb_i, fine_emb_j))
    
    # Apply contrastive plasticity to fine field
    contrastive_plasticity(
        pmfield=encoder.pm_field.fine_field,
        similar_pairs=similar_pairs_emb,
        dissimilar_pairs=dissimilar_pairs_emb,
        mu_lr=mu_lr,
        c_lr=c_lr,
        margin=1.0,
    )
    
    # Batch update for general plasticity - also use fine embeddings
    all_embeddings = []
    for text in corpus:
        with torch.no_grad():
            words = text.lower().split()
            indices = [encoder._get_word_index(w) for w in words]
            word_vecs = encoder.projection[indices]
            doc_vec = torch.mean(word_vecs, dim=0, keepdim=True)
            fine_emb, _, _ = encoder.pm_field(doc_vec)
            all_embeddings.append(fine_emb)
    
    batch_plasticity_update(
        pmfield=encoder.pm_field.fine_field,
        examples=all_embeddings,
        mu_lr=mu_lr * 0.5,
        c_lr=c_lr * 0.5,
        batch_size=32,
    )
    
    # Check if centers changed
    centers_after = encoder.pm_field.fine_field.centers
    center_change = torch.norm(centers_after - centers_before).item()
    print(f"  Center change magnitude: {center_change:.6f}")


def main():
    print("=" * 80)
    print("CONTRASTIVE TRAINING - STANDALONE")
    print("=" * 80)
    
    # Load data
    corpus = get_corpus()
    labels = get_concept_labels()
    test_queries = get_test_queries()
    
    print(f"\nCorpus: {len(corpus)} documents")
    print(f"Categories: {len(set(labels.values()))}")
    print(f"Test queries: {len(test_queries)}")
    
    # Initialize encoder
    print("\nInitializing MultiScalePMField encoder...")
    encoder = SimpleSemanticEncoder(vocab_size=1000, embedding_dim=64, seed=42)
    
    # Baseline evaluation
    print("\n" + "=" * 80)
    print("BASELINE (before training)")
    print("=" * 80)
    
    baseline_silhouette = evaluate_clustering(encoder, corpus, labels)
    baseline_precision = evaluate_retrieval(encoder, corpus, test_queries, k=3)
    
    print(f"Silhouette score: {baseline_silhouette:.4f}")
    print(f"Mean P@3: {baseline_precision:.4f}")
    
    # Training
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    n_epochs = 20  # More epochs
    mu_lr = 1e-3  # Higher learning rate
    c_lr = 1e-3   # Higher learning rate
    
    print(f"Epochs: {n_epochs}")
    print(f"Learning rates: mu={mu_lr:.1e}, c={c_lr:.1e}")
    
    best_silhouette = baseline_silhouette
    
    for epoch in range(n_epochs):
        print(f"\n--- Epoch {epoch + 1}/{n_epochs} ---")
        
        train_epoch(encoder, corpus, labels, mu_lr=mu_lr, c_lr=c_lr, max_pairs=200)
        
        # Evaluate
        silhouette = evaluate_clustering(encoder, corpus, labels)
        precision = evaluate_retrieval(encoder, corpus, test_queries, k=3)
        
        improvement = ((silhouette - baseline_silhouette) / baseline_silhouette * 100) if baseline_silhouette > 0 else 0
        
        print(f"  Silhouette: {silhouette:.4f} (baseline: {baseline_silhouette:.4f}, {improvement:+.1f}%)")
        print(f"  P@3: {precision:.4f} (baseline: {baseline_precision:.4f})")
        
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            print(f"  ✓ New best!")
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    final_silhouette = evaluate_clustering(encoder, corpus, labels)
    final_precision = evaluate_retrieval(encoder, corpus, test_queries, k=3)
    
    print(f"\nClustering:")
    print(f"  Baseline silhouette: {baseline_silhouette:.4f}")
    print(f"  Final silhouette: {final_silhouette:.4f}")
    print(f"  Improvement: {((final_silhouette - baseline_silhouette) / baseline_silhouette * 100):+.1f}%")
    
    print(f"\nRetrieval:")
    print(f"  Baseline P@3: {baseline_precision:.4f}")
    print(f"  Final P@3: {final_precision:.4f}")
    print(f"  Improvement: {((final_precision - baseline_precision) / baseline_precision * 100) if baseline_precision > 0 else 0:+.1f}%")
    
    # Check if targets met
    print("\n" + "=" * 80)
    print("TARGET VALIDATION")
    print("=" * 80)
    
    target_silhouette_min = 0.30
    target_silhouette_max = 0.45
    target_precision = 0.80
    
    print(f"\nSilhouette target: {target_silhouette_min:.2f}-{target_silhouette_max:.2f}")
    if target_silhouette_min <= final_silhouette <= target_silhouette_max:
        print(f"  ✅ ACHIEVED: {final_silhouette:.4f}")
    elif final_silhouette > target_silhouette_max:
        print(f"  ✅ EXCEEDED: {final_silhouette:.4f}")
    else:
        print(f"  ⚠️  Not yet: {final_silhouette:.4f}")
    
    print(f"\nP@3 target: {target_precision:.2f}")
    if final_precision >= target_precision:
        print(f"  ✅ ACHIEVED: {final_precision:.4f}")
    else:
        print(f"  ⚠️  Not yet: {final_precision:.4f} (need {target_precision - final_precision:.2f} more)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
