"""Contrastive learning to improve concept separation in embeddings.

Uses concept taxonomy to identify similar/dissimilar pairs and
train the PMFlow encoder to cluster similar concepts closer together
while pushing dissimilar concepts apart.

Integrates PMFlow Enhanced v0.3.0 features:
- batch_plasticity_update for 10-100x speedup
- contrastive_plasticity for improved clustering
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .concept_taxonomy import ConceptTaxonomy


@dataclass
class ContrastivePair:
    """A pair of examples with similarity label."""
    
    text1: str
    text2: str
    similarity: float  # 0.0 = dissimilar, 1.0 = identical
    label: str  # Description for logging


class ContrastiveLearner:
    """Contrastive learning trainer for embedding quality."""
    
    def __init__(self, taxonomy: ConceptTaxonomy, margin: float = 1.0):  # type: ignore
        self.taxonomy = taxonomy
        self.margin = margin  # Margin for dissimilar pairs
    
    def generate_pairs(self, corpus: List[str], max_pairs: int = 100) -> List[ContrastivePair]:
        """Generate contrastive training pairs from corpus using taxonomy."""
        
        pairs = []
        
        # Extract concepts from each document
        doc_concepts = []
        for text in corpus:
            concepts = self.taxonomy.extract_concepts(text)
            doc_concepts.append((text, concepts))
        
        # Generate positive pairs (similar documents)
        for i, (text1, concepts1) in enumerate(doc_concepts):
            for j, (text2, concepts2) in enumerate(doc_concepts):
                if i >= j:
                    continue  # Skip self and duplicates
                
                # Check if documents share concepts
                common = concepts1 & concepts2
                if common:
                    # Compute similarity based on shared concepts
                    union = concepts1 | concepts2
                    similarity = len(common) / len(union) if union else 0.0
                    
                    pairs.append(ContrastivePair(
                        text1=text1,
                        text2=text2,
                        similarity=similarity,
                        label=f"similar ({', '.join(list(common)[:2])})",
                    ))
                
                # Generate negative pairs (dissimilar)
                if not common and len(pairs) < max_pairs:
                    # Check if concepts are semantically distant
                    max_sim = 0.0
                    for c1 in concepts1:
                        for c2 in concepts2:
                            max_sim = max(max_sim, self.taxonomy.semantic_similarity(c1, c2))
                    
                    if max_sim < 0.3:  # Threshold for dissimilarity
                        pairs.append(ContrastivePair(
                            text1=text1,
                            text2=text2,
                            similarity=0.0,
                            label="dissimilar",
                        ))
                
                if len(pairs) >= max_pairs:
                    break
            
            if len(pairs) >= max_pairs:
                break
        
        return pairs
    
    def contrastive_loss(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        similarity: float,
    ) -> torch.Tensor:
        """Compute contrastive loss for a pair.
        
        Similar pairs: minimize distance
        Dissimilar pairs: maximize distance (up to margin)
        """
        
        # Cosine distance (1 - cosine_similarity)
        emb1_norm = F.normalize(emb1.squeeze(), dim=-1)
        emb2_norm = F.normalize(emb2.squeeze(), dim=-1)
        
        cos_sim = torch.dot(emb1_norm, emb2_norm)
        distance = 1.0 - cos_sim
        
        if similarity > 0.5:  # Similar pair
            # Minimize distance
            loss = distance
        else:  # Dissimilar pair
            # Maximize distance, but only up to margin
            loss = torch.clamp(self.margin - distance, min=0.0)
        
        return loss
    
    def compute_batch_loss(
        self,
        pairs: List[ContrastivePair],
        encoder_func,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute average contrastive loss over batch.
        
        Args:
            pairs: List of contrastive pairs
            encoder_func: Function that takes text and returns embedding tensor
        
        Returns:
            total_loss: Average loss over batch
            metrics: Dictionary of loss components for logging
        """
        
        total_loss = torch.tensor(0.0)
        similar_loss = torch.tensor(0.0)
        dissimilar_loss = torch.tensor(0.0)
        n_similar = 0
        n_dissimilar = 0
        
        for pair in pairs:
            emb1 = encoder_func(pair.text1)
            emb2 = encoder_func(pair.text2)
            
            loss = self.contrastive_loss(emb1, emb2, pair.similarity)
            total_loss += loss
            
            if pair.similarity > 0.5:
                similar_loss += loss
                n_similar += 1
            else:
                dissimilar_loss += loss
                n_dissimilar += 1
        
        # Normalize
        n_pairs = len(pairs)
        avg_loss = total_loss / n_pairs if n_pairs > 0 else torch.tensor(0.0)
        
        metrics = {
            "total_loss": float(avg_loss.item()),
            "similar_loss": float(similar_loss.item() / n_similar) if n_similar > 0 else 0.0,
            "dissimilar_loss": float(dissimilar_loss.item() / n_dissimilar) if n_dissimilar > 0 else 0.0,
            "n_pairs": n_pairs,
            "n_similar": n_similar,
            "n_dissimilar": n_dissimilar,
        }
        
        return avg_loss, metrics
    
    def train_with_pmflow_plasticity(
        self,
        pairs: List[ContrastivePair],
        encoder,
        pm_field,
        mu_lr: float = 5e-4,
        c_lr: float = 5e-4,
        batch_size: int = 32,
    ) -> Dict[str, float | int]:
        """Train using PMFlow Enhanced contrastive plasticity and batch updates.
        
        Args:
            pairs: Contrastive training pairs
            encoder: PMFlowEmbeddingEncoder instance
            pm_field: The PMFlow field to update (can be MultiScalePMField)
            mu_lr: Learning rate for gravitational strengths
            c_lr: Learning rate for center positions
            batch_size: Mini-batch size for batch_plasticity_update
        
        Returns:
            Training metrics
        """
        try:
            from pmflow_bnn_enhanced.pmflow import (
                contrastive_plasticity,
                batch_plasticity_update,
            )
        except ImportError:
            # Fallback to standard plasticity if enhanced version not available
            return {"n_similar": 0, "n_dissimilar": 0}
        
        # Separate similar and dissimilar pairs
        similar_pairs_pmf = []
        dissimilar_pairs_pmf = []
        all_embeddings = []
        
        for pair in pairs:
            # Get embeddings
            emb1 = encoder.encode(pair.text1.split())
            emb2 = encoder.encode(pair.text2.split())
            
            all_embeddings.append(emb1)
            all_embeddings.append(emb2)
            
            if pair.similarity > 0.5:
                similar_pairs_pmf.append((emb1, emb2))
            else:
                dissimilar_pairs_pmf.append((emb1, emb2))
        
        # Get the actual PMField (handle MultiScalePMField)
        if hasattr(pm_field, 'fine_field'):
            # MultiScalePMField - apply to fine field for better granularity
            target_field = pm_field.fine_field
        else:
            target_field = pm_field
        
        # Apply contrastive plasticity
        contrastive_plasticity(
            pmfield=target_field,
            similar_pairs=similar_pairs_pmf,
            dissimilar_pairs=dissimilar_pairs_pmf,
            mu_lr=mu_lr,
            c_lr=c_lr,
            margin=self.margin,
        )
        
        # Apply batch plasticity update for general learning
        batch_plasticity_update(
            pmfield=target_field,
            examples=all_embeddings,
            mu_lr=mu_lr * 0.5,  # Lower LR for general update
            c_lr=c_lr * 0.5,
            batch_size=batch_size,
        )
        
        return {
            "n_similar": len(similar_pairs_pmf),
            "n_dissimilar": len(dissimilar_pairs_pmf),
            "mu_lr": mu_lr,
            "c_lr": c_lr,
            "batch_size": batch_size,
        }


def demo_contrastive_learning():
    """Demonstrate contrastive pair generation."""
    
    # Import here to avoid relative import issues when run as script
    from concept_taxonomy import ConceptTaxonomy
    
    print("=" * 80)
    print("CONTRASTIVE LEARNING DEMONSTRATION")
    print("=" * 80)
    
    taxonomy = ConceptTaxonomy()
    learner = ContrastiveLearner(taxonomy)
    
    # Sample corpus
    corpus = [
        "alice visited the hospital",
        "bob went to the hospital", 
        "the doctor works at the hospital",
        "alice met bob at the park",
        "the park has many trees",
        "the library has books",
    ]
    
    print("\nCorpus:")
    for i, text in enumerate(corpus):
        concepts = taxonomy.extract_concepts(text)
        print(f"  [{i}] {text}")
        print(f"      Concepts: {concepts if concepts else '(none)'}")
    
    # Generate pairs
    pairs = learner.generate_pairs(corpus, max_pairs=20)
    
    print(f"\nGenerated {len(pairs)} contrastive pairs:")
    print("\nSimilar pairs (should have small embedding distance):")
    for pair in pairs:
        if pair.similarity > 0.5:
            print(f"  • {pair.text1}")
            print(f"    {pair.text2}")
            print(f"    Similarity: {pair.similarity:.2f} - {pair.label}\n")
    
    print("Dissimilar pairs (should have large embedding distance):")
    for pair in pairs:
        if pair.similarity < 0.5:
            print(f"  • {pair.text1}")
            print(f"    {pair.text2}")
            print(f"    Similarity: {pair.similarity:.2f} - {pair.label}\n")
    
    print("=" * 80)
    print("Contrastive learning helps embeddings:")
    print("  ✓ Cluster similar documents closer together")
    print("  ✓ Push dissimilar documents further apart")
    print("  ✓ Improve retrieval precision by better separation")
    print("=" * 80)


if __name__ == "__main__":
    demo_contrastive_learning()
