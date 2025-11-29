"""
Contrastive Learning for Semantic Embeddings

Trains the PMFlow field to learn semantic relationships through:
1. Positive pairs: Concepts that should be similar (related, synonyms, categories)
2. Negative pairs: Concepts that should be dissimilar (randomly sampled)
3. Hard negatives: Similar surface form but different meaning

This allows the BNN to understand that "cat" and "dog" are related (both animals)
while "bank" (river) and "bank" (financial) are different despite same surface form.

Sources of training pairs:
- ConceptDatabase relations (is_a, related_to, part_of, etc.)
- Pattern database intent similarities
- User corrections and feedback
- External knowledge sources (Wikipedia categories, WordNet)

Design Philosophy:
- Lightweight training that can run incrementally
- Database stores relationships, BNN learns to embed them
- Direct insertion of relationships is always possible
- Training improves generalization, not required for function
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Iterator
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


@dataclass
class SemanticPair:
    """A pair of concepts with their semantic relationship."""
    anchor: str
    other: str
    relationship: str  # "positive", "negative", "hard_negative"
    weight: float = 1.0  # For importance sampling
    source: str = "unknown"  # Where this pair came from
    

@dataclass
class TrainingMetrics:
    """Metrics from a training run."""
    epoch: int
    loss: float
    positive_similarity: float  # Average sim of positive pairs
    negative_similarity: float  # Average sim of negative pairs
    margin: float  # pos_sim - neg_sim (should be positive and growing)
    num_pairs: int


class SemanticPairDataset(Dataset):
    """Dataset of semantic pairs for contrastive learning."""
    
    def __init__(self, pairs: List[SemanticPair]):
        self.pairs = pairs
        
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[str, str, int, float]:
        pair = self.pairs[idx]
        # Label: 1 for positive, 0 for negative/hard_negative
        label = 1 if pair.relationship == "positive" else 0
        return pair.anchor, pair.other, label, pair.weight


class ContrastiveLearner:
    """
    Contrastive learning trainer for semantic embeddings.
    
    Uses InfoNCE-style loss to pull positive pairs together
    and push negative pairs apart in the latent space.
    
    Key Features:
    - Incremental learning: Add new relationships without full retraining
    - Multiple pair sources: DB relations, patterns, user feedback
    - Hard negative mining: Focus on confusing pairs
    - Persistent: Save/load trained state
    """
    
    def __init__(
        self,
        encoder,  # PMFlowEmbeddingEncoder
        *,
        margin: float = 0.3,
        temperature: float = 0.07,
        learning_rate: float = 1e-3,
        device: Optional[torch.device] = None,
    ):
        self.encoder = encoder
        self.margin = margin
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.device = device or encoder.device
        
        # Training pairs database
        self.pairs: List[SemanticPair] = []
        self.pair_set: Set[Tuple[str, str, str]] = set()  # For deduplication
        
        # Optimizer targets PMFlow field parameters
        self._setup_optimizer()
        
        # Metrics history
        self.metrics_history: List[TrainingMetrics] = []
        
    def _setup_optimizer(self):
        """Setup optimizer for PMFlow field parameters."""
        # Collect trainable parameters from PMFlow field
        params = []
        
        # Handle MultiScalePMField
        if hasattr(self.encoder.pm_field, 'fine_field'):
            params.extend([
                self.encoder.pm_field.fine_field.centers,
                self.encoder.pm_field.fine_field.mus,
                self.encoder.pm_field.coarse_field.centers,
                self.encoder.pm_field.coarse_field.mus,
            ])
            if hasattr(self.encoder.pm_field, 'coarse_projection'):
                params.extend(self.encoder.pm_field.coarse_projection.parameters())
        else:
            # Standard PMField
            params.extend([
                self.encoder.pm_field.centers,
                self.encoder.pm_field.mus,
            ])
        
        # Also train the projection matrix
        self.encoder._projection.requires_grad_(True)
        params.append(self.encoder._projection)
        
        self.optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        
    # ─────────────────────────────────────────────────────────────
    # Pair Collection Methods
    # ─────────────────────────────────────────────────────────────
    
    def add_pair(
        self,
        anchor: str,
        other: str,
        relationship: str = "positive",
        weight: float = 1.0,
        source: str = "manual",
    ) -> bool:
        """
        Add a semantic pair for training.
        
        Args:
            anchor: First concept
            other: Second concept
            relationship: "positive", "negative", or "hard_negative"
            weight: Importance weight for this pair
            source: Origin of this pair (for tracking)
            
        Returns:
            True if added (False if duplicate)
        """
        key = (anchor.lower(), other.lower(), relationship)
        if key in self.pair_set:
            return False
            
        self.pair_set.add(key)
        self.pairs.append(SemanticPair(
            anchor=anchor,
            other=other,
            relationship=relationship,
            weight=weight,
            source=source,
        ))
        return True
    
    def add_symmetric_pair(
        self,
        concept_a: str,
        concept_b: str,
        relationship: str = "positive",
        weight: float = 1.0,
        source: str = "manual",
    ):
        """Add pair in both directions (for symmetric relations like 'similar_to')."""
        self.add_pair(concept_a, concept_b, relationship, weight, source)
        self.add_pair(concept_b, concept_a, relationship, weight, source)
    
    def load_from_concept_database(self, db_path: str):
        """
        Extract training pairs from ConceptDatabase relations.
        
        Relation types mapped to training pairs:
        - is_a, instance_of, subclass_of → positive (hierarchical)
        - related_to, similar_to → positive (associative)
        - opposite_of, antonym → hard_negative (semantically opposed)
        - part_of, has_part → positive (compositional)
        """
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all relations
        cursor.execute("""
            SELECT c.term as anchor, r.relation_type, r.target, r.confidence
            FROM relations r
            JOIN concepts c ON r.concept_id = c.concept_id
        """)
        
        positive_relations = {
            'is_a', 'instance_of', 'subclass_of', 'related_to', 
            'similar_to', 'part_of', 'has_part', 'synonym'
        }
        hard_negative_relations = {'opposite_of', 'antonym', 'different_from'}
        
        count = 0
        for row in cursor.fetchall():
            rel_type = row['relation_type'].lower()
            
            if rel_type in positive_relations:
                relationship = "positive"
            elif rel_type in hard_negative_relations:
                relationship = "hard_negative"
            else:
                continue  # Skip unknown relations
                
            weight = row['confidence']
            if self.add_pair(row['anchor'], row['target'], relationship, weight, "concept_db"):
                count += 1
                
        conn.close()
        print(f"Loaded {count} pairs from ConceptDatabase")
        return count
    
    def load_from_pattern_database(self, db_path: str):
        """
        Extract training pairs from pattern intents.
        
        Patterns with same intent should have similar embeddings.
        Different intents should be dissimilar.
        """
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get patterns grouped by intent
        cursor.execute("""
            SELECT pattern, intent_hint
            FROM patterns
            WHERE intent_hint IS NOT NULL AND intent_hint != ''
        """)
        
        # Group by intent
        intent_patterns: Dict[str, List[str]] = {}
        for row in cursor.fetchall():
            intent = row['intent_hint']
            pattern = row['pattern']
            if intent not in intent_patterns:
                intent_patterns[intent] = []
            intent_patterns[intent].append(pattern)
        
        conn.close()
        
        # Create pairs
        count = 0
        intents = list(intent_patterns.keys())
        
        for intent, patterns in intent_patterns.items():
            # Positive pairs: Same intent patterns
            for i in range(len(patterns)):
                for j in range(i + 1, min(i + 3, len(patterns))):  # Limit pairs
                    if self.add_symmetric_pair(
                        patterns[i], patterns[j], "positive", 0.9, "pattern_db"
                    ):
                        count += 2
            
            # Negative pairs: Different intent patterns
            other_intents = [i for i in intents if i != intent]
            for other_intent in random.sample(other_intents, min(2, len(other_intents))):
                other_patterns = intent_patterns[other_intent]
                for pattern in patterns[:2]:  # Limit
                    for other in random.sample(other_patterns, min(2, len(other_patterns))):
                        if self.add_pair(pattern, other, "negative", 0.7, "pattern_db"):
                            count += 1
        
        print(f"Loaded {count} pairs from PatternDatabase")
        return count
    
    def generate_core_semantic_pairs(self):
        """
        Generate foundational semantic pairs for basic language understanding.
        
        These are universal semantic relationships that any language model should know.
        """
        # Category hierarchies
        category_pairs = [
            # Animals
            ("cat", "animal"), ("dog", "animal"), ("bird", "animal"),
            ("cat", "mammal"), ("dog", "mammal"), ("whale", "mammal"),
            ("eagle", "bird"), ("sparrow", "bird"), ("penguin", "bird"),
            ("salmon", "fish"), ("shark", "fish"), ("tuna", "fish"),
            
            # Technology
            ("python", "programming language"), ("java", "programming language"),
            ("neural network", "machine learning"), ("deep learning", "machine learning"),
            ("machine learning", "artificial intelligence"),
            ("computer", "technology"), ("smartphone", "technology"),
            
            # Abstract concepts
            ("happiness", "emotion"), ("sadness", "emotion"), ("anger", "emotion"),
            ("mathematics", "science"), ("physics", "science"), ("biology", "science"),
        ]
        
        # Similarity relations (symmetric)
        similarity_pairs = [
            ("cat", "dog"),  # Both pets/animals
            ("python", "java"),  # Both programming languages
            ("happy", "joyful"),  # Synonyms
            ("big", "large"),  # Synonyms
            ("machine learning", "deep learning"),  # Related fields
            ("artificial intelligence", "AI"),  # Abbreviation
            ("hello", "hi"),  # Greetings
            ("goodbye", "bye"),  # Farewells
        ]
        
        # Opposites (hard negatives)
        opposite_pairs = [
            ("hot", "cold"), ("big", "small"), ("fast", "slow"),
            ("happy", "sad"), ("good", "bad"), ("light", "dark"),
            ("up", "down"), ("left", "right"), ("yes", "no"),
        ]
        
        # Unrelated pairs (negatives)
        unrelated_pairs = [
            ("cat", "mathematics"), ("python", "happiness"),
            ("computer", "banana"), ("love", "keyboard"),
            ("mountain", "programming"), ("ocean", "algorithm"),
        ]
        
        count = 0
        
        # Add category pairs (asymmetric: X is_a Y, but Y is not X)
        for specific, category in category_pairs:
            if self.add_pair(specific, category, "positive", 0.95, "core_semantic"):
                count += 1
        
        # Add similarity pairs (symmetric)
        for a, b in similarity_pairs:
            self.add_symmetric_pair(a, b, "positive", 0.85, "core_semantic")
            count += 2
        
        # Add opposites as hard negatives (symmetric)
        for a, b in opposite_pairs:
            self.add_symmetric_pair(a, b, "hard_negative", 0.9, "core_semantic")
            count += 2
        
        # Add unrelated as negatives
        for a, b in unrelated_pairs:
            if self.add_pair(a, b, "negative", 0.7, "core_semantic"):
                count += 1
        
        print(f"Generated {count} core semantic pairs")
        return count
    
    def add_user_correction(
        self,
        concept_a: str,
        concept_b: str,
        should_be_similar: bool,
    ):
        """
        Add a pair from user correction/feedback.
        
        Higher weight since explicit user signal.
        """
        relationship = "positive" if should_be_similar else "hard_negative"
        self.add_symmetric_pair(concept_a, concept_b, relationship, 1.5, "user_correction")
    
    # ─────────────────────────────────────────────────────────────
    # Training Methods
    # ─────────────────────────────────────────────────────────────
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding, returning the latent component."""
        tokens = text.lower().split()
        _, latent, _ = self.encoder.encode_with_components(tokens)
        return latent.to(self.device)
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        """Get PMFlow-evolved embedding for text."""
        latent = self._encode_text(text)
        
        # Pass through PMFlow field
        pm_output = self.encoder.pm_field(latent)
        if isinstance(pm_output, tuple):
            # MultiScalePMField: use combined output
            emb = pm_output[2]
        else:
            emb = pm_output
            
        return F.normalize(emb, dim=-1)
    
    def _contrastive_loss(
        self,
        anchor_emb: torch.Tensor,
        other_emb: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Compute contrastive loss with margin.
        
        For positive pairs: minimize distance
        For negative pairs: maximize distance (up to margin)
        
        Returns:
            loss, avg_positive_similarity, avg_negative_similarity
        """
        # Compute cosine similarity
        similarity = F.cosine_similarity(anchor_emb, other_emb)
        
        # Separate positive and negative
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        # Positive loss: pull together (1 - similarity for positives)
        pos_loss = torch.tensor(0.0, device=self.device)
        pos_sim = 0.0
        if pos_mask.any():
            pos_sims = similarity[pos_mask]
            pos_weights = weights[pos_mask]
            pos_loss = ((1 - pos_sims) * pos_weights).mean()
            pos_sim = pos_sims.mean().item()
        
        # Negative loss: push apart with margin
        neg_loss = torch.tensor(0.0, device=self.device)
        neg_sim = 0.0
        if neg_mask.any():
            neg_sims = similarity[neg_mask]
            neg_weights = weights[neg_mask]
            # Hinge loss: penalize if similarity > -margin
            neg_loss = (F.relu(neg_sims + self.margin) * neg_weights).mean()
            neg_sim = neg_sims.mean().item()
        
        total_loss = pos_loss + neg_loss
        return total_loss, pos_sim, neg_sim
    
    def train_epoch(self, batch_size: int = 32, shuffle: bool = True) -> TrainingMetrics:
        """
        Train one epoch on all pairs.
        
        Returns:
            TrainingMetrics for this epoch
        """
        if not self.pairs:
            raise ValueError("No training pairs! Add pairs before training.")
        
        # Enable training mode for PMFlow
        self.encoder.pm_field.train()
        
        dataset = SemanticPairDataset(self.pairs)
        
        # Manual batching to handle string data
        indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(indices)
        
        total_loss = 0.0
        total_pos_sim = 0.0
        total_neg_sim = 0.0
        num_batches = 0
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            # Collect batch data
            anchors = []
            others = []
            labels = []
            weights = []
            
            for idx in batch_indices:
                anchor, other, label, weight = dataset[idx]
                anchors.append(anchor)
                others.append(other)
                labels.append(label)
                weights.append(weight)
            
            # Encode batch
            anchor_embs = torch.cat([self._get_embedding(a) for a in anchors])
            other_embs = torch.cat([self._get_embedding(o) for o in others])
            labels_t = torch.tensor(labels, device=self.device)
            weights_t = torch.tensor(weights, device=self.device)
            
            # Compute loss and backprop
            self.optimizer.zero_grad()
            loss, pos_sim, neg_sim = self._contrastive_loss(
                anchor_embs, other_embs, labels_t, weights_t
            )
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.optimizer.param_groups[0]['params'] if p.requires_grad],
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_pos_sim += pos_sim
            total_neg_sim += neg_sim
            num_batches += 1
        
        # Back to eval mode
        self.encoder.pm_field.eval()
        
        # Compute averages
        avg_loss = total_loss / num_batches
        avg_pos_sim = total_pos_sim / num_batches
        avg_neg_sim = total_neg_sim / num_batches
        margin = avg_pos_sim - avg_neg_sim
        
        metrics = TrainingMetrics(
            epoch=len(self.metrics_history) + 1,
            loss=avg_loss,
            positive_similarity=avg_pos_sim,
            negative_similarity=avg_neg_sim,
            margin=margin,
            num_pairs=len(self.pairs),
        )
        self.metrics_history.append(metrics)
        
        return metrics
    
    def train(
        self,
        epochs: int = 10,
        batch_size: int = 32,
        early_stop_margin: float = 0.5,
        verbose: bool = True,
    ) -> List[TrainingMetrics]:
        """
        Train for multiple epochs with optional early stopping.
        
        Args:
            epochs: Maximum epochs to train
            batch_size: Batch size
            early_stop_margin: Stop if pos-neg margin exceeds this
            verbose: Print progress
            
        Returns:
            List of metrics for all epochs
        """
        metrics_list = []
        
        for epoch in range(epochs):
            metrics = self.train_epoch(batch_size)
            metrics_list.append(metrics)
            
            if verbose:
                print(f"Epoch {metrics.epoch}: loss={metrics.loss:.4f}, "
                      f"pos_sim={metrics.positive_similarity:.3f}, "
                      f"neg_sim={metrics.negative_similarity:.3f}, "
                      f"margin={metrics.margin:.3f}")
            
            # Early stopping
            if metrics.margin >= early_stop_margin:
                if verbose:
                    print(f"Early stopping: margin {metrics.margin:.3f} >= {early_stop_margin}")
                break
        
        return metrics_list
    
    def incremental_update(
        self,
        new_pairs: List[Tuple[str, str, str]],
        steps: int = 5,
    ):
        """
        Quick update with new pairs without full retraining.
        
        Args:
            new_pairs: List of (anchor, other, relationship) tuples
            steps: Number of gradient steps per pair
        """
        self.encoder.pm_field.train()
        
        for anchor, other, relationship in new_pairs:
            self.add_pair(anchor, other, relationship, 1.5, "incremental")
            
            # Quick gradient steps on this pair
            for _ in range(steps):
                anchor_emb = self._get_embedding(anchor)
                other_emb = self._get_embedding(other)
                
                label = torch.tensor([1 if relationship == "positive" else 0], device=self.device)
                weight = torch.tensor([1.5], device=self.device)
                
                self.optimizer.zero_grad()
                loss, _, _ = self._contrastive_loss(anchor_emb, other_emb, label, weight)
                loss.backward()
                self.optimizer.step()
        
        self.encoder.pm_field.eval()
    
    # ─────────────────────────────────────────────────────────────
    # Evaluation Methods
    # ─────────────────────────────────────────────────────────────
    
    def evaluate_pairs(
        self,
        test_pairs: List[Tuple[str, str, bool]],
    ) -> Dict[str, float]:
        """
        Evaluate on test pairs.
        
        Args:
            test_pairs: List of (concept_a, concept_b, should_be_similar)
            
        Returns:
            Dict with accuracy, avg_pos_sim, avg_neg_sim
        """
        self.encoder.pm_field.eval()
        
        correct = 0
        total = 0
        pos_sims = []
        neg_sims = []
        
        with torch.no_grad():
            for a, b, should_be_similar in test_pairs:
                emb_a = self._get_embedding(a)
                emb_b = self._get_embedding(b)
                sim = F.cosine_similarity(emb_a, emb_b).item()
                
                # Threshold at 0.0 for prediction
                predicted_similar = sim > 0.0
                if predicted_similar == should_be_similar:
                    correct += 1
                total += 1
                
                if should_be_similar:
                    pos_sims.append(sim)
                else:
                    neg_sims.append(sim)
        
        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'avg_positive_similarity': sum(pos_sims) / len(pos_sims) if pos_sims else 0.0,
            'avg_negative_similarity': sum(neg_sims) / len(neg_sims) if neg_sims else 0.0,
        }
    
    def similarity(self, concept_a: str, concept_b: str) -> float:
        """Get learned similarity between two concepts."""
        with torch.no_grad():
            emb_a = self._get_embedding(concept_a)
            emb_b = self._get_embedding(concept_b)
            return F.cosine_similarity(emb_a, emb_b).item()
    
    # ─────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────
    
    def save(self, path: Path):
        """Save training state and pairs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save encoder state
        self.encoder.save_state(path.with_suffix('.encoder.pt'))
        
        # Save pairs and metrics
        state = {
            'pairs': [
                {
                    'anchor': p.anchor,
                    'other': p.other,
                    'relationship': p.relationship,
                    'weight': p.weight,
                    'source': p.source,
                }
                for p in self.pairs
            ],
            'metrics_history': [
                {
                    'epoch': m.epoch,
                    'loss': m.loss,
                    'positive_similarity': m.positive_similarity,
                    'negative_similarity': m.negative_similarity,
                    'margin': m.margin,
                    'num_pairs': m.num_pairs,
                }
                for m in self.metrics_history
            ],
            'margin': self.margin,
            'temperature': self.temperature,
            'learning_rate': self.learning_rate,
        }
        
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self, path: Path):
        """Load training state and pairs."""
        path = Path(path)
        
        # Load encoder state
        encoder_path = path.with_suffix('.encoder.pt')
        if encoder_path.exists():
            self.encoder.load_state(encoder_path)
        
        # Load pairs and metrics
        json_path = path.with_suffix('.json')
        if json_path.exists():
            with open(json_path) as f:
                state = json.load(f)
            
            self.pairs = [
                SemanticPair(**p) for p in state.get('pairs', [])
            ]
            self.pair_set = {
                (p.anchor.lower(), p.other.lower(), p.relationship)
                for p in self.pairs
            }
            self.metrics_history = [
                TrainingMetrics(**m) for m in state.get('metrics_history', [])
            ]
    
    def get_training_summary(self) -> str:
        """Get a summary of training progress."""
        if not self.metrics_history:
            return f"ContrastiveLearner: {len(self.pairs)} pairs, not yet trained"
        
        last = self.metrics_history[-1]
        return (
            f"ContrastiveLearner: {len(self.pairs)} pairs, "
            f"{last.epoch} epochs trained\n"
            f"  Last metrics: loss={last.loss:.4f}, margin={last.margin:.3f}\n"
            f"  Positive similarity: {last.positive_similarity:.3f}\n"
            f"  Negative similarity: {last.negative_similarity:.3f}"
        )
