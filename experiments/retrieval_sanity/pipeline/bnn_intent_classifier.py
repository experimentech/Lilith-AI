"""
BNN Intent Clustering - Semantic Intent Detection

Uses BNN embeddings to cluster response patterns by semantic intent,
enabling more accurate intent detection without keyword matching.

Key improvements over keyword-based classification:
- Semantic similarity instead of exact word matches
- Automatic intent discovery from pattern embeddings
- Better handling of paraphrases and variations
- Faster retrieval via intent-based filtering
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict


@dataclass
class IntentCluster:
    """A semantic cluster of patterns with similar intent"""
    intent_label: str
    centroid: np.ndarray  # Average embedding of patterns in cluster
    pattern_ids: List[str]  # Patterns belonging to this cluster
    representative_text: str  # Example pattern for this intent
    confidence: float = 1.0  # Cluster coherence score


class BNNIntentClassifier:
    """
    Semantic intent classification using BNN embeddings.
    
    Instead of keyword matching, we:
    1. Cluster learned patterns by embedding similarity
    2. Identify intent based on nearest cluster centroid
    3. Filter patterns by intent for faster retrieval
    """
    
    def __init__(self, semantic_encoder, min_cluster_size: int = 3):
        """
        Initialize BNN intent classifier.
        
        Args:
            semantic_encoder: Encoder for generating BNN embeddings
            min_cluster_size: Minimum patterns to form a stable cluster
        """
        self.encoder = semantic_encoder
        self.min_cluster_size = min_cluster_size
        self.clusters: Dict[str, IntentCluster] = {}
        
    def cluster_patterns(self, patterns: Dict[str, 'ResponsePattern']) -> Dict[str, IntentCluster]:
        """
        Cluster patterns by semantic similarity of their responses.
        
        Args:
            patterns: Dictionary of pattern_id -> ResponsePattern
            
        Returns:
            Dictionary of intent_label -> IntentCluster
        """
        if not patterns:
            return {}
        
        # Group patterns by their current intent labels
        intent_groups = defaultdict(list)
        for pattern_id, pattern in patterns.items():
            intent_groups[pattern.intent].append((pattern_id, pattern))
        
        clusters = {}
        
        for intent_label, pattern_list in intent_groups.items():
            if len(pattern_list) < self.min_cluster_size:
                # Too few patterns - keep as singleton cluster
                if pattern_list:
                    pattern_id, pattern = pattern_list[0]
                    try:
                        emb = self.encoder.encode(pattern.response_text)
                        if hasattr(emb, 'cpu'):
                            emb = emb.cpu().numpy()
                        emb = emb.flatten()
                        centroid = emb / (np.linalg.norm(emb) + 1e-8)
                    except:
                        centroid = np.zeros(144)  # Default embedding size
                    
                    clusters[intent_label] = IntentCluster(
                        intent_label=intent_label,
                        centroid=centroid,
                        pattern_ids=[p[0] for p in pattern_list],
                        representative_text=pattern.response_text,
                        confidence=0.5  # Low confidence for small cluster
                    )
                continue
            
            # Compute embeddings for all patterns in this intent
            embeddings = []
            valid_patterns = []
            
            for pattern_id, pattern in pattern_list:
                try:
                    emb = self.encoder.encode(pattern.response_text)
                    if hasattr(emb, 'cpu'):
                        emb = emb.cpu().numpy()
                    emb = emb.flatten()
                    emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                    embeddings.append(emb_norm)
                    valid_patterns.append((pattern_id, pattern))
                except:
                    continue
            
            if not embeddings:
                continue
            
            # Compute centroid (average embedding)
            embeddings_array = np.array(embeddings)
            centroid = np.mean(embeddings_array, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            
            # Compute cluster coherence (average similarity to centroid)
            similarities = [float(np.dot(emb, centroid)) for emb in embeddings]
            coherence = np.mean(similarities)
            
            # Find most representative pattern (closest to centroid)
            best_idx = np.argmax(similarities)
            representative = valid_patterns[best_idx][1].response_text
            
            clusters[intent_label] = IntentCluster(
                intent_label=intent_label,
                centroid=centroid,
                pattern_ids=[p[0] for p in valid_patterns],
                representative_text=representative,
                confidence=coherence
            )
        
        self.clusters = clusters
        return clusters
    
    def classify_intent(self, text: str, topk: int = 3) -> List[Tuple[str, float]]:
        """
        Classify intent of text using BNN embeddings.
        
        Args:
            text: Input text to classify
            topk: Number of top intent candidates to return
            
        Returns:
            List of (intent_label, similarity_score) tuples
        """
        if not self.clusters:
            return [("unknown", 0.0)]
        
        try:
            # Encode input text
            text_emb = self.encoder.encode(text)
            if hasattr(text_emb, 'cpu'):
                text_emb = text_emb.cpu().numpy()
            text_emb = text_emb.flatten()
            text_norm = text_emb / (np.linalg.norm(text_emb) + 1e-8)
        except:
            return [("unknown", 0.0)]
        
        # Compute similarity to each cluster centroid
        intent_scores = []
        for intent_label, cluster in self.clusters.items():
            similarity = float(np.dot(text_norm, cluster.centroid))
            # Weight by cluster confidence
            weighted_score = similarity * cluster.confidence
            intent_scores.append((intent_label, weighted_score))
        
        # Sort by score descending
        intent_scores.sort(key=lambda x: x[1], reverse=True)
        
        return intent_scores[:topk]
    
    def get_patterns_by_intent(
        self, 
        intent_label: str, 
        all_patterns: Dict[str, 'ResponsePattern']
    ) -> List['ResponsePattern']:
        """
        Get all patterns belonging to a specific intent cluster.
        
        Args:
            intent_label: Intent to filter by
            all_patterns: All available patterns
            
        Returns:
            List of patterns with this intent
        """
        if intent_label not in self.clusters:
            return []
        
        cluster = self.clusters[intent_label]
        patterns = []
        
        for pattern_id in cluster.pattern_ids:
            if pattern_id in all_patterns:
                patterns.append(all_patterns[pattern_id])
        
        return patterns
    
    def get_stats(self) -> Dict:
        """Get clustering statistics"""
        if not self.clusters:
            return {
                "total_clusters": 0,
                "avg_cluster_size": 0,
                "avg_coherence": 0,
                "intents": []
            }
        
        cluster_sizes = [len(c.pattern_ids) for c in self.clusters.values()]
        coherences = [c.confidence for c in self.clusters.values()]
        
        return {
            "total_clusters": len(self.clusters),
            "avg_cluster_size": np.mean(cluster_sizes),
            "avg_coherence": np.mean(coherences),
            "intents": list(self.clusters.keys()),
            "largest_cluster": max(self.clusters.items(), key=lambda x: len(x[1].pattern_ids))[0] if self.clusters else None,
            "most_coherent": max(self.clusters.items(), key=lambda x: x[1].confidence)[0] if self.clusters else None
        }
