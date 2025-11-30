#!/usr/bin/env python3
"""
Practical Success-Based Learning: Query→Pattern Success Table

Insight: Don't fight PMFlow dynamics. Instead, TRACK what works!

Instead of trying to change embeddings:
1. Track: "When user says X, pattern Y worked/failed"
2. Store: Success weights in a lookup table
3. Use: Boost/penalize patterns based on past success with similar queries

This is STILL "learning to use the index" but more practical.
The BNN embeddings give us semantic similarity, the success table
tells us what actually works in practice.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments/retrieval_sanity'))

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from lilith.embedding import PMFlowEmbeddingEncoder


class QueryPatternSuccessTracker:
    """
    Tracks which patterns work for which types of queries.
    
    This is the "learning to use the index" component:
    - BNN provides semantic similarity (base capability)
    - Success tracker learns adjustments (experience)
    - Together = effective retrieval
    """
    
    def __init__(self, encoder: PMFlowEmbeddingEncoder, decay_factor: float = 0.95):
        """
        Initialize success tracker.
        
        Args:
            encoder: BNN encoder for computing query similarity
            decay_factor: How quickly old experiences fade (0.95 = 5% decay per interaction)
        """
        self.encoder = encoder
        self.decay_factor = decay_factor
        
        # Track success for each (query_cluster, pattern_id) pair
        # query_cluster = the semantic "area" of the query
        # pattern_id = which pattern we tried
        self.success_weights = defaultdict(lambda: {'successes': 0, 'failures': 0, 'total': 0})
        
        # Track query representations for clustering
        self.query_history = []
        
    def compute_query_embedding(self, query: str) -> np.ndarray:
        """Get BNN embedding for a query."""
        tokens = query.lower().split()
        emb = self.encoder.encode(tokens).cpu().detach().numpy().flatten()
        return emb
    
    def find_nearest_cluster(self, query_emb: np.ndarray, threshold: float = 0.7) -> int:
        """
        Find which query cluster this embedding belongs to.
        
        If no cluster is similar enough, create a new one.
        """
        if not self.query_history:
            # First query = first cluster
            self.query_history.append(query_emb)
            return 0
        
        # Find most similar existing cluster
        best_similarity = -1.0
        best_idx = -1
        
        for idx, cluster_emb in enumerate(self.query_history):
            dot = np.dot(query_emb, cluster_emb)
            norm1 = np.linalg.norm(query_emb)
            norm2 = np.linalg.norm(cluster_emb)
            similarity = dot / (norm1 * norm2 + 1e-8)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx
        
        # If similar enough, use existing cluster
        if best_similarity >= threshold:
            return best_idx
        
        # Otherwise, create new cluster
        new_idx = len(self.query_history)
        self.query_history.append(query_emb)
        return new_idx
    
    def record_outcome(self, query: str, pattern_id: str, success: bool):
        """
        Record whether a pattern worked for a query.
        
        Args:
            query: User's query
            pattern_id: ID/trigger of the pattern we retrieved
            success: Whether conversation continued successfully
        """
        # Get query cluster
        query_emb = self.compute_query_embedding(query)
        cluster_id = self.find_nearest_cluster(query_emb)
        
        # Update success weights with decay
        key = (cluster_id, pattern_id)
        
        # Apply decay to existing stats
        self.success_weights[key]['successes'] *= self.decay_factor
        self.success_weights[key]['failures'] *= self.decay_factor
        self.success_weights[key]['total'] *= self.decay_factor
        
        # Add new observation
        if success:
            self.success_weights[key]['successes'] += 1.0
        else:
            self.success_weights[key]['failures'] += 1.0
        
        self.success_weights[key]['total'] += 1.0
    
    def get_pattern_boost(self, query: str, pattern_id: str) -> float:
        """
        Get success-based boost for a pattern given a query.
        
        Returns:
            Boost factor: >1.0 if pattern worked well for similar queries
                         <1.0 if pattern failed for similar queries
                         =1.0 if no history
        """
        query_emb = self.compute_query_embedding(query)
        cluster_id = self.find_nearest_cluster(query_emb)
        
        key = (cluster_id, pattern_id)
        stats = self.success_weights[key]
        
        if stats['total'] < 0.1:  # No significant history
            return 1.0
        
        # Success rate
        success_rate = stats['successes'] / stats['total']
        
        # Convert to boost: 0% success = 0.5x, 50% = 1.0x, 100% = 1.5x
        boost = 0.5 + success_rate
        
        return boost
    
    def get_stats(self) -> Dict:
        """Get tracker statistics."""
        return {
            'num_clusters': len(self.query_history),
            'num_tracked_pairs': len(self.success_weights),
            'total_observations': sum(s['total'] for s in self.success_weights.values())
        }


def test_success_tracking():
    """Test the success tracker with simulated conversations."""
    
    print("=" * 80)
    print("QUERY→PATTERN SUCCESS TRACKING TEST")
    print("=" * 80)
    
    # Initialize
    print("\n1. Initializing tracker...")
    encoder = PMFlowEmbeddingEncoder(dimension=96, latent_dim=64, combine_mode="pm-only", seed=13)
    tracker = QueryPatternSuccessTracker(encoder)
    
    # Simulate learning
    print("\n2. Simulating conversation outcomes...")
    print("-" * 80)
    
    scenarios = [
        # Greetings → greeting patterns work well
        ("hi", "pattern_hello", True),
        ("hello", "pattern_hello", True),
        ("hey there", "pattern_hello", True),
        ("hi", "pattern_hello", True),  # Repeated success
        
        # Greetings → weather patterns fail
        ("hello", "pattern_weather", False),
        ("hi", "pattern_weather", False),
        
        # Weather → weather patterns work
        ("what's the weather", "pattern_weather", True),
        ("how's the weather", "pattern_weather", True),
        ("is it cold", "pattern_weather", True),
        
        # Weather → greeting patterns fail
        ("what's the weather", "pattern_hello", False),
        
        # Goodbyes → goodbye patterns work
        ("goodbye", "pattern_bye", True),
        ("see you later", "pattern_bye", True),
        ("bye", "pattern_bye", True),
    ]
    
    print("Recording outcomes:")
    for query, pattern, success in scenarios:
        tracker.record_outcome(query, pattern, success)
        result = "✓" if success else "✗"
        print(f"  {result} '{query}' + '{pattern}' → {'SUCCESS' if success else 'FAIL'}")
    
    stats = tracker.get_stats()
    print(f"\nTracker stats:")
    print(f"  Query clusters: {stats['num_clusters']}")
    print(f"  Tracked pairs: {stats['num_tracked_pairs']}")
    print(f"  Total observations: {stats['total_observations']:.1f}")
    
    # Test boosting
    print("\n\n3. Testing pattern boosting for new queries...")
    print("-" * 80)
    
    test_queries = [
        ("hi there", ["pattern_hello", "pattern_weather", "pattern_bye"]),
        ("how's the weather today", ["pattern_hello", "pattern_weather", "pattern_bye"]),
        ("goodbye friend", ["pattern_hello", "pattern_weather", "pattern_bye"]),
    ]
    
    for query, patterns in test_queries:
        print(f"\nQuery: '{query}'")
        for pattern in patterns:
            boost = tracker.get_pattern_boost(query, pattern)
            print(f"  {pattern}: {boost:.3f}x boost")
    
    print("\n\n4. Simulating retrieval with success boosting...")
    print("-" * 80)
    
    # Mock retrieval scenario
    def simulate_retrieval(query: str):
        """Simulate retrieving patterns with success boosting."""
        # Pretend these are keyword/semantic scores
        base_scores = {
            "pattern_hello": 0.6,
            "pattern_weather": 0.5,
            "pattern_bye": 0.4,
        }
        
        print(f"\nQuery: '{query}'")
        print("  Base scores (keyword + semantic):")
        for pattern, score in base_scores.items():
            print(f"    {pattern}: {score:.3f}")
        
        # Apply success boosting
        boosted_scores = {}
        print("\n  Success-boosted scores:")
        for pattern, base_score in base_scores.items():
            boost = tracker.get_pattern_boost(query, pattern)
            boosted = base_score * boost
            boosted_scores[pattern] = boosted
            change = "↑" if boost > 1.0 else ("↓" if boost < 1.0 else "=")
            print(f"    {pattern}: {boosted:.3f} ({boost:.2f}x {change})")
        
        # Show best match
        best = max(boosted_scores.items(), key=lambda x: x[1])
        print(f"\n  → Best match: {best[0]} (score: {best[1]:.3f})")
    
    simulate_retrieval("hello there")
    simulate_retrieval("what's the weather like")
    simulate_retrieval("talk to you later")
    
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nSuccess tracking provides 'learned experience':")
    print("  ✓ BNN gives semantic similarity (base capability)")
    print("  ✓ Success tracker learns what works (experience)")
    print("  ✓ Boost/penalize patterns based on past outcomes")
    print("  ✓ No need to fight PMFlow dynamics!")
    print("\nThis IS 'learning to use the index' - learning which")
    print("patterns to retrieve for which queries based on what works.")


if __name__ == "__main__":
    test_success_tracking()
