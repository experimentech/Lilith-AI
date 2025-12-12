#!/usr/bin/env python3
"""
Success-Based BNN Learning: Learn to use the index through conversation outcomes.

Philosophy (Open Book Exam):
- BNN doesn't memorize facts (closed book)
- BNN learns WHICH patterns work for WHICH queries (how to use the book)
- Success signal = conversation continues well
- Failure signal = conversation breaks down

Learning approach:
1. User says something → BNN retrieves patterns
2. System responds with retrieved pattern
3. User's next message signals success/failure:
   - Continues topic → SUCCESS (reinforce query→pattern mapping)
   - Changes topic / confusion → FAILURE (weaken mapping)
4. Apply plasticity to pull/push embeddings accordingly

This is "learning to use the index" - not changing what's stored,
but learning which queries should map to which patterns.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments/retrieval_sanity'))

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
from pmflow import contrastive_plasticity
from lilith.embedding import PMFlowEmbeddingEncoder


class SuccessBasedLearner:
    """
    Learns to map queries to patterns based on conversation success.
    
    This is the "open book exam" learner - it learns how to use the
    database, not what to store in it.
    """
    
    def __init__(self, encoder: PMFlowEmbeddingEncoder, learning_rate: float = 0.05):
        """
        Initialize success-based learner.
        
        Args:
            encoder: PMFlow encoder to train
            learning_rate: How quickly to adapt to success/failure
        """
        self.encoder = encoder
        self.learning_rate = learning_rate
        
        # Track query→pattern interactions
        self.interaction_history = []
        
    def record_interaction(
        self,
        query: str,
        retrieved_pattern_trigger: str,
        success: bool
    ):
        """
        Record a query→pattern interaction and its outcome.
        
        Args:
            query: User's input that triggered retrieval
            retrieved_pattern_trigger: The trigger phrase of the pattern we retrieved
            success: Whether the interaction was successful
        """
        self.interaction_history.append({
            'query': query,
            'pattern_trigger': retrieved_pattern_trigger,
            'success': success
        })
    
    def learn_from_interactions(self, batch_size: int = 10):
        """
        Apply contrastive learning based on recent interactions.
        
        Success → Pull query closer to pattern trigger (they should match)
        Failure → Push query away from pattern trigger (bad mapping)
        """
        if len(self.interaction_history) < 2:
            print("  ⚠️  Not enough interactions to learn from")
            return
        
        # Get recent interactions
        recent = self.interaction_history[-batch_size:]
        
        # Separate successful and failed interactions
        successful = [(i['query'], i['pattern_trigger']) for i in recent if i['success']]
        failed = [(i['query'], i['pattern_trigger']) for i in recent if not i['success']]
        
        if not successful and not failed:
            print("  ⚠️  No labeled interactions to learn from")
            return
        
        print(f"\n  📚 Learning from {len(recent)} interactions:")
        print(f"     ✓ Successful: {len(successful)}")
        print(f"     ✗ Failed: {len(failed)}")
        
        # Encode pairs to latent space
        def encode_pair(phrase1, phrase2):
            """Encode query and pattern trigger to latent space."""
            tokens1 = phrase1.lower().split()
            tokens2 = phrase2.lower().split()
            base1 = self.encoder.base_encoder.encode(tokens1).to(self.encoder.device)
            base2 = self.encoder.base_encoder.encode(tokens2).to(self.encoder.device)
            z1 = base1 @ self.encoder._projection
            z2 = base2 @ self.encoder._projection
            return (z1, z2)
        
        # Successful interactions → similar pairs (pull together)
        similar_pairs = [encode_pair(q, p) for q, p in successful] if successful else []
        
        # Failed interactions → dissimilar pairs (push apart)
        dissimilar_pairs = [encode_pair(q, p) for q, p in failed] if failed else []
        
        # Apply contrastive plasticity
        pm_field = self.encoder.pm_field.fine_field
        
        if similar_pairs or dissimilar_pairs:
            contrastive_plasticity(
                pm_field,
                similar_pairs=similar_pairs,
                dissimilar_pairs=dissimilar_pairs,
                c_lr=self.learning_rate,
                margin=1.5
            )
            print(f"  ✓ Applied plasticity to {pm_field.centers.shape[0]} centers")
    
    def get_stats(self):
        """Get learning statistics."""
        if not self.interaction_history:
            return {
                'total_interactions': 0,
                'success_rate': 0.0,
                'recent_success_rate': 0.0
            }
        
        successes = sum(1 for i in self.interaction_history if i['success'])
        total = len(self.interaction_history)
        
        # Recent 10 interactions
        recent = self.interaction_history[-10:]
        recent_successes = sum(1 for i in recent if i['success'])
        recent_total = len(recent)
        
        return {
            'total_interactions': total,
            'success_rate': successes / total,
            'recent_success_rate': recent_successes / recent_total if recent_total > 0 else 0.0
        }


def simulate_conversation_with_learning():
    """
    Simulate a conversation with success-based learning.
    
    This demonstrates how the BNN would learn in practice:
    1. User says something
    2. System retrieves pattern and responds
    3. User's reaction indicates success/failure
    4. BNN learns from the outcome
    """
    
    print("=" * 80)
    print("SUCCESS-BASED BNN LEARNING SIMULATION")
    print("=" * 80)
    
    # Initialize encoder
    print("\n1. Initializing BNN encoder...")
    encoder = PMFlowEmbeddingEncoder(
        dimension=96,
        latent_dim=64,
        combine_mode="pm-only",  # Focus on learned component
        seed=13
    )
    
    # Initialize learner
    learner = SuccessBasedLearner(encoder, learning_rate=0.1)
    
    # Simulate conversation scenarios
    print("\n2. Simulating conversations with success/failure signals...")
    print("-" * 80)
    
    # Scenario 1: Greetings (SUCCESSFUL)
    print("\n📖 Scenario 1: User greets, we retrieve greeting pattern")
    conversations = [
        # These should learn to map together (successful interactions)
        ("hi", "hello", True),  # User says "hi", we match to "hello" pattern → success
        ("hey", "hello", True),  # User says "hey", we match to "hello" pattern → success
        ("hello there", "hello", True),  # Variation matches → success
        ("how are you", "how are you doing", True),  # Question matches similar → success
        ("how's it going", "how are you", True),  # Another variation → success
        
        # Failed mappings (learn these are wrong)
        ("hello", "goodbye", False),  # Matched goodbye to hello → failure
        ("hi", "what's the weather", False),  # Completely wrong topic → failure
    ]
    
    print("\nRecording interactions:")
    for query, pattern_trigger, success in conversations:
        result = "✓" if success else "✗"
        learner.record_interaction(query, pattern_trigger, success)
        print(f"  {result} '{query}' → pattern '{pattern_trigger}'")
    
    # Learn from initial batch
    print("\n3. Learning from interactions...")
    learner.learn_from_interactions(batch_size=len(conversations))
    
    stats = learner.get_stats()
    print(f"\n  Stats after first batch:")
    print(f"    Total interactions: {stats['total_interactions']}")
    print(f"    Success rate: {stats['success_rate']:.1%}")
    
    # Scenario 2: More conversations
    print("\n\n📖 Scenario 2: Continuing to learn from more interactions...")
    
    more_conversations = [
        # Weather topic
        ("what's the weather", "what's the climate", True),
        ("how's the weather", "what's the weather", True),
        ("is it cold", "what's the weather", True),
        
        # Wrong weather mappings
        ("what's the weather", "hello", False),
        ("is it cold", "goodbye", False),
        
        # Movies topic  
        ("do you like movies", "do you enjoy films", True),
        ("favorite movie", "do you like movies", True),
        
        # Goodbyes
        ("goodbye", "bye", True),
        ("see you later", "goodbye", True),
        ("talk to you later", "bye", True),
    ]
    
    print("\nRecording more interactions:")
    for query, pattern_trigger, success in more_conversations:
        result = "✓" if success else "✗"
        learner.record_interaction(query, pattern_trigger, success)
        print(f"  {result} '{query}' → pattern '{pattern_trigger}'")
    
    print("\n4. Learning from second batch...")
    learner.learn_from_interactions(batch_size=len(more_conversations))
    
    stats = learner.get_stats()
    print(f"\n  Stats after second batch:")
    print(f"    Total interactions: {stats['total_interactions']}")
    print(f"    Overall success rate: {stats['success_rate']:.1%}")
    print(f"    Recent success rate: {stats['recent_success_rate']:.1%}")
    
    # Test if learning worked
    print("\n\n5. Testing if BNN learned better query→pattern mappings...")
    print("-" * 80)
    
    def compute_similarity(encoder, phrase1, phrase2):
        """Compute cosine similarity."""
        tokens1 = phrase1.lower().split()
        tokens2 = phrase2.lower().split()
        emb1 = encoder.encode(tokens1).cpu().detach().numpy().flatten()
        emb2 = encoder.encode(tokens2).cpu().detach().numpy().flatten()
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        return dot / (norm1 * norm2 + 1e-8)
    
    # Test successful mappings (should be similar)
    print("\n✓ Successful mappings (should be high similarity):")
    test_successful = [
        ("hi", "hello"),
        ("how are you", "how are you doing"),
        ("what's the weather", "what's the climate"),
        ("goodbye", "bye"),
    ]
    
    for q, p in test_successful:
        sim = compute_similarity(encoder, q, p)
        print(f"  '{q}' <-> '{p}': {sim:.3f}")
    
    # Test failed mappings (should be dissimilar)
    print("\n✗ Failed mappings (should be low similarity):")
    test_failed = [
        ("hello", "goodbye"),
        ("hi", "what's the weather"),
        ("what's the weather", "hello"),
    ]
    
    for q, p in test_failed:
        sim = compute_similarity(encoder, q, p)
        print(f"  '{q}' <-> '{p}': {sim:.3f}")
    
    # Save learned encoder
    print("\n\n6. Saving learned encoder...")
    save_path = Path("success_trained_encoder.pt")
    encoder.save_state(save_path)
    print(f"  ✓ Saved to: {save_path}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThis demonstrates 'learning to use the index':")
    print("  • BNN doesn't memorize patterns (database stores them)")
    print("  • BNN learns which queries map to which patterns")
    print("  • Learning signal = conversation success/failure")
    print("  • Over time, BNN gets better at retrieval")
    print("\nNext: Integrate this into the actual conversation loop!")
    
    return learner, encoder


if __name__ == "__main__":
    learner, encoder = simulate_conversation_with_learning()
