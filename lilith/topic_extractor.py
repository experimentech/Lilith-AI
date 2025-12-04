"""
Topic Extractor - BNN-based Topic Learning and Extraction

Learns topics from declarations and uses semantic similarity
to extract topics from queries. No hardcoded patterns needed.

Self-learning flow:
1. User teaches: "dogs are mammals" → learns topic "dogs"
2. User asks: "do you know about dogs?" → BNN matches to "dogs"
3. Wikipedia lookup uses "dogs" instead of regex-extracted garbage

This replaces the regex-based _clean_query approach with neural learning.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LearnedTopic:
    """A topic learned from declarations."""
    name: str                      # "dogs", "Python", "machine learning"
    embedding: np.ndarray          # BNN embedding of the topic
    example_contexts: List[str]    # Contexts where this topic appeared
    usage_count: int = 1           # How often this topic has been used
    success_rate: float = 0.5      # How often lookups succeeded


class TopicExtractor:
    """
    BNN-based topic extraction that learns from declarations.
    
    Instead of regex patterns like:
        "do you know about X?" → extract X
        "what is X?" → extract X
        
    We learn topics from teaching and match via semantic similarity:
        User teaches: "dogs are mammals" → store embedding of "dogs"
        User asks: "do you know about dogs?" → match to "dogs" embedding
    """
    
    def __init__(
        self, 
        encoder,  # PMFlowEmbeddingEncoder or similar
        storage_path: Optional[Path] = None,
        similarity_threshold: float = 0.65
    ):
        """
        Args:
            encoder: BNN encoder for semantic embeddings
            storage_path: Where to persist learned topics
            similarity_threshold: Minimum similarity to match a topic
        """
        self.encoder = encoder
        self.storage_path = storage_path or Path("data/topics.json")
        self.similarity_threshold = similarity_threshold
        
        # Topic store: name -> LearnedTopic
        self.topics: Dict[str, LearnedTopic] = {}
        
        # Query scaffolding - common words that wrap around topics
        # These are stripped to get cleaner embeddings, but NOT via regex extraction
        self._scaffolding_words = frozenset([
            'do', 'does', 'did', 'can', 'could', 'would', 'should',
            'you', 'know', 'about', 'tell', 'me', 'what', 'is', 'are',
            'a', 'an', 'the', 'of', 'please', 'i', 'want', 'to', 'learn',
            'explain', 'describe', 'who', 'where', 'when', 'why', 'how',
        ])
        
        # Load persisted topics
        self._load_topics()
        
        logger.info(f"TopicExtractor initialized with {len(self.topics)} learned topics")
    
    def learn_topic(self, topic: str, context: str) -> None:
        """
        Learn a topic from a declaration or teaching.
        
        Called when user teaches something like:
            "dogs are mammals" → topic="dogs", context="dogs are mammals"
            "Python is a programming language" → topic="Python"
        
        Args:
            topic: The subject/topic being taught about
            context: Full declaration for context
        """
        topic_lower = topic.lower().strip()
        
        if not topic_lower:
            return
        
        # Encode the topic (just the topic word/phrase, not full context)
        topic_emb = self._encode(topic_lower)
        
        if topic_lower in self.topics:
            # Update existing topic
            existing = self.topics[topic_lower]
            existing.usage_count += 1
            if context not in existing.example_contexts:
                existing.example_contexts.append(context)
                # Keep only recent examples
                existing.example_contexts = existing.example_contexts[-10:]
            # Rolling average of embedding (allows topic meaning to evolve)
            alpha = 0.1  # Learning rate for embedding updates
            existing.embedding = (1 - alpha) * existing.embedding + alpha * topic_emb
            logger.debug(f"Updated topic '{topic}' (count: {existing.usage_count})")
        else:
            # New topic
            self.topics[topic_lower] = LearnedTopic(
                name=topic,  # Preserve original case
                embedding=topic_emb,
                example_contexts=[context],
                usage_count=1,
                success_rate=0.5
            )
            logger.info(f"Learned new topic: '{topic}'")
        
        # Persist
        self._save_topics()
    
    def extract_topic(self, query: str) -> Tuple[Optional[str], float]:
        """
        Extract the topic from a query using BNN similarity.
        
        Instead of regex, we:
        1. Encode the full query
        2. Find best matching learned topic
        3. Return topic if similarity exceeds threshold
        
        Args:
            query: User query like "do you know about dogs?"
            
        Returns:
            (topic_name, similarity_score) or (None, 0.0) if no match
        """
        if not self.topics:
            return None, 0.0
        
        # Encode the query
        query_emb = self._encode(query.lower())
        
        # Also try encoding with scaffolding stripped (helps with short topics)
        stripped_query = self._strip_scaffolding(query)
        stripped_emb = self._encode(stripped_query) if stripped_query else None
        
        best_topic = None
        best_score = 0.0
        
        for topic_lower, learned in self.topics.items():
            # Similarity between query and topic
            sim_full = self._cosine_similarity(query_emb, learned.embedding)
            
            # Also try stripped query
            sim_stripped = 0.0
            if stripped_emb is not None:
                sim_stripped = self._cosine_similarity(stripped_emb, learned.embedding)
            
            # Take best of both approaches
            similarity = max(sim_full, sim_stripped)
            
            # Boost by usage (more commonly taught topics are more likely)
            usage_boost = min(0.1, learned.usage_count * 0.01)
            adjusted_score = similarity + usage_boost
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_topic = learned.name
        
        if best_score >= self.similarity_threshold:
            logger.debug(f"Extracted topic '{best_topic}' from query (score: {best_score:.3f})")
            return best_topic, best_score
        
        # No good match - try fallback extraction
        fallback = self._fallback_extract(query)
        if fallback:
            logger.debug(f"Fallback extracted topic: '{fallback}'")
            return fallback, 0.5  # Lower confidence for fallback
        
        return None, 0.0
    
    def update_success(self, topic: str, success: bool) -> None:
        """
        Update topic success rate after a lookup.
        
        This allows the system to learn which topics work well.
        
        Args:
            topic: Topic that was looked up
            success: Whether the lookup was successful/useful
        """
        topic_lower = topic.lower()
        if topic_lower in self.topics:
            learned = self.topics[topic_lower]
            # Exponential moving average
            alpha = 0.2
            learned.success_rate = (1 - alpha) * learned.success_rate + alpha * (1.0 if success else 0.0)
            self._save_topics()
    
    def get_topics(self) -> List[str]:
        """Get all learned topic names."""
        return [t.name for t in self.topics.values()]
    
    def _encode(self, text: str) -> np.ndarray:
        """Encode text to BNN embedding."""
        emb = self.encoder.encode(text)
        if hasattr(emb, 'cpu'):
            emb = emb.cpu().numpy()
        emb = emb.flatten()
        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(a, b))
    
    def _strip_scaffolding(self, query: str) -> str:
        """
        Strip common query scaffolding words.
        
        "do you know about dogs?" → "dogs"
        "tell me about machine learning" → "machine learning"
        
        This is NOT regex extraction - just removing common filler words
        to get a cleaner signal for the BNN.
        """
        words = query.lower().strip('?!.,').split()
        content_words = [w for w in words if w not in self._scaffolding_words]
        return ' '.join(content_words)
    
    def _fallback_extract(self, query: str) -> Optional[str]:
        """
        Fallback topic extraction when no learned topic matches.
        
        Uses simple noun-phrase extraction (not regex patterns).
        Returns the content words as potential topic.
        """
        stripped = self._strip_scaffolding(query)
        if stripped and len(stripped) > 1:
            # Capitalize for Wikipedia lookup
            return stripped.title()
        return None
    
    def _save_topics(self) -> None:
        """Persist topics to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            name: {
                'name': t.name,
                'embedding': t.embedding.tolist(),
                'example_contexts': t.example_contexts,
                'usage_count': t.usage_count,
                'success_rate': t.success_rate
            }
            for name, t in self.topics.items()
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_topics(self) -> None:
        """Load topics from disk."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for name, t_data in data.items():
                self.topics[name] = LearnedTopic(
                    name=t_data['name'],
                    embedding=np.array(t_data['embedding']),
                    example_contexts=t_data.get('example_contexts', []),
                    usage_count=t_data.get('usage_count', 1),
                    success_rate=t_data.get('success_rate', 0.5)
                )
            
            logger.info(f"Loaded {len(self.topics)} topics from {self.storage_path}")
        except Exception as e:
            logger.warning(f"Failed to load topics: {e}")
    
    def get_stats(self) -> Dict:
        """Get statistics about learned topics."""
        if not self.topics:
            return {'total_topics': 0}
        
        return {
            'total_topics': len(self.topics),
            'total_usages': sum(t.usage_count for t in self.topics.values()),
            'avg_success_rate': np.mean([t.success_rate for t in self.topics.values()]),
            'top_topics': sorted(
                [(t.name, t.usage_count) for t in self.topics.values()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
