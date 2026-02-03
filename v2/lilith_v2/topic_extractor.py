"""
Topic Extractor for Lilith v2 (Ported).

Learns topics from declarations (SemanticExtractor output) and matches them in queries
using BioNN embeddings. This enables robust handling of "Tell me about X" queries
without brittle regexes.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

logger = logging.getLogger(__name__)

class LearnedTopic:
    """A topic learned from declarations."""
    __slots__ = ('name', 'embedding', 'example_contexts', 'usage_count', 'success_rate')
    
    def __init__(self, name: str, embedding: np.ndarray, contexts: List[str] = None, usage: int = 1, success_rate: float = 0.5):
        self.name = name
        self.embedding = embedding
        self.example_contexts = contexts or []
        self.usage_count = usage
        self.success_rate = success_rate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "embedding": self.embedding.tolist(),
            "contexts": self.example_contexts[-5:],  # Only save last 5
            "usage": self.usage_count,
            "success_rate": self.success_rate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnedTopic":
        return cls(
            name=data["name"],
            embedding=np.array(data["embedding"]),
            contexts=data.get("contexts", []),
            usage=data.get("usage", 1),
            success_rate=data.get("success_rate", 0.5),
        )

class TopicExtractor:
    """
    BioNN-based topic extraction.
    
    Ported from v1 'lilith/topic_extractor.py'.
    """
    
    def __init__(self, encoder: Any, storage_path: str = "data/topics.json", linguistic_processor: Any = None):
        self.encoder = encoder
        self.linguistic_processor = linguistic_processor  # Optional for POS-based extraction
        self.storage_path = Path(storage_path)
        self.topics: Dict[str, LearnedTopic] = {}
        self.scaffolding = {
            'do', 'does', 'did', 'can', 'could', 'would', 'should',
            'you', 'know', 'about', 'tell', 'me', 'what', 'is', 'are',
            'a', 'an', 'the', 'of', 'please', 'i', 'want', 'to', 'learn',
            'explain', 'describe', 'who', 'where', 'when', 'why', 'how',
            'lilith', 'hello', 'hi', 'hey', 'there', 'very', 'really',
            'just', 'so', 'too', 'also', 'they', 'them', 'it', 'its',
            'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'was', 'were', 'will', 'would', 'shall', 'may', 'might',
        }
        # Common adjectives to deprioritize (not nouns)
        self.adjectives = {
            'wonderful', 'beautiful', 'amazing', 'great', 'good', 'bad',
            'lovely', 'nice', 'excellent', 'terrible', 'awful', 'playful',
            'loyal', 'cute', 'smart', 'intelligent', 'happy', 'sad',
            'fast', 'slow', 'big', 'small', 'large', 'little', 'new', 'old',
        }
        self._load()

    def learn_topic(self, topic_name: str, context_text: str) -> None:
        """
        Learn or update a topic based on a definitive statement.
        E.g. topic="Python", context="Python is a programming language."
        """
        key = topic_name.lower().strip()
        if not key: return

        # Encode topic name itself (context helps definition, but name is the key)
        # Ideally we encode "topic" but use "context" to refine meaning.
        # For simple topic spotting, just encoding the name is sufficient.
        vec = self._encode_text(key)

        if key in self.topics:
            t = self.topics[key]
            t.usage_count += 1
            if context_text not in t.example_contexts:
                t.example_contexts.append(context_text)
                t.example_contexts = t.example_contexts[-5:] # Keep last 5
            
            # Simple moving average for embedding adaptation
            alpha = 0.1
            t.embedding = (1 - alpha) * t.embedding + alpha * vec
        else:
            self.topics[key] = LearnedTopic(topic_name, vec, [context_text])
            logger.info(f"Learned new topic: {topic_name}")

        self._save()

    def extract_topic(self, query: str) -> Tuple[Optional[str], float]:
        """
        Identify the likely topic of a query using embedding similarity.
        Falls back to heuristic extraction if no learned topics exist.
        
        Returns:
            (topic_name, confidence) or (None, 0.0) if no match
        """
        # 1. Clean query (strip scaffolding and punctuation)
        # "Do you know about Python?" -> "Python"
        # "Tell me about machine learning" -> "Machine learning"
        import re
        words = query.lower().split()
        # Strip punctuation from each word
        words = [re.sub(r'[^\w]', '', w) for w in words]
        content = [w for w in words if w and w not in self.scaffolding]
        clean_query = " ".join(content)
        
        if not clean_query:
            return None, 0.0

        # 2. If we have learned topics, use embedding similarity
        if self.topics:
            # Encode query representation
            query_vec = self._encode_text(clean_query)

            # Find best match
            best_topic = None
            best_score = 0.0
            
            for t in self.topics.values():
                score = self._cosine_sim(query_vec, t.embedding)
                # Boost by popularity/familiarity
                usage_boost = min(0.1, t.usage_count * 0.01)
                # Boost by success rate (topics that work well)
                success_boost = (t.success_rate - 0.5) * 0.1
                final_score = score + usage_boost + success_boost
                
                if final_score > best_score:
                    best_score = final_score
                    best_topic = t.name

            threshold = 0.70  # Slightly lower threshold for better recall
            if best_score > threshold:
                logger.info(f"Topic match: '{best_topic}' (score: {best_score:.2f})")
                return best_topic, best_score

        # 3. Bootstrap fallback: extract content word as topic
        # This enables learning from conversations before topics are trained
        if content:
            # Use POS tagging if linguistic processor available
            if self.linguistic_processor:
                try:
                    parsed = self.linguistic_processor.process(query)
                    nouns = [t.text.lower() for t in parsed.parsed.tokens 
                            if t.pos in ('NN', 'NNS', 'NNP', 'NNPS')]
                    if nouns:
                        # Prefer longer nouns
                        nouns.sort(key=len, reverse=True)
                        logger.debug(f"Topic fallback: '{nouns[0]}' (POS-tagged noun)")
                        return nouns[0], 0.5  # Medium confidence for fallback
                except Exception:
                    pass  # Fall through to heuristic
            
            # Heuristic fallback: prefer nouns over adjectives
            # First try non-adjective content words
            non_adj = [c for c in content if c not in self.adjectives]
            if non_adj:
                candidates = sorted(non_adj, key=len, reverse=True)
                candidates = [c for c in candidates if len(c) > 2]
                if candidates:
                    logger.debug(f"Topic fallback: '{candidates[0]}' (heuristic)")
                    return candidates[0], 0.5
            
            # Last resort: any content word
            candidates = sorted(content, key=len, reverse=True)
            candidates = [c for c in candidates if len(c) > 2]
            if candidates:
                logger.debug(f"Topic fallback: '{candidates[0]}' (last resort)")
                return candidates[0], 0.4  # Lower confidence
        
        return None, 0.0

    def update_success(self, topic: str, success: bool) -> None:
        """
        Update topic success rate after a lookup.
        
        Call this after using a topic for external lookup:
        - success=True if lookup returned useful information
        - success=False if lookup failed or was irrelevant
        """
        key = topic.lower()
        if key in self.topics:
            t = self.topics[key]
            # Exponential moving average
            alpha = 0.2
            t.success_rate = (1 - alpha) * t.success_rate + alpha * (1.0 if success else 0.0)
            self._save()
            logger.debug(f"Updated '{topic}' success_rate: {t.success_rate:.2f}")

    def get_topics(self) -> List[str]:
        """Get all learned topic names."""
        return [t.name for t in self.topics.values()]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about learned topics."""
        if not self.topics:
            return {'total_topics': 0}
        
        return {
            'total_topics': len(self.topics),
            'total_usages': sum(t.usage_count for t in self.topics.values()),
            'avg_success_rate': float(np.mean([t.success_rate for t in self.topics.values()])),
            'top_topics': sorted(
                [(t.name, t.usage_count, t.success_rate) for t in self.topics.values()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    def _encode_text(self, text: str) -> np.ndarray:
        """Helper to get numpy embedding from encoder."""
        tensor = self.encoder.encode(text)
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        
        # Determine dimensionality logic: 
        # If encoder returns [Batch, Dim] or [Dim], flatten it.
        vec = tensor.flatten()
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _cosine_sim(self, a, b) -> float:
        # Handle dimension mismatch gracefully
        if a.shape != b.shape:
            # If dimensions don't match, can't compare - return 0
            return 0.0
        return float(np.dot(a, b))

    def _save(self):
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {k: v.to_dict() for k, v in self.topics.items()}
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save topics: {e}")

    def _load(self):
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            for k, v in data.items():
                self.topics[k] = LearnedTopic.from_dict(v)
            logger.info(f"Loaded {len(self.topics)} topics from {self.storage_path}")
        except Exception as e:
            logger.warning(f"Failed to load topics: {e}")

