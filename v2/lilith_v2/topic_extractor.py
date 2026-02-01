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
    def __init__(self, name: str, embedding: np.ndarray, contexts: List[str] = None, usage: int = 1):
        self.name = name
        self.embedding = embedding
        self.example_contexts = contexts or []
        self.usage_count = usage
        self.success_rate = 0.5 

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "embedding": self.embedding.tolist(),
            "contexts": self.example_contexts,
            "usage": self.usage_count
        }

class TopicExtractor:
    """
    BioNN-based topic extraction.
    
    Ported from v1 'lilith/topic_extractor.py'.
    """
    
    def __init__(self, encoder: Any, storage_path: str = "data/topics.json"):
        self.encoder = encoder
        self.storage_path = Path(storage_path)
        self.topics: Dict[str, LearnedTopic] = {}
        self.scaffolding = {
            'do', 'does', 'did', 'can', 'could', 'would', 'should',
            'you', 'know', 'about', 'tell', 'me', 'what', 'is', 'are',
            'a', 'an', 'the', 'of', 'please', 'i', 'want', 'to', 'learn',
            'explain', 'describe', 'who', 'where', 'when', 'why', 'how',
            'lilith', 'hello', 'hi', 'hey'
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

    def extract_topic(self, query: str) -> Optional[str]:
        """
        Identify the likely topic of a query using embedding similarity.
        """
        if not self.topics:
            return None

        # 1. Clean query (strip scaffolding)
        # "Do you know about Python?" -> "Python"
        # "Tell me about machine learning" -> "Machine learning"
        words = query.lower().split()
        content = [w for w in words if w not in self.scaffolding]
        clean_query = " ".join(content)
        
        if not clean_query:
            return None

        # 2. Encode query representation
        query_vec = self._encode_text(clean_query)

        # 3. Find best match
        best_topic = None
        best_score = 0.0
        
        for t in self.topics.values():
            score = self._cosine_sim(query_vec, t.embedding)
            # Boost by popularity/familiarity
            boost = min(0.1, t.usage_count * 0.01)
            final_score = score + boost
            
            if final_score > best_score:
                best_score = final_score
                best_topic = t.name

        threshold = 0.75 # High threshold to match v1 behavior
        if best_score > threshold:
            logger.info(f"Topic match: '{best_topic}' (score: {best_score:.2f})")
            return best_topic
        
        return None

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
                self.topics[k] = LearnedTopic(
                    v["name"], 
                    np.array(v["embedding"]), 
                    v.get("contexts", []), 
                    v.get("usage", 1)
                )
        except Exception as e:
            logger.error(f"Failed to load topics: {e}")

