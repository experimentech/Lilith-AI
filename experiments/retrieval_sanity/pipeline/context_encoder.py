"""
Context Encoder - Build Rich Conversation Context

Encodes conversation history into semantic context representations
using BNN embeddings. This enables:
- Multi-turn coherence
- Topic tracking across turns
- Reference resolution ("it", "that", etc.)
- Contextual response selection
"""

from typing import List, Optional
import numpy as np


class ConversationContextEncoder:
    """
    Encodes conversation history into rich semantic context.
    
    Instead of just using the last user input, we build context from:
    1. Recent conversation turns (with recency weighting)
    2. Active topics from working memory
    3. Semantic embeddings of conversation flow
    """
    
    def __init__(self, semantic_encoder, max_history_turns: int = 5):
        """
        Initialize context encoder.
        
        Args:
            semantic_encoder: Encoder for generating embeddings
            max_history_turns: Number of recent turns to consider
        """
        self.encoder = semantic_encoder
        self.max_history_turns = max_history_turns
        
    def encode_context(
        self,
        user_input: str,
        conversation_history: List[tuple],  # [(user_text, bot_text), ...]
        active_topics: Optional[List] = None
    ) -> tuple:
        """
        Build rich context representation from conversation state.
        
        Args:
            user_input: Current user input
            conversation_history: Recent turns [(user, bot), ...]
            active_topics: Active topics from working memory
            
        Returns:
            (context_text, context_embedding) tuple
        """
        # Detect if this is a simple standalone utterance (greeting, farewell, etc.)
        # that doesn't need history context
        standalone_patterns = [
            'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening',
            'goodbye', 'bye', 'see you', 'farewell', 'take care',
            'thanks', 'thank you', 'ok', 'okay', 'yes', 'no', 'sure', 'alright'
        ]
        
        user_lower = user_input.lower().strip()
        is_standalone = any(pattern in user_lower for pattern in standalone_patterns) and len(user_input.split()) <= 5
        
        # For standalone utterances, don't dilute with history
        if is_standalone or not conversation_history:
            return self.encode_simple(user_input)
        
        # 1. Build textual context with recency weighting
        context_parts = []
        
        # Extract key topics/entities from recent history
        recent_history = conversation_history[-self.max_history_turns:] if conversation_history else []
        
        if recent_history:
            # Extract topics from recent turns (for topic continuity)
            recent_topics = []
            for user_text, bot_text in reversed(recent_history[-2:]):
                # Extract key nouns/topics (simple heuristic: capitalize words, longer words)
                words = user_text.split()
                key_words = [w for w in words if len(w) > 4 and not w.lower() in 
                           ['about', 'that', 'this', 'what', 'which', 'where', 'when', 'how']]
                recent_topics.extend(key_words[:2])  # Take up to 2 key words per turn
            
            # Add topics to context (for continuity)
            if recent_topics:
                # Use most recent topic
                context_parts.append(recent_topics[0])
        
        # Add current input (most important)
        context_parts.append(user_input)
        
        # Build query: "topic current_input" for better matching
        # Example: "machine learning What are the main types?"
        # This helps match patterns about "machine learning types"
        context_text = " ".join(context_parts)
        
        # 2. Build semantic embedding context
        # Combine embeddings with recency weighting
        embeddings_to_combine = []
        weights = []
        
        # Current input: highest weight
        try:
            current_emb = self.encoder.encode(user_input)
            if hasattr(current_emb, 'cpu'):
                current_emb = current_emb.cpu().numpy()
            current_emb = current_emb.flatten()
            embeddings_to_combine.append(current_emb)
            weights.append(0.7)  # 70% weight on current input
        except Exception:
            pass
        
        # Recent history: decaying weight
        for i, (user_text, bot_text) in enumerate(reversed(recent_history[-2:])):
            try:
                # Encode the exchange (user + bot for full context)
                exchange_text = f"{user_text} {bot_text}"
                hist_emb = self.encoder.encode(exchange_text)
                if hasattr(hist_emb, 'cpu'):
                    hist_emb = hist_emb.cpu().numpy()
                hist_emb = hist_emb.flatten()
                
                # Decaying weight: most recent = 0.2, older = 0.1
                weight = 0.2 if i == 0 else 0.1
                embeddings_to_combine.append(hist_emb)
                weights.append(weight)
            except Exception:
                continue
        
        # Combine embeddings with weighted average
        if embeddings_to_combine:
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Weighted combination
            context_embedding = np.zeros_like(embeddings_to_combine[0])
            for emb, weight in zip(embeddings_to_combine, weights):
                context_embedding += emb * weight
            
            # Normalize
            context_embedding = context_embedding / (np.linalg.norm(context_embedding) + 1e-8)
        else:
            context_embedding = None
        
        # 3. Build textual context
        context_text = " | ".join(context_parts)
        
        return context_text, context_embedding
    
    def encode_simple(self, user_input: str) -> tuple:
        """
        Simple encoding for when no history available.
        
        Args:
            user_input: Current user input
            
        Returns:
            (context_text, context_embedding) tuple
        """
        try:
            embedding = self.encoder.encode(user_input)
            if hasattr(embedding, 'cpu'):
                embedding = embedding.cpu().numpy()
            embedding = embedding.flatten()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return user_input, embedding
        except Exception:
            return user_input, None
