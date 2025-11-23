"""
Response Fragment Store - Pure Neuro-Symbolic Response Generation

Stores learned response patterns with PMFlow embeddings.
Retrieves similar patterns based on context.
Updates success scores through plasticity learning.

This implements the OUTPUT side of symmetric neuro-symbolic processing:
  Input: Text → Encode → Retrieve context
  Output: Context → Retrieve patterns → Compose text

No LLM grafting - pure learned behavior!

Note: Uses simple JSON storage for now. Can be upgraded to full SymbolicStore later.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import json
from pathlib import Path


@dataclass
class ResponsePattern:
    """A learned response pattern with metadata"""
    fragment_id: str
    trigger_context: str  # What situation triggers this response
    response_text: str     # What to say
    success_score: float   # How well it works (learned)
    intent: str            # Pattern intent category
    usage_count: int = 0   # How often it's been used
    embedding_cache: Optional[List[float]] = None  # Cached embedding
    
    
class ResponseFragmentStore:
    """
    Stores and retrieves learned response patterns.
    
    Uses PMFlow embeddings to match context → response patterns,
    exactly like how semantic retrieval works for understanding.
    
    This is the core of symmetric neuro-symbolic processing!
    """
    
    def __init__(self, semantic_encoder, storage_path: str = "response_patterns.json"):
        """
        Initialize response fragment store.
        
        Args:
            semantic_encoder: SemanticStage encoder for embeddings
            storage_path: JSON file path for response patterns
        """
        self.encoder = semantic_encoder
        self.storage_path = Path(storage_path)
        self.patterns: Dict[str, ResponsePattern] = {}
        
        # Load existing patterns if available
        self._load_patterns()
        
        # Bootstrap with seed patterns if empty
        if not self.patterns:
            self._bootstrap_seed_patterns()
            self._save_patterns()
        
    def _load_patterns(self):
        """Load patterns from JSON file"""
        if not self.storage_path.exists():
            return
            
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                for pattern_dict in data:
                    pattern = ResponsePattern(**pattern_dict)
                    self.patterns[pattern.fragment_id] = pattern
        except (json.JSONDecodeError, FileNotFoundError):
            pass
            
    def _save_patterns(self):
        """Save patterns to JSON file"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        patterns_list = []
        for pattern in self.patterns.values():
            pattern_dict = asdict(pattern)
            patterns_list.append(pattern_dict)
            
        with open(self.storage_path, 'w') as f:
            json.dump(patterns_list, f, indent=2)
    
    def _bootstrap_seed_patterns(self):
        """
        Seed the system with basic response patterns.
        
        These are NOT templates - they're training examples!
        System will learn variations through plasticity.
        """
            
        seed_patterns = [
            # Greetings
            ("greeting", "Hello", "Hello! How can I help you?"),
            ("greeting", "Hi there", "Hi! What would you like to talk about?"),
            ("greeting", "Good morning", "Good morning! What's on your mind?"),
            
            # Understanding checks
            ("confusion", "I don't understand", "I'm not sure I understand. Could you rephrase that?"),
            ("confusion", "unclear", "That's unclear to me. Can you explain differently?"),
            
            # Acknowledgments
            ("acknowledgment", "I see", "I understand what you mean."),
            ("acknowledgment", "got it", "Got it, that makes sense."),
            ("acknowledgment", "interesting", "That's interesting!"),
            
            # Questions - asking for details
            ("question_detail", "tell me more", "Can you tell me more about that?"),
            ("question_detail", "elaborate", "Could you elaborate on that?"),
            ("question_detail", "explain", "What do you mean by that?"),
            
            # Questions - seeking information
            ("question_info", "what is X", "Let me recall what I know about that..."),
            ("question_info", "who is X", "I'm trying to remember who that is..."),
            ("question_info", "when", "I'll check when that happened..."),
            
            # Topic shifts
            ("topic_shift", "new topic", "Okay, what would you like to discuss?"),
            ("topic_shift", "change subject", "Sure, let's talk about something else."),
            
            # Agreement
            ("agreement", "yes", "Yes, I agree."),
            ("agreement", "correct", "That's correct."),
            
            # Disagreement
            ("disagreement", "no", "I don't think so."),
            ("disagreement", "incorrect", "That doesn't seem right to me."),
            
            # Memory recall
            ("recall", "remember", "Let me check my memory about that..."),
            ("recall", "previous", "I recall we discussed this before..."),
        ]
        
        # Add seed patterns
        for intent, context, response in seed_patterns:
            fragment_id = f"seed_{intent}_{len(self.patterns)}"
            pattern = ResponsePattern(
                fragment_id=fragment_id,
                trigger_context=context,
                response_text=response,
                success_score=0.5,  # Neutral initial score
                intent=intent,
                usage_count=0
            )
            self.patterns[fragment_id] = pattern
            
    def add_pattern(
        self, 
        trigger_context: str, 
        response_text: str, 
        success_score: float = 0.5,
        intent: str = "general"
    ) -> str:
        """
        Learn a new response pattern.
        
        Args:
            trigger_context: Situation that triggers this response
            response_text: What to say in this situation
            success_score: Initial success rating (0.0-1.0)
            intent: Pattern intent category
            
        Returns:
            fragment_id: Unique identifier for this pattern
        """
        # Generate unique ID
        fragment_id = f"pattern_{intent}_{len(self.patterns)}"
        
        # Create pattern
        pattern = ResponsePattern(
            fragment_id=fragment_id,
            trigger_context=trigger_context,
            response_text=response_text,
            success_score=success_score,
            intent=intent,
            usage_count=0
        )
        
        # Store pattern
        self.patterns[fragment_id] = pattern
        self._save_patterns()
        
        return fragment_id
        
    def retrieve_patterns(
        self, 
        context: str, 
        topk: int = 5,
        min_score: float = 0.0
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Retrieve response patterns similar to context.
        
        This is symmetric with semantic retrieval!
        
        Args:
            context: Current conversation context
            topk: Number of patterns to retrieve
            min_score: Minimum similarity threshold
            
        Returns:
            List of (ResponsePattern, similarity_score) tuples
        """
        # Encode context
        try:
            context_emb = self.encoder.encode(context)
            # Convert to numpy for cosine similarity
            if hasattr(context_emb, 'cpu'):
                context_emb = context_emb.cpu().numpy()
            # Flatten to 1D if needed
            context_emb = context_emb.flatten()
            context_norm = context_emb / (np.linalg.norm(context_emb) + 1e-8)
        except Exception:
            # If encoding fails, use simple text matching
            return self._fallback_text_matching(context, topk)
        
        # Calculate similarities with all patterns
        scored_patterns = []
        for pattern in self.patterns.values():
            # Get or compute embedding for pattern
            try:
                pattern_emb = self.encoder.encode(pattern.trigger_context)
                if hasattr(pattern_emb, 'cpu'):
                    pattern_emb = pattern_emb.cpu().numpy()
                # Flatten to 1D if needed
                pattern_emb = pattern_emb.flatten()
                pattern_norm = pattern_emb / (np.linalg.norm(pattern_emb) + 1e-8)
                
                # Cosine similarity
                similarity = float(np.dot(context_norm, pattern_norm))
                
                if similarity >= min_score:
                    scored_patterns.append((pattern, similarity))
            except Exception:
                # Skip patterns that can't be encoded
                continue
        
        # Sort by similarity descending
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return scored_patterns[:topk]
    
    def _fallback_text_matching(self, context: str, topk: int) -> List[Tuple[ResponsePattern, float]]:
        """Simple text-based fallback matching"""
        context_lower = context.lower()
        context_words = set(context_lower.split())
        
        scored_patterns = []
        for pattern in self.patterns.values():
            trigger_lower = pattern.trigger_context.lower()
            trigger_words = set(trigger_lower.split())
            
            # Jaccard similarity
            if trigger_words:
                overlap = len(context_words & trigger_words)
                similarity = overlap / len(trigger_words | context_words)
                scored_patterns.append((pattern, similarity))
        
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        return scored_patterns[:topk]
        
    def update_success(
        self, 
        fragment_id: str, 
        feedback: float,
        plasticity_rate: float = 0.1
    ):
        """
        Update success score based on conversation outcome.
        
        This is where the system LEARNS what works!
        
        Args:
            fragment_id: Which pattern to update
            feedback: Feedback signal (-1.0 to 1.0)
                     Positive = response worked well
                     Negative = response caused confusion
            plasticity_rate: How quickly to update (0.0-1.0)
        """
        # Get pattern
        if fragment_id not in self.patterns:
            return
            
        pattern = self.patterns[fragment_id]
        
        # Update success score with moving average
        new_score = pattern.success_score + plasticity_rate * feedback
        new_score = np.clip(new_score, 0.0, 1.0)
        
        # Update pattern
        pattern.success_score = float(new_score)
        pattern.usage_count += 1
        
        # Save changes
        self._save_patterns()
        
    def get_stats(self) -> Dict:
        """Get statistics about learned patterns"""
        total = len(self.patterns)
        
        if total == 0:
            return {
                "total_patterns": 0,
                "seed_patterns": 0,
                "learned_patterns": 0,
                "average_success": 0.0
            }
        
        # Count seed vs learned
        seed_count = sum(1 for p in self.patterns.values() 
                        if p.fragment_id.startswith("seed_"))
        learned_count = total - seed_count
        
        # Calculate average success score
        avg_success = np.mean([p.success_score for p in self.patterns.values()])
            
        return {
            "total_patterns": total,
            "seed_patterns": seed_count,
            "learned_patterns": learned_count,
            "average_success": float(avg_success)
        }
