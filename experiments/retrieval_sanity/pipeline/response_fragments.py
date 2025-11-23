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
            # Greetings (expanded)
            ("greeting", "Hello", "Hello! How can I help you?"),
            ("greeting", "Hi", "Hi! What would you like to talk about?"),
            ("greeting", "Hi there", "Hi there! How are you doing?"),
            ("greeting", "Hey", "Hey! What's up?"),
            ("greeting", "Good morning", "Good morning! What's on your mind?"),
            ("greeting", "Good afternoon", "Good afternoon! How can I assist you?"),
            ("greeting", "Good evening", "Good evening! What brings you here?"),
            ("greeting", "Greetings", "Greetings! What can I do for you?"),
            
            # Farewells
            ("farewell", "Goodbye", "Goodbye! It was nice talking with you."),
            ("farewell", "Bye", "Bye! Take care!"),
            ("farewell", "See you", "See you later! Have a great day."),
            ("farewell", "Take care", "You too! Goodbye."),
            ("farewell", "Thanks", "You're welcome! Glad I could help."),
            
            # Understanding checks
            ("confusion", "I don't understand", "I'm not sure I understand. Could you rephrase that?"),
            ("confusion", "unclear", "That's unclear to me. Can you explain differently?"),
            ("confusion", "confusing", "I find that confusing. Could you clarify?"),
            ("confusion", "what do you mean", "Let me try to explain that better."),
            ("confusion", "huh", "Sorry, I didn't catch that. Could you repeat?"),
            
            # Acknowledgments (expanded)
            ("acknowledgment", "I see", "I understand what you mean."),
            ("acknowledgment", "got it", "Got it, that makes sense."),
            ("acknowledgment", "okay", "Okay, I follow you."),
            ("acknowledgment", "alright", "Alright, I understand."),
            ("acknowledgment", "understood", "Understood. I've got that now."),
            ("acknowledgment", "makes sense", "Yes, that makes sense to me."),
            ("acknowledgment", "right", "Right, I see what you're saying."),
            
            # Interest and engagement
            ("interest", "interesting", "That's really interesting!"),
            ("interest", "fascinating", "How fascinating! Tell me more."),
            ("interest", "cool", "That's cool! I'd like to know more."),
            ("interest", "wow", "Wow, that's impressive!"),
            ("interest", "amazing", "That's amazing! Please continue."),
            
            # Questions - asking for details (expanded)
            ("question_detail", "tell me more", "Can you tell me more about that?"),
            ("question_detail", "elaborate", "Could you elaborate on that?"),
            ("question_detail", "explain", "What do you mean by that?"),
            ("question_detail", "how so", "How so? I'm curious to understand."),
            ("question_detail", "in what way", "In what way? Could you give an example?"),
            ("question_detail", "like what", "Like what? Can you be more specific?"),
            ("question_detail", "for instance", "Could you give me an example?"),
            
            # Questions - seeking information (expanded)
            ("question_info", "what is", "Let me recall what I know about that..."),
            ("question_info", "who is", "I'm trying to remember who that is..."),
            ("question_info", "when", "I'll check when that happened..."),
            ("question_info", "where", "Let me think about where that was..."),
            ("question_info", "why", "That's a good question. Let me think about why..."),
            ("question_info", "how", "I'm considering how that works..."),
            ("question_info", "which", "Let me think about which one you mean..."),
            
            # Explanations
            ("explain", "because", "The reason is that..."),
            ("explain", "therefore", "Therefore, we can conclude..."),
            ("explain", "this means", "This means that..."),
            ("explain", "in other words", "In other words..."),
            ("explain", "basically", "Basically, what I'm saying is..."),
            
            # Topic shifts (expanded)
            ("topic_shift", "anyway", "Anyway, what else is on your mind?"),
            ("topic_shift", "by the way", "By the way, is there something else you'd like to know?"),
            ("topic_shift", "speaking of", "Speaking of that, let's explore it further."),
            ("topic_shift", "change subject", "Sure, let's talk about something else."),
            ("topic_shift", "new topic", "Okay, what would you like to discuss next?"),
            ("topic_shift", "different question", "Go ahead, ask me something different."),
            
            # Agreement (expanded)
            ("agreement", "yes", "Yes, I agree with that."),
            ("agreement", "correct", "That's correct."),
            ("agreement", "exactly", "Exactly! You've got it."),
            ("agreement", "absolutely", "Absolutely, I think so too."),
            ("agreement", "definitely", "Definitely, that's right."),
            ("agreement", "true", "True, I can see that."),
            ("agreement", "indeed", "Indeed, that's a good point."),
            
            # Disagreement (expanded)
            ("disagreement", "no", "I don't think so."),
            ("disagreement", "incorrect", "That doesn't seem right to me."),
            ("disagreement", "not quite", "Not quite, let me clarify."),
            ("disagreement", "actually", "Actually, I think it's different."),
            ("disagreement", "I disagree", "I respectfully disagree with that."),
            ("disagreement", "false", "I believe that's not accurate."),
            
            # Uncertainty
            ("uncertainty", "maybe", "Maybe, I'm not entirely sure."),
            ("uncertainty", "perhaps", "Perhaps, but I'd need to think about it."),
            ("uncertainty", "possibly", "Possibly, though I can't say for certain."),
            ("uncertainty", "not sure", "I'm not sure about that."),
            ("uncertainty", "don't know", "I don't know enough about that yet."),
            ("uncertainty", "uncertain", "I'm uncertain on that point."),
            
            # Memory and recall (expanded)
            ("recall", "remember", "Let me check my memory about that..."),
            ("recall", "previous", "I recall we discussed this before..."),
            ("recall", "earlier", "Earlier, you mentioned something about..."),
            ("recall", "before", "Before, we were talking about..."),
            ("recall", "you said", "You said something about that, right?"),
            ("recall", "mentioned", "You mentioned that previously."),
            
            # Apologies and politeness
            ("apology", "sorry", "I apologize for any confusion."),
            ("apology", "excuse me", "Excuse me, let me correct that."),
            ("apology", "my mistake", "My mistake, let me try again."),
            ("apology", "pardon", "Pardon me, I misspoke."),
            
            # Gratitude
            ("gratitude", "thank you", "You're very welcome!"),
            ("gratitude", "thanks", "No problem, happy to help!"),
            ("gratitude", "appreciate", "I appreciate your patience."),
            ("gratitude", "helpful", "Glad I could be helpful!"),
            
            # Requests for patience
            ("patience", "wait", "Just a moment while I think about that..."),
            ("patience", "hold on", "Hold on, let me process that..."),
            ("patience", "give me a moment", "Give me a moment to consider..."),
            
            # Clarification requests
            ("clarify", "you mean", "Do you mean...?"),
            ("clarify", "are you asking", "Are you asking about...?"),
            ("clarify", "do you want", "Do you want to know about...?"),
            ("clarify", "referring to", "Are you referring to...?"),
            
            # Meta-conversation
            ("meta", "this conversation", "I'm finding this conversation interesting."),
            ("meta", "talking about", "We're talking about some complex topics."),
            ("meta", "our discussion", "Our discussion has covered a lot."),
            ("meta", "what we're discussing", "What we're discussing is important."),
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
