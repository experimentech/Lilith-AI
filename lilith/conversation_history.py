"""
Conversation History - Sliding Window Memory

Maintains a pure memory buffer of recent conversation turns.
No generation logic - just storage and retrieval.

This is the SHORT-TERM MEMORY layer:
  - Working Memory = ConversationState (PMFlow activations, decay)
  - Short-term Memory = ConversationHistory (recent turns, sliding window)
  - Long-term Memory = SymbolicStore (unlimited, retrieval-based)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
from collections import deque


@dataclass
class ConversationTurn:
    """A single turn in the conversation"""
    timestamp: datetime
    user_input: str
    bot_response: str
    user_embedding: Optional[Any] = None     # Semantic embedding
    response_embedding: Optional[Any] = None  # Response embedding
    working_memory_state: Optional[Dict] = None  # Snapshot of state
    success_score: Optional[float] = None    # Learning feedback
    
    
class ConversationHistory:
    """
    Sliding window buffer of recent conversation turns.
    
    Provides fast access to immediate context without database retrieval.
    Automatically evicts old turns to maintain bounded memory.
    """
    
    def __init__(self, max_turns: int = 10):
        """
        Initialize conversation history.
        
        Args:
            max_turns: Maximum number of turns to keep in memory
        """
        self.max_turns = max_turns
        self.turns = deque(maxlen=max_turns)
        
    def add_turn(
        self,
        user_input: str,
        bot_response: str,
        user_embedding: Optional[Any] = None,
        response_embedding: Optional[Any] = None,
        working_memory_state: Optional[Dict] = None
    ):
        """
        Add a conversation turn to history.
        
        Args:
            user_input: What the user said
            bot_response: What the bot responded
            user_embedding: Semantic embedding of user input
            response_embedding: Semantic embedding of response
            working_memory_state: Snapshot of working memory
        """
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_input=user_input,
            bot_response=bot_response,
            user_embedding=user_embedding,
            response_embedding=response_embedding,
            working_memory_state=working_memory_state
        )
        
        self.turns.append(turn)
        
    def update_last_success(self, success_score: float):
        """
        Update success score of most recent turn.
        
        Called after observing outcome of the response.
        
        Args:
            success_score: Success rating (-1.0 to 1.0)
        """
        if self.turns:
            self.turns[-1].success_score = success_score
            
    def get_recent_turns(self, n: int = 5) -> List[ConversationTurn]:
        """
        Get the N most recent turns.
        
        Args:
            n: Number of recent turns to retrieve
            
        Returns:
            List of recent turns (newest last)
        """
        # Get last N turns
        recent = list(self.turns)[-n:]
        return recent
        
    def get_context_window(
        self, 
        n: int = 5,
        include_embeddings: bool = False
    ) -> str:
        """
        Get recent conversation as formatted text.
        
        Args:
            n: Number of recent turns
            include_embeddings: Whether to include embedding info
            
        Returns:
            Formatted conversation context
        """
        recent = self.get_recent_turns(n)
        
        if not recent:
            return "(no conversation history)"
            
        lines = []
        for i, turn in enumerate(recent):
            lines.append(f"[Turn {i+1}]")
            lines.append(f"User: {turn.user_input}")
            lines.append(f"Bot: {turn.bot_response}")
            
            if include_embeddings and turn.success_score is not None:
                lines.append(f"Success: {turn.success_score:.2f}")
                
            lines.append("")
            
        return "\n".join(lines)
        
    def get_user_inputs(self, n: int = 5) -> List[str]:
        """
        Get recent user inputs only.
        
        Useful for detecting repetition or confusion.
        
        Args:
            n: Number of recent inputs
            
        Returns:
            List of user inputs
        """
        recent = self.get_recent_turns(n)
        return [turn.user_input for turn in recent]
        
    def get_bot_responses(self, n: int = 5) -> List[str]:
        """
        Get recent bot responses only.
        
        Useful for avoiding repetition.
        
        Args:
            n: Number of recent responses
            
        Returns:
            List of bot responses
        """
        recent = self.get_recent_turns(n)
        return [turn.bot_response for turn in recent]
        
    def detect_repetition(self, text: str, window: int = 3) -> bool:
        """
        Detect if text repeats recent user input.
        
        Helps identify when user is confused and repeating question.
        
        Args:
            text: Text to check
            window: Number of recent turns to check
            
        Returns:
            True if text is very similar to recent input
        """
        recent_inputs = self.get_user_inputs(window)
        
        # Simple repetition detection
        text_lower = text.lower().strip()
        for prev_input in recent_inputs:
            prev_lower = prev_input.lower().strip()
            
            # Check for exact match
            if text_lower == prev_lower:
                return True
                
            # Check for high overlap (>80% of words match)
            text_words = set(text_lower.split())
            prev_words = set(prev_lower.split())
            
            if text_words and prev_words:
                overlap = len(text_words & prev_words)
                max_len = max(len(text_words), len(prev_words))
                
                if overlap / max_len > 0.8:
                    return True
                    
        return False
        
    def get_average_success(self, n: int = 5) -> float:
        """
        Get average success score over recent turns.
        
        Args:
            n: Number of recent turns to average
            
        Returns:
            Average success score
        """
        recent = self.get_recent_turns(n)
        scores = [turn.success_score for turn in recent 
                 if turn.success_score is not None]
        
        if not scores:
            return 0.0
            
        return sum(scores) / len(scores)
        
    def clear(self):
        """Clear all conversation history"""
        self.turns.clear()
        
    def get_stats(self) -> Dict:
        """
        Get conversation statistics.
        
        Returns:
            Dict with stats about the conversation
        """
        if not self.turns:
            return {
                "turn_count": 0,
                "average_success": 0.0,
                "has_repetition": False
            }
            
        # Calculate stats
        turn_count = len(self.turns)
        avg_success = self.get_average_success(n=len(self.turns))
        
        # Check for repetition in recent turns
        has_repetition = False
        if len(self.turns) >= 2:
            last_input = self.turns[-1].user_input
            has_repetition = self.detect_repetition(last_input, window=3)
            
        return {
            "turn_count": turn_count,
            "average_success": avg_success,
            "has_repetition": has_repetition,
            "window_size": self.max_turns
        }
