"""
Multi-tenant storage layer for Lilith.

Provides user-isolated databases with base knowledge fallback.
Users can learn and store patterns without corrupting base knowledge.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

from .response_fragments_sqlite import ResponseFragmentStoreSQLite as ResponseFragmentStore, ResponsePattern
from .user_auth import UserIdentity, get_user_data_path


class MultiTenantFragmentStore:
    """
    Multi-tenant response fragment store.
    
    Maintains separate databases:
    - Base store: Teacher knowledge (read-only for users)
    - User store: User-specific learning (read-write for user)
    
    Lookup strategy:
    1. Check user store first (personalized knowledge)
    2. Fall back to base store (shared knowledge)
    
    Writing strategy:
    - Teacher mode: Write to base store
    - User mode: Write only to user store (base protected)
    """
    
    def __init__(
        self,
        encoder,
        user_identity: UserIdentity,
        base_data_path: str = "data",
        enable_fuzzy_matching: bool = True
    ):
        """
        Initialize multi-tenant fragment store.
        
        Args:
            encoder: Semantic encoder for embeddings
            user_identity: User identity information
            base_data_path: Base data directory path
            enable_fuzzy_matching: Enable fuzzy matching
        """
        self.user_identity = user_identity
        self.encoder = encoder
        
        # Base store (teacher knowledge) - using SQLite now
        base_path = Path(base_data_path) / "base" / "response_patterns.db"
        self.base_store = ResponseFragmentStore(
            encoder,
            str(base_path),
            enable_fuzzy_matching=enable_fuzzy_matching
        )
        
        # User store (if not teacher)
        if user_identity.is_teacher():
            # Teacher mode: user_store is same as base_store
            self.user_store = None
            print(f"  👨‍🏫 Teacher mode: Writing to base knowledge")
        else:
            # User mode: separate user database (SQLite)
            # Users start with empty storage, no bootstrap
            user_data_path = get_user_data_path(user_identity, base_data_path)
            user_path = Path(user_data_path) / "response_patterns.db"
            
            self.user_store = ResponseFragmentStore(
                encoder,
                str(user_path),
                enable_fuzzy_matching=enable_fuzzy_matching,
                bootstrap_if_empty=False  # Users start empty
            )
            
            # Note: SQLite stores are self-contained, no manual save needed
            # New users automatically start with empty database
            
            print(f"  👤 User mode: {user_identity.display_name} (isolated storage)")
    
    def retrieve_patterns(
        self,
        context: str,
        topk: int = 5,
        min_score: float = 0.7
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Retrieve best matching patterns across user and base stores.
        
        Strategy:
        1. Get matches from user store (personalized)
        2. Get matches from base store (shared knowledge)
        3. Merge and return top-k by confidence
        
        Args:
            context: Query text to match
            topk: Number of results to return
            min_score: Minimum confidence threshold
        
        Returns:
            List of (pattern, confidence) tuples, sorted by confidence
        """
        all_matches = []
        
        # Get matches from user store (if exists)
        if self.user_store:
            user_matches = self.user_store.retrieve_patterns(
                context,
                topk=topk,
                min_score=min_score
            )
            for pattern, conf in user_matches:
                all_matches.append((pattern, conf, "user"))
        
        # Get matches from base store
        base_matches = self.base_store.retrieve_patterns(
            context,
            topk=topk,
            min_score=min_score
        )
        for pattern, conf in base_matches:
            all_matches.append((pattern, conf, "base"))
        
        # Sort by confidence and take top-k
        all_matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = all_matches[:topk]
        
        # Debug info for best match
        if top_matches:
            _, _, source = top_matches[0]
            marker = "📘" if source == "base" else "📗"
            print(f"     {marker} Retrieved from {source} knowledge")
        
        # Return without source marker
        return [(pattern, conf) for pattern, conf, _ in top_matches]
    
    def add_pattern(
        self,
        trigger_context: str,
        response_text: str,
        success_score: float = 0.5,
        intent: str = "general"
    ) -> str:
        """
        Learn a new response pattern.
        
        Writing strategy:
        - Teacher mode: Write to base store
        - User mode: Write to user store only (base protected)
        
        Args:
            trigger_context: Input context that triggers response
            response_text: Response text
            success_score: Success rating (0.0-1.0)
            intent: Intent category
        
        Returns:
            Fragment ID of learned pattern
        """
        if self.user_identity.is_teacher():
            # Teacher writes to base
            return self.base_store.add_pattern(
                trigger_context,
                response_text,
                success_score=success_score,
                intent=intent
            )
        else:
            # User writes to their own store
            if not self.user_store:
                raise RuntimeError("User store not initialized")
            
            return self.user_store.add_pattern(
                trigger_context,
                response_text,
                success_score=success_score,
                intent=intent
            )
    
    def get_all_patterns(self) -> List[ResponsePattern]:
        """
        Get all patterns (user + base).
        
        Returns:
            List of all patterns user can access
        """
        patterns = []
        
        # User patterns first
        if self.user_store:
            patterns.extend(self.user_store.patterns.values())
        
        # Base patterns
        patterns.extend(self.base_store.patterns.values())
        
        return patterns
    
    def get_pattern_count(self) -> dict:
        """
        Get pattern counts by source.
        
        Returns:
            Dict with 'user', 'base', and 'total' counts
        """
        user_count = len(self.user_store.patterns) if self.user_store else 0
        base_count = len(self.base_store.patterns)
        
        return {
            'user': user_count,
            'base': base_count,
            'total': user_count + base_count
        }
    
    def clear_user_patterns(self):
        """
        Clear user's learned patterns (user mode only).
        Does NOT affect base knowledge.
        """
        if self.user_identity.is_teacher():
            raise PermissionError("Cannot clear patterns in teacher mode")
        
        if self.user_store:
            # Clear all patterns from SQLite database
            conn = self.user_store._get_connection()
            conn.execute("DELETE FROM response_patterns")
            conn.commit()
            conn.close()
            print(f"  🗑️  Cleared user patterns for {self.user_identity.display_name}")
