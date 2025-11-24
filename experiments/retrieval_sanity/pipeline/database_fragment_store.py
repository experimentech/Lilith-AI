"""
Database-backed Response Fragment Store

Replaces JSON file storage with SQLite database queries.
Keeps same API interface for compatibility with existing code.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys

# Import the database infrastructure
sys.path.insert(0, str(Path(__file__).parent.parent))
from pattern_database import PatternDatabase, extract_keywords

# Import the original ResponsePattern dataclass
from .response_fragments import ResponsePattern


class DatabaseBackedFragmentStore:
    """
    Database-backed version of ResponseFragmentStore.
    
    Uses SQLite with indexed queries instead of in-memory pattern storage.
    Maintains same API interface for drop-in replacement.
    """
    
    def __init__(self, semantic_encoder, storage_path: str = "conversation_patterns.db"):
        """
        Initialize database-backed fragment store.
        
        Args:
            semantic_encoder: SemanticStage encoder (kept for compatibility)
            storage_path: SQLite database file path
        """
        self.encoder = semantic_encoder
        self.storage_path = storage_path
        self.db = PatternDatabase(storage_path)
        
        # Check if database is populated
        stats = self.db.get_stats()
        if stats['total_patterns'] == 0:
            print("⚠️  Database is empty. Run migration first:")
            print("   python pattern_database.py migrate conversation_patterns.json")
    
    @property
    def patterns(self) -> Dict[str, ResponsePattern]:
        """
        Compatibility property - returns all patterns as dict.
        
        WARNING: This loads all patterns into memory. Use sparingly!
        For new code, use query methods instead.
        """
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT * FROM patterns")
        
        patterns_dict = {}
        for row in cursor.fetchall():
            pattern = ResponsePattern(
                fragment_id=row['fragment_id'],
                trigger_context=row['trigger_context'],
                response_text=row['response_text'],
                intent=row['intent'],
                success_score=row['success_score'],
                usage_count=row['usage_count']
            )
            patterns_dict[pattern.fragment_id] = pattern
        
        return patterns_dict
    
    def retrieve_patterns(
        self,
        context: str,
        topk: int = 5,
        min_score: float = 0.0
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Retrieve response patterns using database queries.
        
        Extracts keywords from context and queries database with indexed lookups.
        Much faster than embedding similarity for large pattern sets!
        
        Args:
            context: Current conversation context
            topk: Number of patterns to retrieve
            min_score: Minimum success score threshold
            
        Returns:
            List of (ResponsePattern, relevance_score) tuples
        """
        # Extract keywords from context
        keyword_tuples = extract_keywords(context, 'query')
        keywords = [kw for kw, _, _ in keyword_tuples]
        
        if not keywords:
            # No keywords - fallback to high success patterns
            return self._get_top_patterns(topk, min_score)
        
        # Query database with keyword matching
        pattern_rows = self.db.query_patterns(
            keywords=keywords,
            min_success=min_score,
            limit=topk * 3  # Get more candidates for scoring
        )
        
        if not pattern_rows:
            # No keyword matches - fallback to high success patterns
            return self._get_top_patterns(topk, min_score)
        
        # Score patterns by keyword overlap
        scored_patterns = []
        context_keywords = set(keywords)
        
        for row in pattern_rows:
            pattern = ResponsePattern(
                fragment_id=row['fragment_id'],
                trigger_context=row['trigger_context'],
                response_text=row['response_text'],
                intent=row['intent'],
                success_score=row['success_score'],
                usage_count=row['usage_count']
            )
            
            # Get pattern keywords from database
            cursor = self.db.conn.cursor()
            cursor.execute("""
                SELECT keyword FROM pattern_keywords 
                WHERE pattern_id = ?
            """, (row['id'],))
            pattern_keywords = set(kw_row[0] for kw_row in cursor.fetchall())
            
            # Calculate keyword overlap score
            overlap = len(context_keywords & pattern_keywords)
            overlap_score = overlap / max(len(context_keywords), 1)
            
            # Combine with success score
            combined_score = (overlap_score * 0.7) + (pattern.success_score * 0.3)
            
            scored_patterns.append((pattern, combined_score))
        
        # Sort by score and return top-k
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        return scored_patterns[:topk]
    
    def _get_top_patterns(self, topk: int, min_score: float) -> List[Tuple[ResponsePattern, float]]:
        """Fallback: Get top patterns by success score."""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT * FROM patterns 
            WHERE success_score >= ?
            ORDER BY success_score DESC, usage_count DESC
            LIMIT ?
        """, (min_score, topk))
        
        patterns = []
        for row in cursor.fetchall():
            pattern = ResponsePattern(
                fragment_id=row['fragment_id'],
                trigger_context=row['trigger_context'],
                response_text=row['response_text'],
                intent=row['intent'],
                success_score=row['success_score'],
                usage_count=row['usage_count']
            )
            patterns.append((pattern, row['success_score']))
        
        return patterns
    
    def add_pattern(
        self,
        fragment_id: str,
        trigger_context: str,
        response_text: str,
        intent: str = "learned",
        success_score: float = 0.5
    ) -> str:
        """
        Add a new pattern to the database.
        
        Args:
            fragment_id: Unique identifier
            trigger_context: Context that triggers this response
            response_text: The response text
            intent: Pattern intent category
            success_score: Initial success score
            
        Returns:
            fragment_id
        """
        # Extract keywords
        trigger_keywords = extract_keywords(trigger_context, 'trigger')
        response_keywords = extract_keywords(response_text, 'response')
        all_keywords = trigger_keywords + response_keywords
        
        # Insert into database
        self.db.insert_pattern(
            fragment_id=fragment_id,
            trigger_context=trigger_context,
            response_text=response_text,
            intent=intent,
            success_score=success_score,
            keywords=all_keywords
        )
        
        return fragment_id
    
    def update_success(self, fragment_id: str, feedback: float, plasticity_rate: float = 0.1):
        """
        Update pattern success score based on feedback.
        
        Args:
            fragment_id: Pattern identifier
            feedback: Feedback signal (-1.0 to 1.0)
            plasticity_rate: How quickly to update
        """
        # Get pattern ID from fragment_id
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT id, success_score FROM patterns WHERE fragment_id = ?", (fragment_id,))
        row = cursor.fetchone()
        
        if row:
            pattern_id = row['id']
            current_score = row['success_score']
            
            # Update with moving average
            new_score = current_score + plasticity_rate * feedback
            new_score = max(0.0, min(1.0, new_score))  # Clip to [0, 1]
            
            self.db.update_success(pattern_id, new_score)
    
    def get_stats(self) -> Dict:
        """Get storage statistics."""
        return self.db.get_stats()
    
    def _save_patterns(self):
        """Compatibility method - database auto-saves."""
        pass  # No-op: database commits automatically
    
    def _load_patterns(self):
        """Compatibility method - database loads automatically."""
        pass  # No-op: database is always loaded


if __name__ == "__main__":
    """Test database-backed store."""
    print("Testing DatabaseBackedFragmentStore...")
    
    # Mock encoder for testing
    class MockEncoder:
        def encode(self, text):
            return None
    
    # Create store
    store = DatabaseBackedFragmentStore(
        semantic_encoder=MockEncoder(),
        storage_path="conversation_patterns.db"
    )
    
    # Test stats
    stats = store.get_stats()
    print(f"\n📊 Database stats:")
    print(f"   Total patterns: {stats['total_patterns']}")
    print(f"   Unique keywords: {stats['total_keywords']}")
    print(f"   Average success: {stats['avg_success']:.2f}")
    
    # Test retrieval
    print(f"\n🔍 Testing retrieval...")
    results = store.retrieve_patterns("hello how are you", topk=5)
    print(f"   Found {len(results)} patterns for 'hello how are you'")
    for pattern, score in results[:3]:
        print(f"   - {pattern.response_text[:50]}... (score: {score:.3f})")
    
    # Test keyword-based retrieval
    results = store.retrieve_patterns("what's the weather like", topk=5)
    print(f"\n   Found {len(results)} patterns for 'what's the weather like'")
    for pattern, score in results[:3]:
        print(f"   - {pattern.response_text[:50]}... (score: {score:.3f})")
    
    print("\n✅ Test complete!")
