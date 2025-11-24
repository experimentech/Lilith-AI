"""
Database-backed Response Fragment Store

Replaces JSON file storage with SQLite database queries.
Keeps same API interface for compatibility with existing code.

NEW: Success-based learning - tracks which patterns work for which queries.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import sys
import numpy as np

# Import the database infrastructure
sys.path.insert(0, str(Path(__file__).parent.parent))
from pattern_database import PatternDatabase, extract_keywords

# Import the original ResponsePattern dataclass
from .response_fragments import ResponsePattern


class QueryPatternSuccessTracker:
    """
    Tracks which patterns work for which queries - the 'learning to use the index' component.
    
    Uses BNN embeddings to cluster similar queries, then tracks success rates
    for each (query_cluster, pattern) pair. This allows the system to learn
    from experience which patterns work without modifying the BNN embeddings.
    """
    
    def __init__(self, encoder, decay_factor: float = 0.95):
        """
        Initialize success tracker.
        
        Args:
            encoder: BNN encoder for computing query embeddings
            decay_factor: How quickly old experiences fade (0.95 = 5% decay per interaction)
        """
        self.encoder = encoder
        self.decay_factor = decay_factor
        
        # Track success for each (query_cluster, pattern_id) pair
        self.success_weights = defaultdict(lambda: {'successes': 0.0, 'failures': 0.0, 'total': 0.0})
        
        # Track query cluster embeddings
        self.query_clusters = []
        
    def compute_query_embedding(self, query: str) -> np.ndarray:
        """Compute BNN embedding for a query."""
        tokens = query.lower().split()
        try:
            emb = self.encoder.encode(tokens).cpu().detach().numpy().flatten()
        except:
            # Fallback if encoder doesn't work
            emb = np.zeros(96)
        return emb
    
    def find_nearest_cluster(self, query_emb: np.ndarray, threshold: float = 0.7) -> int:
        """Find which query cluster this embedding belongs to."""
        if not self.query_clusters:
            self.query_clusters.append(query_emb)
            return 0
        
        # Find most similar cluster
        best_sim = -1.0
        best_idx = -1
        
        for idx, cluster_emb in enumerate(self.query_clusters):
            dot = np.dot(query_emb, cluster_emb)
            norm1 = np.linalg.norm(query_emb)
            norm2 = np.linalg.norm(cluster_emb)
            similarity = dot / (norm1 * norm2 + 1e-8)
            
            if similarity > best_sim:
                best_sim = similarity
                best_idx = idx
        
        # If similar enough, use existing cluster
        if best_sim >= threshold:
            return best_idx
        
        # Otherwise, create new cluster
        new_idx = len(self.query_clusters)
        self.query_clusters.append(query_emb)
        return new_idx
    
    def record_outcome(self, query: str, pattern_id: str, success: bool):
        """Record whether a pattern worked for a query."""
        query_emb = self.compute_query_embedding(query)
        cluster_id = self.find_nearest_cluster(query_emb)
        
        key = (cluster_id, pattern_id)
        
        # Apply decay to existing stats
        self.success_weights[key]['successes'] *= self.decay_factor
        self.success_weights[key]['failures'] *= self.decay_factor
        self.success_weights[key]['total'] *= self.decay_factor
        
        # Add new observation
        if success:
            self.success_weights[key]['successes'] += 1.0
        else:
            self.success_weights[key]['failures'] += 1.0
        
        self.success_weights[key]['total'] += 1.0
    
    def get_pattern_boost(self, query: str, pattern_id: str) -> float:
        """
        Get success-based boost for a pattern given a query.
        
        Returns:
            Boost factor: >1.0 if pattern worked well, <1.0 if it failed, =1.0 if no history
        """
        query_emb = self.compute_query_embedding(query)
        cluster_id = self.find_nearest_cluster(query_emb)
        
        key = (cluster_id, pattern_id)
        stats = self.success_weights[key]
        
        if stats['total'] < 0.1:  # No significant history
            return 1.0
        
        # Success rate
        success_rate = stats['successes'] / stats['total']
        
        # Convert to boost: 0% success = 0.5x, 50% = 1.0x, 100% = 1.5x
        boost = 0.5 + success_rate
        
        return boost


class DatabaseBackedFragmentStore:
    """
    Database-backed version of ResponseFragmentStore.
    
    Uses SQLite with indexed queries instead of in-memory pattern storage.
    Maintains same API interface for drop-in replacement.
    
    NEW: Success-based learning - tracks query→pattern outcomes to improve retrieval.
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
        
        # Calculate IDF scores for keyword weighting (brain: distinctiveness!)
        # Rare words like 'hi', 'beach' get higher weights than common 'you', 'it'
        self._idf_scores = self.db.calculate_idf_scores()
        
        # Success tracking: Learn which patterns work for which queries
        self.success_tracker = QueryPatternSuccessTracker(semantic_encoder)
        
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
        min_score: float = 0.0,
        intent_hint: Optional[str] = None
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Retrieve response patterns using BNN intent + database queries.
        
        Strategy: Use BNN to understand semantic intent, then query database
        for patterns matching that intent + keywords.
        
        Args:
            context: Current conversation context
            topk: Number of patterns to retrieve
            min_score: Minimum success score threshold
            intent_hint: Optional intent classification from BNN
            
        Returns:
            List of (ResponsePattern, relevance_score) tuples
        """
        # Extract keywords from context
        keyword_tuples = extract_keywords(context, 'query')
        keywords = [kw for kw, _, _ in keyword_tuples]
        
        # Brain insight: Query by most DISTINCTIVE keyword first!
        # Sort keywords by IDF (rare = more informative)
        if keywords:
            keyword_idf_pairs = [(kw, self._idf_scores.get(kw, 0.0)) for kw in keywords]
            keyword_idf_pairs.sort(key=lambda x: x[1], reverse=True)  # Highest IDF first
            sorted_keywords = [kw for kw, _ in keyword_idf_pairs]
        else:
            sorted_keywords = []
        
        # Try intent-based retrieval first if we have an intent hint
        if intent_hint:
            # Query by intent + keywords (TRIGGER keywords only!)
            pattern_rows = self.db.query_patterns(
                intent=intent_hint,
                keywords=sorted_keywords if sorted_keywords else None,
                keyword_source='trigger',  # Match user input to trigger patterns!
                min_success=min_score,
                limit=topk * 3  # Get more candidates for scoring
            )
            
            if pattern_rows:
                return self._score_patterns(pattern_rows, sorted_keywords, topk)
        
        # Fallback: keyword-only retrieval
        if sorted_keywords:
            # Brain strategy: Start with MOST distinctive keyword (highest IDF)
            # This gives us fewer, more relevant candidates
            # CRITICAL: Match user input to TRIGGER keywords only!
            # (Don't match response keywords - that retrieves patterns where someone
            #  ELSE said these words, not patterns activated by these words)
            pattern_rows = []
            
            # Try top keyword first (most distinctive) - TRIGGER ONLY
            if sorted_keywords:
                pattern_rows = self.db.query_patterns(
                    keywords=[sorted_keywords[0]],
                    keyword_source='trigger',  # Only match trigger keywords!
                    min_success=min_score,
                    limit=topk * 3
                )
            
            # If we got good candidates, score them
            if pattern_rows:
                return self._score_patterns(pattern_rows, keywords, topk)
            
            # If top keyword didn't work, try all keywords (trigger only, fallback)
            pattern_rows = self.db.query_patterns(
                keywords=sorted_keywords,
                keyword_source='trigger',  # Still trigger only!
                min_success=min_score,
                limit=topk * 5
            )
            
            if pattern_rows:
                return self._score_patterns(pattern_rows, keywords, topk)
        
        # Last resort: high success patterns
        return self._get_top_patterns(topk, min_score)
    
    def _score_patterns(
        self,
        pattern_rows: List[Dict],
        context_keywords: List[str],
        topk: int
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Score retrieved patterns by TF-IDF weighted keyword overlap + success.
        
        Brain-inspired improvements:
        1. Match user input to TRIGGER patterns (what activates response)
        2. Weight trigger matches > response matches (70/30)
        3. Use TF-IDF: rare words like 'hi' weighted more than common 'you'
        4. Boost EXACT trigger matches (brain recognizes familiar patterns strongly)
        """
        scored_patterns = []
        context_keywords_set = set(context_keywords)
        
        # Calculate IDF-weighted importance for each user keyword
        keyword_weights = {}
        for kw in context_keywords:
            # Get IDF score (0 if keyword not in database)
            idf = self._idf_scores.get(kw, 0.0)
            keyword_weights[kw] = idf
        
        # Normalize weights so they sum to 1.0
        total_weight = sum(keyword_weights.values()) if keyword_weights else 1.0
        if total_weight > 0:
            keyword_weights = {kw: w/total_weight for kw, w in keyword_weights.items()}
        
        for row in pattern_rows:
            pattern = ResponsePattern(
                fragment_id=row['fragment_id'],
                trigger_context=row['trigger_context'],
                response_text=row['response_text'],
                intent=row['intent'],
                success_score=row['success_score'],
                usage_count=row['usage_count']
            )
            
            # Get pattern keywords SEPARATED by source (trigger vs response)
            cursor = self.db.conn.cursor()
            cursor.execute("""
                SELECT keyword, source FROM pattern_keywords 
                WHERE pattern_id = ?
            """, (row['id'],))
            
            trigger_keywords = set()
            response_keywords = set()
            for kw_row in cursor.fetchall():
                if kw_row[1] == 'trigger':
                    trigger_keywords.add(kw_row[0])
                else:
                    response_keywords.add(kw_row[0])
            
            # Calculate TF-IDF weighted overlap scores
            if context_keywords_set:
                # TRIGGER match (primary): Weighted by IDF (rare words count more!)
                trigger_matches = context_keywords_set & trigger_keywords
                trigger_score = sum(keyword_weights.get(kw, 0) for kw in trigger_matches)
                
                # EXACT trigger match bonus: If ALL user keywords match trigger, strong signal!
                # This helps "Hi" match "Hi" even if other patterns match "how you"
                if trigger_keywords and context_keywords_set.issubset(trigger_keywords):
                    trigger_score *= 1.5  # 50% boost for exact matches
                
                # RESPONSE match (secondary): Topic coherence for multi-turn conversations
                response_matches = context_keywords_set & response_keywords
                response_score = sum(keyword_weights.get(kw, 0) for kw in response_matches)
                
                # Brain-like: Weight trigger matching more heavily (70/30)
                overlap_score = (trigger_score * 0.7) + (response_score * 0.3)
            else:
                overlap_score = 0.5  # Neutral score when no keywords
            
            # Combine with success score
            # Overlap 60%, success 40% (pattern has proven useful)
            combined_score = (overlap_score * 0.6) + (pattern.success_score * 0.4)
            
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
    
    def retrieve_patterns_hybrid(
        self,
        context: str,
        topk: int = 5,
        min_score: float = 0.0,
        semantic_weight: float = 0.3
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Hybrid BNN embedding + keyword retrieval (OPEN BOOK EXAM architecture).
        
        This is the key innovation: BNN learns "how to index" patterns,
        database stores "what to retrieve". Like an open book exam where
        the BNN recognizes similarity, but database holds the facts.
        
        Args:
            context: User input/conversation context
            topk: Number of patterns to retrieve
            min_score: Minimum success score threshold
            semantic_weight: How much to weight BNN embeddings vs keywords (0.0-1.0)
                0.0 = pure keywords (current system)
                1.0 = pure semantic similarity
                0.3 = hybrid (recommended)
        
        Returns:
            List of (ResponsePattern, combined_score) tuples
        """
        # Step 1: BNN EMBEDDING - Learn "how to recognize" similar contexts
        # This is the "indexing skill" (open book exam)
        tokens = context.lower().split()
        query_embedding = self.encoder.encode(tokens)  # BNN generates semantic representation
        query_vec = query_embedding.cpu().detach().numpy().flatten()  # Convert to numpy
        
        # Step 2: Get candidate patterns via keywords (fast filter)
        keyword_tuples = extract_keywords(context, 'query')
        keywords = [kw for kw, _, _ in keyword_tuples]
        
        if not keywords:
            # No keywords - fall back to top patterns
            return self._get_top_patterns(topk, min_score)
        
        # Retrieve more candidates than needed for re-ranking
        keyword_patterns = self.db.query_patterns(
            keywords=keywords,
            keyword_source='trigger',
            min_success=min_score,
            limit=topk * 10  # Get many candidates for semantic re-ranking
        )
        
        if not keyword_patterns:
            return self._get_top_patterns(topk, min_score)
        
        # Step 3: SEMANTIC SIMILARITY - BNN computes similarity to each candidate
        # This is where BNN "looks up" which patterns are relevant
        scored_patterns = []
        
        for row in keyword_patterns:
            pattern = ResponsePattern(
                fragment_id=row['fragment_id'],
                trigger_context=row['trigger_context'],
                response_text=row['response_text'],
                intent=row['intent'],
                success_score=row['success_score'],
                usage_count=row['usage_count']
            )
            
            # Compute BNN semantic similarity
            trigger_tokens = pattern.trigger_context.lower().split()
            pattern_embedding = self.encoder.encode(trigger_tokens)
            pattern_vec = pattern_embedding.cpu().detach().numpy().flatten()
            
            # Cosine similarity between query and pattern embeddings
            dot_product = np.dot(query_vec, pattern_vec)
            query_norm = np.linalg.norm(query_vec)
            pattern_norm = np.linalg.norm(pattern_vec)
            
            if query_norm > 0 and pattern_norm > 0:
                semantic_sim = dot_product / (query_norm * pattern_norm)
                # Normalize to 0-1 range (cosine is -1 to 1)
                semantic_sim = (semantic_sim + 1.0) / 2.0
            else:
                semantic_sim = 0.0
            
            # Compute keyword overlap score (existing logic)
            cursor = self.db.conn.cursor()
            cursor.execute("""
                SELECT keyword, source FROM pattern_keywords 
                WHERE pattern_id = ?
            """, (row['id'],))
            
            trigger_keywords_set = set()
            for kw_row in cursor.fetchall():
                if kw_row[1] == 'trigger':
                    trigger_keywords_set.add(kw_row[0])
            
            # TF-IDF weighted keyword overlap
            keywords_set = set(keywords)
            if keywords_set and trigger_keywords_set:
                matches = keywords_set & trigger_keywords_set
                keyword_score = len(matches) / len(keywords_set)
            else:
                keyword_score = 0.0
            
            # Step 4: HYBRID SCORE - Combine BNN similarity + keyword match + success boost
            # This balances learned semantic understanding with explicit matching
            hybrid_score = (
                semantic_weight * semantic_sim +  # BNN: "This feels similar"
                (1.0 - semantic_weight) * keyword_score  # Keywords: "These words match"
            )
            
            # Add database success boost (historical performance)
            success_boost = (pattern.success_score - 0.5) * 0.3  # -0.15 to +0.15
            hybrid_score += success_boost
            
            # Add learned success boost (what actually worked in conversations)
            # This is the "learning to use the index" component
            learned_boost = self.success_tracker.get_pattern_boost(context, pattern.fragment_id)
            hybrid_score *= learned_boost  # Multiply by 0.5x to 1.5x based on experience
            hybrid_score = hybrid_score + success_boost
            
            # Ensure score stays in reasonable range
            hybrid_score = max(0.0, min(2.0, hybrid_score))
            
            scored_patterns.append((pattern, hybrid_score))
        
        # Sort by combined score
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return scored_patterns[:topk]
    
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
    
    def record_conversation_outcome(self, query: str, pattern_id: str, success: bool):
        """
        Record the outcome of using a pattern in conversation.
        
        This is how the system learns - track what works and what doesn't.
        
        Args:
            query: The user's query that triggered retrieval
            pattern_id: The pattern ID that was used
            success: Whether the conversation continued successfully
        """
        self.success_tracker.record_outcome(query, pattern_id, success)
    
    def get_success_stats(self) -> Dict:
        """Get success tracking statistics."""
        return {
            'query_clusters': len(self.success_tracker.query_clusters),
            'tracked_pairs': len(self.success_tracker.success_weights),
            'total_observations': sum(
                stats['total'] 
                for stats in self.success_tracker.success_weights.values()
            )
        }
    
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
