"""
SQLite-backed Response Fragment Store

Production-ready storage for response patterns with:
- ACID transactions for data integrity
- Concurrent read/write with WAL mode
- Efficient indexing and querying
- Thread-safe operations

API-compatible with JSON ResponseFragmentStore but with proper concurrency handling.
"""

import sqlite3
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Optional: Import fuzzy matching for typo tolerance
try:
    from .fuzzy_matcher import FuzzyMatcher
    FUZZY_MATCHING_AVAILABLE = True
except ImportError:
    FUZZY_MATCHING_AVAILABLE = False


@dataclass
class ResponsePattern:
    """A learned response pattern with metadata"""
    fragment_id: str
    trigger_context: str
    response_text: str
    success_score: float
    intent: str
    usage_count: int = 0
    embedding_cache: Optional[List[float]] = None


class ResponseFragmentStoreSQLite:
    """
    SQLite-backed response pattern storage.
    
    Thread-safe with WAL mode for concurrent access.
    Drop-in replacement for JSON-based ResponseFragmentStore.
    """
    
    def __init__(
        self, 
        semantic_encoder, 
        storage_path: str = "response_patterns.db",
        enable_fuzzy_matching: bool = True,
        bootstrap_if_empty: bool = True
    ):
        """
        Initialize SQLite response fragment store.
        
        Args:
            semantic_encoder: Encoder for pattern embeddings
            storage_path: Path to SQLite database file
            enable_fuzzy_matching: Enable typo-tolerant fuzzy matching
        """
        self.encoder = semantic_encoder
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize fuzzy matcher if available
        self.fuzzy_matcher = None
        if enable_fuzzy_matching and FUZZY_MATCHING_AVAILABLE:
            self.fuzzy_matcher = FuzzyMatcher(
                edit_distance_threshold=0.75,
                token_overlap_threshold=0.6,
                enable_phonetic=False
            )
            print("  🔍 Fuzzy matching enabled for typo tolerance!")
        elif enable_fuzzy_matching and not FUZZY_MATCHING_AVAILABLE:
            print("  ⚠️  Fuzzy matching not available")
        
        # Initialize database with WAL mode for concurrency
        self._init_database()
        
        # Bootstrap with seed patterns if empty (optional)
        if bootstrap_if_empty and self._is_empty():
            self._bootstrap_seed_patterns()
    
    def _init_database(self):
        """Initialize database with schema and enable WAL mode."""
        try:
            conn = sqlite3.connect(str(self.storage_path))
            conn.row_factory = sqlite3.Row
            
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            # Verify database integrity
            cursor = conn.execute("PRAGMA integrity_check")
            integrity = cursor.fetchone()[0]
            if integrity != "ok":
                raise sqlite3.DatabaseError(f"Database integrity check failed: {integrity}")
            
            # Create schema
            conn.execute("""
                CREATE TABLE IF NOT EXISTS response_patterns (
                    fragment_id TEXT PRIMARY KEY,
                    trigger_context TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    success_score REAL DEFAULT 0.5,
                    intent TEXT DEFAULT 'general',
                    usage_count INTEGER DEFAULT 0,
                    embedding_cache TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for efficient querying
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_intent 
                ON response_patterns(intent)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_success_score 
                ON response_patterns(success_score DESC)
            """)
            
            conn.commit()
            conn.close()
            
        except sqlite3.DatabaseError as e:
            print(f"⚠️  Database corruption detected: {e}")
            print(f"⚠️  Attempting to recover database: {self.storage_path}")
            self._recover_corrupted_database()
        except Exception as e:
            print(f"❌ Failed to initialize database: {e}")
            raise
    
    def _recover_corrupted_database(self):
        """
        Attempt to recover from database corruption.
        
        Recovery strategy:
        1. Backup corrupted database
        2. Try to dump recoverable data
        3. Create fresh database
        4. Restore what we can
        """
        import shutil
        from datetime import datetime
        
        backup_path = self.storage_path.with_suffix(f'.corrupted.{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')
        
        try:
            # Backup corrupted database
            shutil.copy2(self.storage_path, backup_path)
            print(f"  💾 Backed up corrupted database to: {backup_path}")
            
            # Try to dump recoverable data
            recoverable_patterns = []
            try:
                conn = sqlite3.connect(str(self.storage_path))
                cursor = conn.execute("SELECT * FROM response_patterns")
                for row in cursor:
                    try:
                        recoverable_patterns.append(dict(row))
                    except:
                        pass  # Skip corrupted rows
                conn.close()
                print(f"  ✅ Recovered {len(recoverable_patterns)} patterns")
            except Exception as e:
                print(f"  ⚠️  Could not recover data: {e}")
            
            # Remove corrupted database
            self.storage_path.unlink()
            
            # Create fresh database
            conn = sqlite3.connect(str(self.storage_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            # Create schema
            conn.execute("""
                CREATE TABLE response_patterns (
                    fragment_id TEXT PRIMARY KEY,
                    trigger_context TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    success_score REAL DEFAULT 0.5,
                    intent TEXT DEFAULT 'general',
                    usage_count INTEGER DEFAULT 0,
                    embedding_cache TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("CREATE INDEX idx_intent ON response_patterns(intent)")
            conn.execute("CREATE INDEX idx_success_score ON response_patterns(success_score DESC)")
            
            # Restore recovered data
            for pattern in recoverable_patterns:
                try:
                    conn.execute("""
                        INSERT INTO response_patterns 
                        (fragment_id, trigger_context, response_text, success_score, intent, usage_count)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        pattern['fragment_id'],
                        pattern['trigger_context'],
                        pattern['response_text'],
                        pattern['success_score'],
                        pattern['intent'],
                        pattern.get('usage_count', 0)
                    ))
                except Exception as e:
                    print(f"  ⚠️  Could not restore pattern {pattern.get('fragment_id')}: {e}")
            
            conn.commit()
            conn.close()
            
            print(f"  ✅ Database recovered successfully")
            print(f"  📝 Restored {len(recoverable_patterns)} patterns")
            
        except Exception as e:
            print(f"  ❌ Recovery failed: {e}")
            print(f"  💡 Manual intervention required. Backup at: {backup_path}")
            raise
    
    def reset_database(self, keep_backup: bool = True, bootstrap: bool = False):
        """
        Reset database to fresh state.
        
        Args:
            keep_backup: If True, create backup before reset
            bootstrap: If True, add seed patterns after reset
            
        Returns:
            Path to backup file (if created)
        """
        import shutil
        from datetime import datetime
        
        backup_path = None
        
        if keep_backup and self.storage_path.exists():
            backup_path = self.storage_path.with_suffix(f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')
            shutil.copy2(self.storage_path, backup_path)
            print(f"  💾 Backup created: {backup_path}")
        
        # Get current pattern count
        try:
            conn = self._get_connection()
            cursor = conn.execute("SELECT COUNT(*) FROM response_patterns")
            old_count = cursor.fetchone()[0]
            conn.close()
        except:
            old_count = "unknown"
        
        # Delete database
        if self.storage_path.exists():
            self.storage_path.unlink()
        
        # Recreate fresh database
        self._init_database()
        
        if bootstrap:
            self._bootstrap_seed_patterns()
            new_count = 4
        else:
            new_count = 0
        
        print(f"  🔄 Database reset complete")
        print(f"  📊 Old patterns: {old_count}")
        print(f"  📊 New patterns: {new_count}{' (seed patterns)' if bootstrap else ''}")
        
        return backup_path
    
    def _get_connection(self):
        """Get a new database connection with proper timeout."""
        conn = sqlite3.connect(str(self.storage_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _is_empty(self) -> bool:
        """Check if database has any patterns."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM response_patterns")
        count = cursor.fetchone()[0]
        conn.close()
        return count == 0
    
    def _bootstrap_seed_patterns(self):
        """Bootstrap database with comprehensive seed patterns."""
        seed_patterns = [
            # Greetings
            ("greeting_hello", "hello", "Hello! How can I help you?", 0.9, "greeting"),
            ("greeting_hi", "hi", "Hi there! What's on your mind?", 0.85, "greeting"),
            ("greeting_hey", "hey", "Hey! What's up?", 0.85, "greeting"),
            ("greeting_morning", "good morning", "Good morning! How are you today?", 0.85, "greeting"),
            ("greeting_afternoon", "good afternoon", "Good afternoon!", 0.85, "greeting"),
            ("greeting_evening", "good evening", "Good evening!", 0.85, "greeting"),
            
            # How are you / Well-being
            ("wellbeing_how_are_you", "how are you", "I'm doing well, thank you for asking! How can I assist you?", 0.85, "wellbeing"),
            ("wellbeing_whats_up", "whats up", "Not much! Just here to help. What can I do for you?", 0.8, "wellbeing"),
            
            # Capabilities / What can you do
            ("capability_what_can_do", "what can you do", "I can chat with you, answer questions, help with math problems, and look up information. What would you like to explore?", 0.9, "capability"),
            ("capability_can_you_help", "can you help", "Of course! I'm here to help. What do you need?", 0.85, "capability"),
            ("capability_what_are_you", "what are you", "I'm Lilith, a conversational AI that learns from our interactions. I can chat, answer questions, and help with various tasks.", 0.85, "capability"),
            
            # Identity / Name
            ("identity_name", "what is your name", "My name is Lilith. Nice to meet you!", 0.9, "identity"),
            ("identity_who_are_you", "who are you", "I'm Lilith, an AI assistant designed to learn and adapt to conversations.", 0.85, "identity"),
            
            # Gratitude
            ("gratitude_thank_you", "thank you", "You're welcome! Happy to help.", 0.9, "gratitude"),
            ("gratitude_thanks", "thanks", "No problem! Let me know if you need anything else.", 0.85, "gratitude"),
            
            # Farewell
            ("farewell_goodbye", "goodbye", "Goodbye! Have a great day!", 0.9, "farewell"),
            ("farewell_bye", "bye", "Bye! Take care!", 0.85, "farewell"),
            ("farewell_see_you", "see you later", "See you later! Feel free to come back anytime.", 0.85, "farewell"),
            
            # Affirmative
            ("affirmative_yes", "yes", "Great! How can I help further?", 0.7, "affirmative"),
            ("affirmative_ok", "ok", "Sounds good! Anything else?", 0.65, "affirmative"),
            
            # Negative / Disagreement
            ("negative_no", "no", "Okay, no problem. Is there something else I can help with?", 0.7, "negative"),
            
            # Clarification requests
            ("clarification_what_mean", "what do you mean", "Let me try to explain that differently.", 0.7, "clarification"),
            ("clarification_confused", "i dont understand", "I apologize for the confusion. Let me try to clarify.", 0.75, "clarification"),
            
            # Knowledge - simple facts
            ("knowledge_sky_color", "what color is the sky", "The sky is typically blue during the day, though it can appear red or orange at sunrise and sunset.", 0.8, "knowledge"),
            ("knowledge_water_wet", "is water wet", "Yes, water is wet. Wetness is the property of a liquid sticking to a surface.", 0.75, "knowledge"),
            
            # Fallback
            ("unknown_fallback", "", "I'm not sure I understand. Could you rephrase that?", 0.3, "fallback"),
        ]
        
        conn = self._get_connection()
        for fragment_id, trigger, response, score, intent in seed_patterns:
            conn.execute("""
                INSERT OR IGNORE INTO response_patterns 
                (fragment_id, trigger_context, response_text, success_score, intent)
                VALUES (?, ?, ?, ?, ?)
            """, (fragment_id, trigger, response, score, intent))
        
        conn.commit()
        conn.close()
        print(f"  📚 Bootstrapped with {len(seed_patterns)} seed patterns")
    
    def add_pattern(
        self,
        trigger_context: str,
        response_text: str,
        success_score: float = 0.5,
        intent: str = "general"
    ) -> str:
        """
        Add a new response pattern (thread-safe).
        
        Args:
            trigger_context: Context that triggers this response
            response_text: Response text
            success_score: Initial success score (0.0-1.0)
            intent: Intent category
        
        Returns:
            fragment_id of the added pattern
        """
        conn = self._get_connection()
        
        # Generate unique ID using timestamp + random component
        # This prevents collisions in concurrent writes
        timestamp_ms = int(time.time() * 1000)
        random_suffix = random.randint(0, 9999)
        fragment_id = f"pattern_{intent}_{timestamp_ms}_{random_suffix}"
        
        # Insert pattern (will retry if there's a conflict)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                conn.execute("""
                    INSERT INTO response_patterns 
                    (fragment_id, trigger_context, response_text, success_score, intent)
                    VALUES (?, ?, ?, ?, ?)
                """, (fragment_id, trigger_context, response_text, success_score, intent))
                
                conn.commit()
                conn.close()
                return fragment_id
                
            except sqlite3.IntegrityError:
                # ID collision (very rare), generate new ID
                random_suffix = random.randint(0, 9999)
                fragment_id = f"pattern_{intent}_{timestamp_ms}_{random_suffix}"
                
                if attempt == max_retries - 1:
                    conn.close()
                    raise
        
        conn.close()
        return fragment_id
    
    def retrieve_patterns(
        self,
        context: str,
        topk: int = 5,
        min_score: float = 0.0
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Retrieve best matching patterns for context.
        
        Args:
            context: Query context
            topk: Number of results
            min_score: Minimum success score threshold
        
        Returns:
            List of (pattern, similarity_score) tuples
        """
        conn = self._get_connection()
        
        # Get all patterns above min_score
        cursor = conn.execute("""
            SELECT * FROM response_patterns 
            WHERE success_score >= ?
            ORDER BY success_score DESC
        """, (min_score,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        # Convert to ResponsePattern objects
        patterns = []
        for row in rows:
            embedding_cache = None
            if row['embedding_cache']:
                embedding_cache = json.loads(row['embedding_cache'])
            
            pattern = ResponsePattern(
                fragment_id=row['fragment_id'],
                trigger_context=row['trigger_context'],
                response_text=row['response_text'],
                success_score=row['success_score'],
                intent=row['intent'],
                usage_count=row['usage_count'],
                embedding_cache=embedding_cache
            )
            patterns.append(pattern)
        
        # Encode query context
        try:
            tokens = context.split()
            query_embedding = self.encoder.encode(tokens)
            if hasattr(query_embedding, 'numpy'):
                query_embedding = query_embedding.numpy()
            query_embedding = np.array(query_embedding).flatten()
        except Exception as e:
            print(f"  ⚠️  Encoding error for '{context}': {e}")
            print(f"      Tokens: {context.split()}")
            import traceback
            traceback.print_exc()
            return []
        
        # Compute similarities (hybrid: semantic + fuzzy matching)
        scored_patterns = []
        for pattern in patterns:
            # First try fuzzy matching for exact/near-exact matches
            fuzzy_score = 0.0
            if self.fuzzy_matcher:
                is_match, fuzzy_score = self.fuzzy_matcher.fuzzy_match(
                    context, 
                    pattern.trigger_context,
                    min_similarity=0.75
                )
                # If fuzzy match is strong, use it directly
                if is_match and fuzzy_score >= 0.9:
                    scored_patterns.append((pattern, float(fuzzy_score)))
                    continue
            
            # Get or compute embedding for semantic matching
            if pattern.embedding_cache:
                pattern_embedding = np.array(pattern.embedding_cache)
            else:
                try:
                    pattern_embedding = self.encoder.encode(pattern.trigger_context.split())
                    if hasattr(pattern_embedding, 'numpy'):
                        pattern_embedding = pattern_embedding.numpy()
                    pattern_embedding = np.array(pattern_embedding).flatten()
                except Exception:
                    continue
            
            # Compute cosine similarity
            norm_q = np.linalg.norm(query_embedding)
            norm_p = np.linalg.norm(pattern_embedding)
            
            if norm_q > 0 and norm_p > 0:
                semantic_sim = np.dot(query_embedding, pattern_embedding) / (norm_q * norm_p)
                
                # Adaptive weighting based on phrase length
                # For short phrases (<=3 words), trust fuzzy more
                # For long phrases, trust semantic more
                query_words = len(context.split())
                if query_words <= 3:
                    # Short phrase: 70% fuzzy, 30% semantic
                    combined_score = (0.7 * fuzzy_score) + (0.3 * semantic_sim)
                else:
                    # Long phrase: 40% fuzzy, 60% semantic
                    combined_score = (0.4 * fuzzy_score) + (0.6 * semantic_sim)
                
                scored_patterns.append((pattern, float(combined_score)))
        
        # Sort by similarity and take top-k
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        return scored_patterns[:topk]
    
    def update_success(
        self,
        fragment_id: str,
        outcome: bool,
        learning_rate: float = 0.1
    ):
        """
        Update pattern success score based on outcome (thread-safe).
        
        Args:
            fragment_id: Pattern ID to update
            outcome: True if successful, False if not
            learning_rate: How much to adjust score
        """
        conn = self._get_connection()
        
        # Get current score
        cursor = conn.execute(
            "SELECT success_score, usage_count FROM response_patterns WHERE fragment_id = ?",
            (fragment_id,)
        )
        row = cursor.fetchone()
        
        if row:
            current_score = row['success_score']
            usage_count = row['usage_count']
            
            # Update score with plasticity
            if outcome:
                new_score = current_score + learning_rate * (1.0 - current_score)
            else:
                new_score = current_score - learning_rate * current_score
            
            new_score = max(0.0, min(1.0, new_score))
            
            # Update in database
            conn.execute("""
                UPDATE response_patterns 
                SET success_score = ?, 
                    usage_count = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE fragment_id = ?
            """, (new_score, usage_count + 1, fragment_id))
            
            conn.commit()
        
        conn.close()
    
    def upvote(self, fragment_id: str, strength: float = 0.2):
        """
        Manual positive feedback - reward a good response.
        
        Use this when a response is particularly helpful or accurate.
        Stronger than normal positive feedback.
        
        Args:
            fragment_id: Pattern ID to upvote
            strength: How much to boost (default 0.2 = 20% toward 1.0)
        """
        self.update_success(fragment_id, outcome=True, learning_rate=strength)
        print(f"  👍 Upvoted pattern: {fragment_id}")
    
    def downvote(self, fragment_id: str, strength: float = 0.3):
        """
        Manual negative feedback - punish a bad response.
        
        Use this when a response is incorrect, unhelpful, or inappropriate.
        Stronger than normal negative feedback.
        
        Args:
            fragment_id: Pattern ID to downvote
            strength: How much to penalize (default 0.3 = 30% toward 0.0)
        """
        self.update_success(fragment_id, outcome=False, learning_rate=strength)
        print(f"  👎 Downvoted pattern: {fragment_id}")
    
    def mark_helpful(self, fragment_id: str):
        """Mark response as helpful (moderate positive feedback)."""
        self.update_success(fragment_id, outcome=True, learning_rate=0.15)
        print(f"  ✅ Marked helpful: {fragment_id}")
    
    def mark_not_helpful(self, fragment_id: str):
        """Mark response as not helpful (moderate negative feedback)."""
        self.update_success(fragment_id, outcome=False, learning_rate=0.15)
        print(f"  ❌ Marked not helpful: {fragment_id}")
    
    def get_stats(self) -> Dict:
        """Get comprehensive database statistics."""
        conn = self._get_connection()
        
        # Total patterns
        cursor = conn.execute("SELECT COUNT(*) FROM response_patterns")
        total = cursor.fetchone()[0]
        
        # By intent with detailed stats
        cursor = conn.execute("""
            SELECT intent, COUNT(*) as count, AVG(success_score) as avg_score
            FROM response_patterns 
            GROUP BY intent 
            ORDER BY count DESC
        """)
        by_intent = {}
        for row in cursor:
            by_intent[row[0]] = {
                'count': row[1],
                'avg_score': row[2]
            }
        
        # Overall stats
        cursor = conn.execute("""
            SELECT 
                AVG(success_score) as avg_score,
                MIN(success_score) as min_score,
                MAX(success_score) as max_score,
                AVG(usage_count) as avg_usage
            FROM response_patterns
        """)
        stats_row = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_patterns': total,
            'avg_success_score': stats_row[0] or 0.0,
            'min_success_score': stats_row[1] or 0.0,
            'max_success_score': stats_row[2] or 0.0,
            'avg_usage_count': stats_row[3] or 0.0,
            'by_intent': by_intent
        }

    
    @property
    def patterns(self) -> Dict[str, ResponsePattern]:
        """
        Get all patterns as dict (for API compatibility).
        Note: Returns snapshot, not live view.
        """
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM response_patterns")
        rows = cursor.fetchall()
        conn.close()
        
        patterns_dict = {}
        for row in rows:
            embedding_cache = None
            if row['embedding_cache']:
                embedding_cache = json.loads(row['embedding_cache'])
            
            pattern = ResponsePattern(
                fragment_id=row['fragment_id'],
                trigger_context=row['trigger_context'],
                response_text=row['response_text'],
                success_score=row['success_score'],
                intent=row['intent'],
                usage_count=row['usage_count'],
                embedding_cache=embedding_cache
            )
            patterns_dict[pattern.fragment_id] = pattern
        
        return patterns_dict
    
    def prune_low_quality_patterns(self, threshold: float = 0.1) -> int:
        """
        Remove patterns with very low success scores.
        
        Args:
            threshold: Minimum success score to keep (default 0.1)
            
        Returns:
            Number of patterns pruned
        """
        conn = self._get_connection()
        
        # Count patterns to be deleted
        cursor = conn.execute(
            "SELECT COUNT(*) FROM response_patterns WHERE success_score < ?",
            (threshold,)
        )
        count = cursor.fetchone()[0]
        
        # Delete them
        conn.execute(
            "DELETE FROM response_patterns WHERE success_score < ?",
            (threshold,)
        )
        conn.commit()
        conn.close()
        
        return count
    
    def apply_temporal_decay(self, half_life_days: int = 30):
        """
        Apply exponential decay to patterns based on age.
        
        Patterns that haven't been updated recently lose confidence.
        This helps outdated information naturally fade away.
        
        Args:
            half_life_days: Days for pattern to decay to 50% score
        """
        conn = self._get_connection()
        
        # Decay = 0.5 ^ (days_since_update / half_life)
        conn.execute("""
            UPDATE response_patterns
            SET success_score = success_score * 
                pow(0.5, (julianday('now') - julianday(updated_at)) / ?)
            WHERE julianday('now') - julianday(updated_at) > ?
        """, (half_life_days, half_life_days))
        
        conn.commit()
        conn.close()


