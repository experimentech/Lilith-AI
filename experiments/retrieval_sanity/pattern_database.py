"""
Database Schema for Pattern Storage

Creates SQLite database with relational tables for efficient pattern storage
and retrieval based on semantic features (keywords, topics, intents).
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime


class PatternDatabase:
    """SQLite database for storing and querying conversation patterns."""
    
    def __init__(self, db_path: str = "conversation_patterns.db"):
        """Initialize database connection and create schema if needed."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Return dict-like rows
        self.create_schema()
    
    def create_schema(self):
        """Create database tables and indexes."""
        cursor = self.conn.cursor()
        
        # Main patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fragment_id TEXT UNIQUE NOT NULL,
                trigger_context TEXT NOT NULL,
                response_text TEXT NOT NULL,
                intent TEXT NOT NULL,
                success_score REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Keywords extracted from patterns (many-to-many)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id INTEGER NOT NULL,
                keyword TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                source TEXT,  -- 'trigger' or 'response'
                FOREIGN KEY (pattern_id) REFERENCES patterns(id) ON DELETE CASCADE,
                UNIQUE(pattern_id, keyword, source)
            )
        """)
        
        # Topics associated with patterns (many-to-many)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                FOREIGN KEY (pattern_id) REFERENCES patterns(id) ON DELETE CASCADE,
                UNIQUE(pattern_id, topic)
            )
        """)
        
        # Conversation sequences - which patterns follow which
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id INTEGER NOT NULL,
                follows_pattern_id INTEGER NOT NULL,
                frequency INTEGER DEFAULT 1,
                FOREIGN KEY (pattern_id) REFERENCES patterns(id) ON DELETE CASCADE,
                FOREIGN KEY (follows_pattern_id) REFERENCES patterns(id) ON DELETE CASCADE,
                UNIQUE(pattern_id, follows_pattern_id)
            )
        """)
        
        # Create indexes for fast lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_intent ON patterns(intent)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_success ON patterns(success_score DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON pattern_keywords(keyword)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_keywords_pattern ON pattern_keywords(pattern_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_topic ON pattern_topics(topic)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_pattern ON pattern_topics(pattern_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sequences_follows ON pattern_sequences(follows_pattern_id)")
        
        self.conn.commit()
        print("✅ Database schema created successfully")
    
    def insert_pattern(
        self,
        fragment_id: str,
        trigger_context: str,
        response_text: str,
        intent: str,
        success_score: float = 0.5,
        keywords: Optional[List[Tuple[str, str, float]]] = None,
        topics: Optional[List[Tuple[str, float]]] = None
    ) -> int:
        """
        Insert a new pattern into the database.
        
        Args:
            fragment_id: Unique identifier for the pattern
            trigger_context: Context that triggers this response
            response_text: The response text
            intent: Classified intent (greeting, question_info, etc.)
            success_score: How successful this pattern has been
            keywords: List of (keyword, source, weight) tuples
            topics: List of (topic, strength) tuples
            
        Returns:
            Pattern ID
        """
        cursor = self.conn.cursor()
        
        try:
            # Insert main pattern
            cursor.execute("""
                INSERT INTO patterns (fragment_id, trigger_context, response_text, intent, success_score)
                VALUES (?, ?, ?, ?, ?)
            """, (fragment_id, trigger_context, response_text, intent, success_score))
            
            pattern_id = cursor.lastrowid
            
            # Insert keywords if provided
            if keywords:
                for keyword, source, weight in keywords:
                    cursor.execute("""
                        INSERT OR IGNORE INTO pattern_keywords (pattern_id, keyword, source, weight)
                        VALUES (?, ?, ?, ?)
                    """, (pattern_id, keyword.lower(), source, weight))
            
            # Insert topics if provided
            if topics:
                for topic, strength in topics:
                    cursor.execute("""
                        INSERT OR IGNORE INTO pattern_topics (pattern_id, topic, strength)
                        VALUES (?, ?, ?)
                    """, (pattern_id, topic, strength))
            
            self.conn.commit()
            return pattern_id
            
        except sqlite3.IntegrityError as e:
            self.conn.rollback()
            print(f"⚠️  Pattern {fragment_id} already exists")
            # Return existing pattern ID
            cursor.execute("SELECT id FROM patterns WHERE fragment_id = ?", (fragment_id,))
            row = cursor.fetchone()
            return row[0] if row else None
    
    def query_patterns(
        self,
        intent: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        min_success: float = 0.0,
        limit: int = 10,
        keyword_source: Optional[str] = None  # 'trigger', 'response', or None for both
    ) -> List[Dict]:
        """
        Query patterns based on semantic features.
        
        Args:
            intent: Filter by intent
            keywords: Filter by keywords (OR logic)
            topics: Filter by topics (OR logic)
            min_success: Minimum success score
            limit: Maximum results
            keyword_source: Filter keywords by source ('trigger' or 'response')
            
        Returns:
            List of pattern dictionaries
        """
        cursor = self.conn.cursor()
        
        query = "SELECT DISTINCT p.* FROM patterns p"
        where_clauses = []
        params = []
        
        # Join with keywords if needed
        if keywords:
            query += " JOIN pattern_keywords pk ON p.id = pk.pattern_id"
            placeholders = ','.join('?' * len(keywords))
            where_clauses.append(f"pk.keyword IN ({placeholders})")
            params.extend([kw.lower() for kw in keywords])
            
            # Filter by keyword source (trigger vs response)
            if keyword_source:
                where_clauses.append("pk.source = ?")
                params.append(keyword_source)
        
        # Join with topics if needed
        if topics:
            query += " JOIN pattern_topics pt ON p.id = pt.pattern_id"
            placeholders = ','.join('?' * len(topics))
            where_clauses.append(f"pt.topic IN ({placeholders})")
            params.extend(topics)
        
        # Add intent filter
        if intent:
            where_clauses.append("p.intent = ?")
            params.append(intent)
        
        # Add success filter
        where_clauses.append("p.success_score >= ?")
        params.append(min_success)
        
        # Build WHERE clause
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        # Order by success and limit
        query += " ORDER BY p.success_score DESC, p.usage_count DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        patterns = []
        for row in cursor.fetchall():
            patterns.append(dict(row))
        
        return patterns
    
    def update_success(self, pattern_id: int, new_score: float):
        """Update pattern success score and increment usage count."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE patterns 
            SET success_score = ?, 
                usage_count = usage_count + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (new_score, pattern_id))
        self.conn.commit()
    
    def record_sequence(self, pattern_id: int, follows_pattern_id: int):
        """Record that pattern_id followed follows_pattern_id in conversation."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO pattern_sequences (pattern_id, follows_pattern_id, frequency)
            VALUES (?, ?, 1)
            ON CONFLICT(pattern_id, follows_pattern_id) 
            DO UPDATE SET frequency = frequency + 1
        """, (pattern_id, follows_pattern_id))
        self.conn.commit()
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM patterns")
        total_patterns = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT keyword) FROM pattern_keywords")
        total_keywords = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT topic) FROM pattern_topics")
        total_topics = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT intent) FROM patterns")
        total_intents = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(success_score) FROM patterns")
        avg_success = cursor.fetchone()[0] or 0.0
        
        return {
            "total_patterns": total_patterns,
            "total_keywords": total_keywords,
            "total_topics": total_topics,
            "total_intents": total_intents,
            "avg_success": avg_success,
            "learned_patterns": max(total_patterns, 0),
        }
    
    def calculate_idf_scores(self) -> Dict[str, float]:
        """
        Calculate IDF (Inverse Document Frequency) scores for all keywords.
        
        Brain insight: Rare words are more distinctive and informative!
        - 'hi' appears in 4 patterns → high IDF (distinctive greeting signal)
        - 'you' appears in 500 patterns → low IDF (ubiquitous, less informative)
        
        Formula: IDF(word) = log(total_patterns / patterns_containing_word)
        
        Returns:
            Dict mapping keyword → IDF score
        """
        import math
        
        cursor = self.conn.cursor()
        
        # Get total number of patterns (corpus size)
        cursor.execute("SELECT COUNT(*) FROM patterns")
        total_patterns = cursor.fetchone()[0]
        
        if total_patterns == 0:
            return {}
        
        # Count how many patterns each keyword appears in
        cursor.execute("""
            SELECT keyword, COUNT(DISTINCT pattern_id) as doc_freq
            FROM pattern_keywords
            GROUP BY keyword
        """)
        
        idf_scores = {}
        for row in cursor.fetchall():
            keyword = row['keyword']
            doc_freq = row['doc_freq']
            
            # IDF = log(N / df) where N = total docs, df = document frequency
            # Add 1 to avoid log(0) and division by zero
            idf = math.log((total_patterns + 1) / (doc_freq + 1))
            idf_scores[keyword] = idf
        
        return idf_scores
    
    def get_keyword_idf(self, keyword: str) -> float:
        """Get IDF score for a specific keyword."""
        cursor = self.conn.cursor()
        
        # Total patterns
        cursor.execute("SELECT COUNT(*) FROM patterns")
        total_patterns = cursor.fetchone()[0]
        
        if total_patterns == 0:
            return 0.0
        
        # How many patterns contain this keyword
        cursor.execute("""
            SELECT COUNT(DISTINCT pattern_id)
            FROM pattern_keywords
            WHERE keyword = ?
        """, (keyword,))
        
        doc_freq = cursor.fetchone()[0]
        
        if doc_freq == 0:
            return 0.0
        
        import math
        return math.log((total_patterns + 1) / (doc_freq + 1))
    
    def close(self):
        """Close database connection."""
        self.conn.close()


def extract_keywords(text: str, source: str = "trigger") -> List[Tuple[str, str, float]]:
    """
    Extract keywords from text, filtering stopwords.
    
    Brain-inspired: Keep conversational signals (greetings, farewells, polite words)
    that are often filtered as stopwords in document retrieval but are CRITICAL
    for dialogue pattern matching.
    
    Returns:
        List of (keyword, source, weight) tuples
    """
    # Content stopwords only - NOT conversational signals!
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'can', 'could', 'may', 'might', 'must', 
        'it', 'them', 'this', 'that', 'these', 'those', 'to', 'of', 'in',
        'on', 'at', 'by', 'for', 'with', 'about', 'as', 'from', 'up', 'down',
        'out', 'off', 'over', 'under', 'again', 'then', 'once', 'here', 'there',
        'when', 'where', 'why', 'all', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'don', 'm', 'll', 've'
    }
    
    # KEEP these conversational signals (removed from stopwords):
    # - 'hi', 'hello', 'hey', 'goodbye', 'bye' (greetings/farewells)
    # - 'thanks', 'thank' (gratitude)
    # - 'i', 'you', 'we', 'he', 'she' (personal pronouns matter in conversation!)
    # - 'how' (questions: "how are you?", "how's it going?")
    
    words = text.lower().split()
    keywords = []
    
    for word in words:
        # Remove punctuation
        word = ''.join(c for c in word if c.isalnum())
        
        # Skip stopwords and short words (but keep conversational signals!)
        if word and len(word) > 1 and word not in stopwords:  # Reduced from >2 to >1 for "hi"
            keywords.append((word, source, 1.0))
    
    return keywords


def migrate_from_json(json_path: str, db_path: str = "conversation_patterns.db"):
    """
    Migrate patterns from JSON file to SQLite database.
    
    Args:
        json_path: Path to conversation_patterns.json
        db_path: Path to SQLite database file
    """
    print(f"📦 Migrating patterns from {json_path} to {db_path}...")
    
    # Load JSON patterns
    with open(json_path, 'r') as f:
        patterns_list = json.load(f)
    
    print(f"   Found {len(patterns_list)} patterns in JSON")
    
    # Create database
    db = PatternDatabase(db_path)
    
    # Insert each pattern
    migrated = 0
    for pattern in patterns_list:
        # Extract keywords from trigger and response
        trigger_keywords = extract_keywords(pattern['trigger_context'], 'trigger')
        response_keywords = extract_keywords(pattern['response_text'], 'response')
        all_keywords = trigger_keywords + response_keywords
        
        # Insert pattern
        pattern_id = db.insert_pattern(
            fragment_id=pattern['fragment_id'],
            trigger_context=pattern['trigger_context'],
            response_text=pattern['response_text'],
            intent=pattern.get('intent', 'statement'),
            success_score=pattern.get('success_score', 0.5),
            keywords=all_keywords
        )
        
        if pattern_id:
            migrated += 1
        
        if migrated % 100 == 0:
            print(f"   Migrated {migrated}/{len(patterns_list)} patterns...")
    
    # Print statistics
    stats = db.get_stats()
    print()
    print("="*70)
    print("✅ MIGRATION COMPLETE")
    print("="*70)
    print(f"  Total patterns: {stats['total_patterns']}")
    print(f"  Unique keywords: {stats['total_keywords']}")
    print(f"  Unique topics: {stats['total_topics']}")
    print(f"  Intent categories: {stats['total_intents']}")
    print(f"  Average success: {stats['avg_success']:.2f}")
    print("="*70)
    
    db.close()
    return stats


if __name__ == "__main__":
    import sys
    
    # Migration command
    if len(sys.argv) > 1 and sys.argv[1] == "migrate":
        json_file = sys.argv[2] if len(sys.argv) > 2 else "conversation_patterns.json"
        migrate_from_json(json_file)
    else:
        # Test database creation
        print("Creating test database...")
        db = PatternDatabase("test_patterns.db")
        
        # Test insert
        pattern_id = db.insert_pattern(
            fragment_id="test_greeting_1",
            trigger_context="hello how are you",
            response_text="I'm doing great, thanks for asking!",
            intent="greeting",
            keywords=[
                ("hello", "trigger", 1.0),
                ("doing", "response", 1.0),
                ("great", "response", 1.0)
            ],
            topics=[("greeting", 1.0)]
        )
        
        print(f"Inserted pattern with ID: {pattern_id}")
        
        # Test query
        results = db.query_patterns(keywords=["hello"], limit=5)
        print(f"\nQuery results for 'hello': {len(results)} patterns")
        for result in results:
            print(f"  - {result['response_text'][:50]}...")
        
        # Test stats
        stats = db.get_stats()
        print(f"\nDatabase stats: {stats}")
        
        db.close()
        print("\n✅ Test complete! Database: test_patterns.db")
