"""
Vocabulary Tracker - Track word usage and build semantic vocabulary index

Extracts and tracks novel terms from learned content (Wikipedia, etc.) to:
- Build vocabulary frequency index
- Identify technical vs common terms
- Track word co-occurrence patterns
- Support vocabulary-enhanced embeddings

Phase C: Vocabulary Expansion
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, Counter
import re
import sqlite3
import json
from pathlib import Path


@dataclass
class VocabularyEntry:
    """A tracked vocabulary term"""
    term: str
    frequency: int = 1
    contexts: List[str] = field(default_factory=list)  # Sentences where it appears
    related_terms: Set[str] = field(default_factory=set)  # Co-occurring terms
    is_technical: bool = False  # Technical/domain-specific term
    first_seen_source: str = "unknown"  # Where we first learned it


class VocabularyTracker:
    """
    Track vocabulary from learned content.
    
    Features:
    - Term frequency tracking
    - Context extraction (n-gram windows)
    - Co-occurrence analysis
    - Technical term identification
    - SQLite persistence
    """
    
    def __init__(self, db_path: str):
        """
        Initialize vocabulary tracker.
        
        Args:
            db_path: Path to SQLite database for persistence
        """
        self.db_path = db_path
        self._init_database()
        
        # Common stop words (extended list)
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'this', 'but', 'or', 'not', 'can',
            'been', 'have', 'had', 'were', 'what', 'when', 'where', 'who',
            'which', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
            'more', 'most', 'other', 'some', 'such', 'than', 'very', 'also',
            'any', 'because', 'does', 'did', 'do', 'may', 'might', 'must',
            'should', 'could', 'would', 'their', 'them', 'they', 'his', 'her',
            'she', 'you', 'your', 'we', 'our', 'am', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'up', 'down', 'out',
            'over', 'under', 'again', 'further', 'then', 'once'
        }
        
        # Technical indicators (words that suggest technical content)
        self.technical_indicators = {
            'algorithm', 'function', 'method', 'system', 'process', 'data',
            'structure', 'type', 'language', 'framework', 'library', 'compiler',
            'runtime', 'memory', 'performance', 'optimization', 'implementation',
            'interface', 'protocol', 'architecture', 'paradigm', 'concurrency',
            'parallel', 'distributed', 'scalable', 'efficient', 'computational'
        }
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Vocabulary table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vocabulary (
                    term TEXT PRIMARY KEY,
                    frequency INTEGER DEFAULT 1,
                    contexts TEXT,  -- JSON array of context strings
                    related_terms TEXT,  -- JSON array of co-occurring terms
                    is_technical BOOLEAN DEFAULT 0,
                    first_seen_source TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Co-occurrence table (for building semantic relationships)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cooccurrence (
                    term1 TEXT,
                    term2 TEXT,
                    count INTEGER DEFAULT 1,
                    PRIMARY KEY (term1, term2)
                )
            ''')
            
            conn.commit()
    
    def track_text(
        self,
        text: str,
        source: str = "wikipedia",
        max_contexts_per_term: int = 5
    ) -> Dict[str, VocabularyEntry]:
        """
        Extract and track vocabulary from text.
        
        Args:
            text: Text to analyze
            source: Source of the text (for tracking)
            max_contexts_per_term: Max context examples to store per term
            
        Returns:
            Dictionary of term -> VocabularyEntry for new/updated terms
        """
        # Tokenize and normalize
        tokens = self._tokenize(text)
        
        # Extract sentences for context
        sentences = self._split_sentences(text)
        
        # Track term frequencies
        term_freq = Counter(tokens)
        
        # Build co-occurrence matrix (window size = 5)
        cooccurrences = self._extract_cooccurrences(tokens, window_size=5)
        
        # Track each term
        tracked = {}
        
        for term, freq in term_freq.items():
            # Skip stop words and very short terms
            if term in self.stop_words or len(term) < 3:
                continue
            
            # Get existing entry or create new
            entry = self._get_or_create_entry(term, source)
            
            # Update frequency
            entry.frequency += freq
            
            # Extract contexts (sentences containing this term)
            term_contexts = [
                s for s in sentences 
                if term in s.lower()
            ][:max_contexts_per_term]
            
            # Add new contexts (avoid duplicates)
            for ctx in term_contexts:
                if ctx not in entry.contexts:
                    entry.contexts.append(ctx)
                    if len(entry.contexts) > max_contexts_per_term:
                        entry.contexts.pop(0)  # Keep most recent
            
            # Update related terms (co-occurring)
            if term in cooccurrences:
                entry.related_terms.update(cooccurrences[term])
            
            # Determine if technical
            entry.is_technical = self._is_technical_term(term, term_contexts)
            
            # Save to database
            self._save_entry(entry)
            
            tracked[term] = entry
        
        # Save co-occurrences
        self._save_cooccurrences(cooccurrences)
        
        return tracked
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into normalized terms.
        
        Handles:
        - Lowercase normalization
        - Compound terms (e.g., "machine learning")
        - Hyphenated terms
        """
        # Convert to lowercase
        text = text.lower()
        
        # Extract words (including hyphenated)
        words = re.findall(r'\b[\w-]+\b', text)
        
        # Also extract bigrams for compound terms
        tokens = []
        for i, word in enumerate(words):
            # Add single word
            tokens.append(word)
            
            # Add bigram if next word exists
            if i < len(words) - 1:
                bigram = f"{word} {words[i+1]}"
                # Only add if it looks like a compound term (both words substantive)
                if (len(word) > 3 and len(words[i+1]) > 3 and 
                    word not in self.stop_words and words[i+1] not in self.stop_words):
                    tokens.append(bigram)
        
        return tokens
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_cooccurrences(
        self,
        tokens: List[str],
        window_size: int = 5
    ) -> Dict[str, Set[str]]:
        """
        Extract word co-occurrences within a window.
        
        Args:
            tokens: List of tokens
            window_size: Co-occurrence window size
            
        Returns:
            Dict of term -> set of co-occurring terms
        """
        cooccur = defaultdict(set)
        
        for i, token in enumerate(tokens):
            if token in self.stop_words or len(token) < 3:
                continue
            
            # Get window around this token
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            
            for j in range(start, end):
                if i == j:
                    continue
                
                other = tokens[j]
                if other not in self.stop_words and len(other) >= 3:
                    cooccur[token].add(other)
        
        return cooccur
    
    def _is_technical_term(self, term: str, contexts: List[str]) -> bool:
        """
        Determine if a term is technical/domain-specific.
        
        Heuristics:
        - Contains technical indicators in contexts
        - Capitalized (proper noun, likely technical)
        - Contains numbers or special characters
        - Compound term with technical words
        """
        # Check for capitalization (except first word of sentence)
        if term[0].isupper() and len(term) > 1:
            return True
        
        # Check for numbers/special chars (e.g., "C++", "SHA-256")
        if re.search(r'[\d+#-]', term):
            return True
        
        # Check if any technical indicators appear in contexts
        context_text = ' '.join(contexts).lower()
        for indicator in self.technical_indicators:
            if indicator in context_text:
                return True
        
        # Check if term itself contains technical words
        term_words = term.split()
        for word in term_words:
            if word in self.technical_indicators:
                return True
        
        return False
    
    def _get_or_create_entry(self, term: str, source: str) -> VocabularyEntry:
        """Get existing entry from database or create new."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM vocabulary WHERE term = ?', (term,))
            row = cursor.fetchone()
            
            if row:
                return VocabularyEntry(
                    term=row[0],
                    frequency=row[1],
                    contexts=json.loads(row[2]) if row[2] else [],
                    related_terms=set(json.loads(row[3])) if row[3] else set(),
                    is_technical=bool(row[4]),
                    first_seen_source=row[5]
                )
            else:
                return VocabularyEntry(
                    term=term,
                    frequency=0,
                    contexts=[],
                    related_terms=set(),
                    is_technical=False,
                    first_seen_source=source
                )
    
    def _save_entry(self, entry: VocabularyEntry):
        """Save vocabulary entry to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO vocabulary 
                (term, frequency, contexts, related_terms, is_technical, first_seen_source)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                entry.term,
                entry.frequency,
                json.dumps(entry.contexts),
                json.dumps(list(entry.related_terms)),
                entry.is_technical,
                entry.first_seen_source
            ))
            conn.commit()
    
    def _save_cooccurrences(self, cooccur: Dict[str, Set[str]]):
        """Save co-occurrence data to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for term1, related in cooccur.items():
                for term2 in related:
                    # Ensure alphabetical order for consistency
                    t1, t2 = sorted([term1, term2])
                    
                    cursor.execute('''
                        INSERT INTO cooccurrence (term1, term2, count)
                        VALUES (?, ?, 1)
                        ON CONFLICT(term1, term2) DO UPDATE SET count = count + 1
                    ''', (t1, t2))
            
            conn.commit()
    
    def get_vocabulary_stats(self) -> Dict:
        """Get vocabulary statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total terms
            cursor.execute('SELECT COUNT(*) FROM vocabulary')
            total = cursor.fetchone()[0]
            
            # Technical terms
            cursor.execute('SELECT COUNT(*) FROM vocabulary WHERE is_technical = 1')
            technical = cursor.fetchone()[0]
            
            # Most frequent
            cursor.execute('SELECT term, frequency FROM vocabulary ORDER BY frequency DESC LIMIT 10')
            top_frequent = cursor.fetchall()
            
            # Most technical (by context indicators)
            cursor.execute('''
                SELECT term, frequency FROM vocabulary 
                WHERE is_technical = 1 
                ORDER BY frequency DESC 
                LIMIT 10
            ''')
            top_technical = cursor.fetchall()
            
            return {
                'total_terms': total,
                'technical_terms': technical,
                'common_terms': total - technical,
                'top_frequent': top_frequent,
                'top_technical': top_technical
            }
    
    def get_related_terms(self, term: str, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get terms that co-occur with the given term.
        
        Args:
            term: Term to find relations for
            limit: Max number of related terms
            
        Returns:
            List of (related_term, co-occurrence_count) tuples
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Query both directions
            cursor.execute('''
                SELECT term2, count FROM cooccurrence WHERE term1 = ?
                UNION
                SELECT term1, count FROM cooccurrence WHERE term2 = ?
                ORDER BY count DESC
                LIMIT ?
            ''', (term, term, limit))
            
            return cursor.fetchall()


def demo():
    """Demo vocabulary tracker."""
    import tempfile
    import os
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        db_path = f.name
    
    try:
        tracker = VocabularyTracker(db_path)
        
        # Sample text
        text = """
        Rust is a general-purpose programming language emphasizing performance, 
        type safety, and concurrency. It enforces memory safety without using 
        garbage collection. The Rust compiler performs borrow checking to ensure 
        memory safety at compile time.
        """
        
        print("Tracking vocabulary from text...")
        print("-" * 60)
        tracked = tracker.track_text(text, source="demo")
        
        print(f"\nTracked {len(tracked)} terms:\n")
        for term, entry in sorted(tracked.items(), key=lambda x: x[1].frequency, reverse=True):
            print(f"  {term}:")
            print(f"    Frequency: {entry.frequency}")
            print(f"    Technical: {entry.is_technical}")
            if entry.related_terms:
                print(f"    Related: {list(entry.related_terms)[:5]}")
        
        print("\n" + "=" * 60)
        print("Vocabulary Statistics:")
        print("=" * 60)
        stats = tracker.get_vocabulary_stats()
        print(f"Total terms: {stats['total_terms']}")
        print(f"Technical terms: {stats['technical_terms']}")
        print(f"Common terms: {stats['common_terms']}")
        
    finally:
        # Cleanup
        os.unlink(db_path)


if __name__ == "__main__":
    demo()
