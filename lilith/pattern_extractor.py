"""
Pattern Extractor - Learn syntactic patterns from text

Extracts sentence structure patterns from Wikipedia and other text sources
to improve response generation. Learns templates like:
  - "[SUBJECT] is a [TYPE] that [PROPERTY]"
  - "[SUBJECT] was developed in [TIME] by [CREATOR]"
  - "[SUBJECT] is used for [PURPOSE]"

Phase D: Syntactic Pattern Learning
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter, defaultdict
import re
import sqlite3
import json
from pathlib import Path


@dataclass
class SyntacticPattern:
    """A learned sentence structure pattern"""
    pattern_id: str
    template: str                    # "[SUBJECT] is a [TYPE]"
    slots: List[str]                 # ["SUBJECT", "TYPE"]
    examples: List[str]              # Real sentences matching this pattern
    frequency: int = 1               # How often we've seen this pattern
    confidence: float = 0.75         # How reliable it is
    intent: str = "statement"        # statement, definition, explanation


@dataclass
class PatternMatch:
    """Result of matching text to a pattern"""
    pattern: SyntacticPattern
    slot_values: Dict[str, str]      # SUBJECT → "Rust", TYPE → "programming language"
    confidence: float
    

class PatternExtractor:
    """
    Extract and learn syntactic patterns from text.
    
    Features:
    - Pattern extraction from example sentences
    - Template generalization
    - Slot identification and filling
    - SQLite persistence
    """
    
    def __init__(self, db_path: str):
        """
        Initialize pattern extractor.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._init_database()
        
        # Common sentence patterns (bootstrapped)
        self.bootstrap_patterns = [
            {
                "template": "[SUBJECT] is a [TYPE]",
                "regex": r"^(.+?)\s+is\s+a(?:n)?\s+(.+?)(?:\.|$)",
                "slots": ["SUBJECT", "TYPE"],
                "intent": "definition",
                "examples": [
                    "Rust is a programming language",
                    "Python is an interpreted language"
                ]
            },
            {
                "template": "[SUBJECT] is a [TYPE] that [PROPERTY]",
                "regex": r"^(.+?)\s+is\s+a(?:n)?\s+(.+?)\s+that\s+(.+?)(?:\.|$)",
                "slots": ["SUBJECT", "TYPE", "PROPERTY"],
                "intent": "definition",
                "examples": [
                    "Rust is a language that emphasizes safety",
                    "Python is a language that focuses on readability"
                ]
            },
            {
                "template": "[SUBJECT] is used for [PURPOSE]",
                "regex": r"^(.+?)\s+is\s+used\s+for\s+(.+?)(?:\.|$)",
                "slots": ["SUBJECT", "PURPOSE"],
                "intent": "purpose",
                "examples": [
                    "Machine learning is used for prediction",
                    "Blockchain is used for secure transactions"
                ]
            },
            {
                "template": "[SUBJECT] was [ACTION] by [AGENT]",
                "regex": r"^(.+?)\s+was\s+(created|developed|designed|invented)\s+by\s+(.+?)(?:\.|$)",
                "slots": ["SUBJECT", "ACTION", "AGENT"],
                "intent": "history",
                "examples": [
                    "Python was created by Guido van Rossum",
                    "Rust was developed by Mozilla Research"
                ]
            },
            {
                "template": "[SUBJECT] [VERB] [OBJECT]",
                "regex": r"^(.+?)\s+(enables?|allows?|provides?|supports?)\s+(.+?)(?:\.|$)",
                "slots": ["SUBJECT", "VERB", "OBJECT"],
                "intent": "capability",
                "examples": [
                    "Machine learning enables pattern recognition",
                    "Blockchain provides secure transactions"
                ]
            },
            {
                "template": "[SUBJECT] has [PROPERTY]",
                "regex": r"^(.+?)\s+has\s+(.+?)(?:\.|$)",
                "slots": ["SUBJECT", "PROPERTY"],
                "intent": "property",
                "examples": [
                    "Rust has memory safety",
                    "Python has dynamic typing"
                ]
            },
            {
                "template": "[SUBJECT] emphasizing [FEATURES]",
                "regex": r"^(.+?)\s+emphasizing\s+(.+?)(?:\.|$)",
                "slots": ["SUBJECT", "FEATURES"],
                "intent": "emphasis",
                "examples": [
                    "Rust emphasizing performance and safety",
                    "Python emphasizing code readability"
                ]
            }
        ]
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    template TEXT NOT NULL,
                    slots TEXT,              -- JSON array of slot names
                    examples TEXT,           -- JSON array of example sentences
                    frequency INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.75,
                    intent TEXT DEFAULT 'statement',
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Pattern matches table (track which patterns matched which text)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_matches (
                    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id TEXT,
                    source_text TEXT,
                    slot_values TEXT,        -- JSON object of slot→value
                    confidence REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pattern_id) REFERENCES patterns(pattern_id)
                )
            ''')
            
            conn.commit()
    
    def extract_patterns(
        self,
        text: str,
        source: str = "wikipedia"
    ) -> List[PatternMatch]:
        """
        Extract patterns from text.
        
        Args:
            text: Text to analyze
            source: Source of text
            
        Returns:
            List of pattern matches found
        """
        matches = []
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        # Try to match each sentence against known patterns
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Try bootstrap patterns
            for pattern_def in self.bootstrap_patterns:
                match = self._try_match_pattern(sentence, pattern_def)
                if match:
                    # Store/update pattern
                    pattern = self._get_or_create_pattern(pattern_def, sentence)
                    
                    # Record match
                    pattern_match = PatternMatch(
                        pattern=pattern,
                        slot_values=match,
                        confidence=0.85
                    )
                    matches.append(pattern_match)
                    
                    # Save to database
                    self._save_pattern_match(pattern, match, sentence)
                    
                    break  # Only match first pattern per sentence
        
        return matches
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _try_match_pattern(
        self,
        sentence: str,
        pattern_def: Dict
    ) -> Optional[Dict[str, str]]:
        """
        Try to match a sentence against a pattern definition.
        
        Args:
            sentence: Sentence to match
            pattern_def: Pattern definition with regex and slots
            
        Returns:
            Dict of slot→value if match, None otherwise
        """
        regex = pattern_def["regex"]
        slots = pattern_def["slots"]
        
        match = re.match(regex, sentence, re.IGNORECASE)
        if not match:
            return None
        
        # Extract slot values
        slot_values = {}
        for i, slot in enumerate(slots):
            value = match.group(i + 1).strip()
            slot_values[slot] = value
        
        return slot_values
    
    def _get_or_create_pattern(
        self,
        pattern_def: Dict,
        example: str
    ) -> SyntacticPattern:
        """
        Get existing pattern or create new one.
        
        Args:
            pattern_def: Pattern definition
            example: Example sentence
            
        Returns:
            SyntacticPattern instance
        """
        template = pattern_def["template"]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Try to find existing pattern by template
            cursor.execute('SELECT * FROM patterns WHERE template = ?', (template,))
            row = cursor.fetchone()
            
            if row:
                # Update frequency and add example
                pattern_id = row[0]
                examples = json.loads(row[3]) if row[3] else []
                if example not in examples:
                    examples.append(example)
                    if len(examples) > 10:  # Keep max 10 examples
                        examples.pop(0)
                
                frequency = row[4] + 1
                
                cursor.execute('''
                    UPDATE patterns 
                    SET examples = ?, frequency = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE pattern_id = ?
                ''', (json.dumps(examples), frequency, pattern_id))
                conn.commit()
                
                return SyntacticPattern(
                    pattern_id=pattern_id,
                    template=template,
                    slots=pattern_def["slots"],
                    examples=examples,
                    frequency=frequency,
                    confidence=row[5],
                    intent=pattern_def["intent"]
                )
            else:
                # Create new pattern
                pattern_id = f"pattern_{pattern_def['intent']}_{hash(template) % 10000:04d}"
                
                cursor.execute('''
                    INSERT INTO patterns 
                    (pattern_id, template, slots, examples, frequency, confidence, intent)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern_id,
                    template,
                    json.dumps(pattern_def["slots"]),
                    json.dumps([example]),
                    1,
                    0.75,
                    pattern_def["intent"]
                ))
                conn.commit()
                
                return SyntacticPattern(
                    pattern_id=pattern_id,
                    template=template,
                    slots=pattern_def["slots"],
                    examples=[example],
                    frequency=1,
                    confidence=0.75,
                    intent=pattern_def["intent"]
                )
    
    def _save_pattern_match(
        self,
        pattern: SyntacticPattern,
        slot_values: Dict[str, str],
        source_text: str
    ):
        """Save a pattern match to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO pattern_matches (pattern_id, source_text, slot_values, confidence)
                VALUES (?, ?, ?, ?)
            ''', (
                pattern.pattern_id,
                source_text,
                json.dumps(slot_values),
                pattern.confidence
            ))
            conn.commit()
    
    def get_pattern_stats(self) -> Dict:
        """Get pattern statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total patterns
            cursor.execute('SELECT COUNT(*) FROM patterns')
            total = cursor.fetchone()[0]
            
            # Patterns by intent
            cursor.execute('SELECT intent, COUNT(*) FROM patterns GROUP BY intent')
            by_intent = dict(cursor.fetchall())
            
            # Most frequent patterns
            cursor.execute('''
                SELECT template, frequency, intent 
                FROM patterns 
                ORDER BY frequency DESC 
                LIMIT 10
            ''')
            top_patterns = cursor.fetchall()
            
            # Total matches
            cursor.execute('SELECT COUNT(*) FROM pattern_matches')
            total_matches = cursor.fetchone()[0]
            
            return {
                'total_patterns': total,
                'by_intent': by_intent,
                'top_patterns': top_patterns,
                'total_matches': total_matches
            }
    
    def get_patterns_by_intent(self, intent: str) -> List[SyntacticPattern]:
        """Get all patterns for a specific intent."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM patterns WHERE intent = ?', (intent,))
            rows = cursor.fetchall()
            
            patterns = []
            for row in rows:
                patterns.append(SyntacticPattern(
                    pattern_id=row[0],
                    template=row[1],
                    slots=json.loads(row[2]) if row[2] else [],
                    examples=json.loads(row[3]) if row[3] else [],
                    frequency=row[4],
                    confidence=row[5],
                    intent=row[6]
                ))
            
            return patterns
    
    def generate_from_pattern(
        self,
        pattern: SyntacticPattern,
        slot_values: Dict[str, str]
    ) -> str:
        """
        Generate text from a pattern and slot values.
        
        Args:
            pattern: Pattern template to use
            slot_values: Values for each slot
            
        Returns:
            Generated text
        """
        text = pattern.template
        
        # Replace each slot with its value
        for slot, value in slot_values.items():
            placeholder = f"[{slot}]"
            text = text.replace(placeholder, value)
        
        return text


def demo():
    """Demo pattern extractor."""
    import tempfile
    import os
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        db_path = f.name
    
    try:
        extractor = PatternExtractor(db_path)
        
        # Sample text
        text = """
        Rust is a general-purpose programming language. It emphasizes performance,
        type safety, and concurrency. Rust was developed by Mozilla Research.
        Machine learning is used for pattern recognition. Python is an interpreted
        language that focuses on code readability.
        """
        
        print("Extracting patterns from text...")
        print("-" * 60)
        matches = extractor.extract_patterns(text)
        
        print(f"\nFound {len(matches)} pattern matches:\n")
        for match in matches:
            print(f"Pattern: {match.pattern.template}")
            print(f"Intent: {match.pattern.intent}")
            print(f"Slots: {match.slot_values}")
            print(f"Generated: {extractor.generate_from_pattern(match.pattern, match.slot_values)}")
            print()
        
        print("=" * 60)
        print("Pattern Statistics:")
        print("=" * 60)
        stats = extractor.get_pattern_stats()
        print(f"Total patterns: {stats['total_patterns']}")
        print(f"Total matches: {stats['total_matches']}")
        print(f"\nBy intent: {stats['by_intent']}")
        print(f"\nTop patterns:")
        for template, freq, intent in stats['top_patterns']:
            print(f"  {template} ({intent}) - {freq}x")
        
    finally:
        # Cleanup
        os.unlink(db_path)


if __name__ == "__main__":
    demo()
