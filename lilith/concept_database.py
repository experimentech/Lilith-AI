"""
Concept Database - Production persistence for ConceptStore

SQLite-backed storage for semantic concepts with properties and relations.
Similar to PatternDatabase but for compositional architecture.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict


class ConceptDatabase:
    """
    Production database for storing semantic concepts.
    
    Schema:
    - concepts: Core concept data (id, term, confidence, source, usage_count)
    - properties: Concept properties (concept_id, property_text)
    - relations: Semantic relations (concept_id, relation_type, target, confidence)
    """
    
    def __init__(self, db_path: str):
        """
        Initialize concept database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_schema()
    
    def _create_schema(self):
        """Create database schema if it doesn't exist."""
        cursor = self.conn.cursor()
        
        # Concepts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                concept_id TEXT PRIMARY KEY,
                term TEXT NOT NULL,
                confidence REAL DEFAULT 0.85,
                source TEXT DEFAULT 'taught',
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Properties table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_id TEXT NOT NULL,
                property_text TEXT NOT NULL,
                FOREIGN KEY (concept_id) REFERENCES concepts(concept_id) ON DELETE CASCADE
            )
        """)
        
        # Relations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                target TEXT NOT NULL,
                confidence REAL DEFAULT 0.85,
                FOREIGN KEY (concept_id) REFERENCES concepts(concept_id) ON DELETE CASCADE
            )
        """)
        
        # Indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_concepts_term ON concepts(term)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_properties_concept ON properties(concept_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relations_concept ON relations(concept_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_concepts_usage ON concepts(usage_count DESC)")
        
        self.conn.commit()
    
    def add_concept(
        self,
        concept_id: str,
        term: str,
        properties: List[str],
        relations: List[Dict],
        confidence: float = 0.85,
        source: str = "taught"
    ) -> bool:
        """
        Add or update a concept.
        
        Args:
            concept_id: Unique concept identifier
            term: Concept term (e.g., "machine learning")
            properties: List of property strings
            relations: List of relation dicts with keys: relation_type, target, confidence
            confidence: Confidence score
            source: Source of knowledge ("taught", "learned", "wikipedia")
            
        Returns:
            True if successful
        """
        cursor = self.conn.cursor()
        
        try:
            # Insert or update concept (using INSERT OR REPLACE for older SQLite compatibility)
            cursor.execute("""
                INSERT OR REPLACE INTO concepts (concept_id, term, confidence, source, usage_count, updated_at)
                VALUES (?, ?, ?, ?, 
                    COALESCE((SELECT usage_count FROM concepts WHERE concept_id = ?), 0),
                    CURRENT_TIMESTAMP)
            """, (concept_id, term, confidence, source, concept_id))
            
            # Delete old properties and relations (simpler than diffing)
            cursor.execute("DELETE FROM properties WHERE concept_id = ?", (concept_id,))
            cursor.execute("DELETE FROM relations WHERE concept_id = ?", (concept_id,))
            
            # Insert properties
            for prop in properties:
                cursor.execute("""
                    INSERT INTO properties (concept_id, property_text)
                    VALUES (?, ?)
                """, (concept_id, prop))
            
            # Insert relations
            for rel in relations:
                cursor.execute("""
                    INSERT INTO relations (concept_id, relation_type, target, confidence)
                    VALUES (?, ?, ?, ?)
                """, (concept_id, rel['relation_type'], rel['target'], rel.get('confidence', 0.85)))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.conn.rollback()
            print(f"Error adding concept {concept_id}: {e}")
            return False
    
    def get_concept(self, concept_id: str) -> Optional[Dict]:
        """
        Retrieve a concept by ID.
        
        Returns:
            Dict with concept data including properties and relations, or None
        """
        cursor = self.conn.cursor()
        
        # Get concept
        cursor.execute("""
            SELECT concept_id, term, confidence, source, usage_count
            FROM concepts
            WHERE concept_id = ?
        """, (concept_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        concept = dict(row)
        
        # Get properties
        cursor.execute("""
            SELECT property_text
            FROM properties
            WHERE concept_id = ?
        """, (concept_id,))
        concept['properties'] = [r['property_text'] for r in cursor.fetchall()]
        
        # Get relations
        cursor.execute("""
            SELECT relation_type, target, confidence
            FROM relations
            WHERE concept_id = ?
        """, (concept_id,))
        concept['relations'] = [dict(r) for r in cursor.fetchall()]
        
        return concept
    
    def get_all_concepts(self) -> List[Dict]:
        """Get all concepts with their properties and relations."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT concept_id, term, confidence, source, usage_count
            FROM concepts
            ORDER BY usage_count DESC
        """)
        
        concepts = []
        for row in cursor.fetchall():
            concept = dict(row)
            cid = concept['concept_id']
            
            # Get properties
            cursor.execute("""
                SELECT property_text
                FROM properties
                WHERE concept_id = ?
            """, (cid,))
            concept['properties'] = [r['property_text'] for r in cursor.fetchall()]
            
            # Get relations
            cursor.execute("""
                SELECT relation_type, target, confidence
                FROM relations
                WHERE concept_id = ?
            """, (cid,))
            concept['relations'] = [dict(r) for r in cursor.fetchall()]
            
            concepts.append(concept)
        
        return concepts
    
    def delete_concept(self, concept_id: str) -> bool:
        """Delete a concept and its properties/relations."""
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("DELETE FROM concepts WHERE concept_id = ?", (concept_id,))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"Error deleting concept {concept_id}: {e}")
            return False
    
    def increment_usage(self, concept_id: str):
        """Increment usage count for a concept."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE concepts
            SET usage_count = usage_count + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE concept_id = ?
        """, (concept_id,))
        self.conn.commit()
    
    def get_relations_from(self, concept_id: str, relation_type: Optional[str] = None) -> List[Dict]:
        """
        Get all relations from a concept.
        
        Args:
            concept_id: Source concept ID
            relation_type: Optional filter by relation type (e.g., "is_type_of", "has_property", "used_for")
            
        Returns:
            List of relation dicts with keys: relation_type, target, confidence
        """
        cursor = self.conn.cursor()
        
        if relation_type:
            cursor.execute("""
                SELECT relation_type, target, confidence
                FROM relations
                WHERE concept_id = ? AND relation_type = ?
            """, (concept_id, relation_type))
        else:
            cursor.execute("""
                SELECT relation_type, target, confidence
                FROM relations
                WHERE concept_id = ?
            """, (concept_id,))
        
        return [dict(r) for r in cursor.fetchall()]
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as count FROM concepts")
        total_concepts = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM properties")
        total_properties = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM relations")
        total_relations = cursor.fetchone()['count']
        
        cursor.execute("SELECT AVG(usage_count) as avg_usage FROM concepts")
        avg_usage = cursor.fetchone()['avg_usage'] or 0.0
        
        return {
            'total_concepts': total_concepts,
            'total_properties': total_properties,
            'total_relations': total_relations,
            'avg_usage': avg_usage
        }
    
    def close(self):
        """Close database connection."""
        self.conn.close()
