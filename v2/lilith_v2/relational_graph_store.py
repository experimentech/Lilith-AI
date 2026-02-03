import sqlite3
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

class RelationalGraphStore:
    """SQLite-backed knowledge graph store (The Textbook).
    
    Provides schema for Nodes and Edges, allowing explicit graph querying 
    separate from latent/vector similarity.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        
        self._init_schema()

    def _init_schema(self) -> None:
        # Nodes: Concepts, Entities, Patterns
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,       -- 'concept', 'pattern', 'fragment'
                term TEXT,               -- Human readable label (e.g. 'photosynthesis')
                confidence REAL DEFAULT 1.0, 
                data TEXT                -- JSON payload
            )
        """)
        
        # Edges: Typed relationships
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                type TEXT NOT NULL,       -- 'is_a', 'requires', 'causes'
                confidence REAL DEFAULT 1.0,
                FOREIGN KEY(source) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(target) REFERENCES nodes(id) ON DELETE CASCADE,
                PRIMARY KEY (source, target, type)
            )
        """)
        
        # Indexes for traversal speed
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_term ON nodes(term)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)")
        self._conn.commit()

    def add_node(self, node_id: str, node_type: str, term: str, confidence: float = 1.0, data: Dict[str, Any] = None, **kwargs) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO nodes(id, type, term, confidence, data) VALUES (?, ?, ?, ?, ?)",
            (node_id, node_type, term, confidence, json.dumps(data or {}))
        )
        self._conn.commit()

    def add_edge(self, source: str, target: str, edge_type: str, confidence: float = 1.0, **kwargs) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO edges(source, target, type, confidence) VALUES (?, ?, ?, ?)",
            (source, target, edge_type, confidence)
        )
        self._conn.commit()

    def get_node(self, node_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "type": row["type"],
            "term": row["term"],
            "confidence": row["confidence"],
            "data": json.loads(row["data"])
        }

    def get_related(self, source_id: str, edge_type: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Direct neighbors."""
        query = """
            SELECT e.type as relation, e.confidence, n.* 
            FROM edges e
            JOIN nodes n ON e.target = n.id
            WHERE e.source = ?
        """
        params = [source_id]
        if edge_type:
            query += " AND e.type = ?"
            params.append(edge_type)
        
        rows = self._conn.execute(query, tuple(params)).fetchall()
        return [
            {
                "relation": r["relation"],
                "confidence": r["confidence"],
                "target": {
                    "id": r["id"],
                    "term": r["term"],
                    "type": r["type"],
                    "data": json.loads(r["data"])
                }
            } 
            for r in rows
        ]

    def traverse_bfs(self, start_id: str, max_depth: int = 3, edge_types: Optional[List[str]] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Perform BFS traversal to find connected subgraph.
        Returns paths: [{'path': ['A', 'B', 'C'], 'confidence': 0.9}, ...]
        """
        # (current_id, path_list, min_confidence)
        queue = [(start_id, [start_id], 1.0)]
        visited = {start_id}
        valid_paths = []

        type_filter = set(edge_types) if edge_types else None
        
        while queue:
            curr, path, conf = queue.pop(0)
            
            if len(path) > 1:
                valid_paths.append({"path_ids": path, "confidence": conf})
            
            if len(path) >= max_depth:
                continue

            query = "SELECT target, type, confidence FROM edges WHERE source = ?"
            rows = self._conn.execute(query, (curr,)).fetchall()
            
            for row in rows:
                tgt, etype, econf = row["target"], row["type"], row["confidence"]
                
                if type_filter and etype not in type_filter:
                    continue
                
                if tgt not in visited: # Cycle prevention in simple path
                    # Note: strict visited set prevents multiple paths to same node. 
                    # For a true "all paths" we'd move visited check to path-level, but that explodes.
                    # Keeping simple distinct-node traversal for now.
                    visited.add(tgt)
                    new_conf = min(conf, econf) # Weakest link principle
                    queue.append((tgt, path + [tgt], new_conf))
                    
        return sorted(valid_paths, key=lambda x: x['confidence'], reverse=True)

    def find_nodes_by_term(self, term_fragment: str, **kwargs) -> List[Dict[str, Any]]:
        """Find nodes where the term contains the fragment (for broad search)."""
        # Helper for efficient lookup without loading everything
        query = "SELECT * FROM nodes WHERE term LIKE ? ORDER BY confidence DESC LIMIT 20"
        rows = self._conn.execute(query, (f"%{term_fragment}%",)).fetchall()
        return [dict(r) for r in rows]

    def get_all_terms(self) -> List[Tuple[str, str]]:
        """Return (id, term) for all nodes with terms. Used for fuzzy matching index."""
        # Include all node types that have meaningful terms for grounding
        query = "SELECT id, term FROM nodes WHERE term IS NOT NULL AND term != ''"
        rows = self._conn.execute(query).fetchall()
        return [(r["id"], r["term"]) for r in rows]

    def close(self) -> None:
        self._conn.close()

__all__ = ["RelationalGraphStore"]
