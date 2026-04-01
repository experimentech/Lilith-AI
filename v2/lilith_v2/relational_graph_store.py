import sqlite3
import json
import re
import hashlib
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
        # Nodes: Concepts, senses, lexical forms, entities, patterns
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,       -- e.g. 'concept', 'sense', 'lexeme', 'pattern', 'fragment'
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
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)")
        self._conn.commit()

    @staticmethod
    def _slugify(term: str) -> str:
        text = (term or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return text.strip("_") or "concept"

    def find_concept_by_term(self, term: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Find a canonical concept by case-insensitive term match."""
        row = self._conn.execute(
            """
            SELECT * FROM nodes
            WHERE type = 'concept' AND LOWER(term) = LOWER(?)
            ORDER BY confidence DESC, rowid DESC
            LIMIT 1
            """,
            (term,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "type": row["type"],
            "term": row["term"],
            "confidence": row["confidence"],
            "data": json.loads(row["data"]),
        }

    def get_or_create_concept(
        self,
        term: str,
        confidence: float = 1.0,
        data: Optional[Dict[str, Any]] = None,
        alias: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Return stable canonical concept ID for a term and create if absent.

        Canonical IDs follow `concept:<slug>` and avoid collisions by suffixing
        a short hash when needed.
        """
        existing = self.find_concept_by_term(term)
        if existing:
            if alias and alias.strip().lower() != (existing.get("term") or "").strip().lower():
                self._append_concept_alias(existing["id"], alias)
            return existing["id"]

        slug = self._slugify(term)
        candidate = f"concept:{slug}"
        current = self.get_node(candidate)
        if current and (current.get("type") != "concept" or (current.get("term") or "").strip().lower() != term.strip().lower()):
            suffix = hashlib.sha1(term.encode("utf-8")).hexdigest()[:6]
            candidate = f"concept:{slug}:{suffix}"

        payload = dict(data or {})
        payload.setdefault("aliases", [])
        if alias and alias.strip():
            payload["aliases"] = sorted({*payload.get("aliases", []), alias.strip()})

        self.add_node(
            node_id=candidate,
            node_type="concept",
            term=term,
            confidence=confidence,
            data=payload,
        )
        return candidate

    def _append_concept_alias(self, concept_id: str, alias: str) -> None:
        """Attach lexical alias metadata to an existing concept node."""
        if not alias.strip():
            return
        node = self.get_node(concept_id)
        if not node:
            return
        data = dict(node.get("data") or {})
        aliases = set(data.get("aliases") or [])
        aliases.add(alias.strip())
        data["aliases"] = sorted(aliases)
        self.update_node(concept_id, data=data)

    def add_sense(
        self,
        sense_id: str,
        term: str,
        confidence: float = 1.0,
        data: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        """Create or keep an existing sense node.

        Sense nodes represent contextual meanings for a lexeme, e.g.
        `sense:bank#finance` vs `sense:bank#river_edge`.
        """
        payload = dict(data or {})
        payload.setdefault("kind", "sense")
        self.add_node(sense_id, "sense", term, confidence=confidence, data=payload)

    def add_lexeme(
        self,
        lexeme_id: str,
        term: str,
        confidence: float = 1.0,
        data: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        """Create or keep an existing lexeme node (surface form / lemma layer)."""
        payload = dict(data or {})
        payload.setdefault("kind", "lexeme")
        self.add_node(lexeme_id, "lexeme", term, confidence=confidence, data=payload)

    def link_lexeme_to_sense(
        self,
        lexeme_id: str,
        sense_id: str,
        relation: str = "possible_meaning",
        confidence: float = 1.0,
        **kwargs,
    ) -> None:
        """Link a lexeme to a sense.

        Recommended relations:
        - possible_meaning
        - preferred_meaning
        """
        self.add_edge(lexeme_id, sense_id, relation, confidence=confidence)

    def link_sense_to_concept(
        self,
        sense_id: str,
        concept_id: str,
        relation: str = "maps_to",
        confidence: float = 1.0,
        **kwargs,
    ) -> None:
        """Link a sense to a canonical concept.

        Recommended relation: maps_to
        """
        self.add_edge(sense_id, concept_id, relation, confidence=confidence)

    def get_senses_for_lexeme(
        self,
        lexeme_id: str,
        relations: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Return sense neighbors for a lexeme via meaning edges."""
        allowed = relations or ["possible_meaning", "preferred_meaning"]
        placeholders = ",".join(["?"] * len(allowed))
        rows = self._conn.execute(
            f"""
            SELECT e.type AS relation, e.confidence AS edge_confidence,
                   n.id, n.type, n.term, n.confidence, n.data
            FROM edges e
            JOIN nodes n ON n.id = e.target
            WHERE e.source = ?
              AND e.type IN ({placeholders})
              AND n.type = 'sense'
            ORDER BY e.confidence DESC, n.confidence DESC
            """,
            (lexeme_id, *allowed),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "type": r["type"],
                "term": r["term"],
                "confidence": r["confidence"],
                "relation": r["relation"],
                "edge_confidence": r["edge_confidence"],
                "data": json.loads(r["data"]),
            }
            for r in rows
        ]

    def get_concepts_for_sense(
        self,
        sense_id: str,
        relations: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Return canonical concept neighbors for a sense via mapping edges."""
        allowed = relations or ["maps_to"]
        placeholders = ",".join(["?"] * len(allowed))
        rows = self._conn.execute(
            f"""
            SELECT e.type AS relation, e.confidence AS edge_confidence,
                   n.id, n.type, n.term, n.confidence, n.data
            FROM edges e
            JOIN nodes n ON n.id = e.target
            WHERE e.source = ?
              AND e.type IN ({placeholders})
              AND n.type = 'concept'
            ORDER BY e.confidence DESC, n.confidence DESC
            """,
            (sense_id, *allowed),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "type": r["type"],
                "term": r["term"],
                "confidence": r["confidence"],
                "relation": r["relation"],
                "edge_confidence": r["edge_confidence"],
                "data": json.loads(r["data"]),
            }
            for r in rows
        ]

    def add_node(self, node_id: str, node_type: str, term: str, confidence: float = 1.0, data: Dict[str, Any] = None, **kwargs) -> None:
        # Use INSERT OR IGNORE to avoid triggering ON DELETE CASCADE
        # If node exists, we just skip (existing data is preserved)
        self._conn.execute(
            "INSERT OR IGNORE INTO nodes(id, type, term, confidence, data) VALUES (?, ?, ?, ?, ?)",
            (node_id, node_type, term, confidence, json.dumps(data or {}))
        )
        self._conn.commit()

    def add_edge(self, source: str, target: str, edge_type: str, confidence: float = 1.0, **kwargs) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO edges(source, target, type, confidence) VALUES (?, ?, ?, ?)",
            (source, target, edge_type, confidence)
        )
        self._conn.commit()

    def update_edge_confidence(
        self,
        source: str,
        target: str,
        edge_type: str,
        delta: float,
        min_conf: float = 0.0,
        max_conf: float = 1.0,
        **kwargs,
    ) -> Optional[float]:
        """Adjust an edge confidence by delta and return new value when edge exists."""
        row = self._conn.execute(
            "SELECT confidence FROM edges WHERE source = ? AND target = ? AND type = ?",
            (source, target, edge_type),
        ).fetchone()
        if not row:
            return None
        current = float(row["confidence"])
        updated = max(min_conf, min(max_conf, current + float(delta)))
        self._conn.execute(
            "UPDATE edges SET confidence = ? WHERE source = ? AND target = ? AND type = ?",
            (updated, source, target, edge_type),
        )
        self._conn.commit()
        return updated

    def _confidence_policy(self, node_type: str) -> Dict[str, float]:
        """Layer-aware confidence policy parameters by node type."""
        t = (node_type or "").lower()
        if t in {"lexeme", "word", "lexical_token"}:
            return {"step": 0.18, "floor": 0.05, "ceiling": 1.0}
        if t == "sense":
            return {"step": 0.12, "floor": 0.10, "ceiling": 1.0}
        if t in {"concept", "learned_concept", "entity"}:
            return {"step": 0.05, "floor": 0.20, "ceiling": 1.0}
        return {"step": 0.08, "floor": 0.05, "ceiling": 1.0}

    def _is_verified_sense(self, sense_id: str) -> bool:
        """A sense is treated as verified when strongly mapped to a concept.

        This is intentionally simple and conservative for pruning protection.
        """
        row = self._conn.execute(
            """
            SELECT 1
            FROM edges
            WHERE source = ?
              AND type = 'maps_to'
              AND confidence >= 0.75
            LIMIT 1
            """,
            (sense_id,),
        ).fetchone()
        return row is not None

    def apply_feedback_to_node(
        self,
        node_id: str,
        feedback_score: float,
        evidence_weight: float = 1.0,
        protect_rare_senses: bool = True,
        **kwargs,
    ) -> Optional[float]:
        """Apply a feedback pulse to node confidence using layer-aware policy.

        Positive feedback reinforces confidence, negative feedback decays it.
        """
        node = self.get_node(node_id)
        if not node:
            return None

        policy = self._confidence_policy(node.get("type", ""))
        score = max(-1.0, min(1.0, float(feedback_score)))
        weight = max(0.0, float(evidence_weight))
        delta = policy["step"] * score * weight

        current = float(node.get("confidence", 0.5))
        updated = max(policy["floor"], min(policy["ceiling"], current + delta))

        if protect_rare_senses and node.get("type") == "sense" and score < 0 and self._is_verified_sense(node_id):
            # Guardrail: preserve verified but low-frequency senses from aggressive decay.
            updated = max(updated, 0.35)

        self.update_node(node_id, confidence=updated)
        return updated

    def prune_low_confidence_senses(
        self,
        threshold: float = 0.15,
        preserve_verified: bool = True,
        **kwargs,
    ) -> int:
        """Demote very low-confidence senses to dormant, with verification guardrails.

        This avoids deleting canonical concepts while allowing weak senses to be pruned.
        """
        rows = self._conn.execute(
            "SELECT id, data FROM nodes WHERE type = 'sense' AND confidence < ?",
            (float(threshold),),
        ).fetchall()

        updated_count = 0
        for row in rows:
            sid = row["id"]
            if preserve_verified and self._is_verified_sense(sid):
                continue
            data = json.loads(row["data"])
            if data.get("status") == "dormant":
                continue
            data["status"] = "dormant"
            self.update_node(sid, data=data)
            updated_count += 1

        return updated_count

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

    def update_node(
        self,
        node_id: str,
        *,
        term: Optional[str] = None,
        confidence: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update selected node fields. Returns True when a row was updated."""
        updates: List[str] = []
        values: List[Any] = []

        if term is not None:
            updates.append("term = ?")
            values.append(term)
        if confidence is not None:
            updates.append("confidence = ?")
            values.append(confidence)
        if data is not None:
            updates.append("data = ?")
            values.append(json.dumps(data))

        if not updates:
            return False

        values.append(node_id)
        cur = self._conn.execute(
            f"UPDATE nodes SET {', '.join(updates)} WHERE id = ?",
            tuple(values),
        )
        self._conn.commit()
        return cur.rowcount > 0

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

    def list_nodes_by_type(self, node_type: str, limit: int = 1000, **kwargs) -> List[Dict[str, Any]]:
        """List nodes filtered by type, newest first by rowid."""
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE type = ? ORDER BY rowid DESC LIMIT ?",
            (node_type, limit),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "type": r["type"],
                "term": r["term"],
                "confidence": r["confidence"],
                "data": json.loads(r["data"]),
            }
            for r in rows
        ]

    def get_all_terms(self, **kwargs) -> List[Tuple[str, str]]:
        """Return (id, term) for all nodes with terms. Used for fuzzy matching index."""
        query = "SELECT id, term FROM nodes WHERE term IS NOT NULL AND term != ''"
        rows = self._conn.execute(query).fetchall()
        return [(r["id"], r["term"]) for r in rows]

    def get_terms_by_types(
        self,
        allowed_types: Optional[List[str]] = None,
        excluded_types: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Tuple[str, str]]:
        """Return (id, term) filtered by node type for safer grounding.

        Args:
            allowed_types: If provided, only include these node types.
            excluded_types: If provided, exclude these node types.
        """
        base = "SELECT id, term FROM nodes WHERE term IS NOT NULL AND term != ''"
        params: List[Any] = []

        if allowed_types:
            placeholders = ",".join(["?"] * len(allowed_types))
            base += f" AND type IN ({placeholders})"
            params.extend(allowed_types)

        if excluded_types:
            placeholders = ",".join(["?"] * len(excluded_types))
            base += f" AND type NOT IN ({placeholders})"
            params.extend(excluded_types)

        rows = self._conn.execute(base, tuple(params)).fetchall()
        return [(r["id"], r["term"]) for r in rows]

    def close(self) -> None:
        self._conn.close()

__all__ = ["RelationalGraphStore"]
