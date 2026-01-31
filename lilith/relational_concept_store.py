#!/usr/bin/env python3
"""
RelationalConceptStore - thin relational query helper for ConceptStore SQLite DBs.

Read-only helper used by ProductionConceptStore and higher layers to run
relation-aware lookups (term match, properties, relation filter, and bounded
relation chains). Keeps DB access centralized.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class RelationalConceptStore:
    """Run targeted relational queries against a ConceptStore SQLite DB."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def _fetchall_dicts(self, cur: sqlite3.Cursor) -> List[Dict[str, Any]]:
        rows = cur.fetchall()
        return [dict(row) for row in rows]

    def query_concepts(self, term: str, limit: int = 5) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT c.concept_id,
                   c.term,
                   c.confidence,
                   c.source,
                   c.usage_count,
                   COUNT(p.id) AS property_count,
                   COUNT(r.id) AS relation_count
            FROM concepts c
            LEFT JOIN properties p ON c.concept_id = p.concept_id
            LEFT JOIN relations r ON c.concept_id = r.concept_id
            WHERE lower(c.term) LIKE lower(?)
            GROUP BY c.concept_id
            ORDER BY c.usage_count DESC, c.confidence DESC
            LIMIT ?
            """,
            (f"%{term}%", limit),
        )
        rows = cur.fetchall()
        results: List[Dict[str, Any]] = []
        for row in rows:
            cid = row["concept_id"]
            props = self.conn.execute(
                "SELECT property_text FROM properties WHERE concept_id=? LIMIT ?",
                (cid, limit),
            ).fetchall()
            rels = self.conn.execute(
                "SELECT relation_type, target, confidence FROM relations WHERE concept_id=? LIMIT ?",
                (cid, limit),
                ).fetchall()
            results.append(
                {
                    "concept_id": cid,
                    "term": row["term"],
                    "confidence": row["confidence"],
                    "source": row["source"],
                    "usage_count": row["usage_count"],
                    "property_count": row["property_count"],
                    "relation_count": row["relation_count"],
                    "properties": [p[0] for p in props],
                    "relations": [
                        {"relation_type": r[0], "target": r[1], "confidence": r[2]}
                        for r in rels
                    ],
                }
            )
        return results

    def query_properties(self, term: str, limit: int = 5) -> List[str]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT p.property_text
            FROM properties p
            JOIN concepts c ON p.concept_id = c.concept_id
            WHERE lower(c.term) LIKE lower(?)
            LIMIT ?
            """,
            (f"%{term}%", limit),
        )
        return [row[0] for row in cur.fetchall()]

    def query_relations(
        self,
        term: str,
        relation_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        params: List[Any] = [f"%{term}%"]
        rel_filter = ""
        if relation_type:
            rel_filter = "AND lower(r.relation_type) = lower(?)"
            params.append(relation_type)
        params.append(limit)
        cur.execute(
            f"""
            SELECT r.relation_type, r.target, r.confidence
            FROM relations r
            JOIN concepts c ON r.concept_id = c.concept_id
            WHERE lower(c.term) LIKE lower(?) {rel_filter}
            LIMIT ?
            """,
            params,
        )
        rows = cur.fetchall()
        return [
            {"relation_type": row[0], "target": row[1], "confidence": row[2]} for row in rows
        ]

    def reverse_property_lookup(self, substring: str, limit: int = 5) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT c.concept_id, c.term, p.property_text
            FROM properties p
            JOIN concepts c ON p.concept_id = c.concept_id
            WHERE lower(p.property_text) LIKE lower(?)
            LIMIT ?
            """,
            (f"%{substring}%", limit),
        )
        rows = cur.fetchall()
        return [
            {"concept_id": row[0], "term": row[1], "property": row[2]} for row in rows
        ]

    def two_hop_chains(self, term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Return two-hop chains: term -> target concept -> that concept's relations."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT c.term AS src_term,
                   r1.relation_type AS step_relation,
                   r1.target AS mid_term,
                   r2.relation_type AS next_relation,
                   r2.target AS final_target,
                   r1.confidence AS step_confidence,
                   r2.confidence AS next_confidence
            FROM concepts c
            JOIN relations r1 ON c.concept_id = r1.concept_id
            JOIN concepts mid ON lower(mid.term) = lower(r1.target)
            JOIN relations r2 ON mid.concept_id = r2.concept_id
            WHERE lower(c.term) LIKE lower(?)
            LIMIT ?
            """,
            (f"%{term}%", limit),
        )
        rows = cur.fetchall()
        return [
            {
                "src_term": row[0],
                "step_relation": row[1],
                "mid_term": row[2],
                "next_relation": row[3],
                "final_target": row[4],
                "step_confidence": row[5],
                "next_confidence": row[6],
            }
            for row in rows
        ]

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


__all__ = ["RelationalConceptStore"]
