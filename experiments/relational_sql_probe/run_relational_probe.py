#!/usr/bin/env python3
"""
Relational SQL probe against ConceptStore SQLite DB.
Reads only. No mutations.
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any


def load_concepts_naive(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute("SELECT concept_id, term, confidence, source, usage_count FROM concepts")
    concepts = []
    for row in cur.fetchall():
        cid = row[0]
        cur_props = conn.execute(
            "SELECT property_text FROM properties WHERE concept_id=?", (cid,)
        ).fetchall()
        cur_rels = conn.execute(
            "SELECT relation_type, target, confidence FROM relations WHERE concept_id=?",
            (cid,),
        ).fetchall()
        concepts.append(
            {
                "concept_id": cid,
                "term": row[1],
                "confidence": row[2],
                "source": row[3],
                "usage_count": row[4],
                "properties": [p[0] for p in cur_props],
                "relations": [
                    {"relation_type": r[0], "target": r[1], "confidence": r[2]}
                    for r in cur_rels
                ],
            }
        )
    return concepts


def query_concept_by_term(conn: sqlite3.Connection, term: str, limit: int) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.concept_id, c.term, c.confidence, c.source, c.usage_count,
               COUNT(p.id) as prop_count, COUNT(r.id) as rel_count
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
    results = []
    for row in rows:
        cid = row[0]
        props = conn.execute(
            "SELECT property_text FROM properties WHERE concept_id=? LIMIT ?",
            (cid, limit),
        ).fetchall()
        rels = conn.execute(
            "SELECT relation_type, target, confidence FROM relations WHERE concept_id=? LIMIT ?",
            (cid, limit),
        ).fetchall()
        results.append(
            {
                "concept_id": cid,
                "term": row[1],
                "confidence": row[2],
                "source": row[3],
                "usage_count": row[4],
                "property_count": row[5],
                "relation_count": row[6],
                "properties": [p[0] for p in props],
                "relations": [
                    {"relation_type": r[0], "target": r[1], "confidence": r[2]}
                    for r in rels
                ],
            }
        )
    return results


def query_properties_for_term(conn: sqlite3.Connection, term: str, limit: int) -> List[str]:
    cur = conn.cursor()
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


def query_relations_for_term(
    conn: sqlite3.Connection, term: str, relation_type: str, limit: int
) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    params = [f"%{term}%"]
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


def query_relation_chains(
    conn: sqlite3.Connection, term: str, limit: int
) -> List[Dict[str, Any]]:
    """Return two-hop relation chains: term -> target concept -> that concept's relations."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.term as src_term,
               r1.relation_type as step_relation,
               r1.target as mid_term,
               r2.relation_type as next_relation,
               r2.target as final_target,
               r1.confidence as step_confidence,
               r2.confidence as next_confidence
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


def reverse_property_lookup(
    conn: sqlite3.Connection, substring: str, limit: int
) -> List[Dict[str, Any]]:
    cur = conn.cursor()
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


def main():
    parser = argparse.ArgumentParser(description="Relational SQL probe for ConceptStore (isolated experiment)")
    parser.add_argument(
        "--db",
        default="experiments/relational_sql_probe/data/relational_concepts.db",
        help="Path to concepts.db (defaults to isolated experimental DB)",
    )
    parser.add_argument("--term", default="birds", help="Term to query (LIKE match)")
    parser.add_argument("--property-like", dest="prop_like", default=None, help="Substring for reverse property lookup")
    parser.add_argument("--relation-type", dest="rel_type", default=None, help="Relation type filter (e.g., has_property)")
    parser.add_argument("--limit", type=int, default=5, help="Max rows per section")
    parser.add_argument("--trace", default=None, help="Optional JSONL trace output path")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))

    trace_events = []

    print(f"🔎 DB: {db_path}")
    if "data/base" in str(db_path):
        print("⚠️ Using base concepts DB; keep this probe separate from production flows.")
    print(f"Term LIKE: {args.term}")
    if args.prop_like:
        print(f"Property LIKE: {args.prop_like}")
    if args.rel_type:
        print(f"Relation type: {args.rel_type}")

    t0 = time.time()
    relational = query_concept_by_term(conn, args.term, args.limit)
    t_rel = (time.time() - t0) * 1000

    print(f"\nRelational concept hits ({t_rel:.1f} ms):")
    for item in relational:
        print(f"- {item['term']} ({item['confidence']:.2f}, usage {item['usage_count']}) props={item['property_count']} rels={item['relation_count']}")
        for p in item["properties"]:
            print(f"    • {p}")
        for r in item["relations"]:
            print(f"    ↪ {r['relation_type']}: {r['target']} ({r['confidence']:.2f})")
    trace_events.append({"type": "relational_concepts", "term": args.term, "results": relational, "ms": t_rel})

    props = query_properties_for_term(conn, args.term, args.limit)
    print(f"\nTop properties for term (limit {args.limit}):")
    for p in props:
        print(f"- {p}")
    trace_events.append({"type": "properties", "term": args.term, "results": props})

    rels = query_relations_for_term(conn, args.term, args.rel_type, args.limit)
    print(f"\nRelations for term (limit {args.limit}):")
    for r in rels:
        print(f"- {r['relation_type']}: {r['target']} ({r['confidence']:.2f})")
    trace_events.append({"type": "relations", "term": args.term, "results": rels})

    chains = query_relation_chains(conn, args.term, args.limit)
    if chains:
        print(f"\nTwo-hop relation chains (limit {args.limit}):")
        for ch in chains:
            print(
                f"- {ch['src_term']} --{ch['step_relation']}→ {ch['mid_term']} --{ch['next_relation']}→ {ch['final_target']}"
                f" (conf {ch['step_confidence']:.2f}/{ch['next_confidence']:.2f})"
            )
        trace_events.append({"type": "relation_chains", "term": args.term, "results": chains})
    else:
        print("\nTwo-hop relation chains: none")
        trace_events.append({"type": "relation_chains", "term": args.term, "results": []})

    if args.prop_like:
        rev = reverse_property_lookup(conn, args.prop_like, args.limit)
        print(f"\nReverse property lookup containing '{args.prop_like}' (limit {args.limit}):")
        for r in rev:
            print(f"- {r['term']}: {r['property']}")
        trace_events.append({"type": "reverse_properties", "substring": args.prop_like, "results": rev})

    # Naive scan for comparison
    t1 = time.time()
    all_concepts = load_concepts_naive(conn)
    t_naive = (time.time() - t1) * 1000
    matches = [c for c in all_concepts if args.term.lower() in c["term"].lower()]
    print(f"\nNaive scan timing: {t_naive:.1f} ms (concepts loaded: {len(all_concepts)}, term matches: {len(matches)})")
    trace_events.append({"type": "naive_scan", "ms": t_naive, "concepts": len(all_concepts), "matches": len(matches)})

    if args.trace:
        trace_path = Path(args.trace)
        trace_path.write_text("\n".join(json.dumps(e) for e in trace_events))
        print(f"\nTrace written to {trace_path}")


if __name__ == "__main__":
    main()
