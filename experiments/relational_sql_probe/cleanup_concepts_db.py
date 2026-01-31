#!/usr/bin/env python3
"""
Concept DB cleaner for the relational probe.
- Works on a copy by default to avoid mutating production/user DBs.
- Removes overly long properties, deduplicates properties per concept, and can require term matches.
"""

import argparse
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, Tuple


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def load_concepts(conn: sqlite3.Connection) -> Dict[str, str]:
    cur = conn.execute("SELECT concept_id, term FROM concepts")
    return {row[0]: row[1] for row in cur.fetchall()}


def count_props(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM properties")
    return cur.fetchone()[0]


def clean_properties(
    conn: sqlite3.Connection,
    concepts: Dict[str, str],
    max_len: int,
    require_term_match: bool,
) -> Tuple[int, int, int]:
    """Remove long properties and per-concept duplicates. Returns (removed_long, removed_dupe, removed_term_miss)."""
    cur = conn.execute("SELECT id, concept_id, property_text FROM properties")
    removed_long = 0
    removed_dupe = 0
    removed_term_miss = 0
    seen = {}
    to_delete = []
    for pid, cid, prop in cur.fetchall():
        norm = normalize_text(prop)
        term = concepts.get(cid, "")
        # Drop if too long
        if len(prop) > max_len:
            removed_long += 1
            to_delete.append(pid)
            continue
        # Optionally drop if property does not mention the concept term (rough heuristic)
        if require_term_match and term and term.lower() not in norm:
            removed_term_miss += 1
            to_delete.append(pid)
            continue
        # Dedup per concept
        key = (cid, norm)
        if key in seen:
            removed_dupe += 1
            to_delete.append(pid)
            continue
        seen[key] = True
    if to_delete:
        conn.executemany("DELETE FROM properties WHERE id=?", [(pid,) for pid in to_delete])
    return removed_long, removed_dupe, removed_term_miss


def main():
    parser = argparse.ArgumentParser(description="Clean ConceptStore properties for relational probe experiments")
    parser.add_argument("--db", required=True, help="Path to input concepts.db")
    parser.add_argument(
        "--out",
        default=None,
        help="Output DB path (default: copy alongside input with .clean suffix)",
    )
    parser.add_argument("--max-len", type=int, default=320, help="Max property length to keep")
    parser.add_argument(
        "--require-term-match",
        action="store_true",
        help="Drop properties that do not mention their concept term",
    )
    parser.add_argument("--dry-run", action="store_true", help="Analyze without modifying")
    args = parser.parse_args()

    src = Path(args.db)
    if not src.exists():
        raise SystemExit(f"Input DB not found: {src}")

    out_path = Path(args.out) if args.out else src.with_suffix(".clean.db")
    if args.dry_run:
        print(f"[dry-run] Will analyze {src} (no writes)")
    else:
        if out_path == src:
            raise SystemExit("Refusing to overwrite input DB; specify --out or use --dry-run")
        shutil.copy2(src, out_path)
        print(f"Copied {src} -> {out_path}")

    target = src if args.dry_run else out_path
    conn = sqlite3.connect(target)

    concepts = load_concepts(conn)
    before_props = count_props(conn)

    removed_long, removed_dupe, removed_term_miss = clean_properties(
        conn,
        concepts,
        max_len=args.max_len,
        require_term_match=args.require_term_match,
    )

    after_props = count_props(conn)

    if not args.dry_run:
        conn.commit()
    conn.close()

    print("Cleanup summary:")
    print(f"- properties before: {before_props}")
    print(f"- removed (too long): {removed_long}")
    print(f"- removed (dupes): {removed_dupe}")
    if args.require_term_match:
        print(f"- removed (term mismatch): {removed_term_miss}")
    else:
        print(f"- removed (term mismatch): skipped")
    print(f"- properties after: {after_props}")
    if args.dry_run:
        print("Done (dry-run).")
    else:
        print(f"Cleaned DB: {target}")


if __name__ == "__main__":
    main()
