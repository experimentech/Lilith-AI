#!/usr/bin/env python3
"""Export LM adaptation episodes from graph DB to JSONL training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from v2.lilith_v2.relational_graph_store import RelationalGraphStore


def export_lm_adaptation_dataset(
    graph_db_path: str,
    output_path: str,
    status: str = "accepted",
    limit: int = 5000,
) -> int:
    store = RelationalGraphStore(graph_db_path)
    count = 0
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        rows = store.list_nodes_by_type("lm_adaptation_event", limit=limit)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                data = row.get("data") or {}
                if data.get("kind") != "lm_assist_episode":
                    continue
                if status and data.get("status") != status:
                    continue

                record = {
                    "event_id": row.get("id"),
                    "tenant_id": data.get("tenant_id"),
                    "prompt": data.get("normalized_input") or data.get("user_input"),
                    "target": data.get("assistant_response"),
                    "lm_suggestion": data.get("lm_suggestion"),
                    "parse_confidence": data.get("parse_confidence"),
                    "lm_score": data.get("lm_score"),
                    "feedback": data.get("feedback"),
                    "status": data.get("status"),
                    "created_at": data.get("created_at"),
                    "finalized_at": data.get("finalized_at"),
                }
                if not record["prompt"] or not record["target"]:
                    continue

                f.write(json.dumps(record, ensure_ascii=True) + "\n")
                count += 1
    finally:
        store.close()

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LM adaptation dataset from Lilith graph DB")
    parser.add_argument("--graph-db", required=True, help="Path to graph SQLite DB")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument(
        "--status",
        default="accepted",
        choices=["accepted", "rejected", "pending_feedback", ""],
        help="Filter by event status (empty string means all)",
    )
    parser.add_argument("--limit", type=int, default=5000, help="Max events to scan")
    args = parser.parse_args()

    written = export_lm_adaptation_dataset(
        graph_db_path=args.graph_db,
        output_path=args.output,
        status=args.status,
        limit=args.limit,
    )
    print(f"wrote_records={written}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
