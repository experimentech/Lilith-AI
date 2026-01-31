# Relational SQL Probe (Concept Store)

Goal: explore whether leveraging relational queries (joins over concepts/properties/relations) yields better on-topic answers than the current load-and-filter approach. This stays isolated from production code until validated.

## What it does
- Opens a ConceptStore SQLite DB (defaults to the isolated `experiments/relational_sql_probe/data/relational_concepts.db`; you can point to a user DB if needed).
- Runs targeted SQL joins to fetch:
  - Concept by term (exact/LIKE).
  - Top properties for a term.
  - Relations by type (`is_type_of`, `has_property`, `used_for`).
  - Two-hop relation chains: term → related concept → that concept's relations.
  - Reverse lookups: given a property substring, find concepts that have it.
- Compares relational-filtered hits to a naive in-Python scan to measure precision/latency.
- Optional: emits a small JSONL trace of query → results for later inspection.

## Usage
```bash
python3 experiments/relational_sql_probe/run_relational_probe.py \
  --db experiments/relational_sql_probe/data/relational_concepts.db \
  --term "birds" \
  --property-like "feather" \
  --limit 5 \
  --trace trace.jsonl
```

Key flags:
- `--db`: path to concepts.db (defaults to the isolated experimental DB).
- `--term`: concept term to query (exact/LIKE).
- `--property-like`: substring to match in properties for reverse lookup.
- `--relation-type`: filter relations (default: any).
- `--limit`: max rows per section.
- `--trace`: optional JSONL file to write results.

## Notes
- This does **not** modify any data; all queries are read-only. Keep this DB separate from production flows until results are validated.
- The default experimental DB includes a tiny seed graph (`bird`, `wing`, `feathers`, `vertebrate`, `penguin`) to make relational queries non-empty.
- Two-hop chains rely on the relation target matching another concept term (case-insensitive) to walk `term -> target concept -> its relations`.

## Cleanup helper (spurious property slabs)

To trim long/duplicate properties before probing, use:

```bash
python3 experiments/relational_sql_probe/cleanup_concepts_db.py \
  --db data/users/tristan/concepts.db \
  --out experiments/relational_sql_probe/data/tristan.clean.db \
  --max-len 320
```

Notes:
- Operates on a copy by default (`--out`); refuses to overwrite the source.
- Removes per-concept duplicate properties and any over `--max-len` characters.
- Add `--require-term-match` to drop properties that do not mention their concept term.
- It uses direct SQL joins to exploit relational structure instead of loading all rows.
- Keep it isolated; no production wiring.

## Production alignment
- Relational querying has been lifted into `lilith/relational_concept_store.py` and wired behind a flag inside `ProductionConceptStore`.
- Property ingestion is sanitized via `lilith/concept_sanitizer.py`; run the cleanup helper on legacy DBs before enabling relational retrieval.

## Interpreting output
The script prints:
- Concept hits by term (with properties and relation counts).
- Properties for the term (top N).
- Relations for the term filtered by type.
- Reverse property lookup: concepts whose properties contain the substring.
- Timing for SQL vs. naive scan to highlight potential gains.

## Next steps (if promising)
- Add relation-aware retrieval to the production ConceptStore using parameterized SQL (not full table scans).
- Incorporate relation-based scoring into the ranker.
- Gate ingestion to ensure properties stay concise and relation-typed so SQL filters remain sharp.
