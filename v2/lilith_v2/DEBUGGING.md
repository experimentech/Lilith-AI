# Debugging pathways (keep clutter low)

- Prefer structured logs via an `Observability` implementation; emit JSON with `trace_id`, `node_id`, `event_type`, and minimal payloads.
- Use spans around stage encode/retrieve/learn and I/O send/receive to trace latency; avoid ad-hoc prints.
- Gate verbose logs behind a `debug` flag in context/meta; default to info-level summaries.
- Keep per-tenant and per-modality counters to spot hot spots; avoid metric cardinality explosions (bucket IDs).
- Add temporary taps by swapping in `InMemoryIOPort` on a node binding instead of sprinkling logging across nodes.
- For routing issues, log the `RouteDecision` once per change (dedupe by descriptor+policy) rather than per message.
- Health probes: shallow receive/send on each bound port plus a cheap Stage `stats()` call; surface in readiness endpoints.
