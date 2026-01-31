# MCP Integration Plan (Understanding + Path Forward)

## What Lilith Is (from source/docs)
- Layered BioNN + Database architecture: each cognitive layer pairs a PMFlow/BioNN encoder with a database; embeddings drive retrieval instead of monolithic weights. See docs/ARCHITECTURE_VERIFICATION.md and docs/PROJECT_SCOPE.md.
- Cognitive tree metaphor: core reasoning/memory/feedback form the trunk; branches are modality pipelines (language, math, vision, speech, devices, MCP tools). Branches can add their own sublayers while sharing the trunk’s state and learning signals.
- I/O modularity: interface layer is swappable (CLI, Discord, planned REST); core session logic stays constant. See docs/io_modularity.md.
- Core runtime: `LilithSession` wraps encoder, multi-tenant fragment store, conversation state/history, composer, modal routing, grammar, and knowledge augmentation. See lilith/session.py and lilith/response_composer.py.
- Storage: `MultiTenantFragmentStore` provides base and user stores, optional concept store, vocabulary tracker, and pattern extractor. See lilith/multi_tenant_store.py.
- Knowledge augmentation: external sources (Wikipedia, Wiktionary, etc.) are consulted when confidence is low; results are transient unless promoted via teaching/upvote. See lilith/knowledge_augmenter.py and usage in lilith/response_composer.py.

## Current MCP State
- MCP is optional and decoupled: adapter and FastAPI shim live in lilith/mcp_adapter.py and lilith/mcp_server.py. They map MCP-style calls to `LilithSession` (chat/teach/feedback/stats/weather/news) but do not alter core flows by default.
- Tests cover adapter routing and transient tool non-persistence: tests/test_mcp_adapter.py.
- MCP tool stream can be enabled per session as a first-class stage ("MCP tool stage"). It produces a structured artifact (decision/candidates/results/summary/confidence) that is passed into the composer as an explicit input. Default behavior is gated and fail-closed.

## Principles for MCP as a Modality (no hacks)
- Separation: keep MCP transport/adapters outside core; integrate through explicit, intent-gated augmentation hooks.
- Transience first: MCP results should enter working memory for the current turn; persistence only via explicit teaching/upvote to avoid DB bloat/confidence drift.
- Intent/modal routing: use existing modal/intent signals to decide when to call MCP (e.g., weather/news/tool-specific intents). Fail open to existing augmentation (Wiki/dictionary) when MCP is unavailable.
- Safety and QoS: per-source rate limits, health checks, TTL caches, and provenance logging; soft-fail to core retrieval.
- Symmetry: treat MCP sources like other augmentation modalities (not new storage) unless explicitly promoted to stores.

## Proposed API-Safe Path (incremental)
1) Keep the adapter/server thin; expose MCP endpoints via FastAPI (already present) or other runtimes.
2) Integrate MCP through a stage artifact (not string concatenation): pass the artifact to the composer, and keep persistence explicit (teach/upvote only).
3) Provide a small, explicit configuration surface so behavior is not hidden.
4) Add observability: log caller/tool, latency, success/failure, and user feedback links so promotion decisions can be manual/explicit.
5) Extend tests: intent-gated MCP augmentation, transient-only behavior, and fallback to legacy augmentation when MCP unavailable.

## References (key files)
- docs/PROJECT_SCOPE.md
- docs/ARCHITECTURE_VERIFICATION.md
- docs/io_modularity.md
- lilith/session.py
- lilith/response_composer.py
- lilith/knowledge_augmenter.py
- lilith/multi_tenant_store.py
- lilith/mcp_adapter.py
- lilith/mcp_server.py
- lilith/mcp_tool_stream.py
- lilith/mcp_stage.py
- tests/test_mcp_adapter.py
