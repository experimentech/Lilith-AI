# Lilith V2 - Autodidact Cognitive Agent

> **See [v2/docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for detailed architecture documentation.**

## Overview

Protocol-style stubs for the v2 tree: stages, stores, I/O ports, MCP routing, PMFlow state, persistence, observability, and node-to-port bindings for multi-modal/multi-tenant attachment. Includes an in-memory port, bindings resolver, router using modality/tenant-aware decisions, and sample wiring.

## Key Files
- **Somatic Layer ("The Body"):** `somatic_layer.py`, `app.py`
- **Cognitive Core ("The Brain"):** `cognitive_stage.py`, `pmflow_physics.py`
- **Interfaces ("Limbs & Senses"):** `io_port.py`, `mcp_transport.py`, `fs_tools.py`, `terminal_tools.py`
- **Specialized Branches:** `math_stage.py`, `math_system.py`

## Configuration
- `LILITH_ENABLE_FS` (default: true): Enable file system tools.
- `LILITH_ENABLE_TERMINAL` (default: false): Enable terminal execution tools (use with caution).


## Structure Details
- stage.py / store.py / io_port.py / observability.py: core Protocols.
- mcp_router.py: EndpointType, MCPDescriptor, RouteDecision, SimpleMCPRouter.
- bindings.py: NodeBindings, InMemoryBindingResolver.
- in_memory_port.py: simple FIFO port for tests and taps.
- in_memory_store.py: simple in-memory Store with TTL/decay.
- noop_stage.py: minimal Stage for end-to-end wiring tests.
- wiring.py: sample binding map and route invocation.
- observability.py: NoopObservability, JsonLoggerObservability.
- health.py: probes ports and stages for readiness signals.
- vscode_endpoint.py: stub VS Code MCP adapter and binding helper.
- vscode_demo.py: routes VS Code stub through router and port, prints decision.
- app.py: minimal V2Runtime wiring router, bindings, stages, observability.
- health_demo.py: example running probes with JSON logging.
- config_loader.py: load bindings config from JSON/YAML.
- json_file_store.py: file-backed Store for persistence.
- relational_store.py: SQLite-backed Store for relational/SQL path.
- persistent_stage.py: Store-backed Stage persisting events.
- concepts_stage.py: lightweight concept Stage with merge and loose schema.
- test_persistence.py: coverage for file store and persistent stage.
- test_relational_store.py: coverage for relational store.
- test_config_loader.py: coverage for config loader.
- test_concepts_stage.py: coverage for concept stage.
- configs/bindings.example.json: sample bindings config for runtime.
- configs/bindings.concepts.json: sample config enabling concepts stage with file-backed store.
- mcp_transport.py: mockable MCP transport interface.
- mcp_transport_demo.py: demo wiring mock transport through runtime.
- test_mcp_transport.py: transport/adapter coverage.
- cli.py: simple CLI runner for runtime with mock transport and JSON config.
- DEBUGGING.md: low-clutter debug pathways.
- pmflow_sqlite.py: SQLite-backed PMFlow state store (versioned state per branch).
- relational_event_store.py: append/query event log keyed by branch/kind.
- persistence_sqlite.py: minimal persistence wrapper with tx/decay hooks.

Routing policy keys (branch_policy)
- allow / deny: list of node_ids eligible for routing.
- allow_modalities / deny_modalities: filter decision by modality string from context.
- allow_tenants / deny_tenants: filter decision by tenant string from context.
- allow_endpoint_types / deny_endpoint_types: filter by descriptor.type (EndpointType value).
- timeout_ms: per-branch timeout hint; falls back to descriptor.latency_hint_ms.
- concurrency_limit: max concurrent invocations for this decision.
- per_tenant_limit: optional per-tenant cap.
- error_policy: dict with retry/backoff, circuit-breaker, or fail-open flags (shallow stub today).

Binding fields
- node_id: hierarchical binding id.
- ports: list of IOPort instances (e.g., VSCodeMCPAdapter).
- modalities: optional set of allowed modalities for this binding.
- tenants: optional set of allowed tenants for this binding.
- endpoint_types: optional set of allowed EndpointType values for this binding (auto-set by make_vscode_binding).

PMFlow/relational config (optional)
- pmflow.state_path: path to SQLite file for PMFlow state table (pmflow_state).
- pmflow.event_path: path to SQLite file for branch-scoped events table.
- pmflow.enabled: optional flag; any pmflow block enables creation by default.
