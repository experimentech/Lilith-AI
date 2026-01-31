# lilith_v2 stubs

Protocol-style stubs for the v2 tree: stages, stores, I/O ports, MCP routing, PMFlow state, persistence, observability, and node-to-port bindings for multi-modal/multi-tenant attachment. Includes an in-memory port, binding resolver, router using modality/tenant-aware decisions, and sample wiring. Replace ellipses with concrete implementations per branch.

Files of interest:
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
