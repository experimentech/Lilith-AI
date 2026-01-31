Quick notes about v2

Refer to the structure as a tree and not a pipeline because it is more accurate

Draft tree sketch:
- Trunk (purely symbolic / agentic): deliberation, planning, capability routing, goal/trace state.
- Branches (domain stages): language (syntax/pragmatic), concepts+relational, world/grounding, tools/MCP, memory.
- Leaves (I/O and effectors): chat transport, MCP endpoint callers, external stores/logs. Typed ports connect branches to leaves.

Try to utilise something based on cognitive_stage_base.py for all nodes

Core interfaces (lightweight):
- Stage: encode, retrieve, learn, update_success, stats; optional relational sidecar (SQL joins for concepts).
- Store: CRUD with hygiene/decay hooks; SQLite/JSON backend.
- I/O port: typed send/receive, async/stream-friendly, with trace IDs.
- MCP client: enumerate endpoints, type them (knowledge/action/retrieval/generation), expose as ports; router maps classes to branches.

Re-evaluate interfaces. they are based on older versions and are inhibiting development

Name/ID scheme: hierarchical IDs (stage.node.leaf) to avoid duplication; reserve namespaces per branch.

ensure there is no name duplication

Try to have a base module for pmflow and database access to aid in maintenance and development

PMFlow/DB base: shared PMFlow state load/save, latent dims per branch, unified persistence wrapper (SQLite/JSON) to cut bespoke caches.

pre-plan nodes, branches, leaves whatever based on the desired layout of v1

Desired layout example (v1-aligned):
- Trunk: agentic planner + deliberation + capability router.
- Branches: syntax/pragmatic; concepts+relational; world model; tool/MCP; memory (short/long-term); preferences/persona.
- Leaves: chat transport; MCP endpoints grouped by type; file/log writers.

Try to have a more universal API for I/O

I/O API: async-friendly send/receive with metadata (trace IDs, user/session), backpressure/queueing; adapters per transport.

Integrate MCP with enumeration and ability for v2 to detect arbitrary endpoints and utilise them.

not all MCP endpoints will be useful interfacing with the same node type. Keep this in mind.

MCP typing/routing: classify endpoints; allow per-node allowlist/denylist so only relevant branches consume them.

Fully utilise PMFlow including the agentic physics and continuous / not turn operated / not input driven architecture.

Continuous/agentic loop: trunk ticks on a scheduler (not turn-bound); stages consume/emit events; PMFlow states updated incrementally; I/O decoupled.

Ensure core function is decoupled from I/O to allow for background and more complex processing

Migration/compat: decide if v2 starts clean or ships a shim to read v1 stores; run hygiene/decay before migrating.

Observability: standard trace/log/metrics hooks per node/port; structured events with source IDs; minimal counters (latency, hit/miss, errors).

Concrete interfaces to populate
- Stage interface: encode(input, ctx), retrieve(query, ctx), learn(fact/event, ctx), update_success(feedback, ctx), stats() -> metrics; optional relational_sidecar(sql_context) for concept joins.
- Store interface: get(id), put(id, value, ttl=None), list(prefix/filter), delete(id), decay(now), sanitize(value); backends SQLite/JSON share same signature.
- I/O port interface: send(message, meta), receive(meta) -> stream/events, ack(id), nack(id, reason), flush(), attach(adapter) to transport; all carry trace_id/session/user IDs.
- MCP router contract: classify endpoints into knowledge/action/retrieval/generation/ops; router maps endpoint class -> branch allowlist/denylist, returns port bindings and error policy (retry/backoff/fail-open flags).
- Observability hooks: on_event(event_type, node_id, payload, trace_id), counters for latency/hit/miss/error, span IDs propagated across ports.

PMFlow/DB base (shared contract)
- PMFlow state: load_state(branch_id), save_state(branch_id, state, version), bump_version(branch_id), latent_dims(branch_id) -> tuple, compact() for cleanup.
- Unified persistence wrapper: open_store(path, schema_version), migrate(schema_version), tx(fn) transactional helper; hygiene hooks: sanitize(record), decay(now), validate(schema).
- Shim path: read v1 store with adapter, run hygiene/decay pass, emit v2-compatible state; flag to run once on bootstrap.

I/O ports and leaves
- Ports expose async send/receive with backpressure: queue limits, drop/park policy, retry with jitter; adapters for chat, MCP calls, file/log writers.
- Ports carry metadata envelopes: trace_id, session, user, node_id, auth/context; serialization pluggable (JSON/MsgPack) but typed frames.
- Leaves register via typed adapters; trunk/branches bind to ports via config (per branch allow/deny list and capacity).

MCP typing/routing details
- Endpoint classification enums: knowledge (read-only data), retrieval (search/vector), action (writes/side-effects), generation (LLM-ish), ops (admin/health).
- Router inputs: endpoint descriptors (name, type, cost, latency hint, required auth), branch policies (allow/deny, cost budget), context (trace, user intent).
- Router outputs: target port(s), concurrency limit, timeout, fallback chain; error policy (retry/backoff, circuit-breaker, fail-closed vs fail-open).
Migration/observability checklist
- Decide: clean v2 stores vs shimmed v1 import; if shim, run sanitize/decay before commit.
- Standard logs: structured JSON with node_id/stage/trace_id; metrics: latency, hit/miss, errors, queue depth; traces: spans per stage/port.
- Health: per-branch readiness and liveness; synthetic probes for MCP endpoints; alert thresholds on error rate/latency/backlog.

Prototype stubs (Python-ish, non-final)

```python
from typing import Protocol, Any, Iterable, Optional, Dict, Tuple, Callable


class Stage(Protocol):
	id: str  # hierarchical e.g., "trunk.concepts"

	def encode(self, item: Any, ctx: Dict[str, Any]) -> Any: ...
	def retrieve(self, query: Any, ctx: Dict[str, Any]) -> Iterable[Any]: ...
	def learn(self, event: Any, ctx: Dict[str, Any]) -> None: ...
	def update_success(self, feedback: Any, ctx: Dict[str, Any]) -> None: ...
	def stats(self) -> Dict[str, Any]: ...

	# Optional relational sidecar for concept joins
	def relational_sidecar(self, sql_ctx: Any) -> Optional[Any]: ...


class Store(Protocol):
	def get(self, key: str) -> Optional[Any]: ...
	def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None: ...
	def list(self, prefix: Optional[str] = None) -> Iterable[Tuple[str, Any]]: ...
	def delete(self, key: str) -> None: ...
	def decay(self, now: float) -> int: ...  # returns items touched
	def sanitize(self, value: Any) -> Any: ...


class IOPort(Protocol):
	def send(self, message: Any, meta: Dict[str, Any]) -> None: ...
	def receive(self, meta: Dict[str, Any]) -> Iterable[Any]: ...
	def ack(self, message_id: str) -> None: ...
	def nack(self, message_id: str, reason: str) -> None: ...
	def flush(self) -> None: ...
	def attach(self, adapter: Any) -> None: ...


class MCPDescriptor:
	name: str
	type: str  # knowledge/retrieval/action/generation/ops
	cost: float
	latency_hint_ms: int
	auth: Optional[str]


class MCPRouter(Protocol):
	def classify(self, descriptor: MCPDescriptor) -> str: ...
	def route(
		self,
		descriptor: MCPDescriptor,
		context: Dict[str, Any],
		branch_policy: Dict[str, Any],
	) -> Dict[str, Any]: ...  # ports, timeouts, error policy


class PMFlowStateStore(Protocol):
	def load_state(self, branch_id: str) -> Dict[str, Any]: ...
	def save_state(self, branch_id: str, state: Dict[str, Any], version: int) -> None: ...
	def bump_version(self, branch_id: str) -> int: ...
	def latent_dims(self, branch_id: str) -> Tuple[int, ...]: ...
	def compact(self) -> None: ...


class PersistenceWrapper(Protocol):
	def open_store(self, path: str, schema_version: int) -> Store: ...
	def migrate(self, target_version: int) -> None: ...
	def tx(self, fn: Callable[[], Any]) -> Any: ...
	def sanitize(self, record: Any) -> Any: ...
	def decay(self, now: float) -> int: ...
	def validate(self, record: Any) -> bool: ...


class Observability(Protocol):
	def on_event(self, event_type: str, node_id: str, payload: Dict[str, Any], trace_id: str) -> None: ...
	def counter(self, name: str, value: float, meta: Dict[str, Any]) -> None: ...
	def span(self, name: str, trace_id: str, fn: Callable[[], Any]) -> Any: ...
```

