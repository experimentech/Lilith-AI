# MCP Adapter Skeleton

This note captures a minimal plan for exposing Lilith through MCP-style endpoints without changing core behavior.

## Goals
- Keep Lilith core untouched (`LilithSession`, retrieval, learning, stores).
- Provide a thin adapter that maps MCP tool calls to existing session methods.
- Support transient tools (e.g., weather/news) without polluting persistent knowledge.

## Files
- `lilith/mcp_adapter.py`: In-process adapter with handlers for chat, teach, feedback, stats, reset, and transient weather/news with TTL caching and optional live fetches.
- `lilith/mcp_server.py`: FastAPI shim exposing the adapter over HTTP for quick MCP-style testing.

## Usage Sketch
You supply the server/runtime (FastAPI, aiohttp, MCP runtime). The adapter simply orchestrates sessions:

```python
from lilith.mcp_adapter import MCPAdapter

adapter = MCPAdapter()

# Chat
resp = adapter.handle_chat(client_id="alice", message="Hello")
print(resp.text)

# Teach (writes to user store unless caller is teacher)
pattern_id = adapter.handle_teach("alice", trigger="What is Python?", response="A programming language")

# Feedback
adapter.handle_upvote("alice", pattern_id)

# Transient weather (no DB writes)
report = adapter.handle_weather("alice", location="London")
print(report.summary)
```

# Transient vs. Persistent Data
- **Transient tools (weather/news):** Do **not** persist to fragment stores. They return data only; no `add_pattern` calls. This avoids database bloat and confidence drift. TTL cache defaults to 5 minutes.
- **Session isolation:** One session per `(client_id, context_id)` is maintained to keep histories separate, but transient tool results are not stored.
- If later you want temporal memory for time-bound facts, consider a short-TTL cache or a separate transient store, and gate promotion into permanent stores via explicit teaching/confirmation.

## Extending
- Weather/news clients already attempt live fetches (Open-Meteo geocoding/forecast, HN search) with TTL caching and safe fallbacks; swap in your preferred providers or add API keys as needed.
- Add xiaozhi hardware hooks as separate MCP tools behind an allowlist; keep them side-effect guarded and logged.
- Wire modal routing hints (e.g., pass modality if the MCP client supplies it) to reuse existing math/code safeguards.

## Runtime flags
- `LILITH_QUIET=1` reduces noisy stdout logging from the SQLite-backed stores when running the adapter or server.
- `LILITH_MCP_MINIMAL=1` makes the Xiaozhi `/xiaozhi/ws` WebSocket send only `text` in the MCP payload (hiding `pattern_id`, confidence, fallback flags, and source) while leaving HTTP `/chat` responses unchanged.

## Safety Notes
- Keep base knowledge writes restricted to teacher/approved contexts.
- Log caller identity and tool name for auditing.
- Add rate limits for external API-backed tools to avoid abuse.

## Personality & Mood (planned, optional, core-wide)
- Profile schema (per session/client, default neutral/no-op):
	- `personality`: `tone`, `brevity`, `warmth`, `humor`, `interests: [tags]`, `aversions: [tags]`, `proactivity` (0–1).
	- `mood`: `label` (e.g., `neutral|curious|confident|cautious`), `emoji`, `decay` (drift back to neutral). When absent, omit from responses.
- Emission: optionally include `mood` in HTTP `/chat`, Xiaozhi `/xiaozhi/ws`, Discord bot, and CLI responses; gated by a feature flag; neutral omits the field. Implementation should live in core so adapters can share it.
- Retrieval/composer bias (gentle): small positive weight for matching `interests`, small negative for `aversions`; zeroed when neutral/flag off; safety filters remain first-class.
- Style pass (light touch): optional post-pass to adjust tone/verbosity; skipped when neutral/flag off.
- Proactivity (opt-in): if `proactivity > 0`, allow a single short, context-relevant follow-up only on high-confidence turns.
- Mood stub: start static defaults; later, allow simple heuristic updates (e.g., success streak → `confident/😌`, fallback streak → `cautious/🤔`) with decay back to neutral; never affects safety.

## Plasticity coupling (planned, bounded)
- Excitement/mood can gently modulate learning/plasticity per session:
	- Layer promotion: higher excitement may allow promotion to a mid/short-term layer; low excitement keeps items transient; teacher/approval gates still apply.
	- Reinforcement scale: apply a bounded factor (e.g., 0.8–1.2) to usage/success increments; global caps remain.
	- Decay: mildly slow decay for “interested” topics or speed it for cautious mood (e.g., factors 0.9–1.1).
	- Gating: cautious mood can require an extra positive signal before committing; confident mood allows normal thresholds but never bypasses safety.
- Isolation: mood/personality influences are per `(client_id, context_id)`; shared/base layers stay stable unless explicitly enabled.
- Safety: personality/mood never override content filters, score floors/ceilings, or access controls.
