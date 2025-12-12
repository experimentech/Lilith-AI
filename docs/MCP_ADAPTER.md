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
