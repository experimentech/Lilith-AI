"""MCP Tool Stream (remote endpoints as a cognitive I/O stream).

Goal
- Accept a list of MCP JSON-RPC endpoints (websocket) and treat them as a concurrent
  observation stream that can enrich Lilith's text pipeline.
- Avoid hardcoding individual tools/modalities. Selection is driven by tool metadata
  (name/description/schema) + lightweight lexical scoring.

This module is intentionally conservative:
- It only auto-calls tools when it can satisfy required args generically.
- It time-bounds remote calls and fails closed (no tool output) on errors.

Transport
- Uses MCP JSON-RPC methods: initialize, tools/list, tools/call.
- Default implementation uses `websockets` sync client (keeps Lilith session code sync).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass(frozen=True)
class MCPRemoteEndpoint:
    name: str
    url: str


@dataclass(frozen=True)
class MCPToolInfo:
    server: str
    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass(frozen=True)
class MCPToolCallResult:
    tool: MCPToolInfo
    ok: bool
    text: str
    raw: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class MCPJsonRpcError(RuntimeError):
    pass


class MCPWebSocketClient:
    """Minimal MCP JSON-RPC client over websocket.

    This is a per-request client (connect, request, close) to keep lifecycle simple.
    Pooling can be added later if needed.
    """

    def __init__(self, url: str, timeout_seconds: float = 3.0):
        self.url = url
        self.timeout_seconds = timeout_seconds
        self._next_id = 1

        try:
            from websockets.sync.client import connect  # type: ignore

            self._connect = connect
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "websockets sync client not available; ensure websockets>=12 is installed"
            ) from exc

    def _request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1

        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method, "id": req_id}
        if params is not None:
            payload["params"] = params

        with self._connect(
            self.url,
            open_timeout=self.timeout_seconds,
            close_timeout=self.timeout_seconds,
        ) as ws:
            # Some servers (e.g., Lilith Xiaozhi shim) send an initial hello envelope.
            # Peek once with a short timeout to detect this behavior.
            xiaozhi_envelope = False
            try:
                first = ws.recv(timeout=min(self.timeout_seconds, 0.25))
                if isinstance(first, str) and first:
                    maybe = json.loads(first)
                    if isinstance(maybe, dict) and maybe.get("type") == "hello":
                        xiaozhi_envelope = True
            except Exception:
                # Timeout or parse failure -> assume raw JSON-RPC.
                xiaozhi_envelope = False

            if xiaozhi_envelope:
                ws.send(json.dumps({"type": "mcp", "payload": payload}))
                raw = ws.recv(timeout=self.timeout_seconds)
                if raw is None:
                    raise MCPJsonRpcError(f"No response for method={method}")
                outer = json.loads(raw)
                resp = outer.get("payload") if isinstance(outer, dict) else None
                if not isinstance(resp, dict):
                    raise MCPJsonRpcError("Invalid Xiaozhi MCP envelope")
            else:
                ws.send(json.dumps(payload))
                # Sync API returns text frames as str.
                raw = ws.recv(timeout=self.timeout_seconds)
                if raw is None:
                    raise MCPJsonRpcError(f"No response for method={method}")
                resp = json.loads(raw)

        if resp.get("id") != req_id:
            raise MCPJsonRpcError("Mismatched JSON-RPC id")
        if "error" in resp and resp["error"] is not None:
            raise MCPJsonRpcError(str(resp["error"]))
        return resp

    def initialize(self) -> Dict[str, Any]:
        return self._request("initialize")

    def list_tools(self) -> List[Dict[str, Any]]:
        resp = self._request("tools/list")
        result = resp.get("result") or {}
        tools = result.get("tools") or []
        if not isinstance(tools, list):
            return []
        return tools

    def call_tool(self, *, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._request("tools/call", params={"name": name, "arguments": arguments or {}})
        return resp


def _tokenize(text: str) -> List[str]:
    parts = re.split(r"[^a-zA-Z0-9_]+", (text or "").lower())
    return [p for p in parts if p and len(p) >= 2]


def _schema_required_fields(schema: Dict[str, Any]) -> List[str]:
    req = schema.get("required")
    if isinstance(req, list):
        return [str(x) for x in req]
    return []


def _schema_properties(schema: Dict[str, Any]) -> Dict[str, Any]:
    props = schema.get("properties")
    if isinstance(props, dict):
        return props
    return {}


def _extract_location(text: str) -> Optional[str]:
    # Generic heuristic: "weather in X", "in X", "for X".
    m = re.search(r"\b(?:in|for)\s+([a-zA-Z][a-zA-Z\s\-]{1,40})\b", text)
    if not m:
        return None
    loc = m.group(1).strip()

    # Stop at common punctuation if the regex captured beyond it.
    loc = re.split(r"[\?\.!,:;]", loc, maxsplit=1)[0].strip()

    # Trim trailing temporal/command words commonly appended after locations.
    # Examples: "in London today", "in Paris right now", "for Berlin please".
    loc = re.sub(
        r"\s+\b(?:today|tonight|tomorrow|now|currently|please)\b\s*$",
        "",
        loc,
        flags=re.IGNORECASE,
    ).strip()
    loc = re.sub(
        r"\s+\b(?:right\s+now)\b\s*$",
        "",
        loc,
        flags=re.IGNORECASE,
    ).strip()

    # Final trim for any leftover punctuation.
    loc = loc.strip(" \t\r\n\"'“”‘’")
    return loc or None


def build_generic_arguments(user_input: str, input_schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fill required args using generic field-name heuristics.

    This avoids tool-specific hardcoding: only common argument *roles* are recognized.
    If required fields can't be satisfied, returns None.
    """

    required = _schema_required_fields(input_schema)
    props = _schema_properties(input_schema)

    # If no schema or no required fields, call with empty args.
    if not required:
        return {}

    out: Dict[str, Any] = {}
    for field in required:
        f = field.lower()
        if f in {"text", "prompt", "message", "input"}:
            out[field] = user_input
            continue
        if f in {"query", "topic", "subject", "q"}:
            out[field] = user_input
            continue
        if f in {"location", "city", "place"}:
            loc = _extract_location(user_input)
            if loc:
                out[field] = loc
                continue

        # If the field is optional (present in properties but not required), ignore.
        # Here: it's required but we don't know how to fill it.
        _ = props.get(field)
        return None

    return out


def score_tool_for_text(tool: MCPToolInfo, user_input: str) -> float:
    """Lexical score based on token overlap with name/description."""

    q = set(_tokenize(user_input))
    if not q:
        return 0.0

    meta = set(_tokenize(tool.name) + _tokenize(tool.description))
    if not meta:
        return 0.0

    overlap = len(q & meta)
    return float(overlap) / float(len(q))


class MCPToolStream:
    """Multi-endpoint MCP tool registry + concurrent caller."""

    def __init__(
        self,
        endpoints: Sequence[MCPRemoteEndpoint],
        *,
        timeout_seconds: float = 3.0,
        cache_ttl_seconds: float = 60.0,
        max_workers: int = 8,
    ) -> None:
        self.endpoints = list(endpoints)
        self.timeout_seconds = float(timeout_seconds)
        self.cache_ttl_seconds = float(cache_ttl_seconds)
        self.max_workers = int(max_workers)

        self._tool_cache: Tuple[float, List[MCPToolInfo]] = (0.0, [])

    def list_tools(self, *, force_refresh: bool = False) -> List[MCPToolInfo]:
        now = time.time()
        expires, cached = self._tool_cache
        if (not force_refresh) and cached and now < expires:
            return list(cached)

        tools: List[MCPToolInfo] = []
        for ep in self.endpoints:
            try:
                client = MCPWebSocketClient(ep.url, timeout_seconds=self.timeout_seconds)
                raw_tools = client.list_tools()
                for t in raw_tools:
                    name = str(t.get("name") or "")
                    if not name:
                        continue
                    tools.append(
                        MCPToolInfo(
                            server=ep.name,
                            name=name,
                            description=str(t.get("description") or ""),
                            input_schema=dict(t.get("inputSchema") or {}),
                        )
                    )
            except Exception:
                continue

        self._tool_cache = (now + self.cache_ttl_seconds, tools)
        return list(tools)

    def select_tools(self, user_input: str, *, topk: int = 2, min_score: float = 0.34) -> List[MCPToolInfo]:
        tools = self.list_tools()
        scored: List[Tuple[MCPToolInfo, float]] = [(t, score_tool_for_text(t, user_input)) for t in tools]
        scored = [(t, s) for (t, s) in scored if s >= min_score]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[: max(0, int(topk))]]

    def call_selected(self, user_input: str, *, topk: int = 2) -> List[MCPToolCallResult]:
        selected = self.select_tools(user_input, topk=topk)
        if not selected:
            return []

        # Group by server endpoint.
        endpoint_by_name = {e.name: e for e in self.endpoints}

        def _call(tool: MCPToolInfo) -> MCPToolCallResult:
            ep = endpoint_by_name.get(tool.server)
            if not ep:
                return MCPToolCallResult(tool=tool, ok=False, text="", error="unknown_server")

            args = build_generic_arguments(user_input, tool.input_schema)
            if args is None:
                return MCPToolCallResult(tool=tool, ok=False, text="", error="unsatisfied_required_args")

            try:
                client = MCPWebSocketClient(ep.url, timeout_seconds=self.timeout_seconds)
                resp = client.call_tool(name=tool.name, arguments=args)
                result = resp.get("result") or {}
                content = result.get("content") or []
                text_parts: List[str] = []
                if isinstance(content, list):
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "text":
                            text_parts.append(str(item.get("text") or ""))
                text_out = "\n".join([p for p in text_parts if p.strip()]).strip()
                return MCPToolCallResult(tool=tool, ok=True, text=text_out, raw=resp)
            except Exception as exc:
                return MCPToolCallResult(tool=tool, ok=False, text="", error=str(exc))

        results: List[MCPToolCallResult] = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(selected)))) as ex:
            futs = [ex.submit(_call, t) for t in selected]
            for f in as_completed(futs, timeout=max(self.timeout_seconds, 0.1) * max(1, len(futs))):
                try:
                    results.append(f.result())
                except Exception as exc:
                    # Treat as a failed tool call with unknown tool.
                    results.append(
                        MCPToolCallResult(
                            tool=MCPToolInfo(server="", name="", description="", input_schema={}),
                            ok=False,
                            text="",
                            error=str(exc),
                        )
                    )

        # Deterministic order for downstream context: server/name.
        results.sort(key=lambda r: (r.tool.server, r.tool.name))
        return results


def summarize_tool_results(results: Sequence[MCPToolCallResult], *, max_chars: int = 500) -> str:
    """Create a compact, provenance-carrying context snippet."""

    chunks: List[str] = []
    for r in results:
        if not r.ok or not r.text:
            continue
        txt = r.text.strip().replace("\n", " ")
        if len(txt) > max_chars:
            txt = txt[: max_chars - 3].rstrip() + "..."
        chunks.append(f"MCP[{r.tool.server}:{r.tool.name}]: {txt}")

    return "\n".join(chunks).strip()


__all__ = [
    "MCPRemoteEndpoint",
    "MCPToolInfo",
    "MCPToolCallResult",
    "MCPToolStream",
    "summarize_tool_results",
    "build_generic_arguments",
]
