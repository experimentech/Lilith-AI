"""MCPStage: treat MCP tool access as a first-class pipeline stage.

This stage sits alongside other "perception"-like stages: it can decide whether
(and which) remote MCP tools to call, and returns a structured artifact that can
be logged, traced, and credit-assigned via eligibility traces.

Design goals
- Conservative by default: fail-closed and avoid calling tools on chit-chat.
- Data-driven: selection uses tool metadata + lightweight lexical scoring.
- Structured: returns an artifact (decision + candidates + results + summary).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lilith.mcp_tool_stream import MCPToolCallResult, MCPToolInfo, MCPToolStream, score_tool_for_text, summarize_tool_results


@dataclass(frozen=True)
class MCPCandidate:
    tool: MCPToolInfo
    score: float
    args_satisfiable: bool


@dataclass
class MCPStageArtifact:
    enabled: bool
    mode: str
    decision: str  # "skipped" | "sensed" | "called"
    reason: str
    candidates: List[MCPCandidate]
    results: List[MCPToolCallResult]
    summary: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "mode": str(self.mode),
            "decision": str(self.decision),
            "reason": str(self.reason),
            "confidence": float(self.confidence),
            "candidates": [
                {
                    "server": c.tool.server,
                    "name": c.tool.name,
                    "description": c.tool.description,
                    "score": float(c.score),
                    "args_satisfiable": bool(c.args_satisfiable),
                }
                for c in (self.candidates or [])
            ],
            "results": [
                {
                    "server": r.tool.server,
                    "name": r.tool.name,
                    "ok": bool(r.ok),
                    "text": r.text,
                    "error": r.error,
                }
                for r in (self.results or [])
            ],
            "summary": self.summary,
        }


class MCPStage:
    """Gated MCP tool-calling stage."""

    def __init__(
        self,
        tool_stream: MCPToolStream,
        *,
        mode: str = "gated",
        topk: int = 2,
        min_score: float = 0.34,
        direct_answer_enabled: bool = True,
        direct_answer_max_chars: int = 600,
    ) -> None:
        self.tool_stream = tool_stream
        self.mode = str(mode or "gated").strip().lower()
        self.topk = int(topk)
        self.min_score = float(min_score)
        self.direct_answer_enabled = bool(direct_answer_enabled)
        self.direct_answer_max_chars = int(direct_answer_max_chars)

    def process(self, user_input: str) -> MCPStageArtifact:
        if not user_input or not user_input.strip():
            return MCPStageArtifact(
                enabled=False,
                mode=self.mode,
                decision="skipped",
                reason="empty_input",
                candidates=[],
                results=[],
                summary="",
                confidence=0.0,
            )

        if self.mode in {"off", "disabled", "false", "0"}:
            return MCPStageArtifact(
                enabled=False,
                mode=self.mode,
                decision="skipped",
                reason="disabled",
                candidates=[],
                results=[],
                summary="",
                confidence=0.0,
            )

        tools = self.tool_stream.list_tools()
        scored: List[Tuple[MCPToolInfo, float]] = [(t, score_tool_for_text(t, user_input)) for t in tools]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Keep only plausible candidates (but preserve ordering for diagnostics).
        top_scored = scored[: max(1, self.topk * 3)]
        candidates: List[MCPCandidate] = []
        for tool, score in top_scored:
            if score <= 0.0:
                continue
            # Check required args satisfiable using the same generic builder logic.
            try:
                from lilith.mcp_tool_stream import build_generic_arguments

                args_ok = build_generic_arguments(user_input, tool.input_schema) is not None
            except Exception:
                args_ok = False
            candidates.append(MCPCandidate(tool=tool, score=float(score), args_satisfiable=bool(args_ok)))

        best_score = float(candidates[0].score) if candidates else 0.0

        if self.mode == "always_sense":
            return MCPStageArtifact(
                enabled=True,
                mode=self.mode,
                decision="sensed",
                reason="always_sense",
                candidates=candidates,
                results=[],
                summary="",
                confidence=min(1.0, best_score),
            )

        # Select call set: satisfiable + score threshold.
        viable = [c for c in candidates if c.args_satisfiable and c.score >= self.min_score]
        viable = viable[: max(0, self.topk)]

        if not viable:
            return MCPStageArtifact(
                enabled=True,
                mode=self.mode,
                decision="skipped",
                reason="no_viable_tools",
                candidates=candidates,
                results=[],
                summary="",
                confidence=min(1.0, best_score),
            )

        # Gating: in gated mode, only call when input looks like it benefits from tools.
        if self.mode == "gated" and not self._should_call_tools(user_input, best_score):
            return MCPStageArtifact(
                enabled=True,
                mode=self.mode,
                decision="skipped",
                reason="gated_off",
                candidates=candidates,
                results=[],
                summary="",
                confidence=min(1.0, best_score),
            )

        # always_call or gated+passed.
        try:
            results = self.tool_stream.call_selected(user_input, topk=self.topk, min_score=self.min_score)
        except Exception:
            results = []

        summary = summarize_tool_results(results)
        ok_count = len([r for r in results if getattr(r, "ok", False) and getattr(r, "text", "").strip()])
        confidence = 0.0
        if ok_count > 0:
            confidence = max(best_score, 0.65)
        else:
            confidence = best_score

        return MCPStageArtifact(
            enabled=True,
            mode=self.mode,
            decision="called",
            reason="called_tools",
            candidates=candidates,
            results=results,
            summary=summary,
            confidence=min(1.0, float(confidence)),
        )

    def _should_call_tools(self, user_input: str, best_score: float) -> bool:
        """Heuristic gating for tool calls.

        Rationale
        - Tools are expensive/fragile vs internal pattern retrieval.
        - Call tools when the request is likely time-sensitive, lookup-oriented,
          or when tool metadata match is strong.
        """

        t = (user_input or "").strip().lower()
        if not t:
            return False

        # Strong metadata match -> call even if phrased like a command.
        if best_score >= (self.min_score + 0.10):
            return True

        if "?" in t:
            return True

        # Generic lookup intent signals (keep short and domain-agnostic).
        starters = (
            "what ",
            "who ",
            "when ",
            "where ",
            "why ",
            "how ",
            "show ",
            "find ",
            "lookup ",
            "search ",
            "get ",
            "current ",
            "latest ",
            "today ",
            "now ",
        )
        if t.startswith(starters):
            return True

        return False


__all__ = [
    "MCPCandidate",
    "MCPStage",
    "MCPStageArtifact",
]
