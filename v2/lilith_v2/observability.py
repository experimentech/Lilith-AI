import json
import sys
import time
from typing import Any, Callable, Dict, Protocol


class Observability(Protocol):
    """Observability hooks for logs/metrics/traces."""

    def on_event(self, event_type: str, node_id: str, payload: Dict[str, Any], trace_id: str) -> None:
        ...

    def counter(self, name: str, value: float, meta: Dict[str, Any]) -> None:
        ...

    def span(self, name: str, trace_id: str, fn: Callable[[], Any]) -> Any:
        ...


class NoopObservability(Observability):
    """No-op implementation useful for tests."""

    def on_event(self, event_type: str, node_id: str, payload: Dict[str, Any], trace_id: str) -> None:
        return None

    def counter(self, name: str, value: float, meta: Dict[str, Any]) -> None:
        return None

    def span(self, name: str, trace_id: str, fn: Callable[[], Any]) -> Any:
        return fn()


class JsonLoggerObservability(Observability):
    """Structured JSON logger; writes to stdout by default."""

    def __init__(self, stream=None) -> None:
        self._stream = stream or sys.stdout

    def on_event(self, event_type: str, node_id: str, payload: Dict[str, Any], trace_id: str) -> None:
        entry = {
            "type": event_type,
            "node_id": node_id,
            "trace_id": trace_id,
            "payload": payload,
            "ts": time.time(),
        }
        self._stream.write(json.dumps(entry) + "\n")

    def counter(self, name: str, value: float, meta: Dict[str, Any]) -> None:
        entry = {
            "type": "counter",
            "name": name,
            "value": value,
            "meta": meta,
            "ts": time.time(),
        }
        self._stream.write(json.dumps(entry) + "\n")

    def span(self, name: str, trace_id: str, fn: Callable[[], Any]) -> Any:
        start = time.time()
        try:
            return fn()
        finally:
            duration_ms = (time.time() - start) * 1000.0
            entry = {
                "type": "span",
                "name": name,
                "trace_id": trace_id,
                "duration_ms": duration_ms,
                "ts": time.time(),
            }
            self._stream.write(json.dumps(entry) + "\n")
