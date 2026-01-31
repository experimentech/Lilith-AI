from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Iterable, Optional

from .store import Store


def _jsonify(value: Any) -> Any:
    """Best-effort conversion to JSON-friendly structures."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "squeeze"):
            value = value.squeeze()
        if hasattr(value, "tolist"):
            return value.tolist()
    except Exception:
        return None
    return value


class PatternStoreAdapter:
    """Adapter that maps the legacy PatternStore protocol onto the v2 Store.

    Patterns are persisted as JSON-serializable dicts keyed under
    "{namespace}:pattern:{id}". Values are sanitized via the underlying Store
    to keep backend-specific constraints (e.g., SQLite JSON) intact.
    """

    def __init__(self, store: Store, namespace: str) -> None:
        self.store = store
        self.namespace = namespace

    def add_pattern(
        self,
        fragment_id: Optional[str],
        trigger_context: str,
        response_text: str,
        intent: str,
        success_score: float,
        embedding: Optional[Any] = None,
        pm_latent: Optional[Any] = None,
        pm_raw: Optional[Any] = None,
    ) -> str:
        pid = fragment_id or uuid.uuid4().hex
        payload: Dict[str, Any] = {
            "id": pid,
            "trigger_context": trigger_context,
            "response_text": response_text,
            "intent": intent,
            "success_score": success_score,
            "created_at": time.time(),
        }
        if embedding is not None:
            payload["embedding"] = _jsonify(embedding)
        if pm_latent is not None:
            payload["pm_latent"] = _jsonify(pm_latent)
        if pm_raw is not None:
            payload["pm_raw"] = _jsonify(pm_raw)
        sanitized = self.store.sanitize(payload)
        self.store.put(self._key(pid), sanitized)
        return pid

    def update_success(self, fragment_id: str, feedback: float, plasticity_rate: float) -> None:
        key = self._key(fragment_id)
        entry = self.store.get(key)
        if not entry:
            return
        score = float(entry.get("success_score", 0.0))
        entry["success_score"] = score + feedback * plasticity_rate
        sanitized = self.store.sanitize(entry)
        self.store.put(key, sanitized)

    def get(self, fragment_id: str) -> Optional[Dict[str, Any]]:
        return self.store.get(self._key(fragment_id))

    def list(self) -> Iterable[Dict[str, Any]]:
        for _, value in self.store.list(prefix=self._prefix()):
            yield value

    def _prefix(self) -> str:
        return f"{self.namespace}:pattern:"

    def _key(self, fragment_id: str) -> str:
        return f"{self._prefix()}{fragment_id}"
