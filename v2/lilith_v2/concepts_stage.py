from typing import Any, Dict, Iterable, Optional

from .stage import Stage
from .store import Store


class ConceptStage(Stage):
    """Lightweight concept stage with merge-friendly learn and simple retrieve.

    Concepts are stored as dicts with optional 'id', 'properties', and 'relations'.
    Storage is delegated to a Store; schema is intentionally loose to avoid over-constraining v2.
    """

    def __init__(self, stage_id: str, store: Store):
        self.id = stage_id
        self.store = store
        self._successes = 0

    def encode(self, item: Any, ctx: Dict[str, Any]) -> Any:
        return item

    def retrieve(self, query: Any, ctx: Dict[str, Any]) -> Iterable[Any]:
        # If query contains an id, try that first; otherwise return all concepts.
        if isinstance(query, dict) and query.get("id"):
            cid = str(query["id"])
            hit = self.store.get(self._key(cid))
            if hit is not None:
                return [hit]
        return [v for _, v in self.store.list(prefix=self._prefix())]

    def learn(self, event: Any, ctx: Dict[str, Any]) -> None:
        concept = dict(event) if isinstance(event, dict) else {"value": event}
        cid = str(concept.get("id", self._next_id()))
        key = self._key(cid)
        existing = self.store.get(key) or {}
        merged = self._merge(existing, concept)
        self.store.put(key, merged)

    def update_success(self, feedback: Any, ctx: Dict[str, Any]) -> None:
        self._successes += 1

    def stats(self) -> Dict[str, Any]:
        count = sum(1 for _ in self.store.list(prefix=self._prefix()))
        return {"concepts": count, "successes": self._successes}

    def relational_sidecar(self, sql_ctx: Any) -> Optional[Any]:
        return None

    def _key(self, cid: str) -> str:
        return f"{self.id}:{cid}"

    def _prefix(self) -> str:
        return f"{self.id}:"

    def _next_id(self) -> str:
        return "concept"  # caller can overwrite; keeps schema loose

    def _merge(self, existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(existing)
        for k, v in incoming.items():
            if k == "properties" and isinstance(v, dict):
                props = dict(existing.get("properties", {}))
                props.update(v)
                merged["properties"] = props
            elif k == "relations" and isinstance(v, list):
                rels = list(existing.get("relations", []))
                rels.extend(v)
                merged["relations"] = rels
            else:
                merged[k] = v
        return merged
