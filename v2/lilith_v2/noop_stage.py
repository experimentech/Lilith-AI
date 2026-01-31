from typing import Any, Dict, Iterable, List, Optional

from .stage import Stage


class NoopStage(Stage):
    """Minimal Stage that stores events and echoes queries."""

    def __init__(self, stage_id: str) -> None:
        self.id = stage_id
        self._events: List[Any] = []
        self._successes: int = 0

    def encode(self, item: Any, ctx: Dict[str, Any]) -> Any:
        return item

    def retrieve(self, query: Any, ctx: Dict[str, Any]) -> Iterable[Any]:
        # Return all events for now; could filter by query in the future
        return list(self._events)

    def learn(self, event: Any, ctx: Dict[str, Any]) -> None:
        self._events.append(event)

    def update_success(self, feedback: Any, ctx: Dict[str, Any]) -> None:
        self._successes += 1

    def stats(self) -> Dict[str, Any]:
        return {"events": len(self._events), "successes": self._successes}

    def relational_sidecar(self, sql_ctx: Any) -> Optional[Any]:
        return None
