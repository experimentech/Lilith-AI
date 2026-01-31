import time
from typing import Any, Dict, Iterable, Optional

from .stage import Stage
from .store import Store


class PersistentStage(Stage):
    """Stage backed by a Store; stores events with timestamp keys."""

    def __init__(self, stage_id: str, store: Store) -> None:
        self.id = stage_id
        self.store = store
        self._successes = 0

    def encode(self, item: Any, ctx: Dict[str, Any]) -> Any:
        return item

    def retrieve(self, query: Any, ctx: Dict[str, Any]) -> Iterable[Any]:
        prefix = f"{self.id}:"
        return [v for _, v in self.store.list(prefix=prefix)]

    def learn(self, event: Any, ctx: Dict[str, Any]) -> None:
        key = f"{self.id}:{time.time_ns()}"
        self.store.put(key, event)

    def update_success(self, feedback: Any, ctx: Dict[str, Any]) -> None:
        self._successes += 1

    def stats(self) -> Dict[str, Any]:
        prefix = f"{self.id}:"
        count = sum(1 for _ in self.store.list(prefix=prefix))
        return {"events": count, "successes": self._successes}

    def relational_sidecar(self, sql_ctx: Any) -> Optional[Any]:
        return None
