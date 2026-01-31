from typing import Any, Dict, Iterable, Optional, Protocol


class Stage(Protocol):
    """Abstract interface for a stage node in the v2 tree."""

    id: str  # hierarchical e.g., "trunk.concepts"

    def encode(self, item: Any, ctx: Dict[str, Any]) -> Any:
        ...

    def retrieve(self, query: Any, ctx: Dict[str, Any]) -> Iterable[Any]:
        ...

    def learn(self, event: Any, ctx: Dict[str, Any]) -> None:
        ...

    def update_success(self, feedback: Any, ctx: Dict[str, Any]) -> None:
        ...

    def stats(self) -> Dict[str, Any]:
        ...

    def relational_sidecar(self, sql_ctx: Any) -> Optional[Any]:
        """Optional relational hook for concept joins."""
        ...
