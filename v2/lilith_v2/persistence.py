from typing import Any, Callable, Protocol


class PersistenceWrapper(Protocol):
    """Unified persistence wrapper with hygiene hooks."""

    def open_store(self, path: str, schema_version: int):
        ...

    def migrate(self, target_version: int) -> None:
        ...

    def tx(self, fn: Callable[[], Any]) -> Any:
        ...

    def sanitize(self, record: Any) -> Any:
        ...

    def decay(self, now: float) -> int:
        ...

    def validate(self, record: Any) -> bool:
        ...
