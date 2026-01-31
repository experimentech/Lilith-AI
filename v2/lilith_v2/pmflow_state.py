from typing import Any, Dict, Protocol, Tuple


class PMFlowStateStore(Protocol):
    """Persistent PMFlow state per branch."""

    def load_state(self, branch_id: str) -> Dict[str, Any]:
        ...

    def save_state(self, branch_id: str, state: Dict[str, Any], version: int) -> None:
        ...

    def bump_version(self, branch_id: str) -> int:
        ...

    def latent_dims(self, branch_id: str) -> Tuple[int, ...]:
        ...

    def compact(self) -> None:
        ...
