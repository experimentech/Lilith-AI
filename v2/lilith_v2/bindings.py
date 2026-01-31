from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Set

from .io_port import IOPort


@dataclass
class NodeBindings:
    """Ports attached to a specific node in the tree."""

    node_id: str
    ports: List[IOPort] = field(default_factory=list)
    modalities: Optional[Set[str]] = None  # allowed modalities, None = all
    tenants: Optional[Set[str]] = None  # allowed tenants, None = all


BindingMap = Dict[str, NodeBindings]


class BindingResolver(Protocol):
    """Resolve which ports to use for a node/modality/tenant combo."""

    def for_node(
        self,
        node_id: str,
        modality: Optional[str] = None,
        tenant: Optional[str] = None,
    ) -> List[IOPort]:
        ...


class InMemoryBindingResolver:
    """Simple binding resolver filtering by modality and tenant."""

    def __init__(self, bindings: BindingMap):
        self.bindings = bindings

    def for_node(
        self,
        node_id: str,
        modality: Optional[str] = None,
        tenant: Optional[str] = None,
    ) -> List[IOPort]:
        binding = self.bindings.get(node_id)
        if not binding:
            return []

        if binding.modalities and modality and modality not in binding.modalities:
            return []

        if binding.tenants and tenant and tenant not in binding.tenants:
            return []

        return list(binding.ports)
