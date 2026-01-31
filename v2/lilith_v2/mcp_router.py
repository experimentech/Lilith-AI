from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

from .io_port import IOPort
from .bindings import BindingResolver


class EndpointType(str, Enum):
    KNOWLEDGE = "knowledge"
    RETRIEVAL = "retrieval"
    ACTION = "action"
    GENERATION = "generation"
    OPS = "ops"


@dataclass
class MCPDescriptor:
    name: str
    type: EndpointType
    cost: float = 0.0
    latency_hint_ms: int = 0
    auth: Optional[str] = None


@dataclass
class RouteDecision:
    target_nodes: List[str]
    ports: Dict[str, IOPort]
    timeout_ms: int
    concurrency_limit: int
    per_tenant_limit: Optional[int]
    error_policy: Dict[str, Any]
    metadata: Dict[str, Any]


class MCPRouter(Protocol):
    """Router from endpoint descriptors to branch/port bindings."""

    def classify(self, descriptor: MCPDescriptor) -> EndpointType:
        ...

    def route(
        self,
        descriptor: MCPDescriptor,
        context: Dict[str, Any],
        branch_policy: Dict[str, Any],
    ) -> RouteDecision:
        ...


class SimpleMCPRouter:
    """Minimal router honoring modality/tenant and branch allow/deny lists."""

    def __init__(self, binding_resolver: BindingResolver):
        self.binding_resolver = binding_resolver

    def classify(self, descriptor: MCPDescriptor) -> EndpointType:
        return descriptor.type

    def route(
        self,
        descriptor: MCPDescriptor,
        context: Dict[str, Any],
        branch_policy: Dict[str, Any],
    ) -> RouteDecision:
        modality = context.get("modality")
        tenant = context.get("tenant")

        allowed_nodes: List[str] = branch_policy.get("allow", [])
        denied_nodes: List[str] = branch_policy.get("deny", [])
        target_nodes = [n for n in allowed_nodes if n not in denied_nodes]

        ports: Dict[str, IOPort] = {}
        for node in target_nodes:
            resolved = self.binding_resolver.for_node(node, modality=modality, tenant=tenant)
            # last write wins if duplicate ids; realistic impl could merge by priority
            if resolved:
                ports[node] = resolved[0]

        timeout_ms = branch_policy.get("timeout_ms", descriptor.latency_hint_ms or 0)
        concurrency_limit = branch_policy.get("concurrency_limit", 1)
        per_tenant_limit = branch_policy.get("per_tenant_limit")
        error_policy = branch_policy.get("error_policy", {"retry": True, "backoff_ms": 100})

        return RouteDecision(
            target_nodes=target_nodes,
            ports=ports,
            timeout_ms=timeout_ms,
            concurrency_limit=concurrency_limit,
            per_tenant_limit=per_tenant_limit,
            error_policy=error_policy,
            metadata={"modality": modality, "tenant": tenant},
        )
