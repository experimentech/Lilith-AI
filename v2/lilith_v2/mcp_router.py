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
        endpoint_type = getattr(descriptor.type, "value", str(descriptor.type))

        allow_modalities = set(branch_policy.get("allow_modalities", []) or []) or None
        deny_modalities = set(branch_policy.get("deny_modalities", []) or []) or None
        allow_tenants = set(branch_policy.get("allow_tenants", []) or []) or None
        deny_tenants = set(branch_policy.get("deny_tenants", []) or []) or None
        allow_endpoint_types = set(branch_policy.get("allow_endpoint_types", []) or []) or None
        deny_endpoint_types = set(branch_policy.get("deny_endpoint_types", []) or []) or None

        # Early modality/tenant/endpoint filtering
        modality_blocked = (allow_modalities and modality and modality not in allow_modalities) or (
            deny_modalities and modality in deny_modalities
        )
        tenant_blocked = (allow_tenants and tenant and tenant not in allow_tenants) or (
            deny_tenants and tenant in deny_tenants
        )
        endpoint_blocked = (allow_endpoint_types and endpoint_type not in allow_endpoint_types) or (
            deny_endpoint_types and endpoint_type in deny_endpoint_types
        )

        allowed_nodes: List[str] = branch_policy.get("allow", [])
        denied_nodes: List[str] = branch_policy.get("deny", [])
        target_nodes = [n for n in allowed_nodes if n not in denied_nodes]

        if modality_blocked or tenant_blocked or endpoint_blocked:
            target_nodes = []

        # Ensure deterministic ordering and uniqueness
        seen = set()
        target_nodes = [n for n in target_nodes if not (n in seen or seen.add(n))]

        ports: Dict[str, IOPort] = {}
        for node in target_nodes:
            resolved = self.binding_resolver.for_node(
                node,
                modality=modality,
                tenant=tenant,
                endpoint_type=endpoint_type,
            )
            if resolved:
                ports[node] = resolved[0]

        timeout_ms = branch_policy.get("timeout_ms", descriptor.latency_hint_ms or 0)
        concurrency_limit = branch_policy.get("concurrency_limit", 1)
        per_tenant_limit = branch_policy.get("per_tenant_limit")
        error_policy = branch_policy.get("error_policy", {"retry": True, "backoff_ms": 100})

        metadata: Dict[str, Any] = {
            "modality": modality,
            "tenant": tenant,
            "endpoint_type": endpoint_type,
            "filtered_nodes": [n for n in target_nodes if n not in ports],
        }

        return RouteDecision(
            target_nodes=target_nodes,
            ports=ports,
            timeout_ms=timeout_ms,
            concurrency_limit=concurrency_limit,
            per_tenant_limit=per_tenant_limit,
            error_policy=error_policy,
            metadata=metadata,
        )
