from typing import Dict

from .bindings import BindingMap, InMemoryBindingResolver, NodeBindings
from .in_memory_port import InMemoryIOPort
from .mcp_router import EndpointType, MCPDescriptor, RouteDecision, SimpleMCPRouter
from .observability import JsonLoggerObservability, NoopObservability
from .vscode_endpoint import make_vscode_binding


def sample_binding_map() -> BindingMap:
    return {
        "trunk.tools": NodeBindings(node_id="trunk.tools", ports=[InMemoryIOPort()], modalities={"text"}),
        "branch.vision": NodeBindings(node_id="branch.vision", ports=[InMemoryIOPort()], modalities={"image"}),
        "trunk.vscode": make_vscode_binding("trunk.vscode"),
    }


def sample_route(modality: str = "text", tenant: str = "tenant_a") -> RouteDecision:
    bindings = sample_binding_map()
    resolver = InMemoryBindingResolver(bindings)
    router = SimpleMCPRouter(binding_resolver=resolver)
    obs = JsonLoggerObservability()

    descriptor = MCPDescriptor(name="search", type=EndpointType.RETRIEVAL, latency_hint_ms=200)
    context: Dict[str, str] = {"modality": modality, "tenant": tenant}
    branch_policy: Dict[str, object] = {
        "allow": ["trunk.tools", "branch.vision", "trunk.vscode"],
        "deny": [],
        "timeout_ms": 300,
        "concurrency_limit": 4,
        "per_tenant_limit": 2,
        "error_policy": {"retry": True, "backoff_ms": 100},
    }

    decision = router.route(descriptor, context=context, branch_policy=branch_policy)
    obs.on_event("route_decision", "router", decision.metadata, trace_id=context.get("tenant", ""))
    return decision


__all__ = ["sample_binding_map", "sample_route"]
