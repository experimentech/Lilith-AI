from typing import Dict

from .bindings import InMemoryBindingResolver
from .mcp_router import EndpointType, MCPDescriptor, SimpleMCPRouter
from .noop_stage import NoopStage
from .observability import JsonLoggerObservability
from .vscode_endpoint import make_vscode_binding


def run_demo() -> None:
    """Route a VS Code MCP stub through the router and echo via noop stage."""

    obs = JsonLoggerObservability()

    # Bind a VS Code endpoint to an intermediate node
    binding = make_vscode_binding("trunk.vscode", modalities={"text"})
    bindings = {binding.node_id: binding}
    resolver = InMemoryBindingResolver(bindings)

    router = SimpleMCPRouter(binding_resolver=resolver)
    descriptor = MCPDescriptor(name="vscode", type=EndpointType.ACTION, latency_hint_ms=50)
    context: Dict[str, str] = {"modality": "text", "tenant": "tenant_vs"}
    branch_policy: Dict[str, object] = {
        "allow": [binding.node_id],
        "deny": [],
        "timeout_ms": 200,
        "concurrency_limit": 2,
        "per_tenant_limit": 1,
        "error_policy": {"retry": True, "backoff_ms": 50},
    }

    decision = router.route(descriptor, context=context, branch_policy=branch_policy)
    obs.on_event("route_decision", "router", decision.metadata, trace_id=context.get("tenant", ""))
    print("Route decision:", decision)

    port = decision.ports[binding.node_id]

    # Drive a noop stage to show flow
    stage = NoopStage(binding.node_id)
    stage.learn({"source": "vscode", "intent": "ping"}, ctx=context)

    # Simulate sending a request through the port
    port.send({"action": "list_files"}, meta={"trace_id": "t1", "tenant": context["tenant"]})
    received = list(port.receive(meta={}))

    print("Port received:", received)
    print("Stage stats:", stage.stats())


if __name__ == "__main__":
    run_demo()
