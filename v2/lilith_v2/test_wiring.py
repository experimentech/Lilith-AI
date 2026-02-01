import io

from .wiring import sample_route
from .in_memory_port import InMemoryIOPort
from .bindings import NodeBindings, InMemoryBindingResolver
from .mcp_router import SimpleMCPRouter, MCPDescriptor, EndpointType


def test_sample_route_text_modality_logs_and_routes(monkeypatch):
    buf = io.StringIO()

    # Route with text modality; trunk.tools should be chosen, vision skipped
    decision = sample_route(modality="text", tenant="tenant_test")

    assert "trunk.tools" in decision.ports
    assert "branch.vision" not in decision.ports  # filtered by modality
    assert decision.metadata["modality"] == "text"
    assert decision.metadata["tenant"] == "tenant_test"
    assert decision.concurrency_limit == 4

    port = decision.ports["trunk.tools"]
    assert isinstance(port, InMemoryIOPort)

    # Ensure send/receive round-trip works
    port.send({"msg": "hello"}, meta={"trace_id": "t1"})
    received = list(port.receive(meta={}))
    assert received and received[0]["message"] == {"msg": "hello"}


def test_sample_route_image_hits_vision(monkeypatch):
    decision = sample_route(modality="image", tenant="tenant_b")
    assert "branch.vision" in decision.ports
    assert decision.metadata["modality"] == "image"


def test_route_filters_by_endpoint_type():
    bindings = {
        "branch.action": NodeBindings(node_id="branch.action", ports=[InMemoryIOPort()], endpoint_types={EndpointType.ACTION.value}),
        "branch.retrieval": NodeBindings(node_id="branch.retrieval", ports=[InMemoryIOPort()], endpoint_types={EndpointType.RETRIEVAL.value}),
    }
    resolver = InMemoryBindingResolver(bindings)
    router = SimpleMCPRouter(binding_resolver=resolver)

    descriptor = MCPDescriptor(name="do", type=EndpointType.ACTION)
    context = {"modality": "text", "tenant": "t"}
    policy = {"allow": list(bindings.keys()), "deny": []}

    decision = router.route(descriptor, context=context, branch_policy=policy)
    assert "branch.action" in decision.ports
    assert "branch.retrieval" not in decision.ports


def test_route_blocked_by_allow_modalities_policy():
    bindings = {
        "branch.any": NodeBindings(node_id="branch.any", ports=[InMemoryIOPort()]),
    }
    resolver = InMemoryBindingResolver(bindings)
    router = SimpleMCPRouter(binding_resolver=resolver)

    descriptor = MCPDescriptor(name="search", type=EndpointType.RETRIEVAL)
    context = {"modality": "text", "tenant": "t"}
    policy = {"allow": ["branch.any"], "allow_modalities": ["image"]}

    decision = router.route(descriptor, context=context, branch_policy=policy)
    assert decision.ports == {}
