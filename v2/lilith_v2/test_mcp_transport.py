import pytest

from .mcp_transport import MockMCPTransport
from .mcp_router import EndpointType, MCPDescriptor, SimpleMCPRouter
from .vscode_endpoint import make_vscode_binding
from .bindings import InMemoryBindingResolver


def test_transport_invokes_handler_and_returns_response():
    transport = MockMCPTransport()

    def handler(message, meta):
        return {"ok": True, "seen": message}

    transport.register("vscode", handler)
    binding = make_vscode_binding("trunk.vscode", transport=transport)
    bindings = {binding.node_id: binding}
    resolver = InMemoryBindingResolver(bindings)
    router = SimpleMCPRouter(binding_resolver=resolver)

    descriptor = MCPDescriptor(name="vscode", type=EndpointType.ACTION)
    decision = router.route(descriptor, context={"tenant": "t", "modality": "text"}, branch_policy={"allow": [binding.node_id], "deny": []})

    port = decision.ports[binding.node_id]
    port.send({"action": "ping"}, meta={"trace_id": "t1"})

    received = list(port.receive(meta={}))
    # first item is the request, second item is transport response
    assert len(received) >= 2
    assert received[0]["message"] == {"action": "ping"}
    assert "response" in received[1]["message"]
    assert received[1]["message"]["response"]["ok"] is True


def test_transport_error_handler():
    transport = MockMCPTransport()
    transport.register_error("fail", "boom")
    with pytest.raises(RuntimeError):
        transport.call("fail", {}, {})
