from .bindings import InMemoryBindingResolver
from .mcp_router import EndpointType, MCPDescriptor, SimpleMCPRouter
from .vscode_endpoint import make_vscode_binding


def test_vscode_binding_routes_through_router():
    binding = make_vscode_binding("trunk.vscode", descriptor=MCPDescriptor(name="vscode", type=EndpointType.ACTION))
    bindings = {binding.node_id: binding}
    resolver = InMemoryBindingResolver(bindings)
    router = SimpleMCPRouter(binding_resolver=resolver)

    descriptor = MCPDescriptor(name="vscode", type=EndpointType.ACTION)
    context = {"modality": "text", "tenant": "tenant_vs"}
    branch_policy = {"allow": [binding.node_id], "deny": []}

    decision = router.route(descriptor, context=context, branch_policy=branch_policy)
    assert binding.node_id in decision.ports
    assert decision.metadata["tenant"] == "tenant_vs"
