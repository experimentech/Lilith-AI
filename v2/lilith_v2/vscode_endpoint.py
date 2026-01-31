from dataclasses import dataclass
from typing import Optional, Set

from .bindings import NodeBindings
from .in_memory_port import InMemoryIOPort
from .mcp_router import EndpointType, MCPDescriptor
from .mcp_transport import MCPTransport


def make_descriptor(
    name: str = "vscode",
    endpoint_type: EndpointType = EndpointType.ACTION,
    cost: float = 0.0,
    latency_hint_ms: int = 50,
    auth: Optional[str] = None,
) -> MCPDescriptor:
    return MCPDescriptor(name=name, type=endpoint_type, cost=cost, latency_hint_ms=latency_hint_ms, auth=auth)


@dataclass
class VSCodeMCPAdapter(InMemoryIOPort):
    """Stub MCP adapter for VS Code; extends in-memory port for testing."""

    descriptor: MCPDescriptor

    def __init__(self, descriptor: MCPDescriptor, transport: Optional[MCPTransport] = None):
        super().__init__()
        self.descriptor = descriptor
        self._transport = transport

    def send(self, message: object, meta: dict) -> None:
        # Queue the outbound request
        super().send(message, meta)
        # If a transport is present, synchronously invoke handler and enqueue response
        if self._transport:
            response = self._transport.call(self.descriptor.name, message, meta)
            super().send({"response": response}, {"transport": True, **meta})


def make_vscode_binding(
    node_id: str,
    descriptor: Optional[MCPDescriptor] = None,
    transport: Optional[MCPTransport] = None,
    modalities: Optional[Set[str]] = None,
    tenants: Optional[Set[str]] = None,
) -> NodeBindings:
    desc = descriptor or make_descriptor()
    adapter = VSCodeMCPAdapter(desc, transport=transport)
    return NodeBindings(node_id=node_id, ports=[adapter], modalities=modalities, tenants=tenants)
