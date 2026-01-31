from typing import Dict

from .app import V2Runtime
from .config_loader import load_config
from .mcp_router import EndpointType, MCPDescriptor
from .mcp_transport import MockMCPTransport
from .persistent_stage import PersistentStage
from .json_file_store import JsonFileStore
from .stage import Stage


def run_demo(config_path: str = "v2/lilith_v2/configs/bindings.example.json") -> None:
    """Load bindings from config, wire a mock transport, and route a request."""

    cfg = load_config(config_path)

    transport = MockMCPTransport()
    transport.register("vscode", lambda msg, meta: {"ok": True, "handled": msg, "meta": meta})

    def factory(node_id: str) -> Stage:
        store = JsonFileStore(f"/tmp/{node_id}.json")
        return PersistentStage(node_id, store)

    runtime = V2Runtime.from_config(
        cfg,
        stage_factory=factory,
        observability=None,
    )

    # Inject transport into any VSCode bindings
    for binding in runtime.bindings.values():
        for port in binding.ports:
            if hasattr(port, "_transport"):
                port._transport = transport  # type: ignore[attr-defined]

    descriptor = MCPDescriptor(name="vscode", type=EndpointType.ACTION)
    branch_policy: Dict[str, object] = {"allow": list(runtime.bindings.keys()), "deny": []}
    context: Dict[str, str] = {"tenant": "tenant_demo", "modality": "text"}

    decision = runtime.route_and_dispatch(descriptor, context=context, branch_policy=branch_policy)
    print("Route decision:", decision)

    for node_id, port in decision.ports.items():
        received = list(port.receive(meta={}))
        print(f"Received on {node_id}:", received)

    health = runtime.health_check(modality="text", tenant="tenant_demo")
    print("Health:", health)


if __name__ == "__main__":
    run_demo()
