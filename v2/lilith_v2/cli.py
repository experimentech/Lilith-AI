import argparse
import sys
from typing import Dict

from .app import V2Runtime
from .config_loader import load_config
from .mcp_router import EndpointType, MCPDescriptor
from .mcp_transport import MockMCPTransport
from .json_file_store import JsonFileStore
from .persistent_stage import PersistentStage
from .stage import Stage


def build_runtime(config_path: str) -> V2Runtime:
    cfg = load_config(config_path)
    transport = MockMCPTransport()
    transport.register("vscode", lambda msg, meta: {"ok": True, "handled": msg, "meta": meta})
    transport.register_error("vscode.error", "simulated transport error")

    def factory(node_id: str) -> Stage:
        store = JsonFileStore(f"/tmp/{node_id}.json")
        return PersistentStage(node_id, store)

    runtime = V2Runtime.from_config(cfg, stage_factory=factory, observability=None)

    # Inject transport into any VSCode bindings
    for binding in runtime.bindings.values():
        for port in binding.ports:
            if hasattr(port, "_transport"):
                port._transport = transport  # type: ignore[attr-defined]

    return runtime


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run v2 runtime with config and mock transport")
    parser.add_argument("--config", default="v2/lilith_v2/configs/bindings.example.json", help="Path to JSON config")
    parser.add_argument("--tenant", default="tenant_demo")
    parser.add_argument("--modality", default="text")
    args = parser.parse_args(argv)

    runtime = build_runtime(args.config)

    descriptor = MCPDescriptor(name="vscode", type=EndpointType.ACTION)
    branch_policy: Dict[str, object] = {}
    context: Dict[str, str] = {"tenant": args.tenant, "modality": args.modality}

    decision = runtime.route_and_dispatch(descriptor, context=context, branch_policy=branch_policy)
    print("Route decision:", decision)

    for node_id, port in decision.ports.items():
        received = list(port.receive(meta={}))
        print(f"Received on {node_id}:", received)

    health = runtime.health_check(modality=args.modality, tenant=args.tenant)
    print("Health:", health)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
