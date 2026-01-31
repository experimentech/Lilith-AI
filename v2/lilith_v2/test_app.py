import json
from pathlib import Path
from typing import Any, Dict

from .app import V2Runtime
from .mcp_router import EndpointType, MCPDescriptor
from .stage import Stage
from .persistent_stage import PersistentStage
from .json_file_store import JsonFileStore


def test_route_and_health_flow():
    runtime = V2Runtime.default()

    descriptor = MCPDescriptor(name="vscode", type=EndpointType.ACTION)
    branch_policy = {"allow": list(runtime.bindings.keys()), "deny": [], "concurrency_limit": 2}
    context = {"tenant": "tenant_demo", "modality": "text"}

    decision = runtime.route_and_dispatch(descriptor, context=context, branch_policy=branch_policy)
    assert decision.ports, "should have routed at least one port"

    health = runtime.health_check(modality="text", tenant="tenant_demo")
    assert health["ports"]
    assert health["stages"]


def test_config_policy_defaults_populate_allow():
    cfg = {
        "bindings": [
            {"node_id": "trunk.one", "modalities": ["text"]},
        ],
        "policy": {"timeout_ms": 123, "concurrency_limit": 5},
    }

    runtime = V2Runtime.from_config(cfg)
    descriptor = MCPDescriptor(name="vscode", type=EndpointType.ACTION)
    decision = runtime.route_and_dispatch(descriptor, context={"tenant": "t", "modality": "text"})
    # even with minimal policy, allow should be populated
    assert "trunk.one" in runtime.branch_policy["allow"]


class RecordingStage(Stage):
    def __init__(self, stage_id: str) -> None:
        self.id = stage_id
        self.seen: list[Dict[str, Any]] = []

    def encode(self, item: Any, ctx: Dict[str, Any]) -> Any:
        return item

    def retrieve(self, query: Any, ctx: Dict[str, Any]):
        return list(self.seen)

    def learn(self, event: Any, ctx: Dict[str, Any]) -> None:
        self.seen.append({"event": event, "ctx": ctx})

    def update_success(self, feedback: Any, ctx: Dict[str, Any]) -> None:
        return None

    def stats(self) -> Dict[str, Any]:
        return {"seen": len(self.seen)}

    def relational_sidecar(self, sql_ctx: Any):
        return None


def test_from_config_with_custom_stage_factory():
    cfg = {
        "bindings": [
            {"node_id": "trunk.tools", "modalities": ["text"], "tenants": ["tenant_x"]},
        ]
    }

    runtime = V2Runtime.from_config(cfg, stage_factory=lambda nid: RecordingStage(nid))

    descriptor = MCPDescriptor(name="vscode", type=EndpointType.ACTION)
    branch_policy = {"allow": ["trunk.tools"], "deny": []}
    context = {"tenant": "tenant_x", "modality": "text"}

    decision = runtime.route_and_dispatch(descriptor, context=context, branch_policy=branch_policy)
    assert "trunk.tools" in decision.ports

    stage = runtime.stages["trunk.tools"]
    assert isinstance(stage, RecordingStage)
    assert stage.stats()["seen"] == 1


def test_from_file_uses_loader(tmp_path: Path):
    cfg = {
        "bindings": [
            {"node_id": "branch.persist", "modalities": ["text"]},
        ]
    }
    f = tmp_path / "cfg.json"
    f.write_text(json.dumps(cfg), encoding="utf-8")

    def factory(nid: str) -> Stage:
        store = JsonFileStore(str(tmp_path / f"{nid}.json"))
        return PersistentStage(nid, store)

    runtime = V2Runtime.from_file(str(f), stage_factory=factory)
    descriptor = MCPDescriptor(name="vscode", type=EndpointType.ACTION)
    branch_policy = {"allow": ["branch.persist"], "deny": []}
    context = {"tenant": "tenant_file", "modality": "text"}

    decision = runtime.route_and_dispatch(descriptor, context=context, branch_policy=branch_policy)
    assert "branch.persist" in decision.ports

    stage = runtime.stages["branch.persist"]
    retrieved = list(stage.retrieve(query=None, ctx={}))
    assert len(retrieved) == 1
