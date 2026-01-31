from typing import Any, Callable, Dict, Optional

from .bindings import BindingMap, InMemoryBindingResolver
from .config_loader import load_config
from .health import probe_ports, probe_stages
from .in_memory_port import InMemoryIOPort
from .mcp_router import EndpointType, MCPDescriptor, RouteDecision, SimpleMCPRouter
from .noop_stage import NoopStage
from .observability import JsonLoggerObservability, NoopObservability, Observability
from .stage import Stage
from .vscode_endpoint import make_vscode_binding
from .concepts_stage import ConceptStage
from .gpl_concept_stage import GPLConceptStage
from .json_file_store import JsonFileStore
from .relational_store import RelationalStore


class V2Runtime:
    """Minimal runtime wiring router, bindings, stages, and observability."""

    def __init__(
        self,
        bindings: BindingMap,
        resolver: InMemoryBindingResolver,
        router: SimpleMCPRouter,
        stages: Dict[str, Stage],
        observability: Observability,
        branch_policy: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.bindings = bindings
        self.resolver = resolver
        self.router = router
        self.stages = stages
        self.observability = observability
        self.branch_policy = branch_policy or {}

    @classmethod
    def default(cls) -> "V2Runtime":
        bindings: BindingMap = {
            "trunk.tools": make_vscode_binding("trunk.tools", modalities={"text"}),
            "trunk.vscode": make_vscode_binding("trunk.vscode", modalities={"text"}),
            "branch.vision": make_vscode_binding("branch.vision", modalities={"image"}),
            "branch.audio": make_vscode_binding("branch.audio", modalities={"audio"}),
        }
        resolver = InMemoryBindingResolver(bindings)
        router = SimpleMCPRouter(binding_resolver=resolver)
        stages: Dict[str, Stage] = {node_id: NoopStage(node_id) for node_id in bindings}
        observability: Observability = JsonLoggerObservability()
        policy = {
            "allow": list(bindings.keys()),
            "deny": [],
            "timeout_ms": 0,
            "concurrency_limit": 1,
            "error_policy": {"retry": True, "backoff_ms": 100},
        }
        return cls(bindings, resolver, router, stages, observability, branch_policy=policy)

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        stage_factory: Optional[Callable[[str], Stage]] = None,
        observability: Optional[Observability] = None,
    ) -> "V2Runtime":
        """Construct runtime from config dict specifying bindings.

        Config shape example:
        {
            "bindings": [
                {"node_id": "trunk.tools", "modalities": ["text"], "tenants": ["tenant_a"]},
                {"node_id": "branch.vision", "modalities": ["image"]},
            ]
        }
        """

        binding_map: BindingMap = {}
        for entry in config.get("bindings", []):
            node_id = entry.get("node_id")
            if not node_id:
                continue
            modalities = set(entry.get("modalities", [])) or None
            tenants = set(entry.get("tenants", [])) or None
            binding_map[node_id] = make_vscode_binding(node_id, modalities=modalities, tenants=tenants)

        resolver = InMemoryBindingResolver(binding_map)
        router = SimpleMCPRouter(binding_resolver=resolver)
        def build_store(store_type: str, path: str):
            if store_type == "sqlite":
                return RelationalStore(path)
            return JsonFileStore(path)

        def _make_stage(nid: str) -> Stage:
            if stage_factory:
                return stage_factory(nid)
            binding_cfg = next((b for b in config.get("bindings", []) if b.get("node_id") == nid), {})
            stage_type = binding_cfg.get("stage") or "noop"
            store_path = binding_cfg.get("store_path")
            store_type = binding_cfg.get("store_type", "json")
            if stage_type in {"concepts", "concepts_relational"}:
                default_path = f"/tmp/{nid}.concepts.sqlite" if store_type == "sqlite" else f"/tmp/{nid}.concepts.json"
                path = store_path or default_path
                store = build_store(store_type, path)
                return ConceptStage(nid, store)
            if stage_type in {"gpl_concepts", "gpl_concepts_relational"}:
                default_path = f"/tmp/{nid}.gpl_concepts.sqlite" if store_type == "sqlite" else f"/tmp/{nid}.gpl_concepts.json"
                path = store_path or default_path
                store = build_store(store_type, path)
                learner_cfg = binding_cfg.get("learner", {})
                pmflow_cfg = binding_cfg.get("pmflow", {})
                retrieval_cfg = binding_cfg.get("retrieval", {})
                return GPLConceptStage(
                    nid,
                    store,
                    learner_config=learner_cfg,
                    pmflow_config=pmflow_cfg,
                    retrieval_config=retrieval_cfg,
                )
            return NoopStage(nid)

        stages: Dict[str, Stage] = {node_id: _make_stage(node_id) for node_id in binding_map}
        obs: Observability = observability or JsonLoggerObservability()
        policy = config.get("policy", {})
        if "allow" not in policy:
            policy["allow"] = list(binding_map.keys())
        if "deny" not in policy:
            policy.setdefault("deny", [])
        policy.setdefault("timeout_ms", 0)
        policy.setdefault("concurrency_limit", 1)
        policy.setdefault("error_policy", {"retry": True, "backoff_ms": 100})
        return cls(binding_map, resolver, router, stages, obs, branch_policy=policy)

    @classmethod
    def from_file(
        cls,
        path: str,
        stage_factory: Optional[Callable[[str], Stage]] = None,
        observability: Optional[Observability] = None,
    ) -> "V2Runtime":
        cfg = load_config(path)
        return cls.from_config(cfg, stage_factory=stage_factory, observability=observability)

    def route_and_dispatch(
        self,
        descriptor: MCPDescriptor,
        context: Dict[str, Any],
        branch_policy: Optional[Dict[str, Any]] = None,
    ) -> RouteDecision:
        resolved_policy = dict(self.branch_policy)
        if branch_policy:
            resolved_policy.update(branch_policy)
        # Always ensure allow list covers current bindings if not overridden
        resolved_policy.setdefault("allow", list(self.bindings.keys()))
        resolved_policy.setdefault("deny", [])
        decision = self.router.route(descriptor, context=context, branch_policy=resolved_policy)
        self.observability.on_event("route_decision", "router", decision.metadata, trace_id=context.get("tenant", ""))

        # Optionally push a synthetic payload through the first port for sanity
        for node_id, port in decision.ports.items():
            port.send({"descriptor": descriptor.name, "node_id": node_id}, meta={"trace_id": context.get("tenant", "")})
            # Touch stage as part of the flow
            stage = self.stages.get(node_id)
            if stage:
                stage.learn({"descriptor": descriptor.name, "node_id": node_id}, ctx=context)
        return decision

    def health_check(
        self,
        modality: Optional[str] = None,
        tenant: Optional[str] = None,
    ) -> Dict[str, Any]:
        port_results = probe_ports(list(self.bindings.keys()), self.resolver, modality=modality, tenant=tenant, observability=self.observability)
        stage_results = probe_stages(self.stages, observability=self.observability)
        return {"ports": port_results, "stages": stage_results}


__all__ = ["V2Runtime"]
