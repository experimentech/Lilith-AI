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
from .pmflow_sqlite import SQLitePMFlowStateStore
from .relational_event_store import RelationalEventStore
from .relational_graph_store import RelationalGraphStore
from .multi_tenant_store import MultiTenantPMFlowManager, MultiTenantGraphManager
from .cognitive_stage import CognitiveStage
from .math_stage import MathStage
if False: # Try import full v1 encoder if present, else fallback
     try:
         from lilith.embedding import PMFlowEmbeddingEncoder
     except ImportError:
         PMFlowEmbeddingEncoder = None


from .mcp_transport_tools import LocalToolsTransport
from .fs_tools import FileSystemTools
from .terminal_tools import TerminalTools
from .weather_tools import get_weather
import os

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
        pmflow_state_store: Optional[SQLitePMFlowStateStore] = None,
        event_store: Optional[RelationalEventStore] = None,
    ) -> None:
        self.bindings = bindings
        self.resolver = resolver
        self.router = router
        self.stages = stages
        self.observability = observability
        self.branch_policy = branch_policy or {}
        self.pmflow_state_store = pmflow_state_store
        self.event_store = event_store

    @classmethod
    def default(cls) -> "V2Runtime":
        # Setup Limb Capabilities based on Env Vars (Safeguards)
        enable_fs = os.getenv("LILITH_ENABLE_FS", "true").lower() == "true"
        enable_terminal = os.getenv("LILITH_ENABLE_TERMINAL", "false").lower() == "true"
        workspace_root = os.getcwd()

        tools_transport = LocalToolsTransport()
        
        # 1. File System Limb
        if enable_fs:
            fs = FileSystemTools(workspace_root)
            tools_transport.register_tool("read_file", fs.read_file)
            tools_transport.register_tool("write_file", fs.write_file)
            tools_transport.register_tool("list_dir", fs.list_dir)
            
        # 2. Terminal Limb
        term = TerminalTools(workspace_root, enabled=enable_terminal)
        if enable_terminal:
            tools_transport.register_tool("run_command", term.run_command)
            
        # 3. Weather Limb
        tools_transport.register_tool("get_weather", get_weather)

        bindings: BindingMap = {
            "trunk.tools": make_vscode_binding("trunk.tools", modalities={"text"}, transport=tools_transport),
            "trunk.vscode": make_vscode_binding("trunk.vscode", modalities={"text"}),
            "branch.vision": make_vscode_binding("branch.vision", modalities={"image"}),
            "branch.audio": make_vscode_binding("branch.audio", modalities={"audio"}),
            "branch.math": make_vscode_binding("branch.math", modalities={"math", "text"}), # Text fallback
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

        pmflow_cfg = config.get("pmflow", {}) or {}
        pmflow_enabled = bool(pmflow_cfg)
        pmflow_state_store = None
        event_store = None
        if pmflow_enabled:
            state_path = pmflow_cfg.get("state_path") or "/tmp/pmflow_state.sqlite"
            event_path = pmflow_cfg.get("event_path") or "/tmp/pmflow_events.sqlite"
            pmflow_state_store = SQLitePMFlowStateStore(state_path)
            event_store = RelationalEventStore(event_path)
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
            if stage_type == "cognitive":
                data_root = binding_cfg.get("data_root")
                
                if data_root:
                    base_root = os.path.join(data_root, "base")
                    users_root = os.path.join(data_root, "users")
                    pmflow_s = MultiTenantPMFlowManager(base_root, users_root)
                    graph_s = MultiTenantGraphManager(base_root, users_root)
                elif not pmflow_state_store:
                    # Fallback or error - simplistic fallback for now
                    store_p = f"/tmp/{nid}.pmflow.sqlite"
                    pmflow_s = SQLitePMFlowStateStore(store_p)
                    graph_p = store_path or f"/tmp/{nid}.graph.sqlite"
                    graph_s = RelationalGraphStore(graph_p)
                else:
                    pmflow_s = pmflow_state_store
                    graph_p = store_path or f"/tmp/{nid}.graph.sqlite"
                    graph_s = RelationalGraphStore(graph_p)
                
                # Encoder factory - simplistic mock if not configured
                class MockEncoder:
                    def encode(self, x):
                        import torch 
                        return torch.zeros(128)
                
                encoder = MockEncoder() 
                # In real scenario, load encoder config from binding_cfg.get("encoder")
                
                return CognitiveStage(
                    nid,
                    pmflow_store=pmflow_s,
                    graph_store=graph_s,
                    encoder=encoder
                )
            if stage_type == "math":
                return MathStage(nid)
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
        return cls(
            binding_map,
            resolver,
            router,
            stages,
            obs,
            branch_policy=policy,
            pmflow_state_store=pmflow_state_store,
            event_store=event_store,
        )

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
        payload: Optional[Any] = None,
    ) -> RouteDecision:
        resolved_policy = dict(self.branch_policy)
        if branch_policy:
            resolved_policy.update(branch_policy)
        # Always ensure allow list covers current bindings if not overridden
        resolved_policy.setdefault("allow", list(self.bindings.keys()))
        resolved_policy.setdefault("deny", [])
        decision = self.router.route(descriptor, context=context, branch_policy=resolved_policy)
        self.observability.on_event("route_decision", "router", decision.metadata, trace_id=context.get("tenant", ""))

        # Optionally push a synthetic payload through the ports for sanity
        for node_id, port in decision.ports.items():
            meta = {"trace_id": context.get("tenant", ""), "tenant": context.get("tenant"), "modality": context.get("modality")}
            body = payload if payload is not None else {"descriptor": descriptor.name, "node_id": node_id}
            port.send(body, meta=meta)
            # Touch stage as part of the flow
            stage = self.stages.get(node_id)
            if stage:
                stage.learn(body, ctx=context)
                
                # Efferent Flow (Somatic Bridge)
                # If the stage produced a response/impulse, send it back through the ports
                # Note: In a real SomaticLayer loop, this would be decoupled (tick),
                # but for V2Runtime dispatch we mimic the immediate reflex.
                if hasattr(stage, "last_interaction") and stage.last_interaction:
                    response = stage.last_interaction
                    
                    # Determine where to send. Generally to the port that handled the input.
                    # Or broadcast to all ports in the decision? 
                    # Simpler is to use the specific port for this node.
                    if port:
                         # Ensure payload is serializable/expected format
                         # VSCodeAdapter wraps logic but raw port expects dict/msg
                         # We wrap in standard Lilith envelope if not already
                         if isinstance(response, dict) and "response" in response:
                             out_payload = response["response"]
                             out_meta = response.get("meta", {})
                         else:
                             out_payload = response
                             out_meta = {}
                             
                         out_meta.update({"source": node_id, "ref": context.get("trace_id")})
                         port.send(out_payload, meta=out_meta)
                    
                    # Clear impulse
                    stage.last_interaction = None

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
