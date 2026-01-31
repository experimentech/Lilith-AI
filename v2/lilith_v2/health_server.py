import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from typing import Optional

from .app import V2Runtime
from .config_loader import load_config
from .json_file_store import JsonFileStore
from .persistent_stage import PersistentStage
from .stage import Stage
from .mcp_router import EndpointType, MCPDescriptor


class HealthHandler(BaseHTTPRequestHandler):
    runtime: Optional[V2Runtime] = None
    tenant: str = "tenant_demo"
    modality: str = "text"
    descriptor_name: str = "vscode"

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health" or parsed.path == "/":
            self._write_json(self._health())
            return
        if parsed.path == "/route":
            self._write_json(self._route())
            return
        self.send_response(404)
        self.end_headers()

    def _health(self):
        if not HealthHandler.runtime:
            return {"error": "runtime not initialized"}
        return HealthHandler.runtime.health_check(modality=self.modality, tenant=self.tenant)

    def _route(self):
        if not HealthHandler.runtime:
            return {"error": "runtime not initialized"}
        descriptor = MCPDescriptor(name=self.descriptor_name, type=EndpointType.ACTION)
        decision = HealthHandler.runtime.route_and_dispatch(
            descriptor,
            context={"tenant": self.tenant, "modality": self.modality},
            branch_policy={},
        )
        return {
            "target_nodes": decision.target_nodes,
            "ports": list(decision.ports.keys()),
            "metadata": decision.metadata,
            "timeout_ms": decision.timeout_ms,
            "concurrency_limit": decision.concurrency_limit,
            "per_tenant_limit": decision.per_tenant_limit,
        }

    def _write_json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_health_server(config_path: str, host: str = "127.0.0.1", port: int = 8080) -> HTTPServer:
    cfg = load_config(config_path)

    def factory(node_id: str) -> Stage:
        store = JsonFileStore(f"/tmp/{node_id}.json")
        return PersistentStage(node_id, store)

    runtime = V2Runtime.from_config(cfg, stage_factory=factory, observability=None)
    HealthHandler.runtime = runtime
    server = HTTPServer((host, port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


if __name__ == "__main__":
    run_health_server("v2/lilith_v2/configs/bindings.example.json")
    print("Health server running on http://127.0.0.1:8080/health")
    thread = threading.Event()
    thread.wait()
