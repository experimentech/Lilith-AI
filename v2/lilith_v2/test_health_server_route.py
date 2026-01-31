import json
import time
import urllib.request

from .health_server import run_health_server


def test_route_endpoint_returns_ports(tmp_path):
    server = run_health_server("v2/lilith_v2/configs/bindings.example.json", host="127.0.0.1", port=8091)
    time.sleep(0.1)
    try:
        with urllib.request.urlopen("http://127.0.0.1:8091/route") as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            assert "ports" in data
            assert "metadata" in data
    finally:
        server.shutdown()
        server.server_close()
