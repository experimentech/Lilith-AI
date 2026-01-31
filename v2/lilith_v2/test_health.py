from .bindings import InMemoryBindingResolver, NodeBindings
from .health import probe_ports, probe_stages
from .in_memory_port import InMemoryIOPort
from .noop_stage import NoopStage


def test_probe_ports_with_in_memory_binding():
    bindings = {
        "node.a": NodeBindings(node_id="node.a", ports=[InMemoryIOPort()]),
    }
    resolver = InMemoryBindingResolver(bindings)

    results = probe_ports(["node.a"], resolver, modality=None, tenant="tenant_x")
    assert results and results[0]["ok"] is True
    assert results[0]["ports"][0]["ok"] is True


def test_probe_stages_calls_stats():
    stages = {
        "node.b": NoopStage("node.b"),
    }
    results = probe_stages(stages)
    assert results and results[0]["ok"] is True
    assert "stats" in results[0]
