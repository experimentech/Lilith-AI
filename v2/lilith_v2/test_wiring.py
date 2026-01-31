import io

from .wiring import sample_route
from .in_memory_port import InMemoryIOPort


def test_sample_route_text_modality_logs_and_routes(monkeypatch):
    buf = io.StringIO()

    # Route with text modality; trunk.tools should be chosen, vision skipped
    decision = sample_route(modality="text", tenant="tenant_test")

    assert "trunk.tools" in decision.ports
    assert "branch.vision" not in decision.ports  # filtered by modality
    assert decision.metadata["modality"] == "text"
    assert decision.metadata["tenant"] == "tenant_test"
    assert decision.concurrency_limit == 4

    port = decision.ports["trunk.tools"]
    assert isinstance(port, InMemoryIOPort)

    # Ensure send/receive round-trip works
    port.send({"msg": "hello"}, meta={"trace_id": "t1"})
    received = list(port.receive(meta={}))
    assert received and received[0]["message"] == {"msg": "hello"}


def test_sample_route_image_hits_vision(monkeypatch):
    decision = sample_route(modality="image", tenant="tenant_b")
    assert "branch.vision" in decision.ports
    assert decision.metadata["modality"] == "image"
