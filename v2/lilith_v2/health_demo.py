from typing import Dict

from .bindings import InMemoryBindingResolver, NodeBindings
from .health import probe_ports, probe_stages
from .in_memory_port import InMemoryIOPort
from .noop_stage import NoopStage
from .observability import JsonLoggerObservability


def run_demo() -> None:
    """Run a simple health probe with JSON logs to stdout."""

    obs = JsonLoggerObservability()

    bindings = {
        "trunk.tools": NodeBindings(node_id="trunk.tools", ports=[InMemoryIOPort()]),
        "branch.vision": NodeBindings(node_id="branch.vision", ports=[InMemoryIOPort()], modalities={"image"}),
    }
    resolver = InMemoryBindingResolver(bindings)

    stages: Dict[str, NoopStage] = {
        "trunk.tools": NoopStage("trunk.tools"),
        "branch.vision": NoopStage("branch.vision"),
    }

    port_results = probe_ports(list(bindings.keys()), resolver, modality="text", tenant="demo", observability=obs)
    stage_results = probe_stages(stages, observability=obs)

    print("Port results:", port_results)
    print("Stage results:", stage_results)


if __name__ == "__main__":
    run_demo()
