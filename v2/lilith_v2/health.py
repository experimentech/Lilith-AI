from typing import Any, Dict, Iterable, List, Optional

from .bindings import BindingResolver
from .observability import NoopObservability, Observability
from .stage import Stage


def probe_ports(
    node_ids: List[str],
    resolver: BindingResolver,
    modality: Optional[str] = None,
    tenant: Optional[str] = None,
    observability: Optional[Observability] = None,
) -> List[Dict[str, Any]]:
    """Shallow probe of ports bound to each node.

    Sends a lightweight ping and attempts a receive; exceptions are caught and returned.
    """

    obs = observability or NoopObservability()
    results: List[Dict[str, Any]] = []
    for node in node_ids:
        try:
            ports = resolver.for_node(node, modality=modality, tenant=tenant)
            if not ports:
                results.append({"node_id": node, "ok": True, "ports": 0})
                continue

            port_results: List[Dict[str, Any]] = []
            for port in ports:
                port_status: Dict[str, Any] = {"ok": True}
                try:
                    # best-effort ping; not all ports will round-trip
                    port.send({"ping": True, "node": node}, meta={"health": True})
                    recv = port.receive(meta={"health": True})
                    # Attempt to consume one item if available
                    if isinstance(recv, Iterable):
                        try:
                            next(iter(recv))
                        except StopIteration:
                            pass
                except Exception as exc:  # noqa: BLE001
                    port_status = {"ok": False, "error": str(exc)}
                port_results.append(port_status)
            results.append({"node_id": node, "ok": all(r.get("ok", False) for r in port_results), "ports": port_results})
        except Exception as exc:  # noqa: BLE001
            results.append({"node_id": node, "ok": False, "error": str(exc)})
    obs.on_event("health_ports", "probe", {"results": results}, trace_id=tenant or "")
    return results


def probe_stages(
    stages: Dict[str, Stage],
    observability: Optional[Observability] = None,
) -> List[Dict[str, Any]]:
    """Call stats() on provided stages and capture errors."""

    obs = observability or NoopObservability()
    results: List[Dict[str, Any]] = []
    for node_id, stage in stages.items():
        try:
            stats = stage.stats()
            results.append({"node_id": node_id, "ok": True, "stats": stats})
        except Exception as exc:  # noqa: BLE001
            results.append({"node_id": node_id, "ok": False, "error": str(exc)})
    obs.on_event("health_stages", "probe", {"results": results}, trace_id="")
    return results
