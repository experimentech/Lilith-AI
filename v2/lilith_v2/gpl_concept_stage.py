from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .general_purpose_concept import ConceptGPLearner
from .pattern_store_adapter import PatternStoreAdapter
from .stage import Stage
from .store import Store

try:  # Optional PMFlow import; stage can run without it
    from pmflow.encoder import PMFlowEmbeddingEncoder
    from pmflow.core.retrieval import CompositionalRetrievalPMField
    from pmflow.core.pmflow import MultiScalePMField
except ImportError:  # pragma: no cover - exercised when pmflow is absent
    PMFlowEmbeddingEncoder = None  # type: ignore
    CompositionalRetrievalPMField = None  # type: ignore
    MultiScalePMField = None  # type: ignore


def _to_list(vector: Any) -> Optional[List[float]]:
    if vector is None:
        return None
    try:
        if hasattr(vector, "detach"):
            vector = vector.detach().cpu()
        arr = vector.squeeze().tolist()
        return [float(x) for x in arr]
    except Exception:
        return None


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class GPLConceptStage(Stage):
    """Concept stage that wraps the General Purpose Learner with PMFlow embeddings."""

    def __init__(
        self,
        stage_id: str,
        store: Store,
        *,
        encoder: Optional[Any] = None,
        learner_config: Optional[Dict[str, Any]] = None,
        pmflow_config: Optional[Dict[str, Any]] = None,
        retrieval_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.id = stage_id
        self.store = store
        self.pattern_store = PatternStoreAdapter(store, stage_id)
        self.encoder = encoder or self._maybe_build_encoder(pmflow_config or {})
        lcfg = learner_config or {}
        self.learner = ConceptGPLearner(
            pattern_store=self.pattern_store,
            learning_rate=float(lcfg.get("learning_rate", 0.1)),
            learning_mode=str(lcfg.get("learning_mode", "moderate")),
        )
        self.retrieval_config = retrieval_config or {}
        self._successes = 0
        self._compositional = None

    def encode(self, item: Any, ctx: Dict[str, Any]) -> Any:
        text = self._as_text(item)
        combined, latent, pm_raw = self._encode_components(text)
        ctx["embedding"] = combined
        ctx["pm_latent"] = latent
        ctx["pm_raw"] = pm_raw
        return {"text": text, "embedding": combined, "pm_latent": latent, "pm_raw": pm_raw}

    def retrieve(self, query: Any, ctx: Dict[str, Any]) -> Iterable[Any]:
        if isinstance(query, dict) and query.get("id"):
            hit = self.pattern_store.get(str(query["id"]))
            return [hit] if hit else []

        qtext = self._as_text(query)
        combined, latent, _ = self._encode_components(qtext)
        patterns = list(self.pattern_store.list())

        # PMFlow-enhanced retrieval when enabled and available
        if self._can_use_pmflow() and self.retrieval_config.get("pmflow", False) and latent is not None:
            results = self._pmflow_retrieve(latent, patterns)
            if results is not None:
                return results

        qvec = combined
        if not qvec:
            return patterns

        ranked = []
        for pat in patterns:
            pvec = pat.get("embedding")
            if not isinstance(pvec, list):
                continue
            score = _cosine(qvec, [float(x) for x in pvec])
            ranked.append((score, pat))

        if not ranked:
            return patterns

        ranked.sort(key=lambda tup: tup[0], reverse=True)
        k = int(self.retrieval_config.get("top_k", 5))
        return [pat for _, pat in ranked[:k]]

    def learn(self, event: Any, ctx: Dict[str, Any]) -> None:
        payload = event if isinstance(event, dict) else {"text": self._as_text(event)}
        if "embedding" not in ctx:
            combined, latent, pm_raw = self._encode_components(self._as_text(payload))
            ctx["embedding"] = combined
            ctx["pm_latent"] = latent
            ctx["pm_raw"] = pm_raw
        # Reuse payload as both input/output for GPL loop
        self.learner.observe_interaction(payload, payload, ctx)

    def update_success(self, feedback: Any, ctx: Dict[str, Any]) -> None:
        pid = None
        delta = 1.0
        if isinstance(feedback, dict):
            pid = feedback.get("id") or feedback.get("pattern_id")
            delta = float(feedback.get("delta", delta))
        elif isinstance(feedback, (int, float)):
            delta = float(feedback)
        if pid:
            self.pattern_store.update_success(str(pid), delta, plasticity_rate=1.0)
        self._successes += 1

    def stats(self) -> Dict[str, Any]:
        count = sum(1 for _ in self.pattern_store.list())
        learner_stats = self.learner.get_learning_stats()
        learner_stats.update({"patterns": count, "successes": self._successes})
        return learner_stats

    def relational_sidecar(self, sql_ctx: Any) -> Optional[Any]:
        if self.store.__class__.__name__ == "RelationalStore":
            return {"table": "kv", "prefix": f"{self.id}:pattern:"}
        return None

    def _maybe_build_encoder(self, cfg: Dict[str, Any]) -> Optional[Any]:
        if PMFlowEmbeddingEncoder is None:
            return None
        encoder = PMFlowEmbeddingEncoder(
            dimension=int(cfg.get("dimension", 96)),
            latent_dim=int(cfg.get("latent_dim", 48)),
            seed=int(cfg.get("seed", 13)),
            combine_mode=str(cfg.get("combine_mode", "concat")),
            target_pm_dim=cfg.get("target_pm_dim"),
        )
        state_path = cfg.get("state_path")
        if state_path:
            encoder.attach_state_path(Path(state_path))
        return encoder

    def _encode_components(self, text: str) -> tuple[Optional[List[float]], Optional[Any], Optional[Any]]:
        if not text or not self.encoder:
            return None, None, None
        tokens = text.split()
        try:
            if hasattr(self.encoder, "encode_with_components"):
                combined, latent, raw = self.encoder.encode_with_components(tokens)
            else:
                combined = self.encoder.encode(tokens)
                latent, raw = None, None
            return _to_list(combined), latent, raw
        except Exception:
            return None, None, None

    def _as_text(self, item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("text") or item.get("value") or item.get("descriptor") or "")
        return str(item)

    def _can_use_pmflow(self) -> bool:
        return PMFlowEmbeddingEncoder is not None and self.encoder is not None and hasattr(self.encoder, "pm_field")

    def _pmflow_retrieve(self, query_latent: Any, patterns: List[Dict[str, Any]]) -> Optional[Iterable[Any]]:
        if CompositionalRetrievalPMField is None or MultiScalePMField is None:
            return None
        pm_field = getattr(self.encoder, "pm_field", None)
        if pm_field is None or not isinstance(pm_field, MultiScalePMField):
            return None
        # Ensure we have latents for all patterns
        candidate_latents = []
        candidates = []
        for pat in patterns:
            lat = pat.get("pm_latent")
            if isinstance(lat, list):
                if lat and isinstance(lat[0], list):
                    lat = lat[0]
                candidate_latents.append(lat)
                candidates.append(pat)
        if not candidates:
            return None

        import torch

        qz = torch.as_tensor(query_latent, dtype=torch.float32)
        if qz.dim() == 1:
            qz = qz.unsqueeze(0)
        cz = torch.as_tensor(candidate_latents, dtype=torch.float32)
        if cz.dim() == 1:
            cz = cz.unsqueeze(0)
        if self._compositional is None:
            self._compositional = CompositionalRetrievalPMField(pm_field)
        expand = bool(self.retrieval_config.get("expand_query", True))
        use_h = bool(self.retrieval_config.get("hierarchical", True))
        min_sim = float(self.retrieval_config.get("min_similarity", 0.4))
        results = self._compositional.retrieve_concepts(
            qz,
            cz,
            expand_query=expand,
            use_hierarchical=use_h,
            min_similarity=min_sim,
        )
        if not results:
            return candidates
        top_k = int(self.retrieval_config.get("top_k", len(results)))
        ranked = [candidates[idx] for idx, _ in results[:top_k]]
        return ranked
