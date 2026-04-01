from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from .relational_graph_store import RelationalGraphStore


@dataclass
class SenseResolution:
    concept_id: str
    confidence: float
    sense_id: Optional[str] = None
    path: str = "direct"
    lm_score: Optional[float] = None
    structural_score: Optional[float] = None
    fused_score: Optional[float] = None


class SenseResolver:
    """Resolve lexeme/sense nodes to canonical concepts.

    This service keeps grounding compatible with legacy concept IDs while allowing
    a sense-first path:
      lexeme -> sense -> concept
      sense -> concept
    """

    def __init__(
        self,
        graph_store: RelationalGraphStore,
        ambiguity_margin: float = 0.10,
        *,
        language_model: Optional[Any] = None,
        enable_lm_rerank: bool = False,
        lm_rerank_min_ambiguity: float = 0.12,
        lm_rerank_weight: float = 0.25,
    ):
        self.graph = graph_store
        self.ambiguity_margin = max(0.0, float(ambiguity_margin))
        self.language_model = language_model
        self.enable_lm_rerank = bool(enable_lm_rerank)
        self.lm_rerank_min_ambiguity = max(0.0, float(lm_rerank_min_ambiguity))
        self.lm_rerank_weight = max(0.0, min(1.0, float(lm_rerank_weight)))

    def resolve(self, node_id: str, base_confidence: float = 1.0, context_query: Optional[str] = None) -> List[SenseResolution]:
        node = self.graph.get_node(node_id)
        if not node:
            return [SenseResolution(concept_id=node_id, confidence=base_confidence, path="missing_node")]

        node_type = node.get("type")

        if node_type in {"concept", "learned_concept", "entity"}:
            return [SenseResolution(concept_id=node_id, confidence=base_confidence, path="direct")]

        if node_type == "sense":
            concept_neighbors = self.graph.get_concepts_for_sense(node_id)
            if concept_neighbors:
                out = []
                for c in concept_neighbors:
                    conf = base_confidence * float(c.get("edge_confidence", 1.0))
                    out.append(
                        SenseResolution(
                            concept_id=c["id"],
                            confidence=min(1.0, conf),
                            sense_id=node_id,
                            path="sense_to_concept",
                            structural_score=min(1.0, conf),
                            fused_score=min(1.0, conf),
                        )
                    )
                out = sorted(out, key=lambda x: x.confidence, reverse=True)
                return self._maybe_lm_rerank(out, context_query)
            return [SenseResolution(concept_id=node_id, confidence=base_confidence, sense_id=node_id, path="sense_unmapped")]

        if node_type in {"lexeme", "word"}:
            sense_neighbors = self.graph.get_senses_for_lexeme(node_id)
            out: List[SenseResolution] = []
            for sense in sense_neighbors:
                sense_id = sense["id"]
                sense_conf = base_confidence * float(sense.get("edge_confidence", 1.0))
                concept_neighbors = self.graph.get_concepts_for_sense(sense_id)
                if concept_neighbors:
                    for c in concept_neighbors:
                        conf = sense_conf * float(c.get("edge_confidence", 1.0))
                        out.append(
                            SenseResolution(
                                concept_id=c["id"],
                                confidence=min(1.0, conf),
                                sense_id=sense_id,
                                path="lexeme_to_sense_to_concept",
                                structural_score=min(1.0, conf),
                                fused_score=min(1.0, conf),
                            )
                        )
                else:
                    out.append(
                        SenseResolution(
                            concept_id=node_id,
                            confidence=min(1.0, sense_conf * 0.7),
                            sense_id=sense_id,
                            path="lexeme_to_sense_unmapped",
                            structural_score=min(1.0, sense_conf * 0.7),
                            fused_score=min(1.0, sense_conf * 0.7),
                        )
                    )
            if out:
                out = sorted(out, key=lambda x: x.confidence, reverse=True)
                return self._maybe_lm_rerank(out, context_query)
            return [SenseResolution(concept_id=node_id, confidence=base_confidence, path="lexeme_unmapped")]

        return [SenseResolution(concept_id=node_id, confidence=base_confidence, path="unhandled_type")]

    def is_ambiguous(self, resolutions: List[SenseResolution]) -> bool:
        if len(resolutions) < 2:
            return False
        top = resolutions[0].confidence
        second = resolutions[1].confidence
        return (top - second) <= self.ambiguity_margin

    def _maybe_lm_rerank(self, resolutions: List[SenseResolution], context_query: Optional[str]) -> List[SenseResolution]:
        """Fuse LM score with structural score when candidates are near-tied."""
        if not self.enable_lm_rerank:
            return resolutions
        if self.language_model is None or not hasattr(self.language_model, "score_text"):
            return resolutions
        if not context_query or len(resolutions) < 2:
            return resolutions

        gap = resolutions[0].confidence - resolutions[1].confidence
        if gap > self.lm_rerank_min_ambiguity:
            return resolutions

        scored: List[SenseResolution] = []
        raw_scores: List[float] = []
        for r in resolutions:
            lm_score = self._lm_candidate_score(context_query, r)
            rr = SenseResolution(
                concept_id=r.concept_id,
                confidence=r.confidence,
                sense_id=r.sense_id,
                path=r.path,
                lm_score=lm_score,
                structural_score=r.structural_score if r.structural_score is not None else r.confidence,
                fused_score=r.confidence,
            )
            scored.append(rr)
            raw_scores.append(lm_score if lm_score is not None else float("-inf"))

        finite_scores = [s for s in raw_scores if s != float("-inf")]
        if not finite_scores:
            return resolutions

        lo = min(finite_scores)
        hi = max(finite_scores)
        span = hi - lo

        reranked: List[SenseResolution] = []
        for r in scored:
            if r.lm_score is None:
                norm = 0.0
            elif span <= 1e-9:
                norm = 0.5
            else:
                norm = (r.lm_score - lo) / span

            fused = (1.0 - self.lm_rerank_weight) * r.confidence + self.lm_rerank_weight * norm
            reranked.append(
                SenseResolution(
                    concept_id=r.concept_id,
                    confidence=min(1.0, max(0.0, fused)),
                    sense_id=r.sense_id,
                    path=f"{r.path}:lm_rerank",
                    lm_score=r.lm_score,
                    structural_score=r.structural_score if r.structural_score is not None else r.confidence,
                    fused_score=min(1.0, max(0.0, fused)),
                )
            )

        return sorted(reranked, key=lambda x: x.confidence, reverse=True)

    def _lm_candidate_score(self, context_query: str, resolution: SenseResolution) -> Optional[float]:
        """Score a candidate concept/sense in context via LM adapter."""
        sense_term = ""
        if resolution.sense_id:
            sense_node = self.graph.get_node(resolution.sense_id)
            if sense_node:
                sense_term = sense_node.get("term") or ""

        concept_term = ""
        concept_node = self.graph.get_node(resolution.concept_id)
        if concept_node:
            concept_term = concept_node.get("term") or ""

        prompt = (
            f"query: {context_query}\n"
            f"sense: {sense_term}\n"
            f"concept: {concept_term}"
        )
        try:
            lm = self.language_model
            if lm is None:
                return None
            return lm.score_text(prompt)
        except Exception:
            return None
