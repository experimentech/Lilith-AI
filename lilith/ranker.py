"""Lightweight candidate scorer for cross-source response selection."""

from dataclasses import dataclass
from typing import Dict, Optional
import json
import os
from pathlib import Path


@dataclass
class RankingFeatures:
    source: str  # pattern | concept | template | ka | pragmatic | weather | math
    retrieval_confidence: float
    pmflow_activation: float
    recency: float
    feedback_score: float
    length_penalty: float
    syntax_score: float
    concept_overlap: float
    is_fallback: bool
    approach_success_rate: float


class LinearScorer:
    """Tiny linear scorer with JSON-configurable weights and event logging."""

    DEFAULT_WEIGHTS = {
        "bias": 0.0,
        "source.pattern": 0.15,
        "source.concept": 0.18,
        "source.template": 0.12,
        "source.ka": 0.10,
        "source.pragmatic": 0.10,
        "source.weather": 0.08,
        "source.math": 0.10,
        "retrieval_confidence": 0.45,
        "pmflow_activation": 0.25,
        "recency": 0.10,
        "feedback_score": 0.20,
        "length_penalty": -0.10,
        "syntax_score": 0.08,
        "concept_overlap": 0.12,
        "is_fallback": -0.25,
        "approach_success_rate": 0.25,
    }

    def __init__(self, weights_path: str = "data/ranker_weights.json", log_path: str = "data/ranker_events.jsonl"):
        self.weights_path = Path(weights_path)
        self.log_path = Path(log_path)
        self.weights = self._load_weights()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_weights(self) -> Dict[str, float]:
        if self.weights_path.exists():
            try:
                data = json.loads(self.weights_path.read_text())
                if isinstance(data, dict):
                    return {**self.DEFAULT_WEIGHTS, **data}
            except Exception:
                pass
        return dict(self.DEFAULT_WEIGHTS)

    def score(self, feats: RankingFeatures) -> float:
        w = self.weights
        score = w.get("bias", 0.0)
        score += w.get(f"source.{feats.source}", 0.0)
        score += w.get("retrieval_confidence", 0.0) * feats.retrieval_confidence
        score += w.get("pmflow_activation", 0.0) * feats.pmflow_activation
        score += w.get("recency", 0.0) * feats.recency
        score += w.get("feedback_score", 0.0) * feats.feedback_score
        score += w.get("length_penalty", 0.0) * feats.length_penalty
        score += w.get("syntax_score", 0.0) * feats.syntax_score
        score += w.get("concept_overlap", 0.0) * feats.concept_overlap
        score += w.get("approach_success_rate", 0.0) * feats.approach_success_rate
        if feats.is_fallback:
            score += w.get("is_fallback", 0.0)
        return score

    def log(self, feats: RankingFeatures, raw_score: float, chosen: bool) -> None:
        try:
            entry = {
                "source": feats.source,
                "retrieval_confidence": feats.retrieval_confidence,
                "pmflow_activation": feats.pmflow_activation,
                "recency": feats.recency,
                "feedback_score": feats.feedback_score,
                "length_penalty": feats.length_penalty,
                "syntax_score": feats.syntax_score,
                "concept_overlap": feats.concept_overlap,
                "approach_success_rate": feats.approach_success_rate,
                "is_fallback": feats.is_fallback,
                "score": raw_score,
                "chosen": chosen,
            }
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
