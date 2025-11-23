"""PMFlow plasticity helpers for adaptive retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch

from .base import PipelineArtifact
from .embedding import PMFlowEmbeddingEncoder

try:  # pragma: no cover - validated indirectly via integration tests
    from pmflow_bnn.pmflow import pm_local_plasticity  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback when pmflow isn't available
    try:
        from pmflow_bnn.pmflow import vectorized_pm_plasticity as pm_local_plasticity  # type: ignore[attr-defined]
    except Exception:
        pm_local_plasticity = None  # type: ignore


@dataclass
class PlasticityReport:
    """Summary of a plasticity update."""

    recall_score: Optional[float]
    threshold: float
    delta_centers: float
    delta_mus: float

    def as_dict(self) -> Dict[str, float]:
        payload: Dict[str, float] = {
            "threshold": float(self.threshold),
            "delta_centers": float(self.delta_centers),
            "delta_mus": float(self.delta_mus),
        }
        if self.recall_score is not None:
            payload["recall_score"] = float(self.recall_score)
        return payload


class PlasticityController:
    """Trigger PMFlow plasticity steps based on recall quality."""

    def __init__(
        self,
        encoder: PMFlowEmbeddingEncoder,
        *,
        threshold: float = 0.55,
        mu_lr: float = 5e-4,
        center_lr: float = 5e-4,
    ) -> None:
        self.encoder = encoder
        self.threshold = threshold
        self.mu_lr = mu_lr
        self.center_lr = center_lr
        self._log = logging.getLogger(__name__)
        if pm_local_plasticity is None:
            raise RuntimeError("pmflow_bnn is required for PlasticityController")

    def maybe_update(
        self,
        artefact: PipelineArtifact,
        *,
        recall_score: Optional[float],
    ) -> Optional[PlasticityReport]:
        if recall_score is None:
            return None
        if recall_score >= self.threshold:
            return None

        tokens: Iterable[str] = [token.text for token in artefact.parsed.tokens]
        _, latent_cpu, refined_cpu = self.encoder.encode_with_components(tokens)

        pm_field = self.encoder.pm_field
        device = pm_field.centers.device

        latent = latent_cpu.to(device)
        refined = refined_cpu.to(device)

        before_centers = pm_field.centers.detach().clone()
        before_mus = pm_field.mus.detach().clone()

        pm_local_plasticity(pm_field, latent, refined, mu_lr=self.mu_lr, c_lr=self.center_lr)  # type: ignore

        delta_centers = torch.norm(pm_field.centers - before_centers, p=2).item()
        delta_mus = torch.norm(pm_field.mus - before_mus, p=2).item()

        report = PlasticityReport(
            recall_score=recall_score,
            threshold=self.threshold,
            delta_centers=delta_centers,
            delta_mus=delta_mus,
        )
        self._log.debug(
            "Applied PMFlow plasticity: recall=%.3f threshold=%.3f Δcenters=%.4f Δmus=%.4f",
            recall_score,
            self.threshold,
            delta_centers,
            delta_mus,
        )
        return report