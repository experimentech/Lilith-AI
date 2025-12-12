"""High-level pipeline orchestrator that links all components."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

from . import symbolic
from .base import PipelineArtifact, Utterance
from .embedding import HashedEmbeddingEncoder, PMFlowEmbeddingEncoder
from .intake import NoiseNormalizer
from .parser import parse as parse_sentence


class SymbolicPipeline:
    """Run the language-to-symbol pipeline for a batch of utterances."""

    def __init__(
        self,
        normalizer: NoiseNormalizer | None = None,
        encoder: HashedEmbeddingEncoder | PMFlowEmbeddingEncoder | None = None,
        *,
        use_pmflow: bool = True,
        pmflow_kwargs: Optional[dict] = None,
        pmflow_state_path: Path | None = None,
    ) -> None:
        self.normalizer = normalizer or NoiseNormalizer()
        self.pmflow_state_path = pmflow_state_path
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = self._build_encoder(use_pmflow=use_pmflow, pmflow_kwargs=pmflow_kwargs)
        self._attach_pmflow_state()

    def _build_encoder(
        self,
        *,
        use_pmflow: bool,
        pmflow_kwargs: Optional[dict],
    ) -> HashedEmbeddingEncoder | PMFlowEmbeddingEncoder:
        if not use_pmflow:
            return HashedEmbeddingEncoder()
        try:
            return PMFlowEmbeddingEncoder(**(pmflow_kwargs or {}))
        except (RuntimeError, ImportError) as exc:
            logging.getLogger(__name__).warning(
                "PMFlow embeddings unavailable (%s); falling back to hashed encoder.",
                exc,
            )
            return HashedEmbeddingEncoder()

    def _attach_pmflow_state(self) -> None:
        if isinstance(self.encoder, PMFlowEmbeddingEncoder):
            self.encoder.attach_state_path(self.pmflow_state_path)

    def process(self, utterance: Utterance) -> PipelineArtifact:
        normalised = self.normalizer.normalise(utterance)
        candidates = self.normalizer.generate_candidates(normalised)[:8]
        parsed = parse_sentence(normalised)
        frame = symbolic.build_frame(utterance, parsed, normalised)
        embedding = self.encoder.encode([token.text for token in parsed.tokens])
        return PipelineArtifact(
            utterance=utterance,
            normalised_text=normalised,
            candidates=candidates,
            parsed=parsed,
            frame=frame,
            embedding=embedding,
            confidence=parsed.confidence,
        )

    def run(self, utterances: Iterable[Utterance]) -> List[PipelineArtifact]:
        return [self.process(utterance) for utterance in utterances]