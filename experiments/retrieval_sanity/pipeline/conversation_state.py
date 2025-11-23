"""PMFlow-driven working memory tracker for the conversation responder."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from .base import PipelineArtifact, SymbolicFrame
from .embedding import PMFlowEmbeddingEncoder


@dataclass
class WorkingMemoryTopic:
    """Compact representation of a short-lived conversation topic."""

    signature: Tuple[int, ...]
    strength: float
    summary: str
    mentions: int
    last_strength: float


@dataclass
class ConversationStateSnapshot:
    """Immutable snapshot of the current working-memory state."""

    active: bool
    activation_energy: float
    novelty: float
    topics: List[WorkingMemoryTopic]

    @property
    def dominant(self) -> Optional[WorkingMemoryTopic]:
        return self.topics[0] if self.topics else None


class ConversationState:
    """Track PMFlow activations as a lightweight working memory."""

    def __init__(
        self,
        encoder: PMFlowEmbeddingEncoder | None,
        *,
        decay: float = 0.75,
        max_topics: int = 5,
        topk: int = 4,
        novelty_alpha: float = 0.4,
        bucket_stride: int = 16,  # Increased from 8 for looser matching
    ) -> None:
        self.encoder = encoder
        self.decay = decay
        self.max_topics = max_topics
        self.topk = topk
        self.novelty_alpha = max(0.0, min(1.0, novelty_alpha))
        self.bucket_stride = max(1, bucket_stride)
        self._topics: Dict[Tuple[Tuple[int, ...], Tuple[str, str, str]], WorkingMemoryTopic] = {}
        self._last_activation: Optional[torch.Tensor] = None
        self._last_novelty: Optional[float] = None

    def is_active(self) -> bool:
        return self.encoder is not None

    def update(self, artefact: PipelineArtifact) -> ConversationStateSnapshot:
        if self.encoder is None:
            return ConversationStateSnapshot(
                active=False,
                activation_energy=0.0,
                novelty=0.0,
                topics=[],
            )

        tokens: Iterable[str] = [token.text for token in artefact.parsed.tokens]
        if not tokens:
            tokens = artefact.normalised_text.split()

        _, _latent_cpu, raw_cpu = self.encoder.encode_with_components(tokens)
        activation = raw_cpu.squeeze(0).to(torch.float32)
        energy = float(torch.norm(activation, p=2).item())

        signature = self._build_signature(activation)
        bucket = self._bucket_signature(signature)
        frame_key = _frame_signature(artefact.frame)
        topic_key = (bucket, frame_key)
        summary = _summarise_topic(artefact)

        topic = self._topics.get(topic_key)
        if topic:
            updated_strength = topic.strength * self.decay + energy * (1.0 - self.decay)
            topic.strength = updated_strength
            topic.summary = summary
            topic.mentions += 1
            topic.last_strength = energy
            topic.signature = signature
        else:
            topic = WorkingMemoryTopic(
                signature=signature,
                strength=energy,
                summary=summary,
                mentions=1,
                last_strength=energy,
            )
            self._topics[topic_key] = topic

        stale_keys: List[Tuple[Tuple[int, ...], Tuple[str, str, str]]] = []
        for key, existing in self._topics.items():
            if key == topic_key:
                continue
            existing.strength *= self.decay
            if existing.strength < 1e-4:
                stale_keys.append(key)
        for key in stale_keys:
            self._topics.pop(key, None)

        items = sorted(self._topics.items(), key=lambda pair: pair[1].strength, reverse=True)
        if len(items) > self.max_topics:
            for key, _ in items[self.max_topics :]:
                self._topics.pop(key, None)
            items = items[: self.max_topics]
        sorted_topics = [topic for _, topic in items]

        raw_novelty = self._compute_novelty(activation)
        if self._last_novelty is None:
            novelty = raw_novelty
        else:
            novelty = self.novelty_alpha * raw_novelty + (1.0 - self.novelty_alpha) * self._last_novelty
        self._last_novelty = novelty
        self._last_activation = activation.detach().clone()

        snapshot_topics = [replace(topic) for topic in sorted_topics]
        return ConversationStateSnapshot(
            active=True,
            activation_energy=energy,
            novelty=novelty,
            topics=snapshot_topics,
        )

    def _build_signature(self, activation: torch.Tensor) -> Tuple[int, ...]:
        k = min(self.topk, activation.numel())
        if k == 0:
            return tuple()
        indices = torch.topk(activation.abs(), k=k).indices.cpu().tolist()
        return tuple(indices)

    def _bucket_signature(self, signature: Tuple[int, ...]) -> Tuple[int, ...]:
        """Bucket signature indices to group similar activation patterns.
        
        Only buckets the most dominant component (first index) to allow
        more variation in secondary activations. Relies on frame_signature
        to provide semantic matching.
        """
        if not signature:
            return signature
        # Only bucket the dominant activation, allow variation in others
        return (signature[0] // self.bucket_stride,)

    def _compute_novelty(self, activation: torch.Tensor) -> float:
        if self._last_activation is None:
            return 0.0
        prev = self._last_activation
        denom = torch.norm(prev, p=2) * torch.norm(activation, p=2)
        if denom.item() == 0.0:
            return 0.0
        cosine = torch.dot(prev, activation) / denom
        cosine_clamped = torch.clamp(cosine, -1.0, 1.0).item()
        return float((1.0 - cosine_clamped) / 2.0)


def _frame_signature(frame: SymbolicFrame) -> Tuple[str, str, str]:
    actor = (frame.actor or "").strip().lower()
    action = (frame.action or "").strip().lower()
    target = (frame.target or "").strip().lower()
    return actor, action, target


def _summarise_topic(artefact: PipelineArtifact) -> str:
    frame = artefact.frame
    primary: List[str] = []
    if frame.actor:
        primary.append(str(frame.actor))
    if frame.action:
        primary.append(str(frame.action))
    if frame.target and frame.target not in primary:
        primary.append(str(frame.target))

    attribute_keys = (
        "time_phrase",
        "location",
        "object",
        "topic",
        "subject",
    )
    extras: List[str] = []
    for key in attribute_keys:
        value = frame.attributes.get(key)
        if value:
            extras.append(str(value))
    if not extras:
        modifiers = list(frame.modifiers.values())
        if modifiers:
            extras.append(str(modifiers[0]))

    tokens: List[str] = list(primary + extras)
    summary = " ".join(token for token in tokens if token).strip()

    if len(summary.split()) < 2:
        fallback = artefact.normalised_text.strip() or artefact.utterance.text.strip()
        if fallback:
            summary = " ".join(fallback.split()[:8]).strip()

    if not summary:
        summary = frame.raw_text.strip()
    if not summary:
        summary = "this topic"
    return summary
