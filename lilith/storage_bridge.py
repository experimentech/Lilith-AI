"""Helper functions to persist symbolic frames alongside embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from ..storage.sqlite_store import SQLiteVectorStore
from .base import PipelineArtifact


class SymbolicStore:
    """Store embeddings and dump symbolic frames for inspection."""

    def __init__(self, sqlite_path: Path, *, scenario: str) -> None:
        self.vector_store = SQLiteVectorStore(sqlite_path)
        self.scenario = scenario
        self._frames_path = Path("runs") / f"{scenario}_frames.json"

    @property
    def frames_path(self) -> Path:
        return self._frames_path

    def load_frames(self) -> List[dict]:
        if not self._frames_path.exists():
            return []
        try:
            return json.loads(self._frames_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def persist(self, artefacts: Iterable[PipelineArtifact]) -> None:
        artefact_list = list(artefacts)
        if not artefact_list:
            return

        embeddings: List[torch.Tensor] = []
        labels: List[int] = []
        frames: List[Dict[str, Any]] = []

        base_offset = self.vector_store.count(scenario=self.scenario)

        for idx, artefact in enumerate(artefact_list):
            embeddings.append(artefact.embedding)
            labels.append(base_offset + idx)
            frame_payload: Dict[str, Any] = dict(artefact.frame.as_dict())
            frame_payload["label"] = base_offset + idx
            frame_payload["normalised_text"] = artefact.normalised_text
            frame_payload["tokens"] = [token.text for token in artefact.parsed.tokens]
            frames.append(frame_payload)

        stacked_embeddings = torch.cat(embeddings, dim=0)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        self.vector_store.add(stacked_embeddings, label_tensor, scenario=self.scenario)

        dump_path = self._frames_path
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        existing: List[dict] = []
        if dump_path.exists():
            try:
                existing = json.loads(dump_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = []
        existing.extend(frames)
        dump_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    def refresh_embeddings(self, encoder: Any) -> int:
        """Re-encode stored embeddings for the current scenario using ``encoder``.

        Returns the number of entries that were re-encoded; entries lacking token
        metadata retain their previous vectors.
        """

        frames = self.load_frames()
        if not frames:
            return 0

        try:
            existing_vectors, existing_labels = self.vector_store.fetch(scenario=self.scenario)
        except RuntimeError:
            existing_vectors = torch.empty(0, dtype=torch.float32)
            existing_labels = torch.empty(0, dtype=torch.long)

        label_to_vector = {
            int(label.item()): existing_vectors[idx : idx + 1].clone()
            for idx, label in enumerate(existing_labels)
        }

        updated_embeddings: List[torch.Tensor] = []
        updated_labels: List[int] = []
        refreshed_count = 0

        for frame in frames:
            label_value = frame.get("label")
            if label_value is None:
                continue
            label_int = int(label_value)
            tokens = frame.get("tokens")
            if tokens:
                embedding = encoder.encode(tokens)
                refreshed_count += 1
            else:
                embedding = label_to_vector.get(label_int)
                if embedding is None:
                    continue
            updated_embeddings.append(embedding.cpu())
            updated_labels.append(label_int)

        if not updated_embeddings:
            return 0

        embeddings_tensor = torch.cat(updated_embeddings, dim=0)
        label_tensor = torch.tensor(updated_labels, dtype=torch.long)
        self.vector_store.clear(scenario=self.scenario)
        self.vector_store.add(embeddings_tensor, label_tensor, scenario=self.scenario)
        return refreshed_count