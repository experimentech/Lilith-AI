"""Embedding encoders for the language-to-symbol pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F


class HashedEmbeddingEncoder:
    """Convert tokens into a fixed-width embedding via hashing.

    This keeps dependencies minimal while providing deterministic vectors that play
    nicely with the existing SQLiteVectorStore.
    """

    def __init__(self, *, dimension: int = 64) -> None:
        self.dimension = dimension

    def _hash_token(self, token: str) -> int:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dimension

    def encode(self, tokens: Iterable[str] | str) -> torch.Tensor:
        # Convenience: many call sites pass raw text.
        if isinstance(tokens, str):
            tokens = tokens.split()
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in tokens:
            index = self._hash_token(token)
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return torch.from_numpy(vector).unsqueeze(0)


class PMFlowEmbeddingEncoder:
    """Blend hashed bag-of-words features with a PMFlow latent field.

    The PMFlow field injects a touch of learned-like structure without requiring
    any training loop. We initialise it deterministically so embeddings are
    stable across runs.
    """

    def __init__(
        self,
        *,
        dimension: int = 96,
        latent_dim: int = 48,
        seed: int = 13,
        combine_mode: str = "concat",
        device: Optional[torch.device] = None,
        base_encoder: Optional[HashedEmbeddingEncoder] = None,
    ) -> None:
        if combine_mode not in {"concat", "pm-only"}:
            raise ValueError("combine_mode must be 'concat' or 'pm-only'.")
        self.combine_mode = combine_mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_encoder = base_encoder or HashedEmbeddingEncoder(dimension=dimension)
        self.dimension = self.base_encoder.dimension
        self.latent_dim = latent_dim
        self._projection = self._build_projection_matrix(self.dimension, latent_dim, seed).to(self.device)
        self.pm_field = self._init_pm_field(latent_dim, seed)
        self.pm_field.to(self.device)
        self.pm_field.eval()
        self._state_path: Optional[Path] = None

    @staticmethod
    def _build_projection_matrix(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        matrix = rng.standard_normal((input_dim, output_dim), dtype=np.float32)
        return torch.from_numpy(matrix)

    @staticmethod
    def _init_pm_field(latent_dim: int, seed: int):
        """Create a deterministic PMFlow field using bundled implementations."""
        from pmflow.core.pmflow import MultiScalePMField, ParallelPMField

        def _seed_multiscale(field):
            generator = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                centres_fine = torch.randn(
                    field.fine_field.centers.shape,
                    generator=generator,
                    device=field.fine_field.centers.device,
                ) * 0.5
                mus_fine = torch.full(
                    field.fine_field.mus.shape,
                    0.35,
                    device=field.fine_field.mus.device,
                )
                omegas_fine = torch.randn(
                    field.fine_field.mus.shape,
                    generator=generator,
                    device=field.fine_field.mus.device,
                ) * 0.01

                field.fine_field.centers.copy_(centres_fine)
                field.fine_field.mus.copy_(mus_fine)
                if hasattr(field.fine_field, "omegas"):
                    field.fine_field.omegas.copy_(omegas_fine)

                centres_coarse = torch.randn(
                    field.coarse_field.centers.shape,
                    generator=generator,
                    device=field.coarse_field.centers.device,
                ) * 0.5
                mus_coarse = torch.full(
                    field.coarse_field.mus.shape,
                    0.35,
                    device=field.coarse_field.mus.device,
                )
                omegas_coarse = torch.randn(
                    field.coarse_field.mus.shape,
                    generator=generator,
                    device=field.coarse_field.mus.device,
                ) * 0.01

                field.coarse_field.centers.copy_(centres_coarse)
                field.coarse_field.mus.copy_(mus_coarse)
                if hasattr(field.coarse_field, "omegas"):
                    field.coarse_field.omegas.copy_(omegas_coarse)
            return field

        # Prefer MultiScale; if a TypeError indicates older pmflow signature, retry without enable_flow.
        multiscale_kwargs = dict(
            d_latent=latent_dim,
            n_centers_fine=128,
            n_centers_coarse=32,
            steps_fine=5,
            steps_coarse=3,
            dt=0.15,
            beta=1.2,
            clamp=3.0,
            enable_flow=True,  # Enable agentic frame-dragging
        )

        try:
            field = MultiScalePMField(**multiscale_kwargs)
            return _seed_multiscale(field)
        except TypeError as exc:
            # Older pmflow versions may not accept enable_flow; retry without it.
            if "enable_flow" in str(exc):
                multiscale_kwargs.pop("enable_flow", None)
                field = MultiScalePMField(**multiscale_kwargs)
                return _seed_multiscale(field)
            raise
        except Exception:
            # As a last resort, fall back to parallel to keep embeddings usable (hierarchical retrieval will be disabled elsewhere).
            field = ParallelPMField(d_latent=latent_dim, steps=5, dt=0.08, beta=0.9, clamp=2.5, enable_flow=True)
            generator = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                centres = torch.randn(
                    field.centers.shape,
                    generator=generator,
                    device=field.centers.device,
                ) * 0.5
                mus = torch.full(field.mus.shape, 0.35, device=field.mus.device)
                omegas = torch.randn(field.mus.shape, generator=generator, device=field.mus.device) * 0.01

                field.centers.copy_(centres)
                field.mus.copy_(mus)
                if hasattr(field, "omegas"):
                    field.omegas.copy_(omegas)
            return field

    def encode(self, tokens: Iterable[str] | str) -> torch.Tensor:
        combined, _, _ = self._encode_internal(tokens)
        return combined

    def encode_with_components(self, tokens: Iterable[str] | str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return embedding together with PMFlow latent and raw activations.

        The latent corresponds to the input fed into the PMFlow field and the
        raw activation is the unnormalised PMFlow output before concatenation.
        """

        combined, latent, raw_refined = self._encode_internal(tokens)
        return combined, latent, raw_refined

    def attach_state_path(self, path: Optional[Path]) -> None:
        """Opt into persistence of the PMFlow field parameters."""

        self._state_path = path
        if path and path.exists():
            self.load_state(path)

    def save_state(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = self._state_path
        if path is None:
            return
        
        # Handle MultiScalePMField (has fine_field and coarse_field)
        if hasattr(self.pm_field, 'fine_field') and hasattr(self.pm_field, 'coarse_field'):
            payload = {
                "type": "multiscale",
                "fine_centers": self.pm_field.fine_field.centers.detach().cpu(),
                "fine_mus": self.pm_field.fine_field.mus.detach().cpu(),
                "fine_omegas": getattr(self.pm_field.fine_field, 'omegas', torch.tensor([])).detach().cpu(),
                "coarse_centers": self.pm_field.coarse_field.centers.detach().cpu(),
                "coarse_mus": self.pm_field.coarse_field.mus.detach().cpu(),
                "coarse_omegas": getattr(self.pm_field.coarse_field, 'omegas', torch.tensor([])).detach().cpu(),
                "coarse_projection": self.pm_field.coarse_projection.weight.detach().cpu(),
            }
        else:
            # Standard PMField
            payload = {
                "type": "standard",
                "centers": self.pm_field.centers.detach().cpu(),
                "mus": self.pm_field.mus.detach().cpu(),
                "omegas": getattr(self.pm_field, 'omegas', torch.tensor([])).detach().cpu(),
            }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_state(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = self._state_path
        if path is None or not path.exists():
            return
        payload = torch.load(path, map_location=self.device)
        
        with torch.no_grad():
            # Handle MultiScalePMField
            if payload.get("type") == "multiscale":
                if hasattr(self.pm_field, 'fine_field') and hasattr(self.pm_field, 'coarse_field'):
                    self.pm_field.fine_field.centers.copy_(payload["fine_centers"].to(self.device))
                    self.pm_field.fine_field.mus.copy_(payload["fine_mus"].to(self.device))
                    if "fine_omegas" in payload and hasattr(self.pm_field.fine_field, 'omegas'):
                         self.pm_field.fine_field.omegas.copy_(payload["fine_omegas"].to(self.device))
                    
                    self.pm_field.coarse_field.centers.copy_(payload["coarse_centers"].to(self.device))
                    self.pm_field.coarse_field.mus.copy_(payload["coarse_mus"].to(self.device))
                    if "coarse_omegas" in payload and hasattr(self.pm_field.coarse_field, 'omegas'):
                        self.pm_field.coarse_field.omegas.copy_(payload["coarse_omegas"].to(self.device))
                    
                    self.pm_field.coarse_projection.weight.copy_(payload["coarse_projection"].to(self.device))
            # Handle standard PMField (backward compatibility)
            elif "centers" in payload:
                if hasattr(self.pm_field, 'centers'):
                    self.pm_field.centers.copy_(payload["centers"].to(self.device))
                if "mus" in payload and hasattr(self.pm_field, 'mus'):
                    self.pm_field.mus.copy_(payload["mus"].to(self.device))
                if "omegas" in payload and hasattr(self.pm_field, 'omegas'):
                    self.pm_field.omegas.copy_(payload["omegas"].to(self.device))

    def _encode_internal(self, tokens: Iterable[str] | str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(tokens, str):
            tokens = tokens.split()
        with torch.no_grad():
            base = self.base_encoder.encode(tokens).to(self.device)
            latent = base @ self._projection
            
            # Handle MultiScalePMField which returns (fine, coarse, combined) tuple
            pm_output = self.pm_field(latent)
            if isinstance(pm_output, tuple) and len(pm_output) == 3:
                # MultiScalePMField returns (fine_emb, coarse_emb, combined)
                # Use combined for hierarchical concept representation
                raw_refined: torch.Tensor = pm_output[2]  # Combined multi-scale embedding
            else:
                # Standard PMField returns single tensor
                raw_refined: torch.Tensor = pm_output
            
            refined = F.normalize(raw_refined, p=2, dim=1)
            if self.combine_mode == "concat":
                hashed = F.normalize(base, p=2, dim=1)
                combined = torch.cat([hashed, refined], dim=1)
            else:
                combined = refined
            return combined.cpu(), latent.detach().cpu(), raw_refined.detach().cpu()