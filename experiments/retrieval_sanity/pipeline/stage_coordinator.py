"""Multi-stage coordinator for distributed PMFlow BNN pipeline.

Each cognitive stage has:
- Dedicated PMFlow encoder (specialized BNN)
- Dedicated database namespace
- Independent plasticity learning
- Specific task focus

Stages communicate via standardized artifacts, enabling:
- Incremental specialization
- Independent evaluation
- Modular replacement
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .base import PipelineArtifact, Utterance
from .embedding import PMFlowEmbeddingEncoder


class StageType(Enum):
    """Cognitive processing stages in the pipeline."""
    
    INTAKE = "intake"           # Noise normalization, typo learning
    SEMANTIC = "semantic"       # Concept embedding, retrieval
    REASONING = "reasoning"     # Inference, logical composition
    RESPONSE = "response"       # Generation quality, coherence


@dataclass
class StageConfig:
    """Configuration for a single processing stage."""
    
    stage_type: StageType
    encoder_config: Dict[str, Any] = field(default_factory=dict)
    db_namespace: Optional[str] = None
    state_path: Optional[Path] = None
    plasticity_enabled: bool = True
    plasticity_lr: float = 5e-4
    
    def __post_init__(self) -> None:
        if self.db_namespace is None:
            self.db_namespace = f"{self.stage_type.value}_memory"


@dataclass
class StageArtifact:
    """Output from a single processing stage.
    
    Each stage produces embeddings + metadata that downstream stages can use.
    """
    
    stage: StageType
    embedding: torch.Tensor
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional stage-specific data
    tokens: Optional[List[str]] = None
    activations: Optional[torch.Tensor] = None  # Raw PMFlow activations
    centers_snapshot: Optional[torch.Tensor] = None  # PMFlow center positions


class CognitiveStage:
    """Single specialized BNN processing stage."""
    
    def __init__(
        self,
        config: StageConfig,
        encoder: Optional[PMFlowEmbeddingEncoder] = None,
    ) -> None:
        self.config = config
        self.stage_type = config.stage_type
        self._log = logging.getLogger(f"{__name__}.{config.stage_type.value}")
        
        # Each stage has its own PMFlow encoder
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = PMFlowEmbeddingEncoder(**config.encoder_config)
            if config.state_path:
                self.encoder.attach_state_path(config.state_path)
        
        self._log.info(
            "Initialized %s stage (plasticity=%s, namespace=%s)",
            self.stage_type.value,
            config.plasticity_enabled,
            config.db_namespace,
        )
    
    def process(
        self,
        input_data: Any,
        upstream_artifacts: Optional[List[StageArtifact]] = None,
    ) -> StageArtifact:
        """Process input through this stage's specialized BNN.
        
        Args:
            input_data: Stage-specific input (tokens, frames, etc.)
            upstream_artifacts: Results from previous stages (for context)
        
        Returns:
            StageArtifact with embedding and metadata
        """
        raise NotImplementedError(f"{self.stage_type.value} stage must implement process()")
    
    def encode_with_context(
        self,
        tokens: List[str],
        context_embeddings: Optional[List[torch.Tensor]] = None,
    ) -> StageArtifact:
        """Encode tokens using this stage's BNN, optionally conditioned on context.
        
        Context embeddings from upstream stages can guide this stage's processing.
        """
        # Get full PMFlow components for plasticity and analysis
        embedding, latent, raw_activations = self.encoder.encode_with_components(tokens)
        
        # Optionally blend with upstream context (weighted average, attention, etc.)
        if context_embeddings:
            # Simple weighted blend for now - could use attention later
            context = torch.stack(context_embeddings).mean(dim=0)
            
            # Handle dimension mismatch by padding or projecting
            if context.shape != embedding.shape:
                # Simple approach: only blend if dimensions match
                # More sophisticated: learn a projection matrix
                self._log.debug(
                    "Context dimension mismatch (%s vs %s), skipping blend",
                    context.shape,
                    embedding.shape,
                )
            else:
                # Preserve this stage's embedding but incorporate context signal
                embedding = 0.8 * embedding + 0.2 * context
        
        confidence = self._compute_confidence(raw_activations)
        
        # Get centers snapshot (handle MultiScalePMField)
        centers_snapshot = None
        if hasattr(self.encoder, 'pm_field'):
            pm_field = self.encoder.pm_field
            if hasattr(pm_field, 'fine_field') and hasattr(pm_field.fine_field, 'centers'):
                # MultiScalePMField - snapshot fine field centers
                centers_snapshot = pm_field.fine_field.centers.detach().clone()
            elif hasattr(pm_field, 'centers'):
                # Standard PMField
                centers_snapshot = pm_field.centers.detach().clone()
        
        return StageArtifact(
            stage=self.stage_type,
            embedding=embedding,
            confidence=confidence,
            tokens=tokens,
            activations=raw_activations,
            centers_snapshot=centers_snapshot,
            metadata={
                "latent_norm": float(torch.norm(latent).item()),
                "activation_energy": float(torch.norm(raw_activations).item()),
            },
        )
    
    def _compute_confidence(self, activations: torch.Tensor) -> float:
        """Compute confidence score from activation patterns.
        
        Higher activation energy = more confident representation.
        This is stage-specific and can be tuned per task.
        """
        energy = float(torch.norm(activations, p=2).item())
        # Normalize to [0, 1] - these thresholds are task-specific
        return min(1.0, energy / 10.0)
    
    def apply_plasticity(
        self,
        tokens: List[str],
        reward_signal: float,
    ) -> Dict[str, Any]:
        """Update this stage's BNN based on reward signal.
        
        Args:
            tokens: Input that produced the reward
            reward_signal: Quality metric (retrieval score, inference accuracy, etc.)
        
        Returns:
            Plasticity report with update statistics
        """
        if not self.config.plasticity_enabled:
            return {"plasticity_enabled": 0.0}
        
        try:
            # Re-encode to get gradients - need to enable gradients on centers
            self.encoder.pm_field.train()
            
            # Enable gradients for PMFlow centers (handle MultiScalePMField)
            pm_field = self.encoder.pm_field
            if hasattr(pm_field, 'fine_field'):
                # MultiScalePMField - update both fields
                if hasattr(pm_field.fine_field, 'centers'):
                    pm_field.fine_field.centers.requires_grad_(True)
                if hasattr(pm_field.coarse_field, 'centers'):
                    pm_field.coarse_field.centers.requires_grad_(True)
            elif hasattr(pm_field, 'centers'):
                # Standard PMField
                pm_field.centers.requires_grad_(True)
            
            embedding, latent, raw = self.encoder.encode_with_components(tokens)
            
            # Simple reward-based loss: push representation toward higher activation
            # when reward is low (need better discrimination)
            loss = -reward_signal * torch.norm(raw, p=2)
            loss.backward()
            
            # Manual gradient step on PMFlow parameters
            grad_norm = 0.0
            with torch.no_grad():
                if hasattr(pm_field, 'fine_field'):
                    # MultiScalePMField - update both fields
                    if hasattr(pm_field.fine_field, 'centers') and pm_field.fine_field.centers.grad is not None:
                        grad = pm_field.fine_field.centers.grad
                        pm_field.fine_field.centers -= self.config.plasticity_lr * grad
                        grad_norm += float(torch.norm(grad).item())
                        pm_field.fine_field.centers.grad.zero_()
                    
                    if hasattr(pm_field.coarse_field, 'centers') and pm_field.coarse_field.centers.grad is not None:
                        grad = pm_field.coarse_field.centers.grad
                        pm_field.coarse_field.centers -= self.config.plasticity_lr * grad
                        grad_norm += float(torch.norm(grad).item())
                        pm_field.coarse_field.centers.grad.zero_()
                elif hasattr(pm_field, 'centers'):
                    # Standard PMField
                    grad = pm_field.centers.grad
                    if grad is not None:
                        pm_field.centers -= self.config.plasticity_lr * grad
                        grad_norm = float(torch.norm(grad).item())
                        pm_field.centers.grad.zero_()
            
            self.encoder.pm_field.eval()
            
            return {
                "reward_signal": float(reward_signal),
                "gradient_norm": grad_norm,
                "learning_rate": self.config.plasticity_lr,
            }
        except RuntimeError as exc:
            # Gradient computation may fail if encoder doesn't support backprop
            # This is expected for current PMFlowEmbeddingEncoder implementation
            self._log.debug("Plasticity update failed (gradient issue): %s", exc)
            return {
                "reward_signal": float(reward_signal),
                "gradient_norm": 0.0,
                "learning_rate": self.config.plasticity_lr,
                "error": "gradient_unavailable",
            }


class IntakeStage(CognitiveStage):
    """Specialized for noise normalization and typo pattern learning."""
    
    def process(
        self,
        input_data: Any,
        upstream_artifacts: Optional[List[StageArtifact]] = None,
    ) -> StageArtifact:
        """Process raw utterance, learn common variations."""
        from .intake import NoiseNormalizer
        
        utterance = input_data if isinstance(input_data, Utterance) else Utterance(text=str(input_data), language="unknown")
        normalizer = NoiseNormalizer()
        normalised = normalizer.normalise(utterance)
        candidates = normalizer.generate_candidates(normalised)[:8]
        
        # Encode the normalized text to capture intake patterns
        tokens = normalised.split()
        artifact = self.encode_with_context(tokens)
        artifact.metadata["normalised"] = normalised
        artifact.metadata["candidates"] = candidates
        
        return artifact


class SemanticStage(CognitiveStage):
    """Specialized for semantic embedding and concept retrieval with taxonomy."""
    
    def __init__(self, config: StageConfig):
        super().__init__(config)
        
        # Initialize concept taxonomy for query expansion
        from .concept_taxonomy import ConceptTaxonomy, CompositionalQuery
        self.taxonomy = ConceptTaxonomy()
        self.compositor = CompositionalQuery(self.taxonomy)
    
    def process(
        self,
        input_data: Any,
        upstream_artifacts: Optional[List[StageArtifact]] = None,
    ) -> StageArtifact:
        """Process parsed tokens into semantic embeddings with concept expansion."""
        from .parser import parse as parse_sentence
        from . import symbolic
        
        # Get normalized text from intake stage if available
        if upstream_artifacts and upstream_artifacts[0].metadata.get("normalised"):
            text = upstream_artifacts[0].metadata["normalised"]
        else:
            text = input_data if isinstance(input_data, str) else str(input_data)
        
        # Parse and extract symbolic frame
        parsed = parse_sentence(text)
        tokens = [token.text for token in parsed.tokens]
        
        # Extract and expand concepts using taxonomy
        concepts = self.taxonomy.extract_concepts(text)
        expanded_concepts = self.taxonomy.expand_query(list(concepts))
        
        # Augment tokens with expanded concepts for richer embedding
        augmented_tokens = tokens.copy()
        for concept in expanded_concepts:
            # Add concept if not already in tokens
            concept_words = concept.replace("_", " ")
            if concept_words not in text.lower():
                augmented_tokens.append(concept.replace("_", " "))
        
        # Use intake embedding as context if available
        context_embs = [art.embedding for art in (upstream_artifacts or []) if art.stage == StageType.INTAKE]
        
        artifact = self.encode_with_context(augmented_tokens, context_embeddings=context_embs if context_embs else None)
        artifact.metadata["parsed"] = parsed
        artifact.metadata["num_tokens"] = len(tokens)
        artifact.metadata["concepts"] = list(concepts)
        artifact.metadata["expanded_concepts"] = list(expanded_concepts)
        artifact.metadata["augmented_tokens"] = augmented_tokens
        
        return artifact


class StageCoordinator:
    """Coordinates multi-stage PMFlow BNN pipeline.
    
    Manages data flow between specialized stages, each with dedicated BNN and database.
    """
    
    def __init__(
        self,
        stage_configs: Optional[List[StageConfig]] = None,
        base_state_dir: Optional[Path] = None,
    ) -> None:
        self._log = logging.getLogger(__name__)
        self.base_state_dir = base_state_dir or Path("runs/stage_states")
        self.base_state_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize stages
        self.stages: Dict[StageType, CognitiveStage] = {}
        
        if stage_configs is None:
            stage_configs = self._default_configs()
        
        for config in stage_configs:
            self._add_stage(config)
        
        self._log.info("StageCoordinator initialized with %d stages", len(self.stages))
    
    def _default_configs(self) -> List[StageConfig]:
        """Create default 2-stage configuration (intake + semantic)."""
        return [
            StageConfig(
                stage_type=StageType.INTAKE,
                db_namespace="intake_memory",
                state_path=self.base_state_dir / "intake_pmflow.pt",
                encoder_config={"latent_dim": 16, "dimension": 32},
            ),
            StageConfig(
                stage_type=StageType.SEMANTIC,
                db_namespace="semantic_memory",
                state_path=self.base_state_dir / "semantic_pmflow.pt",
                encoder_config={"latent_dim": 32, "dimension": 64},
            ),
        ]
    
    def _add_stage(self, config: StageConfig) -> None:
        """Add a new processing stage."""
        if config.stage_type == StageType.INTAKE:
            stage = IntakeStage(config)
        elif config.stage_type == StageType.SEMANTIC:
            stage = SemanticStage(config)
        else:
            # Generic stage for future extensions
            stage = CognitiveStage(config)
        
        self.stages[config.stage_type] = stage
        self._log.info("Added %s stage", config.stage_type.value)
    
    def process(self, utterance: Utterance) -> Dict[StageType, StageArtifact]:
        """Process utterance through all stages sequentially.
        
        Args:
            utterance: Input to process
        
        Returns:
            Dict mapping stage type to its artifact
        """
        results: Dict[StageType, StageArtifact] = {}
        upstream_artifacts: List[StageArtifact] = []
        
        # Process through stages in order
        stage_order = [StageType.INTAKE, StageType.SEMANTIC, StageType.REASONING, StageType.RESPONSE]
        
        for stage_type in stage_order:
            if stage_type not in self.stages:
                continue
            
            stage = self.stages[stage_type]
            
            try:
                artifact = stage.process(utterance, upstream_artifacts=upstream_artifacts)
                results[stage_type] = artifact
                upstream_artifacts.append(artifact)
                
                self._log.debug(
                    "%s stage: confidence=%.3f, energy=%.3f",
                    stage_type.value,
                    artifact.confidence,
                    artifact.metadata.get("activation_energy", 0.0),
                )
            except Exception as exc:
                self._log.error("Stage %s failed: %s", stage_type.value, exc)
                # Continue processing remaining stages
        
        return results
    
    def get_stage(self, stage_type: StageType) -> Optional[CognitiveStage]:
        """Get a specific processing stage."""
        return self.stages.get(stage_type)
    
    def apply_plasticity(
        self,
        stage_type: StageType,
        tokens: List[str],
        reward_signal: float,
    ) -> Optional[Dict[str, float]]:
        """Apply plasticity update to a specific stage."""
        stage = self.stages.get(stage_type)
        if stage is None:
            return None
        return stage.apply_plasticity(tokens, reward_signal)
    
    def save_all_states(self) -> None:
        """Persist all stage PMFlow states."""
        for stage_type, stage in self.stages.items():
            if stage.config.state_path:
                stage.encoder.save_state()
                self._log.info("Saved %s stage state", stage_type.value)
