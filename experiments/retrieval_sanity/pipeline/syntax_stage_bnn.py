"""
Syntax Stage - BNN-based Grammatical Pattern Processing

Full cognitive stage implementation using PMFlow BNN for syntactic structures.
Learns grammatical patterns through reinforcement, just like semantic stage.

Stage flow:
  Input (tokens + POS tags) → PMFlow Encoder → Syntax Embedding
  Syntax Embedding → Retrieve similar patterns → Composition templates
  Feedback → Plasticity updates → Better grammar over time
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import json

import torch
import numpy as np

from .embedding import PMFlowEmbeddingEncoder
from .stage_coordinator import StageType, StageConfig, StageArtifact


logger = logging.getLogger(__name__)


@dataclass
class SyntacticPattern:
    """A learned grammatical pattern stored in syntax_memory."""
    
    pattern_id: str
    pos_sequence: List[str]          # ["DT", "JJ", "NN", "VBZ"]
    embedding: torch.Tensor          # PMFlow encoding of POS sequence
    template: str                     # "The {adj} {noun} {verb}"
    example: str                      # Actual instance
    success_score: float = 0.5       # Reinforcement score
    usage_count: int = 0
    intent: str = "general"          # statement, question, compound


@dataclass
class CompositionTemplate:
    """Template for grammatically combining fragments."""
    
    template_id: str
    pattern_a_type: str              # e.g., "statement"
    pattern_b_type: str              # e.g., "question"
    connector: str                   # "and", ".", "because"
    embedding: torch.Tensor          # PMFlow encoding of template structure
    example: str
    success_score: float = 0.5


class SyntaxStage:
    """
    Cognitive stage for grammatical pattern processing using BNN.
    
    This is a FULL stage implementation, parallel to INTAKE/SEMANTIC:
    - Dedicated PMFlow encoder for POS sequences
    - Separate syntax_memory database namespace  
    - Independent plasticity learning
    - Reinforcement-based grammar improvement
    """
    
    def __init__(
        self,
        config: Optional[StageConfig] = None,
        encoder: Optional[PMFlowEmbeddingEncoder] = None,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize syntax stage.
        
        Args:
            config: Stage configuration (uses defaults if None)
            encoder: PMFlow encoder for POS sequences (creates new if None)
            storage_path: Where to store learned patterns
        """
        # Default config
        if config is None:
            config = StageConfig(
                stage_type=StageType.REASONING,  # Use REASONING slot for syntax
                encoder_config={
                    "latent_dim": 32,  # Smaller than semantic (simpler patterns)
                    "num_centers": 64,  # Enough for common POS sequences
                },
                db_namespace="syntax_memory",
                plasticity_enabled=True,
                plasticity_lr=1e-3  # Faster learning for discrete patterns
            )
        
        self.config = config
        self.stage_type = config.stage_type
        
        # Initialize PMFlow encoder for POS sequences
        if encoder is None:
            logger.info("Initializing PMFlow encoder for syntax stage...")
            self.encoder = PMFlowEmbeddingEncoder(
                latent_dim=config.encoder_config.get("latent_dim", 32),
                seed=config.encoder_config.get("seed", 42),
            )
        else:
            self.encoder = encoder
            
        # Storage
        self.storage_path = storage_path or Path("syntax_patterns.json")
        
        # Pattern stores
        self.patterns: Dict[str, SyntacticPattern] = {}
        self.templates: Dict[str, CompositionTemplate] = {}
        
        # Load or bootstrap
        if self.storage_path.exists():
            self._load_patterns()
        else:
            self._bootstrap_patterns()
            
        logger.info(
            f"Syntax stage initialized: {len(self.patterns)} patterns, "
            f"{len(self.templates)} templates"
        )
    
    def process(
        self,
        tokens: List[str],
        pos_tags: Optional[List[str]] = None
    ) -> StageArtifact:
        """
        Process tokens through syntax stage using BNN.
        
        Args:
            tokens: Input tokens
            pos_tags: Part-of-speech tags (extracts if None)
            
        Returns:
            StageArtifact with syntax embedding and matched patterns
        """
        # Extract POS if not provided
        if pos_tags is None:
            pos_tags = self._extract_pos_tags(tokens)
        
        # Encode POS sequence using PMFlow BNN
        pos_string = " ".join(pos_tags)
        embedding, latent, activations = self.encoder.encode_with_components(pos_string)
        
        # Calculate confidence from activation energy
        confidence = self._compute_confidence(activations)
        
        # Retrieve similar grammatical patterns
        matched_patterns = self._retrieve_patterns(embedding, topk=5)
        
        # Determine syntactic intent
        intent = self._classify_intent(pos_tags, tokens)
        
        return StageArtifact(
            stage=self.stage_type,
            embedding=embedding,
            confidence=confidence,
            tokens=tokens,
            activations=activations,
            metadata={
                "pos_sequence": pos_tags,
                "pos_string": pos_string,
                "intent": intent,
                "matched_patterns": [
                    {"id": p.pattern_id, "score": score, "template": p.template}
                    for p, score in matched_patterns
                ],
                "activation_energy": float(torch.norm(activations).item()),
            }
        )
    
    def _extract_pos_tags(self, tokens: List[str]) -> List[str]:
        """
        Extract POS tags from tokens.
        
        Simplified version - in practice would use spaCy or similar.
        For now, uses heuristics.
        """
        pos_tags = []
        for token in tokens:
            token_lower = token.lower()
            
            # Question words
            if token_lower in ['who', 'what', 'where', 'when', 'why', 'how']:
                pos_tags.append('WRB')
            # Pronouns
            elif token_lower in ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'that', 'this']:
                pos_tags.append('PRON')
            # Determiners
            elif token_lower in ['the', 'a', 'an']:
                pos_tags.append('DT')
            # Coordinators
            elif token_lower in ['and', 'or', 'but']:
                pos_tags.append('CC')
            # Auxiliary verbs
            elif token_lower in ['is', 'are', 'was', 'were', 'does', 'do', 'did', 'can', 'could']:
                pos_tags.append('VBZ')
            # Common adjectives
            elif token_lower in ['good', 'bad', 'interesting', 'cool', 'nice', 'great']:
                pos_tags.append('ADJ')
            # Adverbs
            elif token.endswith('ly'):
                pos_tags.append('ADV')
            # Plural nouns
            elif token.endswith('s') and len(token) > 2:
                pos_tags.append('NNS')
            # Default to noun
            else:
                pos_tags.append('NN')
                
        return pos_tags
    
    def _compute_confidence(self, activations: torch.Tensor) -> float:
        """Compute confidence from PMFlow activation strength."""
        energy = float(torch.norm(activations, p=2).item())
        # Normalize to [0, 1]
        return min(1.0, energy / 8.0)
    
    def _retrieve_patterns(
        self,
        query_embedding: torch.Tensor,
        topk: int = 5
    ) -> List[Tuple[SyntacticPattern, float]]:
        """
        Retrieve similar grammatical patterns using BNN similarity.
        
        This is the key: grammar patterns are retrieved via learned embeddings!
        """
        if not self.patterns:
            return []
        
        query_np = query_embedding.detach().cpu().numpy().flatten()
        query_norm = query_np / (np.linalg.norm(query_np) + 1e-8)
        
        scored_patterns = []
        for pattern in self.patterns.values():
            pattern_np = pattern.embedding.detach().cpu().numpy().flatten()
            pattern_norm = pattern_np / (np.linalg.norm(pattern_np) + 1e-8)
            
            # Cosine similarity
            similarity = float(np.dot(query_norm, pattern_norm))
            
            # Weight by success score
            weighted_score = similarity * pattern.success_score
            
            scored_patterns.append((pattern, weighted_score))
        
        # Sort by score
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return scored_patterns[:topk]
    
    def _classify_intent(self, pos_tags: List[str], tokens: List[str]) -> str:
        """Classify syntactic intent from POS sequence."""
        # Question
        if pos_tags[0] == 'WRB' or any(t.strip().endswith('?') for t in tokens):
            return "question"
        # Exclamation  
        elif any(t.strip().endswith('!') for t in tokens):
            return "exclamation"
        # Coordination (compound)
        elif 'CC' in pos_tags:
            return "compound"
        # Default statement
        else:
            return "statement"
    
    def learn_pattern(
        self,
        tokens: List[str],
        pos_tags: List[str],
        success_feedback: float
    ) -> str:
        """
        Learn a new grammatical pattern from observation.
        
        This is reinforcement learning for grammar!
        
        Args:
            tokens: Example tokens
            pos_tags: POS sequence
            success_feedback: How well this pattern worked (-1.0 to 1.0)
            
        Returns:
            pattern_id of learned pattern
        """
        # Encode POS sequence
        pos_string = " ".join(pos_tags)
        embedding = self.encoder.encode(pos_string)
        
        # Create pattern
        pattern_id = f"syntax_learned_{len(self.patterns)}"
        intent = self._classify_intent(pos_tags, tokens)
        template = self._generalize_template(tokens, pos_tags)
        
        pattern = SyntacticPattern(
            pattern_id=pattern_id,
            pos_sequence=pos_tags,
            embedding=embedding,
            template=template,
            example=" ".join(tokens),
            success_score=0.5 + (success_feedback * 0.3),  # Start near success
            usage_count=0,
            intent=intent
        )
        
        self.patterns[pattern_id] = pattern
        self._save_patterns()
        
        logger.info(f"Learned syntax pattern: {template} (intent: {intent})")
        
        return pattern_id
    
    def update_pattern_success(
        self,
        pattern_id: str,
        feedback: float,
        learning_rate: float = 0.1
    ):
        """
        Update pattern success score via reinforcement.
        
        This is how grammar improves over time!
        """
        if pattern_id not in self.patterns:
            return
            
        pattern = self.patterns[pattern_id]
        pattern.success_score += feedback * learning_rate
        pattern.success_score = np.clip(pattern.success_score, 0.0, 1.0)
        pattern.usage_count += 1
        
        # Apply plasticity to BNN if enabled
        if self.config.plasticity_enabled:
            self._apply_plasticity(pattern, feedback)
    
    def _apply_plasticity(self, pattern: SyntacticPattern, feedback: float):
        """Apply BNN plasticity update for this pattern."""
        # Re-encode with gradients enabled
        pos_string = " ".join(pattern.pos_sequence)
        
        # Simple plasticity: update centers based on feedback
        # Positive feedback → reinforce, negative → weaken
        # This would be more sophisticated in practice
        pass  # TODO: Implement full plasticity like semantic stage
    
    def _generalize_template(self, tokens: List[str], pos_tags: List[str]) -> str:
        """Create generalized template from example."""
        parts = []
        for token, pos in zip(tokens, pos_tags):
            if pos in ['DT', 'PRON', 'CC', 'VBZ']:  # Function words
                parts.append(token.lower())
            else:
                parts.append(f"{{{pos.lower()}}}")
        return " ".join(parts)
    
    def _bootstrap_patterns(self):
        """Bootstrap with basic syntactic patterns using BNN encoding."""
        basic_patterns = [
            (["PRON", "VBZ"], "it works", "statement"),
            (["PRON", "VBZ", "ADJ"], "that's interesting", "statement"),
            (["WRB", "VBZ", "PRON", "VB"], "how does that work", "question"),
            (["DT", "ADJ", "NN"], "the good point", "phrase"),
        ]
        
        for pos_seq, example, intent in basic_patterns:
            pos_string = " ".join(pos_seq)
            embedding = self.encoder.encode(pos_string)
            
            pattern_id = f"syntax_seed_{len(self.patterns)}"
            template = self._generalize_template(example.split(), pos_seq)
            
            pattern = SyntacticPattern(
                pattern_id=pattern_id,
                pos_sequence=pos_seq,
                embedding=embedding,
                template=template,
                example=example,
                intent=intent
            )
            
            self.patterns[pattern_id] = pattern
    
    def _save_patterns(self):
        """Save learned patterns to storage."""
        # Convert to serializable format
        data = {
            "patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "pos_sequence": p.pos_sequence,
                    "embedding": p.embedding.tolist(),
                    "template": p.template,
                    "example": p.example,
                    "success_score": p.success_score,
                    "usage_count": p.usage_count,
                    "intent": p.intent,
                }
                for p in self.patterns.values()
            ]
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_patterns(self):
        """Load patterns from storage."""
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
        
        for p_data in data.get("patterns", []):
            pattern = SyntacticPattern(
                pattern_id=p_data["pattern_id"],
                pos_sequence=p_data["pos_sequence"],
                embedding=torch.tensor(p_data["embedding"]),
                template=p_data["template"],
                example=p_data["example"],
                success_score=p_data["success_score"],
                usage_count=p_data["usage_count"],
                intent=p_data["intent"],
            )
            self.patterns[pattern.pattern_id] = pattern
