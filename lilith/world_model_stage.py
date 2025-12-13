"""
World Model Stage - Grounded Spatial/Temporal/Causal Reasoning

This stage addresses Lilith's "aphantasia" - the missing world grounding layer.
While semantic stage handles abstract concepts and their relationships,
world model stage handles:

- Spatial relations: "in the box", "next to the door", "above the table"
- Temporal sequences: "before breakfast", "after the meeting", "during the storm"
- Causal chains: "because of the rain", "leads to flooding", "prevents growth"
- Entity states: "the door is open", "water is hot", "the cat is sleeping"
- Physical properties: size, color, shape, location, orientation

This enables Lilith to reason about concrete situations and maintain a
grounded understanding of what's being discussed, not just abstract symbols.

Architecture:
    Input → Extract Entities/Relations → BNN Encoding → Latent Space (64-dim)
    → Retrieve Similar Situations → Reason About State/Causality → Output

The larger latent_dim (64 vs syntax's 32) allows for richer grounded representations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any
from pathlib import Path

import torch
import torch.nn.functional as F

from .cognitive_stage_base import CognitiveStageBase, CognitivePattern, RetrievalResult
from .embedding import PMFlowEmbeddingEncoder

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# World Model Data Structures
# ──────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    """An entity in the world model."""
    entity_id: str
    name: str
    entity_type: str              # "object", "person", "place", "event"
    properties: Dict[str, Any] = field(default_factory=dict)
    state: str = "unknown"        # Current state


@dataclass
class SpatialRelation:
    """A spatial relationship between entities."""
    relation_type: str            # "in", "on", "above", "below", "near", "inside", "outside"
    entity_a: str                 # Reference entity
    entity_b: str                 # Related entity
    confidence: float = 1.0


@dataclass
class TemporalRelation:
    """A temporal relationship between events."""
    relation_type: str            # "before", "after", "during", "while", "since", "until"
    event_a: str                  # Reference event
    event_b: str                  # Related event
    confidence: float = 1.0


@dataclass
class CausalRelation:
    """A causal relationship between events/states."""
    relation_type: str            # "causes", "prevents", "enables", "leads_to"
    cause: str                    # Cause entity/event
    effect: str                   # Effect entity/event
    confidence: float = 1.0


@dataclass
class WorldSituation:
    """A complete situation in the world model."""
    situation_id: str
    description: str              # Natural language description
    entities: List[Entity]
    spatial_relations: List[SpatialRelation] = field(default_factory=list)
    temporal_relations: List[TemporalRelation] = field(default_factory=list)
    causal_relations: List[CausalRelation] = field(default_factory=list)
    
    def to_text_representation(self) -> str:
        """Convert situation to text for encoding."""
        parts = [self.description]
        
        # Add entity information
        for entity in self.entities:
            entity_str = f"{entity.name} ({entity.entity_type})"
            if entity.state != "unknown":
                entity_str += f" is {entity.state}"
            parts.append(entity_str)
        
        # Add spatial relations
        for rel in self.spatial_relations:
            parts.append(f"{rel.entity_a} {rel.relation_type} {rel.entity_b}")
        
        # Add temporal relations
        for rel in self.temporal_relations:
            parts.append(f"{rel.event_a} {rel.relation_type} {rel.event_b}")
        
        # Add causal relations
        for rel in self.causal_relations:
            parts.append(f"{rel.cause} {rel.relation_type} {rel.effect}")
        
        return " | ".join(parts)


# ──────────────────────────────────────────────────────────────────
# World Model Stage Implementation
# ──────────────────────────────────────────────────────────────────

class WorldModelStage(CognitiveStageBase):
    """
    Cognitive stage for grounded world representation.
    
    Uses BNN with 64-dim latent space to represent spatial, temporal,
    and causal structures. This is what gives Lilith "world understanding"
    beyond pure linguistic patterns.
    
    Key Features:
    - Entity tracking (objects, people, places, events)
    - Spatial reasoning (locations, containment, proximity)
    - Temporal reasoning (sequences, durations, simultaneity)
    - Causal reasoning (cause-effect chains, prevention, enablement)
    - State tracking (property changes over time)
    """
    
    # Relation markers for pattern extraction
    SPATIAL_MARKERS = {
        "in", "on", "at", "inside", "outside", "above", "below", "under",
        "over", "near", "next to", "beside", "between", "behind", "in front of",
        "around", "through", "across", "along", "into", "onto", "out of"
    }
    
    TEMPORAL_MARKERS = {
        "before", "after", "during", "while", "when", "then", "since", "until",
        "as", "whenever", "once", "now", "later", "earlier", "previously",
        "subsequently", "meanwhile", "simultaneously"
    }
    
    CAUSAL_MARKERS = {
        "because", "so", "therefore", "thus", "hence", "causes", "leads to",
        "results in", "due to", "owing to", "thanks to", "as a result",
        "consequently", "accordingly", "prevents", "enables", "allows"
    }
    
    STATE_VERBS = {
        "is", "are", "was", "were", "be", "being", "been", "become", "became",
        "remains", "stays", "keeps", "continues", "gets", "turns", "grows"
    }
    
    def __init__(
        self,
        encoder: Optional[PMFlowEmbeddingEncoder] = None,
        storage_path: Optional[Path] = None,
        use_sqlite: bool = True,
        plasticity_enabled: bool = True,
        enable_tracking: bool = True,
    ):
        """
        Initialize world model stage.
        
        Args:
            encoder: PMFlow encoder (creates new if None)
            storage_path: Pattern storage location
            use_sqlite: Use SQLite vs JSON
            plasticity_enabled: Enable BNN plasticity
            enable_tracking: Track entities across conversations
        """
        # Create encoder with appropriate latent_dim for world modeling
        if encoder is None:
            encoder = PMFlowEmbeddingEncoder(
                latent_dim=64,  # Larger than syntax (32) for complex grounded reps
                seed=42,
            )
        
        # Initialize base class
        super().__init__(
            encoder=encoder,
            stage_name="world_model",
            latent_dim=64,
            storage_path=storage_path,
            use_sqlite=use_sqlite,
            plasticity_enabled=plasticity_enabled,
            plasticity_lr=5e-4,  # Slightly slower learning for world knowledge
        )
        
        self.enable_tracking = enable_tracking
        
        # Entity tracking across conversation
        self.active_entities: Dict[str, Entity] = {}
        self.entity_history: List[Entity] = []
        
        # Relation tracking
        self.active_spatial_relations: List[SpatialRelation] = []
        self.active_temporal_relations: List[TemporalRelation] = []
        self.active_causal_relations: List[CausalRelation] = []
        
        logger.info(
            f"World model stage initialized: latent_dim=64, "
            f"entity_tracking={enable_tracking}"
        )
    
    # ──────────────────────────────────────────────────────────────
    # Abstract Method Implementations
    # ──────────────────────────────────────────────────────────────
    
    def _encode_content(self, content: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode world situation to embedding and latent.
        
        Args:
            content: Text representation of situation
            
        Returns:
            (full_embedding, latent_representation)
        """
        # Use PMFlow encoder
        embedding = self.encoder.encode(content)
        
        # Extract latent portion (first latent_dim dimensions)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        latent = embedding[:, :self.latent_dim]
        
        return embedding, latent
    
    def _bootstrap_patterns(self) -> List[CognitivePattern]:
        """
        Create seed world model patterns.
        
        These are basic spatial/temporal/causal patterns that bootstrap
        the world understanding system.
        """
        seed_situations = [
            # Spatial patterns - containment
            ("The book is on the table", "spatial", {
                "relation_type": "spatial", "pattern": "X on Y", "type": "support"
            }),
            ("The cat is in the box", "spatial", {
                "relation_type": "spatial", "pattern": "X in Y", "type": "containment"
            }),
            ("The keys are under the pillow", "spatial", {
                "relation_type": "spatial", "pattern": "X under Y", "type": "below"
            }),
            ("The lamp is above the desk", "spatial", {
                "relation_type": "spatial", "pattern": "X above Y", "type": "above"
            }),
            ("The car is near the house", "spatial", {
                "relation_type": "spatial", "pattern": "X near Y", "type": "proximity"
            }),
            
            # Spatial patterns - locations
            ("The phone is in the kitchen", "spatial", {
                "relation_type": "spatial", "pattern": "X in location", "type": "location"
            }),
            ("The meeting is at the office", "spatial", {
                "relation_type": "spatial", "pattern": "event at location", "type": "location"
            }),
            
            # Temporal patterns - sequences
            ("Breakfast before work", "temporal", {
                "relation_type": "temporal", "pattern": "X before Y", "type": "sequence"
            }),
            ("Call after lunch", "temporal", {
                "relation_type": "temporal", "pattern": "X after Y", "type": "sequence"
            }),
            ("Sleep during the night", "temporal", {
                "relation_type": "temporal", "pattern": "X during Y", "type": "duration"
            }),
            ("Exercise while listening to music", "temporal", {
                "relation_type": "temporal", "pattern": "X while Y", "type": "simultaneous"
            }),
            
            # Temporal patterns - states over time
            ("The door was closed", "temporal", {
                "relation_type": "temporal", "pattern": "X was state", "type": "past_state"
            }),
            ("The water is hot", "temporal", {
                "relation_type": "temporal", "pattern": "X is state", "type": "current_state"
            }),
            
            # Causal patterns - cause-effect
            ("Rain causes flooding", "causal", {
                "relation_type": "causal", "pattern": "X causes Y", "type": "causation"
            }),
            ("Exercise leads to fitness", "causal", {
                "relation_type": "causal", "pattern": "X leads to Y", "type": "outcome"
            }),
            ("Cold prevents growth", "causal", {
                "relation_type": "causal", "pattern": "X prevents Y", "type": "prevention"
            }),
            ("Practice enables mastery", "causal", {
                "relation_type": "causal", "pattern": "X enables Y", "type": "enablement"
            }),
            
            # Causal patterns - conditions
            ("Because of the storm, roads are closed", "causal", {
                "relation_type": "causal", "pattern": "because X, Y", "type": "reason"
            }),
            ("Due to rain, event is cancelled", "causal", {
                "relation_type": "causal", "pattern": "due to X, Y", "type": "reason"
            }),
            
            # State change patterns
            ("The door opened", "state_change", {
                "relation_type": "state_change", "pattern": "X changed", "type": "action"
            }),
            ("Water becomes ice when frozen", "state_change", {
                "relation_type": "state_change", "pattern": "X becomes Y", "type": "transformation"
            }),
            ("The light turned on", "state_change", {
                "relation_type": "state_change", "pattern": "X turned state", "type": "transition"
            }),
            
            # Entity property patterns
            ("The big red ball", "entity_property", {
                "relation_type": "property", "pattern": "size color object", "type": "description"
            }),
            ("The old wooden door", "entity_property", {
                "relation_type": "property", "pattern": "age material object", "type": "description"
            }),
            ("A small blue bird", "entity_property", {
                "relation_type": "property", "pattern": "size color object", "type": "description"
            }),
        ]
        
        patterns = []
        for content, intent, metadata in seed_situations:
            # Encode situation
            embedding, latent = self._encode_content(content)
            
            pattern = CognitivePattern(
                pattern_id=f"world_seed_{len(patterns)}",
                content=content,
                embedding=embedding,
                latent=latent,
                success_score=0.5,
                usage_count=0,
                intent=intent,
                metadata=metadata,
            )
            patterns.append(pattern)
        
        logger.info(f"Bootstrapped {len(patterns)} world model seed patterns")
        return patterns
    
    # ──────────────────────────────────────────────────────────────
    # World Model Processing
    # ──────────────────────────────────────────────────────────────
    
    def process_utterance(self, text: str) -> WorldSituation:
        """
        Process utterance to extract world model information.
        
        Args:
            text: Input utterance
            
        Returns:
            WorldSituation with extracted entities and relations
        """
        # Extract entities
        entities = self._extract_entities(text)
        
        # Extract relations
        spatial_relations = self._extract_spatial_relations(text, entities)
        temporal_relations = self._extract_temporal_relations(text, entities)
        causal_relations = self._extract_causal_relations(text, entities)
        
        # Create situation
        situation = WorldSituation(
            situation_id=f"situation_{len(self.entity_history)}",
            description=text,
            entities=entities,
            spatial_relations=spatial_relations,
            temporal_relations=temporal_relations,
            causal_relations=causal_relations,
        )
        
        # Update tracking if enabled
        if self.enable_tracking:
            self._update_tracking(situation)
        
        return situation
    
    def retrieve_similar_situations(
        self,
        query_text: str,
        topk: int = 5,
        relation_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve similar world situations.
        
        Args:
            query_text: Query description
            topk: Number of results
            relation_filter: Optional filter by relation type
            
        Returns:
            List of similar situations
        """
        return self.retrieve_similar(
            query_content=query_text,
            topk=topk,
            min_similarity=0.3,
            intent_filter=relation_filter,
        )
    
    def learn_situation(
        self,
        situation: WorldSituation,
        success_feedback: float = 0.5,
    ) -> CognitivePattern:
        """
        Learn a new world situation pattern.
        
        Args:
            situation: WorldSituation to learn
            success_feedback: Initial success score
            
        Returns:
            Created pattern
        """
        # Convert situation to text representation
        content = situation.to_text_representation()
        
        # Determine primary intent
        if situation.spatial_relations:
            intent = "spatial"
        elif situation.temporal_relations:
            intent = "temporal"
        elif situation.causal_relations:
            intent = "causal"
        else:
            intent = "general"
        
        # Create metadata
        metadata = {
            "num_entities": len(situation.entities),
            "num_spatial": len(situation.spatial_relations),
            "num_temporal": len(situation.temporal_relations),
            "num_causal": len(situation.causal_relations),
            "original_description": situation.description,
        }
        
        # Add pattern using base class
        pattern = self.add_pattern(
            content=content,
            intent=intent,
            success_score=success_feedback,
            metadata=metadata,
        )
        
        logger.debug(f"Learned world situation: {situation.description[:50]}...")
        return pattern
    
    # ──────────────────────────────────────────────────────────────
    # Entity and Relation Extraction
    # ──────────────────────────────────────────────────────────────
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text.
        
        Simple noun-phrase based extraction. Could be enhanced with NER.
        """
        entities = []
        words = text.lower().split()
        
        # Look for noun phrases with determiners
        determiners = {"the", "a", "an", "this", "that", "these", "those", "my", "your"}
        
        i = 0
        while i < len(words):
            if words[i] in determiners and i + 1 < len(words):
                # Found determiner + noun pattern
                noun_phrase = []
                j = i + 1
                
                # Collect adjectives and nouns
                while j < len(words) and words[j] not in {",", ".", "is", "are", "was", "were", "in", "on", "at"}:
                    noun_phrase.append(words[j])
                    j += 1
                
                if noun_phrase:
                    entity_name = " ".join(noun_phrase)
                    entity = Entity(
                        entity_id=f"entity_{entity_name.replace(' ', '_')}",
                        name=entity_name,
                        entity_type="object",  # Default to object
                        properties={},
                        state="unknown",
                    )
                    entities.append(entity)
                
                i = j
            else:
                i += 1
        
        return entities
    
    def _extract_spatial_relations(self, text: str, entities: List[Entity]) -> List[SpatialRelation]:
        """Extract spatial relations from text."""
        relations = []
        text_lower = text.lower()
        
        # Look for spatial markers
        for marker in self.SPATIAL_MARKERS:
            if marker in text_lower:
                # Simple pattern: entity_a marker entity_b
                # This is very basic - could be much more sophisticated
                parts = text_lower.split(marker)
                if len(parts) >= 2 and len(entities) >= 2:
                    relation = SpatialRelation(
                        relation_type=marker,
                        entity_a=entities[0].name if entities else "unknown",
                        entity_b=entities[1].name if len(entities) > 1 else "unknown",
                        confidence=0.7,  # Medium confidence for heuristic extraction
                    )
                    relations.append(relation)
        
        return relations
    
    def _extract_temporal_relations(self, text: str, entities: List[Entity]) -> List[TemporalRelation]:
        """Extract temporal relations from text."""
        relations = []
        text_lower = text.lower()
        
        # Look for temporal markers
        for marker in self.TEMPORAL_MARKERS:
            if marker in text_lower:
                parts = text_lower.split(marker)
                if len(parts) >= 2:
                    # Extract event descriptions from before and after marker
                    event_a = parts[0].strip()[-50:] if parts[0] else "unknown"
                    event_b = parts[1].strip()[:50] if parts[1] else "unknown"
                    
                    relation = TemporalRelation(
                        relation_type=marker,
                        event_a=event_a,
                        event_b=event_b,
                        confidence=0.7,
                    )
                    relations.append(relation)
        
        return relations
    
    def _extract_causal_relations(self, text: str, entities: List[Entity]) -> List[CausalRelation]:
        """Extract causal relations from text."""
        relations = []
        text_lower = text.lower()
        
        # Look for causal markers
        for marker in self.CAUSAL_MARKERS:
            if marker in text_lower:
                parts = text_lower.split(marker)
                if len(parts) >= 2:
                    cause = parts[0].strip()[-50:] if parts[0] else "unknown"
                    effect = parts[1].strip()[:50] if parts[1] else "unknown"
                    
                    relation = CausalRelation(
                        relation_type=marker,
                        cause=cause,
                        effect=effect,
                        confidence=0.7,
                    )
                    relations.append(relation)
        
        return relations
    
    def _update_tracking(self, situation: WorldSituation):
        """Update entity and relation tracking."""
        # Add entities to active set
        for entity in situation.entities:
            self.active_entities[entity.entity_id] = entity
            self.entity_history.append(entity)
        
        # Keep only recent entities (working memory limit)
        max_active = 20
        if len(self.active_entities) > max_active:
            # Remove oldest entities
            oldest_ids = list(self.active_entities.keys())[:-max_active]
            for entity_id in oldest_ids:
                del self.active_entities[entity_id]
        
        # Track relations
        self.active_spatial_relations.extend(situation.spatial_relations)
        self.active_temporal_relations.extend(situation.temporal_relations)
        self.active_causal_relations.extend(situation.causal_relations)
        
        # Keep only recent relations
        max_relations = 50
        self.active_spatial_relations = self.active_spatial_relations[-max_relations:]
        self.active_temporal_relations = self.active_temporal_relations[-max_relations:]
        self.active_causal_relations = self.active_causal_relations[-max_relations:]
    
    def clear_tracking(self):
        """Clear all tracked entities and relations."""
        self.active_entities.clear()
        self.active_spatial_relations.clear()
        self.active_temporal_relations.clear()
        self.active_causal_relations.clear()
        logger.debug("Cleared world model tracking")
    
    def get_active_context(self) -> Dict[str, Any]:
        """Get current active world context."""
        return {
            "num_active_entities": len(self.active_entities),
            "num_spatial_relations": len(self.active_spatial_relations),
            "num_temporal_relations": len(self.active_temporal_relations),
            "num_causal_relations": len(self.active_causal_relations),
            "entities": [
                {
                    "name": e.name,
                    "type": e.entity_type,
                    "state": e.state,
                }
                for e in self.active_entities.values()
            ],
        }
