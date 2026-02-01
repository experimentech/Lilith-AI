"""
World Model Component for Lilith v2.

Addresses "aphantasia" by providing grounded spatial, temporal, and causal reasoning.
Keeps track of "entity states" (Object Permanence) across the conversation.

Ported and adapted from lilith/world_model_stage.py
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)

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
    description: str              # Natural language description
    entities: List[Entity]
    spatial_relations: List[SpatialRelation] = field(default_factory=list)
    temporal_relations: List[TemporalRelation] = field(default_factory=list)
    causal_relations: List[CausalRelation] = field(default_factory=list)

class WorldModel:
    """
    Maintains a grounded representation of the world discussed in conversation.
    Tracks entities, their states, and relationships over time.
    """

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

    def __init__(self):
        # Entity tracking across conversation (Short-term / Working Memory)
        self.active_entities: Dict[str, Entity] = {}
        
        # Relation tracking
        self.active_spatial_relations: List[SpatialRelation] = []
        self.active_temporal_relations: List[TemporalRelation] = []
        self.active_causal_relations: List[CausalRelation] = []
        
        logger.info("WorldMode initialized.")

    def process_utterance(self, text: str) -> WorldSituation:
        """
        Process utterance to extract world model information and update state.
        """
        # Extract entities
        entities = self._extract_entities(text)
        
        # Extract relations (heuristic)
        spatial_relations = self._extract_spatial_relations(text, entities)
        temporal_relations = self._extract_temporal_relations(text, entities)
        causal_relations = self._extract_causal_relations(text, entities)
        
        # Create situation artifact
        situation = WorldSituation(
            description=text,
            entities=entities,
            spatial_relations=spatial_relations,
            temporal_relations=temporal_relations,
            causal_relations=causal_relations,
        )
        
        # Update internal tracking (Object Permanence)
        self._update_tracking(situation)
        
        return situation

    def get_context(self) -> Dict[str, Any]:
        """Return the current state of the world to be used in reasoning/generation."""
        return {
            "entities": [
                {"name": e.name, "state": e.state, "type": e.entity_type} 
                for e in self.active_entities.values()
            ],
            "spatial": [
                f"{r.entity_a} {r.relation_type} {r.entity_b}" 
                for r in self.active_spatial_relations
            ],
            "temporal": [
                f"{r.event_a} {r.relation_type} {r.event_b}" 
                for r in self.active_temporal_relations
            ],
            "causal": [
                f"{r.cause} -> {r.effect}" 
                for r in self.active_causal_relations
            ]
        }

    # ──────────────────────────────────────────────────────────────
    # Extraction Logic (Heuristic / Rule-based)
    # ──────────────────────────────────────────────────────────────
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities via simple NP heuristics (Determiner + Noun).
        """
        entities = []
        # Better tokenization: remove punctuation for heuristic matching
        import re
        # This splits by space but keeps words clean. 
        # Alternatively use nltk if we want to be fancy, but regex is light.
        # We replace punctuation with spaces to avoid 'table.' being one token
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean_text.split()
        
        # Basic list of determiners to identify NP starts
        determiners = {"the", "a", "an", "this", "that", "these", "those", "my", "your"}
        
        i = 0
        while i < len(words):
            if words[i] in determiners and i + 1 < len(words):
                # Found determiner + noun pattern candidate
                noun_phrase = []
                j = i + 1
                
                # Consume adjectives/nouns until a stopword or verb
                # This is a very rough heuristic compared to a real parser
                stoppers = {"is", "are", "was", "were", "in", "on", "at", "to", "for", "with", "from"}
                
                while j < len(words) and words[j] not in stoppers:
                    noun_phrase.append(words[j])
                    j += 1
                
                if noun_phrase:
                    entity_name = " ".join(noun_phrase)
                    # Deduplicate if we already saw this name in this utterance
                    if not any(e.name == entity_name for e in entities):
                        entity = Entity(
                            entity_id=f"entity_{entity_name.replace(' ', '_')}",
                            name=entity_name,
                            entity_type="object",  # Default
                        )
                        entities.append(entity)
                
                i = j
            else:
                i += 1
        
        return entities
    
    def _extract_spatial_relations(self, text: str, entities: List[Entity]) -> List[SpatialRelation]:
        relations = []
        text_lower = text.lower()
        
        for marker in self.SPATIAL_MARKERS:
            if marker in text_lower:
                parts = text_lower.split(f" {marker} ") # spaces to avoid partial word matches
                if len(parts) >= 2:
                    # Very rough: Subject is in part 0, Object is in part 1
                    # Try to map to extracted entities
                    subj = self._find_entity_in_text(parts[0], entities) or "something"
                    obj = self._find_entity_in_text(parts[1], entities) or "something"
                    
                    if subj != "something" or obj != "something":
                        relations.append(SpatialRelation(marker, subj, obj))
        return relations

    def _extract_temporal_relations(self, text: str, entities: List[Entity]) -> List[TemporalRelation]:
        relations = []
        text_lower = text.lower()
        
        for marker in self.TEMPORAL_MARKERS:
            if marker in text_lower:
                parts = text_lower.split(f" {marker} ")
                if len(parts) >= 2:
                    val_a = parts[0].strip()[-30:] # Last 30 chars of pre-text
                    val_b = parts[1].strip()[:30]  # First 30 chars of post-text
                    relations.append(TemporalRelation(marker, val_a, val_b))
        return relations
        
    def _extract_causal_relations(self, text: str, entities: List[Entity]) -> List[CausalRelation]:
        relations = []
        text_lower = text.lower()
        
        for marker in self.CAUSAL_MARKERS:
            if marker in text_lower:
                parts = text_lower.split(f" {marker} ")
                if len(parts) >= 2:
                    cause = parts[0].strip()[-40:]
                    effect = parts[1].strip()[:40]
                    relations.append(CausalRelation(marker, cause, effect))
        return relations

    def _find_entity_in_text(self, segment: str, entities: List[Entity]) -> Optional[str]:
        """Finds if any of the extracted entity names appear in the text segment."""
        for e in entities:
            if e.name in segment:
                return e.name
        return None

    def _update_tracking(self, situation: WorldSituation):
        """Update active persistence state."""
        # 1. Update/Add Entities
        for e in situation.entities:
            # We use name as key for simplicity in this version, though ID is better
            # If entity exists, update it (maybe state changed)
            # Here we just overwrite or add
            self.active_entities[e.name] = e
            
        # 2. Update Relations (Tracking "State of the World")
        # For simplicity, we just extend the list, but we should cap it
        self.active_spatial_relations.extend(situation.spatial_relations)
        self.active_temporal_relations.extend(situation.temporal_relations)
        self.active_causal_relations.extend(situation.causal_relations)
        
        # Memory Limit (Working Memory)
        limit = 10
        if len(self.active_spatial_relations) > limit:
            self.active_spatial_relations = self.active_spatial_relations[-limit:]
        if len(self.active_temporal_relations) > limit:
            self.active_temporal_relations = self.active_temporal_relations[-limit:]
        if len(self.active_causal_relations) > limit:
            self.active_causal_relations = self.active_causal_relations[-limit:]
