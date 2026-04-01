"""
Linguistic Relation Extractor

Extracts semantic relationships from natural language statements like:
- "hot is the opposite of cold" → antonym(hot, cold)
- "a dog is an animal" → hypernym(dog, animal)
- "search means find" → synonym(search, find)

This enables contrastive learning through conversation - users can teach
the system semantic relationships using natural language.

Usage:
    extractor = LinguisticRelationExtractor()
    relations = extractor.extract("red is not blue")
    # [ExtractedRelation(type='negative', term1='red', term2='blue', confidence=0.9)]
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExtractedRelation:
    """A semantic relation extracted from text."""
    relation_type: str  # antonym, synonym, hypernym, negative, property, causal
    term1: str
    term2: str
    confidence: float
    source_text: str
    pattern_matched: str = ""
    
    def to_training_pair(self) -> Tuple[str, str, float]:
        """
        Convert to a (word1, word2, target_similarity) tuple for encoder training.
        
        Returns:
            (term1, term2, target) where target is:
            - Positive (~0.9) for synonyms, hypernyms
            - Negative (~-0.7) for antonyms
            - Slightly negative (~-0.3) for negatives/different
        """
        if self.relation_type == "antonym":
            return (self.term1, self.term2, -0.7)
        elif self.relation_type == "negative":
            return (self.term1, self.term2, -0.3)
        elif self.relation_type in ("synonym", "hypernym"):
            return (self.term1, self.term2, 0.9)
        elif self.relation_type == "property":
            return (self.term1, self.term2, 0.6)  # Related but not synonymous
        elif self.relation_type == "causal":
            return (self.term1, self.term2, 0.5)  # Associated
        else:
            return (self.term1, self.term2, 0.0)  # Neutral


class LinguisticRelationExtractor:
    """
    Extracts semantic relationships from natural language.
    
    Recognizes patterns like:
    - "X is the opposite of Y" → antonym
    - "X is not Y" → negative 
    - "X means Y" → synonym
    - "X is a Y" → hypernym
    - "X has Y" → property
    - "X causes Y" → causal
    """
    
    # Default patterns if no bootstrap file found
    DEFAULT_PATTERNS = {
        "antonym_patterns": [
            "{X} is the opposite of {Y}",
            "{X} is opposite to {Y}",
            "the opposite of {X} is {Y}",
            # Verb-friendly patterns
            "to {X} is the opposite of to {Y}",
            "{X} is not like {Y}",
        ],
        "negative_patterns": [
            "{X} is not {Y}",
            "{X} isn't {Y}",
            "{X} is different from {Y}",
        ],
        "synonym_patterns": [
            "{X} means {Y}",
            "{X} is like {Y}",
            "{X} is similar to {Y}",
            "{X} is the same as {Y}",
            # Verb-friendly patterns
            "to {X} is to {Y}",
            "to {X} means to {Y}",
            "{X} and {Y} are the same",
            "{X} and {Y} mean the same thing",
        ],
        "hypernym_patterns": [
            "{X} is a {Y}",
            "{X} is a type of {Y}",
            "{X} is a kind of {Y}",
        ],
        "property_patterns": [
            "{X} has {Y}",
            "{X} contains {Y}",
            "{X} is {Y}",  # Simple property assertion
        ],
        "causal_patterns": [
            "{X} causes {Y}",
            "{X} leads to {Y}",
            "if {X} then {Y}",
        ],
    }
    
    def __init__(
        self,
        bootstrap_path: Optional[str] = None,
        custom_patterns: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize the extractor.
        
        Args:
            bootstrap_path: Path to capability_bootstrap.json (or similar)
            custom_patterns: Additional patterns to add
        """
        self.patterns: Dict[str, List[re.Pattern]] = {}
        self._raw_patterns: Dict[str, List[str]] = {}
        
        # Load patterns
        self._load_default_patterns()
        
        if bootstrap_path:
            self._load_from_bootstrap(bootstrap_path)
            
        if custom_patterns:
            self._add_patterns(custom_patterns)
            
        # Compile all patterns
        self._compile_patterns()
        
    def _load_default_patterns(self) -> None:
        """Load default patterns."""
        for pattern_type, patterns in self.DEFAULT_PATTERNS.items():
            relation_type = pattern_type.replace("_patterns", "")
            self._raw_patterns[relation_type] = patterns.copy()
            
    def _load_from_bootstrap(self, path: str) -> None:
        """Load patterns from bootstrap JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            semantic = data.get("semantic_bootstrap", {})
            linguistic = semantic.get("linguistic_patterns", {})
            
            for pattern_type, patterns in linguistic.items():
                if pattern_type.startswith("_"):
                    continue
                if not isinstance(patterns, list):
                    continue
                    
                relation_type = pattern_type.replace("_patterns", "")
                if relation_type not in self._raw_patterns:
                    self._raw_patterns[relation_type] = []
                    
                for pattern in patterns:
                    if pattern not in self._raw_patterns[relation_type]:
                        self._raw_patterns[relation_type].append(pattern)
                        
            logger.info(f"Loaded linguistic patterns from {path}")
            
        except Exception as e:
            logger.debug(f"Could not load bootstrap patterns: {e}")
            
    def _add_patterns(self, patterns: Dict[str, List[str]]) -> None:
        """Add custom patterns."""
        for pattern_type, pattern_list in patterns.items():
            relation_type = pattern_type.replace("_patterns", "")
            if relation_type not in self._raw_patterns:
                self._raw_patterns[relation_type] = []
            self._raw_patterns[relation_type].extend(pattern_list)
            
    def _compile_patterns(self) -> None:
        """Compile pattern strings into regex patterns."""
        for relation_type, patterns in self._raw_patterns.items():
            compiled = []
            for pattern in patterns:
                # Convert {X} and {Y} to named capture groups
                # Handle word boundaries and flexible whitespace
                regex_pattern = pattern
                
                # Escape regex special chars except our placeholders
                regex_pattern = re.escape(regex_pattern)
                
                # Convert escaped placeholders back
                regex_pattern = regex_pattern.replace(r"\{X\}", r"(?P<term1>\w+(?:\s+\w+)?)")
                regex_pattern = regex_pattern.replace(r"\{Y\}", r"(?P<term2>\w+(?:\s+\w+)*)")
                
                # Allow flexible whitespace
                regex_pattern = regex_pattern.replace(r"\ ", r"\s+")
                
                try:
                    compiled.append((
                        re.compile(regex_pattern, re.IGNORECASE),
                        pattern  # Keep original for reference
                    ))
                except re.error as e:
                    logger.warning(f"Invalid pattern '{pattern}': {e}")
                    
            self.patterns[relation_type] = compiled
            
    def extract(self, text: str) -> List[ExtractedRelation]:
        """
        Extract semantic relations from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of extracted relations
        """
        relations = []
        text = text.strip()
        
        # Try each relation type
        for relation_type, compiled_patterns in self.patterns.items():
            for regex, original_pattern in compiled_patterns:
                match = regex.search(text)
                if match:
                    term1 = match.group("term1").strip().lower()
                    term2 = match.group("term2").strip().lower()
                    
                    # Skip if terms are too short or identical
                    if len(term1) < 2 or len(term2) < 2:
                        continue
                    if term1 == term2:
                        continue
                        
                    # Confidence based on pattern specificity
                    confidence = self._compute_confidence(
                        relation_type, original_pattern, text
                    )
                    
                    relations.append(ExtractedRelation(
                        relation_type=relation_type,
                        term1=term1,
                        term2=term2,
                        confidence=confidence,
                        source_text=text,
                        pattern_matched=original_pattern,
                    ))
                    
                    # Only match one pattern per relation type
                    break
                    
        return relations
    
    def _compute_confidence(
        self, 
        relation_type: str, 
        pattern: str, 
        text: str
    ) -> float:
        """Compute confidence score for an extraction."""
        # Base confidence by relation type (some patterns are more reliable)
        base_confidence = {
            "antonym": 0.95,     # "opposite of" is very clear
            "synonym": 0.85,     # "means" can be ambiguous
            "hypernym": 0.90,    # "is a" is fairly clear
            "negative": 0.80,    # "is not" might be contextual
            "property": 0.75,    # "has/is" can be many things
            "causal": 0.85,      # "causes" is usually clear
        }.get(relation_type, 0.7)
        
        # Boost for more specific patterns
        if "opposite" in pattern.lower():
            base_confidence += 0.05
        if "type of" in pattern.lower() or "kind of" in pattern.lower():
            base_confidence += 0.05
            
        # Reduce for very short input (might be fragment)
        if len(text) < 15:
            base_confidence -= 0.1
            
        return min(1.0, max(0.0, base_confidence))
    
    def extract_and_train(
        self,
        text: str,
        encoder,  # Duck-typed encoder with compute_word_pair_loss
        graph_store = None,  # Optional graph store for persistence
        tenant_id: Optional[str] = None,
    ) -> List[ExtractedRelation]:
        """
        Extract relations and immediately train the encoder.
        
        This is the main entry point for learning from conversation.
        
        Args:
            text: User input text
            encoder: Encoder with compute_word_pair_loss method
            graph_store: Optional graph store for persistence
            tenant_id: Optional tenant ID
            
        Returns:
            List of extracted and trained relations
        """
        relations = self.extract(text)
        
        if not relations:
            return []
            
        # Get base encoder if wrapped
        base_encoder = getattr(encoder, 'base_encoder', encoder)
        
        for relation in relations:
            # 1. Add words to vocabulary
            if hasattr(encoder, 'add_words'):
                encoder.add_words([relation.term1, relation.term2])
            elif hasattr(encoder, 'add_word'):
                encoder.add_word(relation.term1)
                encoder.add_word(relation.term2)
                
            # 2. Train encoder on the pair
            if hasattr(base_encoder, 'compute_word_pair_loss'):
                import torch
                import torch.optim as optim
                
                try:
                    term1, term2, target = relation.to_training_pair()
                    
                    # Get trainable parameters
                    params = []
                    if hasattr(base_encoder, 'embedding'):
                        params = list(base_encoder.embedding.parameters())
                    elif hasattr(encoder, 'get_trainable_parameters'):
                        params = encoder.get_trainable_parameters()
                        
                    if params:
                        optimizer = optim.SGD(params, lr=0.05)
                        
                        # A few training steps to nudge the embeddings
                        for _ in range(5):
                            optimizer.zero_grad()
                            loss = base_encoder.compute_word_pair_loss(
                                term1, term2, torch.tensor(target)
                            )
                            if loss.requires_grad:
                                loss.backward()
                                optimizer.step()
                                
                        logger.debug(
                            f"Trained {relation.relation_type}({term1}, {term2}) "
                            f"target={target:.2f}"
                        )
                except Exception as e:
                    logger.debug(f"Training failed for {relation}: {e}")
                    
            # 3. Store in graph if available
            if graph_store:
                try:
                    # Store as edge between concept nodes
                    self._store_relation_in_graph(
                        graph_store, relation, tenant_id
                    )
                except Exception as e:
                    logger.debug(f"Graph storage failed: {e}")
                    
        return relations
    
    def _store_relation_in_graph(
        self,
        graph,
        relation: ExtractedRelation,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Store a relation in the knowledge graph."""
        # Map relation types to edge types
        edge_type = {
            "antonym": "opposite_of",
            "synonym": "same_as",
            "hypernym": "is_a",
            "negative": "not_same_as",
            "property": "has_property",
            "causal": "causes",
        }.get(relation.relation_type, relation.relation_type)
        
        # Create or get canonical concept nodes.
        node1_id = graph.get_or_create_concept(
            relation.term1,
            confidence=relation.confidence,
            alias=relation.term1,
            tenant_id=tenant_id,
        )
        node2_id = graph.get_or_create_concept(
            relation.term2,
            confidence=relation.confidence,
            alias=relation.term2,
            tenant_id=tenant_id,
        )
            
        # Add edge
        try:
            graph.add_edge(
                source=node1_id,
                target=node2_id,
                edge_type=edge_type,
                confidence=relation.confidence,
                tenant_id=tenant_id,
            )
            logger.debug(f"Stored edge: {node1_id} --[{edge_type}]--> {node2_id}")
        except Exception as e:
            logger.warning(f"Edge add failed: {node1_id} --[{edge_type}]--> {node2_id}: {e}")


# Convenience function
def extract_relations(text: str, bootstrap_path: str = "data/seed/capability_bootstrap.json") -> List[ExtractedRelation]:
    """
    Extract semantic relations from text.
    
    Args:
        text: Input text
        bootstrap_path: Path to bootstrap file
        
    Returns:
        List of extracted relations
    """
    extractor = LinguisticRelationExtractor(bootstrap_path=bootstrap_path)
    return extractor.extract(text)
