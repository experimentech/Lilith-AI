"""
Automatic Semantic Learning from Conversations

Extracts semantic relationships from natural conversation without explicit teaching.
This enables the BioNN to notice patterns like:
- "cats and dogs are both pets" → (cat, dog, positive), (cat, pet, positive)
- "happy is the opposite of sad" → (happy, sad, hard_negative)
- "Python is a programming language" → (Python, programming language, positive)

Works with ANY text input - CLI, voice transcripts, API calls, etc.
The I/O layer just needs to call `process_text()`.

Design Philosophy:
- Zero configuration: Just feed it text
- Modality agnostic: Works with any text source
- Incremental: Learns continuously, no batch retraining needed
- Lightweight: Pattern matching + optional NLP, not a separate model
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict
from pathlib import Path


@dataclass
class ExtractedRelation:
    """A semantic relation extracted from text."""
    concept_a: str
    concept_b: str
    relation_type: str  # "positive", "negative", "hard_negative"
    confidence: float
    source_pattern: str  # The pattern that matched
    source_text: str  # Original text snippet


class SemanticExtractor:
    """
    Extract semantic relationships from natural language.
    
    Uses pattern matching to identify statements about relationships:
    - Hierarchical: "X is a Y", "X is a type of Y"
    - Similarity: "X is like Y", "X and Y are both Z", "X is similar to Y"
    - Opposition: "X is the opposite of Y", "X is not Y", "unlike X, Y"
    - Equivalence: "X is also called Y", "X, or Y", "X (Y)"
    
    Can be enhanced with NLP (spaCy, etc.) but works standalone.
    """
    
    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
        
        # Patterns that indicate POSITIVE relationships (similar/related)
        self.positive_patterns = [
            # Hierarchical (is-a) - order matters, most specific first
            (r"(\w+(?:\s+\w+)?)\s+is\s+a\s+type\s+of\s+(\w+(?:\s+\w+)?)", 0.95, "is_a"),
            (r"(\w+(?:\s+\w+)?)\s+is\s+a\s+kind\s+of\s+(\w+(?:\s+\w+)?)", 0.95, "is_a"),
            (r"(\w+(?:\s+\w+)?)\s+is\s+a\s+subset\s+of\s+(\w+(?:\s+\w+)?)", 0.95, "is_a"),
            (r"(\w+(?:\s+\w+)?)\s+is\s+a\s+form\s+of\s+(\w+(?:\s+\w+)?)", 0.90, "is_a"),
            (r"(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:an?\s+)?(?:example|instance)s?\s+of\s+(\w+(?:\s+\w+)?)", 0.9, "instance_of"),
            (r"(\w+(?:\s+\w+)?)\s+is\s+an?\s+(\w+(?:\s+\w+)?)", 0.85, "is_a"),  # Generic "is a" last
            
            # Similarity
            (r"(\w+(?:\s+\w+)?)\s+is\s+like\s+(\w+(?:\s+\w+)?)", 0.8, "similar_to"),
            (r"(\w+(?:\s+\w+)?)\s+(?:is|are)\s+similar\s+to\s+(\w+(?:\s+\w+)?)", 0.85, "similar_to"),
            (r"(\w+(?:\s+\w+)?)\s+(?:and|&)\s+(\w+(?:\s+\w+)?)\s+are\s+(?:both|all)\s+(\w+)", 0.85, "both_are"),
            (r"(\w+(?:\s+\w+)?)\s+is\s+related\s+to\s+(\w+(?:\s+\w+)?)", 0.8, "related_to"),
            
            # Equivalence
            (r"(\w+(?:\s+\w+)?)\s+is\s+also\s+(?:called|known\s+as)\s+(\w+(?:\s+\w+)?)", 0.9, "synonym"),
            (r"(\w+(?:\s+\w+)?),?\s+or\s+(\w+(?:\s+\w+)?),", 0.7, "synonym"),
            (r"(\w+(?:\s+\w+)?)\s+\((\w+(?:\s+\w+)?)\)", 0.75, "synonym"),
            
            # Part-whole
            (r"(\w+(?:\s+\w+)?)\s+is\s+part\s+of\s+(\w+(?:\s+\w+)?)", 0.85, "part_of"),
            (r"(\w+(?:\s+\w+)?)\s+(?:has|contains?|includes?)\s+(\w+(?:\s+\w+)?)", 0.75, "has_part"),
            
            # Association (weaker)
            (r"(\w+(?:\s+\w+)?)\s+(?:involves?|uses?|requires?)\s+(\w+(?:\s+\w+)?)", 0.65, "associated"),
        ]
        
        # Patterns that indicate NEGATIVE relationships (different/opposed)
        self.negative_patterns = [
            # Opposition
            (r"(\w+(?:\s+\w+)?)\s+is\s+the\s+opposite\s+of\s+(\w+(?:\s+\w+)?)", 0.95, "opposite"),
            (r"(\w+(?:\s+\w+)?)\s+is\s+(?:not|n't)\s+(\w+(?:\s+\w+)?)", 0.7, "not_is"),
            (r"unlike\s+(\w+(?:\s+\w+)?),?\s+(\w+(?:\s+\w+)?)", 0.8, "unlike"),
            (r"(\w+(?:\s+\w+)?)\s+(?:differs?|different)\s+from\s+(\w+(?:\s+\w+)?)", 0.85, "different"),
            (r"(\w+(?:\s+\w+)?)\s+(?:vs\.?|versus)\s+(\w+(?:\s+\w+)?)", 0.7, "versus"),
            
            # Antonyms
            (r"(\w+(?:\s+\w+)?)\s+is\s+the\s+antonym\s+of\s+(\w+(?:\s+\w+)?)", 0.95, "antonym"),
        ]
        
        # Compile patterns for efficiency
        self._positive_compiled = [
            (re.compile(p, re.IGNORECASE), conf, name) 
            for p, conf, name in self.positive_patterns
        ]
        self._negative_compiled = [
            (re.compile(p, re.IGNORECASE), conf, name)
            for p, conf, name in self.negative_patterns
        ]
        
        # Stop words to filter out
        self.stop_words = {
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'it', 'they', 'them', 'their', 'its', 'he', 'she',
            'what', 'which', 'who', 'how', 'where', 'when', 'why',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did',
            'very', 'really', 'just', 'also', 'even', 'still',
        }
        
    def _clean_concept(self, concept: str) -> Optional[str]:
        """Clean and validate a concept string."""
        concept = concept.strip().lower()
        
        # Skip if too short or just stop words
        words = concept.split()
        meaningful_words = [w for w in words if w not in self.stop_words]
        
        if not meaningful_words:
            return None
        if len(concept) < 2:
            return None
        if len(concept) > 50:  # Too long, probably a phrase not a concept
            return None
            
        return concept
    
    def extract_from_text(self, text: str) -> List[ExtractedRelation]:
        """
        Extract semantic relations from a piece of text.
        
        Args:
            text: Any text (conversation, article, etc.)
            
        Returns:
            List of extracted relations
        """
        relations = []
        
        # Split into sentences for context
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Try positive patterns
            for pattern, base_conf, pattern_name in self._positive_compiled:
                for match in pattern.finditer(sentence):
                    groups = match.groups()
                    
                    if pattern_name == "both_are" and len(groups) >= 3:
                        # "X and Y are both Z" → (X, Z, positive), (Y, Z, positive)
                        concept_a = self._clean_concept(groups[0])
                        concept_b = self._clean_concept(groups[1])
                        category = self._clean_concept(groups[2])
                        
                        if concept_a and category:
                            relations.append(ExtractedRelation(
                                concept_a=concept_a,
                                concept_b=category,
                                relation_type="positive",
                                confidence=base_conf,
                                source_pattern=pattern_name,
                                source_text=sentence
                            ))
                        if concept_b and category:
                            relations.append(ExtractedRelation(
                                concept_a=concept_b,
                                concept_b=category,
                                relation_type="positive",
                                confidence=base_conf,
                                source_pattern=pattern_name,
                                source_text=sentence
                            ))
                        # Also mark X and Y as related
                        if concept_a and concept_b:
                            relations.append(ExtractedRelation(
                                concept_a=concept_a,
                                concept_b=concept_b,
                                relation_type="positive",
                                confidence=base_conf * 0.9,
                                source_pattern="shared_category",
                                source_text=sentence
                            ))
                    elif len(groups) >= 2:
                        concept_a = self._clean_concept(groups[0])
                        concept_b = self._clean_concept(groups[1])
                        
                        if concept_a and concept_b and concept_a != concept_b:
                            relations.append(ExtractedRelation(
                                concept_a=concept_a,
                                concept_b=concept_b,
                                relation_type="positive",
                                confidence=base_conf,
                                source_pattern=pattern_name,
                                source_text=sentence
                            ))
            
            # Try negative patterns
            for pattern, base_conf, pattern_name in self._negative_compiled:
                for match in pattern.finditer(sentence):
                    groups = match.groups()
                    
                    if len(groups) >= 2:
                        concept_a = self._clean_concept(groups[0])
                        concept_b = self._clean_concept(groups[1])
                        
                        if concept_a and concept_b and concept_a != concept_b:
                            # "not_is" is weaker, use "negative"
                            # "opposite/antonym" are stronger, use "hard_negative"
                            if pattern_name in ('opposite', 'antonym'):
                                rel_type = "hard_negative"
                            else:
                                rel_type = "negative"
                                
                            relations.append(ExtractedRelation(
                                concept_a=concept_a,
                                concept_b=concept_b,
                                relation_type=rel_type,
                                confidence=base_conf,
                                source_pattern=pattern_name,
                                source_text=sentence
                            ))
        
        # Filter by confidence
        relations = [r for r in relations if r.confidence >= self.min_confidence]
        
        return relations
    
    def extract_from_conversation(
        self,
        user_input: str,
        response: str
    ) -> List[ExtractedRelation]:
        """
        Extract relations from a conversation turn.
        
        Analyzes both user input and response for semantic patterns.
        """
        relations = []
        relations.extend(self.extract_from_text(user_input))
        relations.extend(self.extract_from_text(response))
        return relations


class AutoSemanticLearner:
    """
    Automatically learns semantic relationships from any text input.
    
    This is the bridge between I/O and the contrastive learner.
    Any input modality just calls `process_text()` or `process_conversation()`.
    
    Features:
    - Extracts semantic pairs from natural language
    - Deduplicates across sessions
    - Periodically triggers incremental training
    - Works with any text source (CLI, voice, API, etc.)
    """
    
    def __init__(
        self,
        contrastive_learner=None,
        auto_train_threshold: int = 10,
        auto_train_steps: int = 3,
    ):
        """
        Initialize auto semantic learner.
        
        Args:
            contrastive_learner: ContrastiveLearner instance (optional, can set later)
            auto_train_threshold: Trigger training after this many new pairs
            auto_train_steps: Gradient steps per incremental update
        """
        self.learner = contrastive_learner
        self.extractor = SemanticExtractor()
        self.auto_train_threshold = auto_train_threshold
        self.auto_train_steps = auto_train_steps
        
        # Track new pairs since last training
        self.pending_pairs: List[Tuple[str, str, str]] = []
        self.seen_pairs: Set[Tuple[str, str]] = set()
        
        # Statistics
        self.stats = {
            'texts_processed': 0,
            'relations_extracted': 0,
            'pairs_added': 0,
            'auto_trains': 0,
        }
    
    def set_learner(self, contrastive_learner):
        """Set or update the contrastive learner."""
        self.learner = contrastive_learner
    
    def process_text(self, text: str) -> List[ExtractedRelation]:
        """
        Process any text and extract semantic relationships.
        
        This is the main entry point - call it with any text from any source.
        
        Args:
            text: Any text (user input, response, article, transcript, etc.)
            
        Returns:
            List of extracted relations (for inspection/logging)
        """
        self.stats['texts_processed'] += 1
        
        relations = self.extractor.extract_from_text(text)
        self.stats['relations_extracted'] += len(relations)
        
        # Add to pending pairs
        for rel in relations:
            pair_key = (rel.concept_a.lower(), rel.concept_b.lower())
            reverse_key = (rel.concept_b.lower(), rel.concept_a.lower())
            
            # Skip if we've seen this pair
            if pair_key in self.seen_pairs or reverse_key in self.seen_pairs:
                continue
                
            self.seen_pairs.add(pair_key)
            self.pending_pairs.append((rel.concept_a, rel.concept_b, rel.relation_type))
            self.stats['pairs_added'] += 1
        
        # Auto-train if threshold reached
        self._maybe_auto_train()
        
        return relations
    
    def process_conversation(
        self,
        user_input: str,
        response: str
    ) -> List[ExtractedRelation]:
        """
        Process a conversation turn (both user input and response).
        
        Convenience method for dialogue systems.
        """
        relations = []
        relations.extend(self.process_text(user_input))
        relations.extend(self.process_text(response))
        return relations
    
    def _maybe_auto_train(self):
        """Trigger incremental training if threshold reached."""
        if not self.learner:
            return
            
        if len(self.pending_pairs) >= self.auto_train_threshold:
            # Add pairs to learner
            for concept_a, concept_b, rel_type in self.pending_pairs:
                self.learner.add_pair(
                    concept_a, concept_b, rel_type,
                    weight=0.8,  # Slightly lower weight for auto-extracted
                    source="auto_extracted"
                )
            
            # Incremental training
            self.learner.incremental_update(
                self.pending_pairs,
                steps=self.auto_train_steps
            )
            
            self.pending_pairs = []
            self.stats['auto_trains'] += 1
    
    def force_train(self):
        """Force training on all pending pairs."""
        if self.pending_pairs and self.learner:
            for concept_a, concept_b, rel_type in self.pending_pairs:
                self.learner.add_pair(
                    concept_a, concept_b, rel_type,
                    weight=0.8,
                    source="auto_extracted"
                )
            
            self.learner.incremental_update(
                self.pending_pairs,
                steps=self.auto_train_steps
            )
            
            self.pending_pairs = []
            self.stats['auto_trains'] += 1
    
    def get_stats(self) -> Dict:
        """Get learning statistics."""
        return {
            **self.stats,
            'pending_pairs': len(self.pending_pairs),
            'unique_pairs_seen': len(self.seen_pairs),
        }
    
    def save_state(self, path: Path):
        """Save seen pairs for persistence across sessions."""
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump({
                'seen_pairs': list(self.seen_pairs),
                'stats': self.stats,
            }, f)
    
    def load_state(self, path: Path):
        """Load seen pairs from previous session."""
        import json
        path = Path(path)
        
        if path.exists():
            with open(path) as f:
                state = json.load(f)
            self.seen_pairs = set(tuple(p) for p in state.get('seen_pairs', []))
            self.stats = state.get('stats', self.stats)
