"""
Semantic Extractor for Lilith v2.

Extracts semantic relationships (Is-A, Has-A, Equivalent-To) from natural text.
Used to automatically populate the RelationalGraphStore from conversation.
"""

import re
from dataclasses import dataclass
from typing import List, Optional
from .fuzzy_utils import FuzzyUtils

@dataclass
class ExtractedRelation:
    subject: str
    predicate: str  # is_a, has_a, similar_to, etc.
    object: str
    confidence: float
    source: str = "text_extraction"

class SemanticExtractor:
    """Extracts semantic triples from text using rigorous patterns."""
    
    def __init__(self):
        self.patterns = [
            # Is-A (Hierarchy)
            (re.compile(r'(\w+(?:\s+\w+)?) is a type of (\w+(?:\s+\w+)?)', re.I), "is_a", 0.9),
            (re.compile(r'(\w+(?:\s+\w+)?) is a kind of (\w+(?:\s+\w+)?)', re.I), "is_a", 0.9),
            (re.compile(r'(\w+(?:\s+\w+)?) is an? (\w+(?:\s+\w+)?)', re.I), "is_a", 0.7), # Lower confidence for generic "is a"
            
            # Has-A (Composition)
            (re.compile(r'(\w+(?:\s+\w+)?) (?:has|contains|includes) (\w+(?:\s+\w+)?)', re.I), "has_a", 0.8),
            (re.compile(r'(\w+(?:\s+\w+)?) is part of (\w+(?:\s+\w+)?)', re.I), "part_of", 0.9),
            
            # Equivalence / Similarity
            (re.compile(r'(\w+(?:\s+\w+)?) is (?:called|known as) (\w+(?:\s+\w+)?)', re.I), "synonym", 0.9),
            (re.compile(r'(\w+(?:\s+\w+)?) is like (\w+(?:\s+\w+)?)', re.I), "similar_to", 0.7),
            
            # Opposition
            (re.compile(r'(\w+(?:\s+\w+)?) is the opposite of (\w+(?:\s+\w+)?)', re.I), "opposite_of", 0.95),
        ]
        
        self.stop_words = {'the', 'a', 'an', 'this', 'that', 'it', 'they', 'what', 'who', 'where'}

    def extract(self, text: str) -> List[ExtractedRelation]:
        results = []
        
        # 1. Pre-processing: Typo Correction for Keywords
        # This turns "A Beagle is a tyype of Dog" -> "A Beagle is a type of Dog"
        # enabling regex patterns to match.
        text = FuzzyUtils.correct_keywords(text)
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence: continue
            
            for pattern, predicate, conf in self.patterns:
                match = pattern.search(sentence)
                if match:
                    subj, obj = match.groups()
                    subj = self._clean(subj)
                    obj = self._clean(obj)
                    
                    if subj and obj and subj != obj:
                        results.append(ExtractedRelation(
                            subject=subj,
                            predicate=predicate,
                            object=obj,
                            confidence=conf
                        ))
        return results

    def _clean(self, term: str) -> Optional[str]:
        term = term.strip().lower()
        # Strip articles from start
        for article in ["a ", "an ", "the "]:
            if term.startswith(article):
                term = term[len(article):].strip()
        
        if term in self.stop_words: return None
        if len(term) < 2: return None
        return term
