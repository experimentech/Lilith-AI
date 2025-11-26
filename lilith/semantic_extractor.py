"""
Semantic Extractor - Extract structured concepts from natural language text

Extracts semantic relationships, entities, and properties from Wikipedia responses
and other knowledge sources to populate the concept store.

Phase A: Semantic Relationship Extraction
- Pattern: "X is a Y" → Relation(is_type_of, Y)
- Pattern: "X is a Y that Z" → Relation + properties
- Pattern: "X is used for Y" → Relation(used_for, Y)
- Pattern: "X has Y" → Relation(has_property, Y)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import re


@dataclass
class ExtractedConcept:
    """A concept extracted from text"""
    term: str                       # Main subject/entity
    type_relations: List[str]       # "is a X", "is a type of X"
    properties: List[str]           # Adjectives, characteristics
    usage_relations: List[str]      # "used for X", "designed for X"
    has_relations: List[str]        # "has X", "includes X", "features X"
    created_by: Optional[str] = None  # Creator/developer
    entity_type: Optional[str] = None  # Inferred entity type (programming_language, person, etc.)
    confidence: float = 0.85        # Extraction confidence


class SemanticExtractor:
    """
    Extract structured semantic information from text.
    
    Focus: Simple, robust patterns that work well for Wikipedia-style text.
    
    Phase A: Semantic relationships
    Phase B: Entity type recognition
    """
    
    def __init__(self):
        """Initialize semantic extractor with pattern matchers."""
        
        # Pattern: "X is a Y [that/which Z]"
        self.is_a_pattern = re.compile(
            r'^(.+?)\s+is\s+a(?:n)?\s+(.+?)(?:\s+that|\s+which|\s+emphasizing|\s+focusing|\.|$)',
            re.IGNORECASE
        )
        
        # Pattern: "X is a type of Y"
        self.type_of_pattern = re.compile(
            r'^(.+?)\s+is\s+a\s+type\s+of\s+(.+?)(?:\.|$)',
            re.IGNORECASE
        )
        
        # Pattern: "X is used for Y"
        self.used_for_pattern = re.compile(
            r'(?:is\s+)?used\s+for\s+(.+?)(?:\.|,|and|$)',
            re.IGNORECASE
        )
        
        # Pattern: "X emphasizing/focusing on Y, Z, and W"
        self.emphasis_pattern = re.compile(
            r'emphasizing|focusing\s+on|designed\s+for',
            re.IGNORECASE
        )
        
        # Pattern: "X features Y" or "X includes Y"
        self.has_pattern = re.compile(
            r'(?:features?|includes?|has|with)\s+(.+?)(?:\.|,|and|$)',
            re.IGNORECASE
        )
        
        # Pattern: "created/developed/designed by X"
        self.creator_pattern = re.compile(
            r'(?:created|developed|designed|invented)\s+(?:by|in)\s+(.+?)(?:\.|,|in\s+\d{4}|$)',
            re.IGNORECASE
        )
        
        # Entity type mapping from type_relations
        self.entity_type_mappings = {
            'programming language': 'programming_language',
            'language': 'programming_language',
            'software': 'software',
            'framework': 'software_framework',
            'library': 'software_library',
            'tool': 'software_tool',
            'algorithm': 'algorithm',
            'data structure': 'data_structure',
            'concept': 'abstract_concept',
            'theory': 'theory',
            'method': 'method',
            'technique': 'technique',
            'person': 'person',
            'company': 'organization',
            'organization': 'organization',
            'institution': 'organization',
        }
    
    def extract_concepts(
        self, 
        query: str, 
        response: str,
        max_sentences: int = 3
    ) -> List[ExtractedConcept]:
        """
        Extract concepts from a text response.
        
        Args:
            query: The original user query (to identify subject)
            response: The Wikipedia or knowledge response
            max_sentences: How many sentences to analyze (default: first 3)
            
        Returns:
            List of extracted concepts
        """
        # Extract subject from query
        subject = self._extract_subject_from_query(query)
        
        # Split response into sentences
        sentences = self._split_sentences(response, max_sentences)
        
        if not sentences:
            return []
        
        # Extract from first sentence (usually the definition)
        first_sentence = sentences[0]
        
        concept = ExtractedConcept(
            term=subject,
            type_relations=[],
            properties=[],
            usage_relations=[],
            has_relations=[],
            created_by=None,
            entity_type=None,
            confidence=0.85
        )
        
        # Extract "is a" relationships
        type_rel = self._extract_is_a(first_sentence)
        if type_rel:
            concept.type_relations.append(type_rel)
        
        # Infer entity type from type relations
        concept.entity_type = self._infer_entity_type(concept.type_relations, concept.term)
        
        # Extract properties from emphasis clauses
        props = self._extract_properties(first_sentence)
        concept.properties.extend(props)
        
        # Analyze all sentences for additional relations
        for sentence in sentences:
            # Usage relations
            usage = self._extract_usage(sentence)
            concept.usage_relations.extend(usage)
            
            # Has/features/includes relations
            has_rels = self._extract_has_relations(sentence)
            concept.has_relations.extend(has_rels)
            
            # Creator information
            creator = self._extract_creator(sentence)
            if creator and not concept.created_by:
                concept.created_by = creator
        
        return [concept] if concept.type_relations or concept.properties else []
    
    def _extract_subject_from_query(self, query: str) -> str:
        """
        Extract the main subject from a user query.
        
        Examples:
            "what is Python" → "Python"
            "tell me about Rust programming" → "Rust"
            "what is quantum computing" → "quantum computing"
        """
        # Remove question words
        text = re.sub(
            r'\b(what|who|where|when|why|how|is|are|was|were|do|does|did|'
            r'can|could|would|should|tell|me|about|know|explain)\b',
            '',
            query,
            flags=re.IGNORECASE
        )
        
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', '', text, flags=re.IGNORECASE)
        
        # Clean up whitespace
        text = ' '.join(text.split()).strip()
        
        # Capitalize properly (for Wikipedia-style names)
        if text and not text[0].isupper():
            # Don't capitalize if it's a compound phrase
            words = text.split()
            if len(words) == 1:
                text = text.capitalize()
        
        return text if text else "Unknown"
    
    def _split_sentences(self, text: str, max_sentences: int = 3) -> List[str]:
        """Split text into sentences, return first N."""
        # Simple sentence splitting (can be improved)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return sentences[:max_sentences]
    
    def _extract_is_a(self, sentence: str) -> Optional[str]:
        """
        Extract type relationship from "X is a Y" pattern.
        
        Examples:
            "Rust is a programming language" → "programming language"
            "Python is an interpreted language" → "interpreted language"
        """
        match = self.is_a_pattern.match(sentence)
        if match:
            type_term = match.group(2).strip()
            # Clean up common trailing words
            type_term = re.sub(r'\s+(that|which|emphasizing|focusing).*$', '', type_term, flags=re.IGNORECASE)
            return type_term
        
        # Try type_of pattern
        match = self.type_of_pattern.match(sentence)
        if match:
            return match.group(2).strip()
        
        return None
    
    def _extract_properties(self, sentence: str) -> List[str]:
        """
        Extract properties from emphasis clauses.
        
        Examples:
            "...emphasizing performance, type safety, and concurrency" 
            → ["performance", "type safety", "concurrency"]
        """
        properties = []
        
        # Find emphasis keyword
        match = self.emphasis_pattern.search(sentence)
        if not match:
            return properties
        
        # Extract text after the emphasis keyword
        start_pos = match.end()
        remainder = sentence[start_pos:].strip()
        
        # Split by commas and "and"
        parts = re.split(r',|\s+and\s+', remainder)
        
        for part in parts:
            # Clean up
            prop = part.strip().rstrip('.')
            prop = re.sub(r'^\s*(on|in)\s+', '', prop)  # Remove leading "on" or "in"
            
            if prop and len(prop) > 2:  # Ignore very short fragments
                properties.append(prop)
        
        return properties
    
    def _extract_usage(self, sentence: str) -> List[str]:
        """
        Extract usage relationships.
        
        Examples:
            "used for systems programming" → ["systems programming"]
        """
        usage = []
        
        for match in self.used_for_pattern.finditer(sentence):
            use_case = match.group(1).strip()
            if use_case and len(use_case) > 2:
                usage.append(use_case)
        
        return usage
    
    def _extract_has_relations(self, sentence: str) -> List[str]:
        """
        Extract "has/features/includes" relationships.
        
        Examples:
            "features automatic memory management" → ["automatic memory management"]
        """
        relations = []
        
        for match in self.has_pattern.finditer(sentence):
            relation = match.group(1).strip()
            if relation and len(relation) > 2:
                relations.append(relation)
        
        return relations
    
    def _extract_creator(self, sentence: str) -> Optional[str]:
        """
        Extract creator/developer information.
        
        Examples:
            "developed by Mozilla Research" → "Mozilla Research"
            "created by Guido van Rossum in 1991" → "Guido van Rossum"
        """
        match = self.creator_pattern.search(sentence)
        if match:
            creator = match.group(1).strip()
            return creator
        
        return None
    
    def _infer_entity_type(self, type_relations: List[str], term: str) -> Optional[str]:
        """
        Infer entity type from type relations and term characteristics.
        
        Args:
            type_relations: List of "is a X" relations
            term: The entity term
            
        Returns:
            Entity type string or None
            
        Examples:
            ["programming language"] → "programming_language"
            ["person"] → "person"
            ["software framework"] → "software_framework"
        """
        if not type_relations:
            # Try to infer from capitalization (proper nouns)
            if term and term[0].isupper() and ' ' in term:
                # Multi-word capitalized = likely organization or proper noun
                return "proper_noun"
            return None
        
        # Check first type relation
        type_rel = type_relations[0].lower()
        
        # Try exact or partial matches
        for key, entity_type in self.entity_type_mappings.items():
            if key in type_rel:
                return entity_type
        
        # Generic fallback based on common patterns
        if 'language' in type_rel:
            return 'programming_language'
        elif 'software' in type_rel or 'application' in type_rel:
            return 'software'
        elif 'algorithm' in type_rel:
            return 'algorithm'
        elif 'structure' in type_rel:
            return 'data_structure'
        elif 'computing' in type_rel or 'computation' in type_rel:
            return 'computing_concept'
        
        # Default: abstract concept
        return 'concept'
    
    def concept_to_dict(self, concept: ExtractedConcept) -> Dict:
        """
        Convert ExtractedConcept to dictionary for storage.
        
        Returns:
            Dict suitable for ProductionConceptStore.add_concept()
        """
        # Build properties list
        properties = concept.properties.copy()
        
        # Add creator as property if present
        if concept.created_by:
            properties.append(f"created by {concept.created_by}")
        
        # Build relations list (for ProductionConceptStore)
        from .production_concept_store import Relation
        
        relations = []
        
        # Type relations
        for type_rel in concept.type_relations:
            relations.append(Relation("is_type_of", type_rel, 0.90))
        
        # Usage relations
        for usage in concept.usage_relations:
            relations.append(Relation("used_for", usage, 0.85))
        
        # Has relations
        for has_rel in concept.has_relations:
            relations.append(Relation("has_property", has_rel, 0.80))
        
        return {
            "term": concept.term,
            "properties": properties,
            "relations": relations,
            "source": "wikipedia",
            "confidence": concept.confidence
        }


def demo():
    """Demo the semantic extractor."""
    extractor = SemanticExtractor()
    
    # Test cases
    test_cases = [
        (
            "what is Rust programming language",
            "Rust is a general-purpose programming language emphasizing performance, type safety, and concurrency."
        ),
        (
            "what is Python",
            "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability."
        ),
        (
            "tell me about quantum computing",
            "Quantum computing is a type of computing that uses quantum-mechanical phenomena. It is used for solving complex problems."
        ),
    ]
    
    for query, response in test_cases:
        print(f"\nQuery: {query}")
        print(f"Response: {response}")
        print("-" * 60)
        
        concepts = extractor.extract_concepts(query, response)
        
        for concept in concepts:
            print(f"Term: {concept.term}")
            print(f"Type: {concept.type_relations}")
            print(f"Entity Type: {concept.entity_type}")
            print(f"Properties: {concept.properties}")
            print(f"Usage: {concept.usage_relations}")
            print(f"Has: {concept.has_relations}")
            print(f"Creator: {concept.created_by}")
            print()


if __name__ == "__main__":
    demo()
