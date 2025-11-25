"""Concept taxonomy and compositional semantics for improved abstraction.

Provides hierarchical concept relationships and compositional query
understanding to enable semantic generalization beyond keyword matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import torch


@dataclass
class Concept:
    """A semantic concept with hierarchical relationships."""
    
    name: str
    parents: Set[str] = field(default_factory=set)  # Hypernyms (is-a)
    children: Set[str] = field(default_factory=set)  # Hyponyms
    properties: Set[str] = field(default_factory=set)  # Has-a, has-property
    related: Set[str] = field(default_factory=set)  # Related concepts
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Concept):
            return False
        return self.name == other.name


class ConceptTaxonomy:
    """Hierarchical concept taxonomy with relationship traversal."""
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self._initialize_default_taxonomy()
    
    def _initialize_default_taxonomy(self) -> None:
        """Build default concept hierarchy for common domain."""
        
        # Locations
        self.add_concept("location")
        self.add_concept("indoor_location", parents={"location"})
        self.add_concept("outdoor_location", parents={"location"})
        
        self.add_concept("hospital", parents={"indoor_location"}, properties={"medical"})
        self.add_concept("library", parents={"indoor_location"}, properties={"books", "study"})
        self.add_concept("classroom", parents={"indoor_location"}, properties={"teaching", "study"})
        
        self.add_concept("park", parents={"outdoor_location"}, properties={"trees", "nature", "recreation"})
        self.add_concept("garden", parents={"outdoor_location"}, properties={"plants", "nature"})
        
        # People/Roles
        self.add_concept("person")
        self.add_concept("medical_professional", parents={"person"})
        self.add_concept("doctor", parents={"medical_professional"})
        self.add_concept("patient", parents={"person"}, related={"medical", "hospital"})
        
        # Activities
        self.add_concept("activity")
        self.add_concept("medical_activity", parents={"activity"})
        self.add_concept("visit", parents={"activity"})
        self.add_concept("treatment", parents={"medical_activity"})
        self.add_concept("appointment", parents={"activity"}, related={"visit"})
        
        # Properties
        self.add_concept("medical")
        self.add_concept("outdoor")
        self.add_concept("indoor")
        self.add_concept("nature")
        self.add_concept("recreation")
    
    def add_concept(
        self,
        name: str,
        parents: Optional[Set[str]] = None,
        properties: Optional[Set[str]] = None,
        related: Optional[Set[str]] = None,
    ) -> Concept:
        """Add a concept to the taxonomy."""
        
        if name in self.concepts:
            concept = self.concepts[name]
        else:
            concept = Concept(
                name=name,
                parents=parents or set(),
                properties=properties or set(),
                related=related or set(),
            )
            self.concepts[name] = concept
        
        # Update parent-child relationships
        for parent_name in concept.parents:
            if parent_name not in self.concepts:
                self.concepts[parent_name] = Concept(name=parent_name)
            self.concepts[parent_name].children.add(name)
        
        return concept
    
    def get_ancestors(self, concept_name: str, max_depth: int = 10) -> Set[str]:
        """Get all ancestor concepts (transitive closure of parents)."""
        
        if concept_name not in self.concepts:
            return set()
        
        ancestors = set()
        to_visit = [concept_name]
        depth = 0
        
        while to_visit and depth < max_depth:
            current = to_visit.pop(0)
            if current in self.concepts:
                parents = self.concepts[current].parents
                ancestors.update(parents)
                to_visit.extend(parents)
            depth += 1
        
        return ancestors
    
    def get_descendants(self, concept_name: str, max_depth: int = 10) -> Set[str]:
        """Get all descendant concepts (transitive closure of children)."""
        
        if concept_name not in self.concepts:
            return set()
        
        descendants = set()
        to_visit = [concept_name]
        depth = 0
        
        while to_visit and depth < max_depth:
            current = to_visit.pop(0)
            if current in self.concepts:
                children = self.concepts[current].children
                descendants.update(children)
                to_visit.extend(children)
            depth += 1
        
        return descendants
    
    def get_properties(self, concept_name: str) -> Set[str]:
        """Get all properties of a concept (direct + inherited)."""
        
        if concept_name not in self.concepts:
            return set()
        
        # Direct properties
        properties = set(self.concepts[concept_name].properties)
        
        # Inherited properties from ancestors
        ancestors = self.get_ancestors(concept_name)
        for ancestor in ancestors:
            if ancestor in self.concepts:
                properties.update(self.concepts[ancestor].properties)
        
        return properties
    
    def is_a(self, concept: str, parent: str) -> bool:
        """Check if concept is-a parent (direct or transitive)."""
        
        if concept == parent:
            return True
        
        ancestors = self.get_ancestors(concept)
        return parent in ancestors
    
    def semantic_similarity(self, concept1: str, concept2: str) -> float:
        """Compute semantic similarity between two concepts."""
        
        if concept1 == concept2:
            return 1.0
        
        if concept1 not in self.concepts or concept2 not in self.concepts:
            return 0.0
        
        # Check direct relationships
        c1 = self.concepts[concept1]
        c2 = self.concepts[concept2]
        
        # Same parent → high similarity
        common_parents = c1.parents & c2.parents
        if common_parents:
            return 0.8
        
        # One is ancestor of other → medium-high similarity
        if self.is_a(concept1, concept2) or self.is_a(concept2, concept1):
            return 0.7
        
        # Related concepts → medium similarity
        if concept2 in c1.related or concept1 in c2.related:
            return 0.6
        
        # Shared properties → low-medium similarity
        props1 = self.get_properties(concept1)
        props2 = self.get_properties(concept2)
        if props1 & props2:
            overlap = len(props1 & props2) / max(len(props1), len(props2))
            return 0.3 + (overlap * 0.3)  # 0.3 to 0.6 range
        
        # Check if ancestors overlap
        ancestors1 = self.get_ancestors(concept1)
        ancestors2 = self.get_ancestors(concept2)
        common_ancestors = ancestors1 & ancestors2
        if common_ancestors:
            # More specific common ancestor = higher similarity
            depth = min(len(ancestors1), len(ancestors2))
            return 0.2 + (0.3 / (depth + 1))
        
        return 0.0
    
    def expand_query(self, query_terms: List[str]) -> Set[str]:
        """Expand query terms with related concepts."""
        
        expanded = set(query_terms)
        
        for term in query_terms:
            if term in self.concepts:
                # Add children (more specific concepts)
                expanded.update(self.get_descendants(term, max_depth=2))
                
                # Add properties
                expanded.update(self.get_properties(term))
                
                # Add related concepts
                expanded.update(self.concepts[term].related)
        
        return expanded
    
    def extract_concepts(self, text: str) -> Set[str]:
        """Extract recognized concepts from text."""
        
        text_lower = text.lower()
        found = set()
        
        for concept_name in self.concepts:
            # Simple word matching (could be improved with stemming/lemmatization)
            if concept_name.replace("_", " ") in text_lower:
                found.add(concept_name)
            elif concept_name in text_lower:
                found.add(concept_name)
        
        return found


class CompositionalQuery:
    """Compositional query understanding for multi-term concepts."""
    
    def __init__(self, taxonomy: ConceptTaxonomy):
        self.taxonomy = taxonomy
    
    def decompose(self, query: str) -> List[str]:
        """Break query into constituent concepts."""
        
        # Simple tokenization (could be improved with NLP)
        words = query.lower().split()
        
        # Try to form multi-word concepts
        concepts = []
        i = 0
        while i < len(words):
            # Try 2-word concepts first
            if i < len(words) - 1:
                two_word = f"{words[i]}_{words[i+1]}"
                if two_word in self.taxonomy.concepts:
                    concepts.append(two_word)
                    i += 2
                    continue
            
            # Single word
            concepts.append(words[i])
            i += 1
        
        return concepts
    
    def compose(self, concepts: List[str]) -> torch.Tensor:
        """Compose multiple concepts into unified query representation.
        
        Uses concept taxonomy to weight terms by relevance and combine
        into a single query vector.
        """
        
        if not concepts:
            return torch.zeros(1)
        
        # Expand each concept
        all_concepts = set()
        for concept in concepts:
            all_concepts.add(concept)
            all_concepts.update(self.taxonomy.expand_query([concept]))
        
        # Create concept vector (simple bag-of-concepts)
        # In practice, would use learned embeddings
        concept_list = sorted(all_concepts)
        vector = torch.zeros(len(concept_list))
        
        for i, concept in enumerate(concept_list):
            if concept in concepts:
                vector[i] = 1.0  # Original query terms get full weight
            else:
                vector[i] = 0.5  # Expanded terms get half weight
        
        # Normalize
        if vector.sum() > 0:
            vector = vector / vector.sum()
        
        return vector
    
    def interpret(self, query: str) -> Tuple[List[str], Set[str], Dict[str, float]]:
        """Interpret a natural language query.
        
        Returns:
            concepts: Core concepts identified
            expanded: All related concepts
            weights: Relevance weights for each concept
        """
        
        concepts = self.decompose(query)
        expanded = self.taxonomy.expand_query(concepts)
        
        # Compute relevance weights
        weights = {}
        for concept in expanded:
            if concept in concepts:
                weights[concept] = 1.0  # Original term
            else:
                # Average similarity to original concepts
                similarities = [
                    self.taxonomy.semantic_similarity(concept, orig)
                    for orig in concepts
                ]
                weights[concept] = max(similarities) if similarities else 0.5
        
        return concepts, expanded, weights


# Example usage and testing
def demo_taxonomy():
    """Demonstrate concept taxonomy capabilities."""
    
    print("=" * 80)
    print("CONCEPT TAXONOMY DEMONSTRATION")
    print("=" * 80)
    
    taxonomy = ConceptTaxonomy()
    
    # Show hierarchy
    print("\nConcept: 'hospital'")
    print(f"  Ancestors: {taxonomy.get_ancestors('hospital')}")
    print(f"  Properties: {taxonomy.get_properties('hospital')}")
    print(f"  Is-a location? {taxonomy.is_a('hospital', 'location')}")
    print(f"  Is-a indoor_location? {taxonomy.is_a('hospital', 'indoor_location')}")
    
    print("\nConcept: 'outdoor_location'")
    print(f"  Descendants: {taxonomy.get_descendants('outdoor_location')}")
    
    # Semantic similarity
    print("\nSemantic Similarities:")
    pairs = [
        ("hospital", "library"),
        ("hospital", "doctor"),
        ("park", "garden"),
        ("park", "hospital"),
        ("outdoor_location", "park"),
    ]
    
    for c1, c2 in pairs:
        sim = taxonomy.semantic_similarity(c1, c2)
        print(f"  {c1:20s} <-> {c2:20s} : {sim:.2f}")
    
    # Query expansion
    print("\nQuery Expansion:")
    queries = ["outdoor location", "hospital visit", "medical"]
    
    compositor = CompositionalQuery(taxonomy)
    
    for query in queries:
        concepts, expanded, weights = compositor.interpret(query)
        print(f"\nQuery: '{query}'")
        print(f"  Core concepts: {concepts}")
        print(f"  Expanded ({len(expanded)} terms): {sorted(expanded)}")
        print(f"  Top weights:")
        top_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        for term, weight in top_weights:
            print(f"    {term:20s} : {weight:.2f}")


if __name__ == "__main__":
    demo_taxonomy()
