"""
ConceptStore - Semantic Knowledge Storage (PoC)

Stores concepts with properties instead of full response patterns.
Uses BNN embeddings for semantic similarity and consolidation.

This is a minimal proof-of-concept implementation.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np
import json
from pathlib import Path


@dataclass
class Relation:
    """Semantic relation between concepts"""
    relation_type: str    # "is_type_of", "has_property", "used_for"
    target: str           # Target concept or property
    confidence: float     # How certain we are about this relation


@dataclass
class SemanticConcept:
    """A concept with semantic properties"""
    concept_id: str
    term: str                      # "machine learning"
    properties: List[str]          # ["learns from data", "branch of AI"]
    relations: List[Relation]      # Semantic relations to other concepts
    embedding: Optional[np.ndarray] = None  # BNN embedding (not serialized)
    confidence: float = 0.85       # How reliable this knowledge is
    source: str = "taught"         # "taught", "wikipedia", "learned"
    usage_count: int = 0           # Track usage for consolidation
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict"""
        data = asdict(self)
        # Remove embedding (not JSON serializable)
        data.pop('embedding', None)
        return data
    
    @staticmethod
    def from_dict(data: dict) -> 'SemanticConcept':
        """Reconstruct from dict"""
        # Convert relations back to Relation objects
        if 'relations' in data and data['relations']:
            data['relations'] = [
                Relation(**r) if isinstance(r, dict) else r 
                for r in data['relations']
            ]
        return SemanticConcept(**data)


class ConceptStore:
    """
    Stores semantic concepts with properties.
    
    Key differences from ResponseFragmentStore:
    1. Stores structured concepts, not full response texts
    2. Properties can be shared across concepts
    3. Consolidation via semantic similarity
    4. Compositional retrieval instead of exact matching
    """
    
    def __init__(self, semantic_encoder, storage_path: Optional[str] = None):
        """
        Initialize concept store.
        
        Args:
            semantic_encoder: BNN encoder for embeddings
            storage_path: Optional JSON file for persistence
        """
        self.encoder = semantic_encoder
        self.storage_path = Path(storage_path) if storage_path else None
        self.concepts: Dict[str, SemanticConcept] = {}
        
        # Load existing concepts if available
        if self.storage_path and self.storage_path.exists():
            self._load_concepts()
    
    def add_concept(
        self, 
        term: str, 
        properties: List[str],
        relations: Optional[List[Relation]] = None,
        source: str = "taught",
        confidence: float = 0.85
    ) -> str:
        """
        Add a new concept or enhance existing one.
        
        If a very similar concept exists (>0.90 similarity), merges properties.
        Otherwise creates new concept.
        
        Args:
            term: Concept name (e.g., "machine learning")
            properties: List of property strings
            relations: Optional semantic relations
            source: Where this knowledge came from
            confidence: How reliable this is
            
        Returns:
            concept_id of created or merged concept
        """
        # Generate embedding for the term
        embedding = self.encoder.encode(term.split())
        
        # Convert torch tensor to numpy if needed
        if hasattr(embedding, 'cpu'):
            embedding = embedding.cpu().detach().numpy().flatten()
        
        # Check for existing similar concept
        existing = self._find_similar_concept(embedding, threshold=0.90)
        
        if existing:
            # Merge properties into existing concept
            print(f"  🔗 Merging with existing concept: {existing.term}")
            for prop in properties:
                if prop not in existing.properties:
                    existing.properties.append(prop)
            
            # Update relations
            if relations:
                for new_rel in relations:
                    # Check if relation already exists
                    existing_rels = [r for r in existing.relations 
                                    if r.relation_type == new_rel.relation_type 
                                    and r.target == new_rel.target]
                    if not existing_rels:
                        existing.relations.append(new_rel)
            
            # Boost confidence with repeated teaching
            existing.confidence = min(0.95, existing.confidence + 0.02)
            
            self._save_concepts()
            return existing.concept_id
        
        else:
            # Create new concept
            concept_id = f"concept_{len(self.concepts):04d}"
            
            concept = SemanticConcept(
                concept_id=concept_id,
                term=term,
                properties=properties,
                relations=relations or [],
                embedding=embedding,
                confidence=confidence,
                source=source,
                usage_count=0
            )
            
            self.concepts[concept_id] = concept
            print(f"  ➕ Created new concept: {term} ({concept_id})")
            
            self._save_concepts()
            return concept_id
    
    def retrieve_similar(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        min_similarity: float = 0.60
    ) -> List[Tuple[SemanticConcept, float]]:
        """
        Retrieve concepts by semantic similarity.
        
        Args:
            query_embedding: Query embedding from BNN
            top_k: Maximum number to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (concept, similarity_score) tuples
        """
        if not self.concepts:
            return []
        
        # Calculate similarities
        results = []
        for concept in self.concepts.values():
            # Re-encode if embedding not cached
            if concept.embedding is None:
                embedding = self.encoder.encode(concept.term.split())
                # Convert torch tensor to numpy if needed
                if hasattr(embedding, 'cpu'):
                    embedding = embedding.cpu().detach().numpy().flatten()
                concept.embedding = embedding
            
            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, concept.embedding)
            
            if similarity >= min_similarity:
                results.append((concept, similarity))
                concept.usage_count += 1
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def get_concept(self, concept_id: str) -> Optional[SemanticConcept]:
        """Get concept by ID"""
        return self.concepts.get(concept_id)
    
    def get_concept_by_term(self, term: str, threshold: float = 0.85) -> Optional[SemanticConcept]:
        """
        Find concept by term name (with fuzzy matching).
        
        Args:
            term: Term to search for
            threshold: Similarity threshold for matching
            
        Returns:
            Matching concept or None
        """
        query_embedding = self.encoder.encode(term)
        results = self.retrieve_similar(query_embedding, top_k=1, min_similarity=threshold)
        
        if results:
            return results[0][0]
        return None
    
    def merge_similar_concepts(self, threshold: float = 0.92) -> int:
        """
        Consolidate concepts that are very similar.
        
        Args:
            threshold: Similarity threshold for merging
            
        Returns:
            Number of concepts merged
        """
        merged_count = 0
        concepts_to_remove = set()
        
        concept_list = list(self.concepts.values())
        
        for i, concept_a in enumerate(concept_list):
            if concept_a.concept_id in concepts_to_remove:
                continue
            
            for concept_b in concept_list[i+1:]:
                if concept_b.concept_id in concepts_to_remove:
                    continue
                
                # Calculate similarity
                if concept_a.embedding is None:
                    concept_a.embedding = self.encoder.encode(concept_a.term)
                if concept_b.embedding is None:
                    concept_b.embedding = self.encoder.encode(concept_b.term)
                
                similarity = self._cosine_similarity(concept_a.embedding, concept_b.embedding)
                
                if similarity >= threshold:
                    # Merge B into A
                    print(f"  🔗 Merging '{concept_b.term}' into '{concept_a.term}' (similarity: {similarity:.3f})")
                    
                    # Combine properties
                    for prop in concept_b.properties:
                        if prop not in concept_a.properties:
                            concept_a.properties.append(prop)
                    
                    # Combine relations
                    for rel in concept_b.relations:
                        existing = [r for r in concept_a.relations 
                                  if r.relation_type == rel.relation_type 
                                  and r.target == rel.target]
                        if not existing:
                            concept_a.relations.append(rel)
                    
                    # Update metadata
                    concept_a.usage_count += concept_b.usage_count
                    concept_a.confidence = max(concept_a.confidence, concept_b.confidence)
                    
                    # Mark for removal
                    concepts_to_remove.add(concept_b.concept_id)
                    merged_count += 1
        
        # Remove merged concepts
        for concept_id in concepts_to_remove:
            del self.concepts[concept_id]
        
        if merged_count > 0:
            self._save_concepts()
        
        return merged_count
    
    def _find_similar_concept(
        self, 
        embedding: np.ndarray, 
        threshold: float = 0.90
    ) -> Optional[SemanticConcept]:
        """Find if a very similar concept already exists"""
        results = self.retrieve_similar(embedding, top_k=1, min_similarity=threshold)
        if results:
            return results[0][0]
        return None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _save_concepts(self):
        """Save concepts to JSON (if storage path configured)"""
        if not self.storage_path:
            return
        
        data = {
            cid: concept.to_dict() 
            for cid, concept in self.concepts.items()
        }
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_concepts(self):
        """Load concepts from JSON"""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
        
        for cid, concept_data in data.items():
            concept = SemanticConcept.from_dict(concept_data)
            # Note: embeddings will be regenerated on first use
            self.concepts[cid] = concept
        
        print(f"  📚 Loaded {len(self.concepts)} concepts from {self.storage_path}")
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the concept store"""
        total_properties = sum(len(c.properties) for c in self.concepts.values())
        total_relations = sum(len(c.relations) for c in self.concepts.values())
        
        return {
            "total_concepts": len(self.concepts),
            "total_properties": total_properties,
            "total_relations": total_relations,
            "avg_properties_per_concept": total_properties / len(self.concepts) if self.concepts else 0,
            "avg_relations_per_concept": total_relations / len(self.concepts) if self.concepts else 0,
        }
