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
import torch
import torch.nn.functional as F

# Import PMFlow retrieval extensions
try:
    from pmflow.core.retrieval import CompositionalRetrievalPMField, SemanticNeighborhoodPMField
    PMFLOW_EXTENSIONS_AVAILABLE = True
except ImportError:
    PMFLOW_EXTENSIONS_AVAILABLE = False


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
        
        # Initialize PMFlow retrieval extensions if available
        if PMFLOW_EXTENSIONS_AVAILABLE and hasattr(semantic_encoder, 'pm_field'):
            self.compositional_retrieval = CompositionalRetrievalPMField(
                semantic_encoder.pm_field
            )
            self.neighborhood = SemanticNeighborhoodPMField(
                semantic_encoder.pm_field
            )
            print("  ✨ PMFlow retrieval extensions enabled (query expansion + hierarchical + attention)")
        else:
            self.compositional_retrieval = None
            self.neighborhood = None
        
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
        min_similarity: float = 0.60,
        use_expansion: bool = True,
        use_hierarchical: bool = True
    ) -> List[Tuple[SemanticConcept, float]]:
        """
        Retrieve concepts by semantic similarity.
        
        Args:
            query_embedding: Query embedding from BNN
            top_k: Maximum number to retrieve
            min_similarity: Minimum similarity threshold
            use_expansion: Use query expansion (requires query_text via retrieve_by_text)
            use_hierarchical: Use hierarchical filtering (requires query_text via retrieve_by_text)
            
        Returns:
            List of (concept, similarity_score) tuples
            
        Note: PMFlow extensions require working in latent space. Use retrieve_by_text()
              for full PMFlow enhancement support.
        """
        if not self.concepts:
            return []
        
        # Use manual retrieval (PMFlow extensions need latent space access)
        return self._manual_retrieve_similar(query_embedding, top_k, min_similarity)
    
    def retrieve_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        min_similarity: float = 0.60,
        use_expansion: bool = True,
        use_hierarchical: bool = True
    ) -> List[Tuple[SemanticConcept, float]]:
        """
        Retrieve concepts by query text (enables PMFlow enhancements).
        
        Args:
            query_text: Query text to search for
            top_k: Maximum number to retrieve
            min_similarity: Minimum similarity threshold
            use_expansion: Use query expansion for synonym matching
            use_hierarchical: Use hierarchical filtering for speed
            
        Returns:
            List of (concept, similarity_score) tuples
        """
        if not self.concepts:
            return []
        
        # Try PMFlow-enhanced retrieval first
        if self.compositional_retrieval is not None:
            return self._pmflow_retrieve_by_text(
                query_text, top_k, min_similarity,
                use_expansion, use_hierarchical
            )
        
        # Fallback: encode and use manual retrieval
        tokens = query_text.lower().split()
        query_embedding = self.encoder.encode(tokens)
        if hasattr(query_embedding, 'cpu'):
            query_embedding = query_embedding.cpu().detach().numpy().flatten()
        
        return self._manual_retrieve_similar(query_embedding, top_k, min_similarity)
    
    def _pmflow_retrieve_by_text(
        self,
        query_text: str,
        top_k: int,
        min_similarity: float,
        use_expansion: bool,
        use_hierarchical: bool
    ) -> List[Tuple[SemanticConcept, float]]:
        """Enhanced retrieval using PMFlow compositional extensions."""
        
        # Encode query in latent space
        tokens = query_text.lower().split()
        
        # Get base embedding and project to latent
        base_emb = self.encoder.base_encoder.encode(tokens).to(self.encoder.device)
        query_latent = base_emb @ self.encoder._projection  # (1, latent_dim)
        
        # Prepare concept latents
        concept_ids = list(self.concepts.keys())
        concept_latents = []
        
        for cid in concept_ids:
            concept = self.concepts[cid]
            term_tokens = concept.term.lower().split()
            term_base = self.encoder.base_encoder.encode(term_tokens).to(self.encoder.device)
            term_latent = term_base @ self.encoder._projection
            concept_latents.append(term_latent)
        
        # Stack into tensor
        concept_tensor = torch.cat(concept_latents, dim=0)  # (N, latent_dim)
        
        # Use compositional retrieval
        results = self.compositional_retrieval.retrieve_concepts(
            query_latent,
            concept_tensor,
            expand_query=use_expansion,
            use_hierarchical=use_hierarchical,
            min_similarity=min_similarity
        )
        
        # Convert back to (concept, score) tuples
        concept_results = []
        for idx, score in results[:top_k]:
            concept = self.concepts[concept_ids[idx]]
            concept.usage_count += 1
            concept_results.append((concept, score))
        
        return concept_results
    
    def _manual_retrieve_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        min_similarity: float
    ) -> List[Tuple[SemanticConcept, float]]:
        """Fallback: Manual O(N) similarity calculation."""
        
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
    
    def merge_similar_concepts(self, threshold: float = 0.85) -> int:
        """
        Consolidate concepts that are very similar.
        
        Args:
            threshold: Similarity threshold for merging (default 0.85 based on PoC findings)
            
        Returns:
            Number of concepts merged
        """
        # Try PMFlow-enhanced consolidation first
        if self.neighborhood is not None:
            return self._pmflow_merge_concepts(threshold)
        
        # Fallback: Manual O(N²) merging
        return self._manual_merge_concepts(threshold)
    
    def _pmflow_merge_concepts(self, threshold: float) -> int:
        """Enhanced consolidation using semantic neighborhood (field signatures)."""
        
        merged_count = 0
        concepts_to_remove = set()
        
        # Prepare concept latent embeddings (in PMFlow space)
        concept_ids = list(self.concepts.keys())
        concept_latents = []
        
        for cid in concept_ids:
            concept = self.concepts[cid]
            term_tokens = concept.term.lower().split()
            # Get latent representation (before final projection)
            term_base = self.encoder.base_encoder.encode(term_tokens).to(self.encoder.device)
            term_latent = term_base @ self.encoder._projection
            concept_latents.append(term_latent)
        
        concept_tensor = torch.cat(concept_latents, dim=0)  # (N, latent_dim)
        
        # For each concept, find neighbors using field signatures
        for i, cid in enumerate(concept_ids):
            if cid in concepts_to_remove:
                continue
            
            concept_z = concept_latents[i]  # (1, latent_dim)
            
            # Find neighbors with similar gravitational field signatures
            neighbor_indices, scores = self.neighborhood.find_neighbors(
                concept_z,
                concept_tensor,
                threshold=threshold
            )
            
            # Merge neighbors into this concept
            for j, score in zip(neighbor_indices.tolist(), scores.tolist()):
                if j <= i:  # Skip self and already processed
                    continue
                
                neighbor_cid = concept_ids[j]
                if neighbor_cid in concepts_to_remove:
                    continue
                
                # Merge neighbor into current concept
                print(f"  🔗 Merging '{self.concepts[neighbor_cid].term}' "
                      f"into '{self.concepts[cid].term}' "
                      f"(field similarity: {score:.3f})")
                
                self._merge_concept_data(
                    self.concepts[cid],
                    self.concepts[neighbor_cid]
                )
                
                concepts_to_remove.add(neighbor_cid)
                merged_count += 1
        
        # Remove merged concepts
        for cid in concepts_to_remove:
            del self.concepts[cid]
        
        if merged_count > 0:
            self._save_concepts()
        
        return merged_count
    
    def _manual_merge_concepts(self, threshold: float) -> int:
        """Fallback: Brute force O(N²) consolidation."""
        
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
                    embedding = self.encoder.encode(concept_a.term.split())
                    if hasattr(embedding, 'cpu'):
                        embedding = embedding.cpu().detach().numpy().flatten()
                    concept_a.embedding = embedding
                if concept_b.embedding is None:
                    embedding = self.encoder.encode(concept_b.term.split())
                    if hasattr(embedding, 'cpu'):
                        embedding = embedding.cpu().detach().numpy().flatten()
                    concept_b.embedding = embedding
                
                similarity = self._cosine_similarity(concept_a.embedding, concept_b.embedding)
                
                if similarity >= threshold:
                    # Merge B into A
                    print(f"  🔗 Merging '{concept_b.term}' into '{concept_a.term}' (similarity: {similarity:.3f})")
                    
                    self._merge_concept_data(concept_a, concept_b)
                    
                    # Mark for removal
                    concepts_to_remove.add(concept_b.concept_id)
                    merged_count += 1
        
        # Remove merged concepts
        for concept_id in concepts_to_remove:
            del self.concepts[concept_id]
        
        if merged_count > 0:
            self._save_concepts()
        
        return merged_count
    
    def _merge_concept_data(self, target: SemanticConcept, source: SemanticConcept):
        """Merge source concept data into target concept."""
        
        # Combine properties
        for prop in source.properties:
            if prop not in target.properties:
                target.properties.append(prop)
        
        # Combine relations
        for rel in source.relations:
            existing = [r for r in target.relations 
                      if r.relation_type == rel.relation_type 
                      and r.target == rel.target]
            if not existing:
                target.relations.append(rel)
        
        # Update metadata
        target.usage_count += source.usage_count
        target.confidence = max(target.confidence, source.confidence)
    
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
