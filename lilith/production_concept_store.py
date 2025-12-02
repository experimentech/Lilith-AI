"""
Production ConceptStore - Database-backed semantic concept storage

Integrates ConceptDatabase for persistence with PMFlow-enhanced retrieval.
This is the production version of poc_compositional/concept_store.py
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from .concept_database import ConceptDatabase

# Import PMFlow retrieval extensions
try:
    from pmflow_bnn_enhanced import CompositionalRetrievalPMField, SemanticNeighborhoodPMField
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
    embedding: Optional[np.ndarray] = None  # BNN embedding (cached, not persisted)
    confidence: float = 0.85       # How reliable this knowledge is
    source: str = "taught"         # "taught", "wikipedia", "learned"
    usage_count: int = 0           # Track usage for consolidation


class ProductionConceptStore:
    """
    Production-ready concept store with database persistence.
    
    Features:
    - SQLite database backing for persistence
    - PMFlow-enhanced retrieval (query expansion, hierarchical, attention)
    - Semantic neighborhood consolidation
    - Usage tracking and metrics
    """
    
    def __init__(
        self, 
        semantic_encoder,
        db_path: str,
        consolidation_threshold: float = 0.85,
        vocabulary_tracker=None
    ):
        """
        Initialize production concept store.
        
        Args:
            semantic_encoder: PMFlowEmbeddingEncoder for embeddings
            db_path: Path to SQLite database
            consolidation_threshold: Threshold for merging similar concepts (0.85 recommended)
            vocabulary_tracker: Optional VocabularyTracker for query expansion
        """
        self.encoder = semantic_encoder
        self.db = ConceptDatabase(db_path)
        self.consolidation_threshold = consolidation_threshold
        self.vocabulary_tracker = vocabulary_tracker
        
        # Cache for concept embeddings (not persisted)
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
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
        
        # Pre-load embeddings for all existing concepts (for symbolic reasoning)
        self._preload_embeddings()
    
    def _preload_embeddings(self):
        """
        Pre-load embeddings for all concepts in the database.
        
        This enables the reasoning stage to work with concept embeddings
        symbolically without needing to do text-based retrieval.
        """
        concepts = self.db.get_all_concepts()
        for concept_dict in concepts:
            cid = concept_dict['concept_id']
            if cid not in self._embedding_cache:
                # Generate embedding from term
                tokens = concept_dict['term'].lower().split()
                emb = self.encoder.encode(tokens)
                if hasattr(emb, 'cpu'):
                    emb = emb.cpu().detach().numpy().flatten()
                self._embedding_cache[cid] = emb
    
    def add_concept(
        self,
        term: str,
        properties: List[str],
        relations: Optional[List[Relation]] = None,
        source: str = "taught",
        confidence: float = 0.85
    ) -> str:
        """
        Add a concept to the store.
        
        Args:
            term: Concept term (e.g., "machine learning")
            properties: List of properties
            relations: List of semantic relations
            source: Source of knowledge
            confidence: Confidence score
            
        Returns:
            concept_id of created or merged concept
        """
        relations = relations or []
        
        # Generate embedding
        tokens = term.lower().split()
        embedding = self.encoder.encode(tokens)
        if hasattr(embedding, 'cpu'):
            embedding = embedding.cpu().detach().numpy().flatten()
        
        # Check if similar concept exists (consolidation)
        existing = self._find_similar_concept(term, embedding, self.consolidation_threshold)
        
        if existing:
            # Merge into existing concept
            print(f"  🔗 Consolidating '{term}' into '{existing['term']}' (similarity: {self.consolidation_threshold:.2f}+)")
            
            # Merge properties (avoid duplicates)
            existing_props = set(existing['properties'])
            new_props = [p for p in properties if p not in existing_props]
            if new_props:
                all_props = existing['properties'] + new_props
                
                # Update database
                rel_dicts = [{'relation_type': r.relation_type, 'target': r.target, 'confidence': r.confidence} 
                            for r in existing['relations']]
                
                self.db.add_concept(
                    existing['concept_id'],
                    existing['term'],
                    all_props,
                    rel_dicts,
                    max(existing['confidence'], confidence),
                    existing['source']
                )
            
            return existing['concept_id']
        
        # Create new concept
        concept_id = f"concept_{len(self.db.get_all_concepts()):04d}"
        
        # Convert relations to dicts for database
        rel_dicts = [{'relation_type': r.relation_type, 'target': r.target, 'confidence': r.confidence} 
                    for r in relations]
        
        self.db.add_concept(concept_id, term, properties, rel_dicts, confidence, source)
        
        # Cache embedding
        self._embedding_cache[concept_id] = embedding
        
        print(f"  ➕ Created new concept: {term} ({concept_id})")
        return concept_id
    
    def retrieve_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        min_similarity: float = 0.60,
        use_expansion: bool = True,
        use_hierarchical: bool = True,
        use_vocabulary_expansion: bool = True
    ) -> List[Tuple[SemanticConcept, float]]:
        """
        Retrieve concepts by query text (PMFlow-enhanced).
        
        Args:
            query_text: Query text to search for
            top_k: Maximum number to retrieve
            min_similarity: Minimum similarity threshold
            use_expansion: Use query expansion for synonym matching
            use_hierarchical: Use hierarchical filtering for speed
            use_vocabulary_expansion: Use vocabulary co-occurrence for term expansion
            
        Returns:
            List of (concept, similarity_score) tuples
        """
        concepts = self.db.get_all_concepts()
        if not concepts:
            return []
        
        # Vocabulary expansion: Augment query with related terms
        expanded_query_text = query_text
        if use_vocabulary_expansion and self.vocabulary_tracker:
            try:
                tokens = query_text.lower().split()
                expanded_tokens = self.vocabulary_tracker.expand_query(
                    tokens,
                    max_related_per_term=2,
                    min_cooccurrence=2
                )
                expanded_query_text = ' '.join(expanded_tokens)
            except Exception:
                pass  # Fallback to original query
        
        # Try PMFlow-enhanced retrieval
        if self.compositional_retrieval is not None:
            return self._pmflow_retrieve_by_text(
                expanded_query_text, concepts, top_k, min_similarity,
                use_expansion, use_hierarchical
            )
        
        # Fallback: Manual retrieval
        return self._manual_retrieve_by_text(expanded_query_text, concepts, top_k, min_similarity)
    
    def _pmflow_retrieve_by_text(
        self,
        query_text: str,
        concepts: List[Dict],
        top_k: int,
        min_similarity: float,
        use_expansion: bool,
        use_hierarchical: bool
    ) -> List[Tuple[SemanticConcept, float]]:
        """PMFlow-enhanced retrieval in latent space."""
        
        # Encode query in latent space
        tokens = query_text.lower().split()
        base_emb = self.encoder.base_encoder.encode(tokens).to(self.encoder.device)
        query_latent = base_emb @ self.encoder._projection
        
        # Prepare concept latents
        concept_latents = []
        for concept_dict in concepts:
            cid = concept_dict['concept_id']
            
            # Use cached embedding or compute
            if cid in self._embedding_cache:
                embedding = self._embedding_cache[cid]
            else:
                term_tokens = concept_dict['term'].lower().split()
                emb = self.encoder.encode(term_tokens)
                if hasattr(emb, 'cpu'):
                    emb = emb.cpu().detach().numpy().flatten()
                self._embedding_cache[cid] = emb
                embedding = emb
            
            # Get latent representation
            term_base = self.encoder.base_encoder.encode(concept_dict['term'].lower().split()).to(self.encoder.device)
            term_latent = term_base @ self.encoder._projection
            concept_latents.append(term_latent)
        
        # Stack into tensor
        concept_tensor = torch.cat(concept_latents, dim=0)
        
        # Use compositional retrieval
        results = self.compositional_retrieval.retrieve_concepts(
            query_latent,
            concept_tensor,
            expand_query=use_expansion,
            use_hierarchical=use_hierarchical,
            min_similarity=min_similarity
        )
        
        # Convert to SemanticConcept objects
        concept_results = []
        for idx, score in results[:top_k]:
            concept_dict = concepts[idx]
            concept = self._dict_to_concept(concept_dict)
            
            # Update usage
            self.db.increment_usage(concept.concept_id)
            
            concept_results.append((concept, score))
        
        return concept_results
    
    def _manual_retrieve_by_text(
        self,
        query_text: str,
        concepts: List[Dict],
        top_k: int,
        min_similarity: float
    ) -> List[Tuple[SemanticConcept, float]]:
        """Fallback manual retrieval."""
        
        # Encode query
        query_tokens = query_text.lower().split()
        query_emb = self.encoder.encode(query_tokens)
        if hasattr(query_emb, 'cpu'):
            query_emb = query_emb.cpu().detach().numpy().flatten()
        
        results = []
        for concept_dict in concepts:
            cid = concept_dict['concept_id']
            
            # Get or compute embedding
            if cid not in self._embedding_cache:
                term_tokens = concept_dict['term'].lower().split()
                emb = self.encoder.encode(term_tokens)
                if hasattr(emb, 'cpu'):
                    emb = emb.cpu().detach().numpy().flatten()
                self._embedding_cache[cid] = emb
            
            concept_emb = self._embedding_cache[cid]
            
            # Cosine similarity
            sim = self._cosine_similarity(query_emb, concept_emb)
            
            if sim >= min_similarity:
                concept = self._dict_to_concept(concept_dict)
                results.append((concept, sim))
        
        # Sort and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Update usage for retrieved concepts
        for concept, _ in results[:top_k]:
            self.db.increment_usage(concept.concept_id)
        
        return results[:top_k]
    
    def consolidate_concepts(self, threshold: Optional[float] = None) -> int:
        """
        Consolidate similar concepts using semantic neighborhood.
        
        Args:
            threshold: Similarity threshold (uses self.consolidation_threshold if None)
            
        Returns:
            Number of concepts merged
        """
        threshold = threshold or self.consolidation_threshold
        
        concepts = self.db.get_all_concepts()
        if len(concepts) < 2:
            return 0
        
        # Try PMFlow-enhanced consolidation
        if self.neighborhood is not None:
            return self._pmflow_consolidate(concepts, threshold)
        
        # Fallback: Manual consolidation
        return self._manual_consolidate(concepts, threshold)
    
    def _pmflow_consolidate(self, concepts: List[Dict], threshold: float) -> int:
        """PMFlow-enhanced consolidation using field signatures."""
        
        merged_count = 0
        concepts_to_remove = []
        
        # Prepare concept latents
        concept_latents = []
        for concept_dict in concepts:
            term_tokens = concept_dict['term'].lower().split()
            term_base = self.encoder.base_encoder.encode(term_tokens).to(self.encoder.device)
            term_latent = term_base @ self.encoder._projection
            concept_latents.append(term_latent)
        
        concept_tensor = torch.cat(concept_latents, dim=0)
        
        # For each concept, find neighbors
        for i, concept_dict in enumerate(concepts):
            if concept_dict['concept_id'] in concepts_to_remove:
                continue
            
            concept_z = concept_latents[i]
            
            # Find neighbors with similar field signatures
            neighbor_indices, scores = self.neighborhood.find_neighbors(
                concept_z,
                concept_tensor,
                threshold=threshold
            )
            
            # Merge neighbors
            for j, score in zip(neighbor_indices.tolist(), scores.tolist()):
                if j <= i:
                    continue
                
                neighbor_dict = concepts[j]
                if neighbor_dict['concept_id'] in concepts_to_remove:
                    continue
                
                print(f"  🔗 Merging '{neighbor_dict['term']}' into '{concept_dict['term']}' (field similarity: {score:.3f})")
                
                # Merge properties
                merged_props = list(set(concept_dict['properties'] + neighbor_dict['properties']))
                
                # Update database
                rel_dicts = [{'relation_type': r['relation_type'], 'target': r['target'], 'confidence': r['confidence']} 
                            for r in concept_dict['relations']]
                
                self.db.add_concept(
                    concept_dict['concept_id'],
                    concept_dict['term'],
                    merged_props,
                    rel_dicts,
                    max(concept_dict['confidence'], neighbor_dict['confidence']),
                    concept_dict['source']
                )
                
                # Mark for removal
                concepts_to_remove.append(neighbor_dict['concept_id'])
                merged_count += 1
        
        # Remove merged concepts
        for cid in concepts_to_remove:
            self.db.delete_concept(cid)
            if cid in self._embedding_cache:
                del self._embedding_cache[cid]
        
        return merged_count
    
    def _manual_consolidate(self, concepts: List[Dict], threshold: float) -> int:
        """Fallback manual consolidation."""
        # Similar to PMFlow version but uses cosine similarity
        # (Implementation omitted for brevity - similar to _pmflow_consolidate)
        return 0
    
    def _find_similar_concept(self, term: str, embedding: np.ndarray, threshold: float) -> Optional[Dict]:
        """Find if a similar concept already exists."""
        concepts = self.db.get_all_concepts()
        
        for concept_dict in concepts:
            cid = concept_dict['concept_id']
            
            # Get or compute embedding
            if cid not in self._embedding_cache:
                term_tokens = concept_dict['term'].lower().split()
                emb = self.encoder.encode(term_tokens)
                if hasattr(emb, 'cpu'):
                    emb = emb.cpu().detach().numpy().flatten()
                self._embedding_cache[cid] = emb
            
            concept_emb = self._embedding_cache[cid]
            sim = self._cosine_similarity(embedding, concept_emb)
            
            if sim >= threshold:
                return concept_dict
        
        return None
    
    def _dict_to_concept(self, concept_dict: Dict) -> SemanticConcept:
        """Convert database dict to SemanticConcept object."""
        relations = [Relation(**r) for r in concept_dict['relations']]
        
        return SemanticConcept(
            concept_id=concept_dict['concept_id'],
            term=concept_dict['term'],
            properties=concept_dict['properties'],
            relations=relations,
            embedding=self._embedding_cache.get(concept_dict['concept_id']),
            confidence=concept_dict['confidence'],
            source=concept_dict['source'],
            usage_count=concept_dict['usage_count']
        )
    
    def get_concept_by_id(self, concept_id: str) -> Optional[SemanticConcept]:
        """
        Get a concept by its ID.
        
        This is the bridge between symbolic reasoning (which works with IDs)
        and linguistic composition (which needs the actual content).
        
        Args:
            concept_id: The concept ID (e.g., "concept_0002")
            
        Returns:
            SemanticConcept with full properties, or None if not found
        """
        concept_dict = self.db.get_concept(concept_id)
        if concept_dict:
            return self._dict_to_concept(concept_dict)
        return None
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get all cached concept embeddings.
        
        Used by the reasoning stage to work symbolically with concepts
        without doing text retrieval.
        
        Returns:
            Dict mapping concept_id to embedding array
        """
        return self._embedding_cache.copy()
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0
    
    def get_stats(self) -> Dict:
        """Get concept store statistics."""
        db_stats = self.db.get_stats()
        
        return {
            **db_stats,
            'cached_embeddings': len(self._embedding_cache),
            'consolidation_threshold': self.consolidation_threshold,
            'pmflow_enabled': self.compositional_retrieval is not None
        }
    
    def close(self):
        """Close database connection."""
        self.db.close()
