"""
Multi-tenant storage layer for Lilith.

Provides user-isolated databases with base knowledge fallback.
Users can learn and store patterns without corrupting base knowledge.
"""

from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np

from .response_fragments_sqlite import ResponseFragmentStoreSQLite as ResponseFragmentStore, ResponsePattern
from .user_auth import UserIdentity, get_user_data_path

# Optional: Import concept store for semantic knowledge extraction
try:
    from .production_concept_store import ProductionConceptStore
    CONCEPT_STORE_AVAILABLE = True
except ImportError:
    CONCEPT_STORE_AVAILABLE = False

# Optional: Import concept taxonomy for entity type hierarchy
try:
    from .concept_taxonomy import ConceptTaxonomy
    TAXONOMY_AVAILABLE = True
except ImportError:
    TAXONOMY_AVAILABLE = False

# Optional: Import vocabulary tracker
try:
    from .vocabulary_tracker import VocabularyTracker
    VOCABULARY_TRACKER_AVAILABLE = True
except ImportError:
    VOCABULARY_TRACKER_AVAILABLE = False


class MultiTenantFragmentStore:
    """
    Multi-tenant response fragment store.
    
    Maintains separate databases:
    - Base store: Teacher knowledge (read-only for users)
    - User store: User-specific learning (read-write for user)
    
    Lookup strategy:
    1. Check user store first (personalized knowledge)
    2. Fall back to base store (shared knowledge)
    
    Writing strategy:
    - Teacher mode: Write to base store
    - User mode: Write only to user store (base protected)
    """
    
    def __init__(
        self,
        encoder,
        user_identity: UserIdentity,
        base_data_path: str = "data",
        enable_fuzzy_matching: bool = True,
        enable_concept_store: bool = True
    ):
        """
        Initialize multi-tenant fragment store.
        
        Args:
            encoder: Semantic encoder for embeddings
            user_identity: User identity information
            base_data_path: Base data directory path
            enable_fuzzy_matching: Enable fuzzy matching
            enable_concept_store: Enable semantic concept extraction
        """
        self.user_identity = user_identity
        self.encoder = encoder
        
        # Base store (teacher knowledge) - using SQLite now
        base_path = Path(base_data_path) / "base" / "response_patterns.db"
        self.base_store = ResponseFragmentStore(
            encoder,
            str(base_path),
            enable_fuzzy_matching=enable_fuzzy_matching
        )
        
        # User store (if not teacher)
        if user_identity.is_teacher():
            # Teacher mode: user_store is same as base_store
            self.user_store = None
            print(f"  👨‍🏫 Teacher mode: Writing to base knowledge")
        else:
            # User mode: separate user database (SQLite)
            # Users start with empty storage, no bootstrap
            user_data_path = get_user_data_path(user_identity, base_data_path)
            user_path = Path(user_data_path) / "response_patterns.db"
            
            self.user_store = ResponseFragmentStore(
                encoder,
                str(user_path),
                enable_fuzzy_matching=enable_fuzzy_matching,
                bootstrap_if_empty=False  # Users start empty
            )
            
            # Note: SQLite stores are self-contained, no manual save needed
            # New users automatically start with empty database
            
            print(f"  👤 User mode: {user_identity.display_name} (isolated storage)")
        
        # Concept store (if enabled)
        if enable_concept_store and CONCEPT_STORE_AVAILABLE:
            if user_identity.is_teacher():
                # Teacher: use base concept store
                concept_db_path = str(Path(base_data_path) / "base" / "concepts.db")
            else:
                # User: isolated concept store
                user_data_path = get_user_data_path(user_identity, base_data_path)
                concept_db_path = str(Path(user_data_path) / "concepts.db")
            
            self.concept_store = ProductionConceptStore(
                semantic_encoder=encoder,
                db_path=concept_db_path
            )
            print(f"  🧠 Concept store enabled: {concept_db_path}")
        else:
            self.concept_store = None
        
        # Concept taxonomy (if enabled)
        if TAXONOMY_AVAILABLE:
            self.taxonomy = ConceptTaxonomy()
            print(f"  📚 Concept taxonomy initialized")
        else:
            self.taxonomy = None
        
        # Vocabulary tracker (if enabled)
        if VOCABULARY_TRACKER_AVAILABLE:
            if user_identity.is_teacher():
                # Teacher: use base vocabulary
                vocab_db_path = str(Path(base_data_path) / "base" / "vocabulary.db")
            else:
                # User: isolated vocabulary
                user_data_path = get_user_data_path(user_identity, base_data_path)
                vocab_db_path = str(Path(user_data_path) / "vocabulary.db")
            
            self.vocabulary = VocabularyTracker(vocab_db_path)
            print(f"  📖 Vocabulary tracker enabled: {vocab_db_path}")
        else:
            self.vocabulary = None
    
    def retrieve_patterns(
        self,
        context: str,
        topk: int = 5,
        min_score: float = 0.7
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Retrieve best matching patterns across user and base stores.
        
        Strategy:
        1. Get matches from user store (personalized)
        2. Get matches from base store (shared knowledge)
        3. Merge and return top-k by confidence
        
        Args:
            context: Query text to match
            topk: Number of results to return
            min_score: Minimum confidence threshold
        
        Returns:
            List of (pattern, confidence) tuples, sorted by confidence
        """
        all_matches = []
        
        # Get matches from user store (if exists)
        if self.user_store:
            user_matches = self.user_store.retrieve_patterns(
                context,
                topk=topk,
                min_score=min_score
            )
            for pattern, conf in user_matches:
                all_matches.append((pattern, conf, "user"))
        
        # Get matches from base store
        base_matches = self.base_store.retrieve_patterns(
            context,
            topk=topk,
            min_score=min_score
        )
        for pattern, conf in base_matches:
            all_matches.append((pattern, conf, "base"))
        
        # Sort by confidence and take top-k
        all_matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = all_matches[:topk]
        
        # Debug info for best match
        if top_matches:
            _, _, source = top_matches[0]
            marker = "📘" if source == "base" else "📗"
            print(f"     {marker} Retrieved from {source} knowledge")
        
        # Return without source marker
        return [(pattern, conf) for pattern, conf, _ in top_matches]
    
    def add_pattern(
        self,
        trigger_context: str,
        response_text: str,
        success_score: float = 0.5,
        intent: str = "general"
    ) -> str:
        """
        Learn a new response pattern.
        
        Writing strategy:
        - Teacher mode: Write to base store
        - User mode: Write to user store only (base protected)
        
        Args:
            trigger_context: Input context that triggers response
            response_text: Response text
            success_score: Success rating (0.0-1.0)
            intent: Intent category
        
        Returns:
            Fragment ID of learned pattern
        """
        if self.user_identity.is_teacher():
            # Teacher writes to base
            return self.base_store.add_pattern(
                trigger_context,
                response_text,
                success_score=success_score,
                intent=intent
            )
        else:
            # User writes to their own store
            if not self.user_store:
                raise RuntimeError("User store not initialized")
            
            return self.user_store.add_pattern(
                trigger_context,
                response_text,
                success_score=success_score,
                intent=intent
            )
    
    def get_all_patterns(self) -> List[ResponsePattern]:
        """
        Get all patterns (user + base).
        
        Returns:
            List of all patterns user can access
        """
        patterns = []
        
        # User patterns first
        if self.user_store:
            patterns.extend(self.user_store.patterns.values())
        
        # Base patterns
        patterns.extend(self.base_store.patterns.values())
        
        return patterns
    
    def get_pattern_count(self) -> dict:
        """
        Get pattern counts by source.
        
        Returns:
            Dict with 'user', 'base', and 'total' counts
        """
        user_count = len(self.user_store.patterns) if self.user_store else 0
        base_count = len(self.base_store.patterns)
        
        return {
            'user': user_count,
            'base': base_count,
            'total': user_count + base_count
        }
    
    def clear_user_patterns(self):
        """
        Clear user's learned patterns (user mode only).
        Does NOT affect base knowledge.
        """
        if self.user_identity.is_teacher():
            raise PermissionError("Cannot clear patterns in teacher mode")
        
        if self.user_store:
            # Clear all patterns from SQLite database
            conn = self.user_store._get_connection()
            conn.execute("DELETE FROM response_patterns")
            conn.commit()
            conn.close()
            print(f"  🗑️  Cleared user patterns for {self.user_identity.display_name}")
    
    def reset_user_data(self, keep_backup: bool = True, bootstrap: bool = False):
        """
        Reset user's data to clean slate.
        
        Creates backup before reset. Only works in user mode.
        Teacher mode must manually manage base knowledge.
        
        Args:
            keep_backup: If True, create backup before reset
            bootstrap: If True, add seed patterns after reset (default: False for users)
            
        Returns:
            Path to backup file (if created)
        """
        if self.user_identity.is_teacher():
            raise PermissionError(
                "Cannot reset user data in teacher mode. "
                "Use base_store.reset_database() to reset base knowledge."
            )
        
        if not self.user_store:
            raise RuntimeError("User store not initialized")
        
        backup_path = self.user_store.reset_database(keep_backup=keep_backup, bootstrap=bootstrap)
        print(f"  🔄 Reset complete for {self.user_identity.display_name}")
        
        return backup_path
    
    def upvote(self, fragment_id: str, strength: float = 0.2):
        """
        Manually upvote a response (mark as helpful/accurate).
        
        Args:
            fragment_id: Pattern ID to upvote
            strength: Reward strength (default 0.2)
        """
        # Determine which store owns this pattern
        store = self._get_pattern_store(fragment_id)
        if store:
            store.upvote(fragment_id, strength)
    
    def downvote(self, fragment_id: str, strength: float = 0.3):
        """
        Manually downvote a response (mark as unhelpful/incorrect).
        
        Args:
            fragment_id: Pattern ID to downvote
            strength: Penalty strength (default 0.3)
        """
        store = self._get_pattern_store(fragment_id)
        if store:
            store.downvote(fragment_id, strength)
    
    def mark_helpful(self, fragment_id: str):
        """Mark response as helpful (moderate reward)."""
        store = self._get_pattern_store(fragment_id)
        if store:
            store.mark_helpful(fragment_id)
    
    def mark_not_helpful(self, fragment_id: str):
        """Mark response as not helpful (moderate penalty)."""
        store = self._get_pattern_store(fragment_id)
        if store:
            store.mark_not_helpful(fragment_id)
    
    def _get_pattern_store(self, fragment_id: str):
        """Find which store owns a pattern."""
        # Check user store first
        if self.user_store and fragment_id in self.user_store.patterns:
            return self.user_store
        
        # Check base store
        if fragment_id in self.base_store.patterns:
            # Can only update base patterns in teacher mode
            if self.user_identity.is_teacher():
                return self.base_store
            else:
                print(f"  ⚠️  Cannot modify base pattern in user mode: {fragment_id}")
                return None
        
        print(f"  ⚠️  Pattern not found: {fragment_id}")
        return None
    
    def learn_from_wikipedia(
        self,
        query: str,
        response_text: str,
        success_score: float = 0.8,
        intent: str = "learned_knowledge"
    ) -> str:
        """
        Learn from Wikipedia response with semantic concept extraction.
        
        This method:
        1. Stores the response as a pattern (normal learning)
        2. Extracts semantic concepts (if concept store enabled)
        3. Adds concepts to the concept store
        
        Args:
            query: Original user query
            response_text: Wikipedia response text
            success_score: Initial confidence (default 0.8)
            intent: Pattern intent (default "learned_knowledge")
            
        Returns:
            Pattern ID of the learned pattern
        """
        # 1. Store as normal pattern
        pattern_id = self.add_pattern(
            trigger_context=query,
            response_text=response_text,
            success_score=success_score,
            intent=intent
        )
        
        # 2. Extract and store concepts (if enabled)
        if self.concept_store:
            try:
                from .semantic_extractor import SemanticExtractor
                
                extractor = SemanticExtractor()
                concepts = extractor.extract_concepts(query, response_text)
                
                if concepts:
                    print(f"  🧠 Extracting semantic concepts...")
                    for concept in concepts:
                        # Convert to dict format
                        concept_dict = extractor.concept_to_dict(concept)
                        
                        # Add to concept store
                        concept_id = self.concept_store.add_concept(**concept_dict)
                        
                        print(f"     ✓ Learned concept: {concept.term}")
                        if concept.entity_type:
                            print(f"       - Entity type: {concept.entity_type}")
                        if concept.type_relations:
                            print(f"       - Type: {concept.type_relations[0]}")
                        if concept.properties:
                            print(f"       - Properties: {', '.join(concept.properties[:3])}")
                        
                        # Add to taxonomy (Phase B: Entity recognition)
                        if self.taxonomy and concept.entity_type:
                            self._add_to_taxonomy(concept)
                
            except Exception as e:
                print(f"  ⚠️  Concept extraction failed: {e}")
        
        # 3. Track vocabulary (Phase C: Vocabulary expansion)
        if self.vocabulary:
            try:
                print(f"  📖 Tracking vocabulary...")
                tracked = self.vocabulary.track_text(response_text, source="wikipedia")
                
                # Show technical terms found
                technical_terms = [term for term, entry in tracked.items() if entry.is_technical]
                if technical_terms:
                    print(f"     ✓ Tracked {len(tracked)} terms ({len(technical_terms)} technical)")
                    # Show top 3 technical terms
                    for term in list(technical_terms)[:3]:
                        print(f"       - {term}")
                
            except Exception as e:
                print(f"  ⚠️  Vocabulary tracking failed: {e}")
        
        return pattern_id
    
    def _add_to_taxonomy(self, concept):
        """
        Add extracted concept to taxonomy with inferred relationships.
        
        Args:
            concept: ExtractedConcept with entity_type and type_relations
        """
        if not self.taxonomy:
            return
        
        try:
            # Prepare parent types
            parents = set()
            
            # Add entity type as parent
            if concept.entity_type:
                parents.add(concept.entity_type)
            
            # Add type relations as parents (normalize to taxonomy format)
            for type_rel in concept.type_relations:
                # Convert "programming language" → "programming_language"
                normalized = type_rel.lower().replace(' ', '_').replace('-', '_')
                parents.add(normalized)
            
            # Add to taxonomy
            self.taxonomy.add_concept(
                name=concept.term.lower().replace(' ', '_'),
                parents=parents,
                properties=set(concept.properties)
            )
            
            print(f"       - Added to taxonomy with parents: {parents}")
            
        except Exception as e:
            print(f"       ⚠️  Taxonomy add failed: {e}")
    
    def get_vocabulary_stats(self) -> Optional[Dict]:
        """
        Get vocabulary statistics.
        
        Returns:
            Dictionary with vocabulary stats or None if not enabled
        """
        if not self.vocabulary:
            return None
        
        return self.vocabulary.get_vocabulary_stats()
