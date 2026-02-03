import logging
import re
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass

try:
    # RapidFuzz 3.x+
    from rapidfuzz import process, fuzz
    msg = "RapidFuzz available"
except ImportError:
    msg = "RapidFuzz not available"
    process = None
    fuzz = None

try:
    import nltk
    from nltk.corpus import wordnet as wn
    NLTK_AVAILABLE = True
except (ImportError, LookupError):
    NLTK_AVAILABLE = False

from .relational_graph_store import RelationalGraphStore

logger = logging.getLogger(__name__)

@dataclass
class GroundedConcept:
    concept_id: str
    term: str
    confidence: float
    source: str # 'exact', 'fuzzy', 'synonym'

class ConceptGrounder:
    """
    Connects raw input text/vectors to specific Graph Nodes.
    
    This is the "Symbolic Bridge" in the neuro-symbolic architecture.
    It resolves "dog", "Dog", and "canine" to the same node ID "dog".
    """

    def __init__(self, graph_store: RelationalGraphStore):
        self.graph = graph_store
        # Cache of (id, term) for fuzzy matching
        self._term_cache: List[Tuple[str, str]] = []
        self._term_cache_dirty = True

    def ground(self, text: str, threshold: float = 80.0, tenant_id: str = None) -> List[GroundedConcept]:
        """
        Identify concepts in the text that exist in the graph.
        
        Priority (highest first):
          1. Full input phrase matches (QA questions)
          2. Multi-word phrase matches
          3. Individual word matches (pattern triggers)
        """
        self._refresh_cache(tenant_id)
        
        found_concepts = []
        words = self._tokenize(text)
        
        # 0. PRIORITY: Full input text matching (for QA pairs)
        # Try to match the ENTIRE input against known question terms
        # This ensures "Do you like movies?" matches Q:"Do you like movies?" over trigger:"like"
        full_match = self._find_best_match(text.strip(), threshold=70.0)
        if full_match and full_match.confidence > 0.75:
            # Boost confidence for full-phrase matches
            full_match.confidence = min(1.0, full_match.confidence + 0.15)
            full_match.source = "full_phrase"
            found_concepts.append(full_match)
            # If we have a strong full match, we likely don't need token matches
            if full_match.confidence > 0.85:
                return found_concepts
        
        # 1. Exact & Fuzzy matching against Graph
        # We try to match each word (and bigrams?) against known terms.
        # For simplicity in v2, we scan known terms and see if they appear in text
        # OR we scan words and fuzzy match them to known terms.
        # Let's do the latter (Words -> Graph)
        
        for word in words:
            if len(word) < 3: continue # Skip 'a', 'is' (primitive stopword filter)
            
            # Exact/Fuzzy search in known terms
            match = self._find_best_match(word, threshold)
            if match:
                # Penalize very short single-word trigger matches
                # "like", "what", "how" shouldn't override full phrase matches
                if len(match.term) < 6 and " " not in match.term:
                    match.confidence *= 0.5  # Heavy penalty for single short words
                found_concepts.append(match)
            elif NLTK_AVAILABLE:
                # 2. Synonym expansion
                # If "canine" is not in graph, but "dog" is, and they are synonyms...
                synonyms = self._get_synonyms(word)
                for syn in synonyms:
                    match_syn = self._find_best_match(syn, threshold)
                    if match_syn:
                        # Found via synonym
                        # Penalize confidence slightly
                        match_syn.confidence *= 0.9 
                        match_syn.source = "synonym"
                        match_syn.term = word # Keep original term for context? No, map to ID
                        found_concepts.append(match_syn)
                        break # Only need one anchor per word usually
                        
        # 3. Handle multi-word concepts (computer science)
        # RapidFuzz extractOne might find "computer science" given "computer"
        # but the score would be low token sort ratio.
        # We need to scan bigrams or check full phrase match against text?
        # For prototype simplicity:
        # Check against full text (slow but works for "computer science")
        if process and self._term_cache and isinstance(self._term_cache, list) and len(self._term_cache) > 0:
             # Try to match the WHOLE text against concepts too?
             try:
                 ids, terms = zip(*self._term_cache)
             except (ValueError, TypeError):
                 # Empty or malformed cache
                 return found_concepts
             # This is expensive if we have 10k terms. For small graph in test it is fine.
             # Find concepts that appear in the text
             # Actually, better: if the text contains a concept term.
             # Let's iterate cache.
             for idx, term in enumerate(terms):
                 # If term is multi-word and present in text
                 if " " in term and term.lower() in text.lower():
                      found_concepts.append(GroundedConcept(
                        concept_id=ids[idx],
                        term=term,
                        confidence=1.0,
                        source="exact_phrase"
                    ))
                 # Or if fuzzy match of term is in text?
                 # "compputer science" in "I study compputer science"
                 elif " " in term:
                     ratio = fuzz.partial_ratio(term.lower(), text.lower())
                     if ratio > threshold:
                          found_concepts.append(GroundedConcept(
                            concept_id=ids[idx],
                            term=term,
                            confidence=ratio / 100.0,
                            source="fuzzy_phrase"
                        ))

        # Post-processing: Filter out low-confidence matches that shouldn't trigger responses
        # This prevents single-word pattern triggers from hijacking unrelated conversations
        min_useful_confidence = 0.6
        found_concepts = [c for c in found_concepts if c.confidence >= min_useful_confidence]
        
        return found_concepts

    def _find_best_match(self, query: str, threshold: float) -> Optional[GroundedConcept]:
        """Find best matching node in graph for a single word/term."""
        if not self._term_cache:
            return None
            
        # Unpack terms for rapidfuzz - guard against empty list
        try:
            ids, terms = zip(*self._term_cache)
        except ValueError:
            # Empty cache
            return None
        
        if process:
            # RapidFuzz extraction
            # extracts: (match_string, score, index)
            result = process.extractOne(query, terms, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
            
            if result:
                match_str, score, idx = result
                node_id = ids[idx]
                
                return GroundedConcept(
                    concept_id=node_id,
                    term=match_str,
                    confidence=score / 100.0,
                    source="fuzzy" if score < 100 else "exact"
                )
        else:
            # Simple exact matching fallback if no rapidfuzz
            # This is slow O(N) but functional
            for idx, term in enumerate(terms):
                if term.lower() == query.lower():
                     return GroundedConcept(
                        concept_id=ids[idx],
                        term=term,
                        confidence=1.0,
                        source="exact"
                    )
        return None

    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms from WordNet."""
        synonyms = set()
        try:
            for syn in wn.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().replace('_', ' '))
        except Exception:
            pass
        return list(synonyms)

    def _refresh_cache(self, tenant_id: str = None):
        """Reload terms from graph if needed (e.g. periodically)."""
        # In a real system, do this async or incrementally.
        # Here we just fetch all terms (assuming <10k for prototype).
        if self._term_cache_dirty:
            # Protocol check: does graph support tenant_id?
            # If so, pass it. Use try/except to handle mocks gracefully.
            try:
                method = getattr(self.graph, 'get_all_terms', None)
                if method and hasattr(method, '__code__') and 'tenant_id' in method.__code__.co_varnames:
                    self._term_cache = self.graph.get_all_terms(tenant_id=tenant_id)
                else:
                    self._term_cache = self.graph.get_all_terms()
            except (AttributeError, TypeError):
                # Fallback for mocks or objects without proper introspection
                self._term_cache = self.graph.get_all_terms()
            
            self._term_cache_dirty = False 
            # In production set dirty=True after every learn() call via callback?
            # For now we'll just pull fresh on every ground() call or rely on manual refresh.
            pass

    def _tokenize(self, text: str) -> List[str]:
        # Simple tokenizer
        clean = re.sub(r'[^\w\s]', '', text.lower())
        return clean.split()
