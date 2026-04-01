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
from .sense_resolver import SenseResolver

logger = logging.getLogger(__name__)

@dataclass
class GroundedConcept:
    concept_id: str
    term: str
    confidence: float
    source: str # 'exact', 'fuzzy', 'synonym'
    sense_id: Optional[str] = None

class ConceptGrounder:
    """
    Connects raw input text/vectors to specific Graph Nodes.
    
    This is the "Symbolic Bridge" in the neuro-symbolic architecture.
    It resolves "dog", "Dog", and "canine" to the same node ID "dog".
    """

    def __init__(
        self,
        graph_store: RelationalGraphStore,
        *,
        language_model: Any = None,
        enable_lm_rerank: bool = False,
        lm_rerank_min_ambiguity: float = 0.12,
        lm_rerank_weight: float = 0.25,
    ):
        self.graph = graph_store
        self.sense_resolver = SenseResolver(
            graph_store,
            language_model=language_model,
            enable_lm_rerank=enable_lm_rerank,
            lm_rerank_min_ambiguity=lm_rerank_min_ambiguity,
            lm_rerank_weight=lm_rerank_weight,
        )
        # Cache of (id, term) for fuzzy matching
        self._term_cache: List[Tuple[str, str]] = []
        self._term_cache_dirty = True
        self._last_trace: Dict[str, Any] = {}
        # Restrict grounding to semantically meaningful node types.
        self.allowed_grounding_types = [
            "concept",
            "learned_concept",
            "entity",
            "word",
            "lexeme",
            "sense",
        ]
        self.excluded_grounding_types = [
            "lexical_token",
            "utterance_parsed",
            "symbolic_frame",
            "pos_pattern",
            "syntax_pattern",
            "fragment",
            "pattern",
        ]

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
        self._last_trace = {
            "query": text,
            "threshold": threshold,
            "tenant_id": tenant_id,
            "raw_matches": [],
            "resolved_candidates": [],
            "selected": [],
        }
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
            self._last_trace["raw_matches"].append({
                "node_id": full_match.concept_id,
                "term": full_match.term,
                "confidence": full_match.confidence,
                "source": full_match.source,
            })
        
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
                self._last_trace["raw_matches"].append({
                    "node_id": match.concept_id,
                    "term": match.term,
                    "confidence": match.confidence,
                    "source": match.source,
                })
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
                        self._last_trace["raw_matches"].append({
                            "node_id": match_syn.concept_id,
                            "term": match_syn.term,
                            "confidence": match_syn.confidence,
                            "source": match_syn.source,
                        })
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
                      self._last_trace["raw_matches"].append({
                          "node_id": ids[idx],
                          "term": term,
                          "confidence": 1.0,
                          "source": "exact_phrase",
                      })
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
                          self._last_trace["raw_matches"].append({
                              "node_id": ids[idx],
                              "term": term,
                              "confidence": ratio / 100.0,
                              "source": "fuzzy_phrase",
                          })

        # 4. Resolve through sense layer to canonical concept IDs when available.
        found_concepts = self._resolve_matches(found_concepts, context_query=text)

        # Post-processing: Filter out low-confidence matches that shouldn't trigger responses
        # This prevents single-word pattern triggers from hijacking unrelated conversations
        min_useful_confidence = 0.6
        found_concepts = [c for c in found_concepts if c.confidence >= min_useful_confidence]
        self._last_trace["selected"] = [
            {
                "concept_id": c.concept_id,
                "sense_id": c.sense_id,
                "confidence": c.confidence,
                "source": c.source,
            }
            for c in found_concepts
        ]
        
        return found_concepts

    def _resolve_matches(self, matches: List[GroundedConcept], context_query: Optional[str] = None) -> List[GroundedConcept]:
        """Resolve lexeme/sense hits into canonical concept IDs with score propagation."""
        resolved: List[GroundedConcept] = []

        for m in matches:
            candidates = self.sense_resolver.resolve(
                m.concept_id,
                base_confidence=m.confidence,
                context_query=context_query,
            )
            ambiguous = self.sense_resolver.is_ambiguous(candidates)
            for c in candidates:
                source = m.source if c.path == "direct" else f"{m.source}:{c.path}"
                if ambiguous and len(candidates) > 1:
                    source = f"{source}:ambiguous"
                self._last_trace["resolved_candidates"].append(
                    {
                        "raw_node_id": m.concept_id,
                        "concept_id": c.concept_id,
                        "sense_id": c.sense_id,
                        "path": c.path,
                        "structural_score": c.structural_score,
                        "lm_score": c.lm_score,
                        "fused_score": c.fused_score if c.fused_score is not None else c.confidence,
                        "ambiguous": ambiguous,
                    }
                )
                resolved.append(
                    GroundedConcept(
                        concept_id=c.concept_id,
                        term=m.term,
                        confidence=c.confidence,
                        source=source,
                        sense_id=c.sense_id,
                    )
                )

        # Deduplicate by concept id, keep highest confidence.
        best: Dict[str, GroundedConcept] = {}
        for r in resolved:
            current = best.get(r.concept_id)
            if current is None or r.confidence > current.confidence:
                best[r.concept_id] = r

        return sorted(best.values(), key=lambda x: x.confidence, reverse=True)

    def get_last_trace(self) -> Dict[str, Any]:
        """Return structured trace for the most recent grounding call."""
        return dict(self._last_trace)

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
            # Prefer type-filtered term retrieval when available.
            try:
                method = getattr(self.graph, 'get_terms_by_types', None)
                if callable(method):
                    self._term_cache = method(
                        allowed_types=self.allowed_grounding_types,
                        excluded_types=self.excluded_grounding_types,
                        tenant_id=tenant_id,
                    )
                else:
                    self._term_cache = self.graph.get_all_terms(tenant_id=tenant_id)
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
