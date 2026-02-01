import logging
import random
import re
from typing import List, Dict, Any, Optional, Tuple

try:
    import nltk
except ImportError:
    nltk = None

from .pragmatic_system import PragmaticSystem

logger = logging.getLogger(__name__)

class GenerativeSystem:
    """
    The 'Broca's Area' of Lilith v2.
    
    Responsibilities:
    1. Pattern Acquisition: Learning sentence structures from input (Grammar).
    2. Response Composition: Synthesizing thoughts into text (Output).
    3. Style Modulation: Adjusting output based on Affective State.
    """

    def __init__(self, graph_store: Any):
        self.graph = graph_store
        self.pragmatics = PragmaticSystem()
        # Simple cache of templates: Predicate -> List[Template Strings]
        # e.g. "is_a" -> ["{subject} is a type of {object}", "{subject}s are {object}s"]
        self._template_cache: Dict[str, List[str]] = {}
        self._hydrate_templates()
        
        self._ensure_nltk_resources()

    def _ensure_nltk_resources(self):
        """Ensure required NLTK data is available."""
        if nltk:
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger_eng')
            except LookupError:
                try:
                    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
                except Exception as e:
                    logger.warning(f"Failed to download NLTK tagger: {e}")

    def _hydrate_templates(self):
        """Load known patterns from graph store."""
        # In a real system, we'd query the graph for nodes of type 'syntax_pattern'.
        # For bootstrapping, we add some defaults if missing, or load what's there.
        # This simulates "learning" being persistent.
        
        # 1. Try to load from graph
        # Assuming schema: Node(type='syntax_pattern', term='{subject} is a {object}', data={'predicate': 'is_a'})
        # This is a simplification; v1 had a complex fragment store.
        pass

    def learn_grammar(self, text: str, extracted_relations: List[Any]) -> None:
        """
        Observe input text and valid extracted relations to learn HOW to express those relations.
        
        Example:
          Text: "A Beagle is a type of dog."
          Relation: (Beagle, Dog, is_a_type_of)
          Learned Template: "A {subject} is a type of {object}."
        """
        for rel in extracted_relations:
            # Simple generalization strategy:
            # Replace the literal subject/object in the text with placeholders.
            # We use case-insensitive replacement for the pattern.
            
            pattern = text
            
            # Escape regex meta characters in subject/object for safety
            subj_re = re.escape(rel.subject)
            obj_re = re.escape(rel.object)
            
            # Replace longest first to avoid partial matches
            # (Very naive implementation of "anti-unification")
            pattern = re.sub(f"(?i){subj_re}", "{subject}", pattern)
            pattern = re.sub(f"(?i){obj_re}", "{object}", pattern)
            
            # If we successfully generalized both, store it
            if "{subject}" in pattern and "{object}" in pattern:
                # OPTIONAL: Extract POS info if NLTK is available
                pos_seq = self._extract_pos_sequence(text) if nltk else []
                
                self._persist_pattern(rel.predicate, pattern, pos_seq)
                logger.debug(f"Learned syntax pattern for '{rel.predicate}': {pattern} (POS: {pos_seq})")

    def _extract_pos_sequence(self, text: str) -> List[str]:
        """Extract Part-of-Speech tags for the raw text."""
        try:
            tokens = nltk.word_tokenize(text)
            tags = nltk.pos_tag(tokens)
            return [tag for word, tag in tags]
        except Exception:
            return []

    def _persist_pattern(self, predicate: str, template: str, pos_sequence: List[str] = None):
        """Save pattern to Graph Store so it's available for Output."""
        # Pattern ID: hash of template
        pattern_id = f"pattern_{abs(hash(template))}"
        
        data = {"predicate": predicate}
        if pos_sequence:
            data["pos_sequence"] = pos_sequence
        
        # Add to Graph
        self.graph.add_node(
            node_id=pattern_id, 
            node_type="syntax_pattern", 
            term=template, 
            confidence=1.0,
            data=data
        )
        
        # Add to local cache
        if predicate not in self._template_cache:
            self._template_cache[predicate] = []
        if template not in self._template_cache[predicate]:
            self._template_cache[predicate].append(template)

    def compose(self, thought_context: Dict[str, Any]) -> str:
        """
        Synthesize a response based on the cognitive state (Perception + Inference + Affect).
        """
        inference = thought_context.get("inference", [])
        mood = thought_context.get("affect", {}).get("mood", "neutral")
        pragmatic_intent = thought_context.get("pragmatic_intent")
        
        extracted = thought_context.get("extracted_knowledge", [])
        
        # Priority 0: Explicit Pragmatic Intent (Teaching)
        if pragmatic_intent == "teaching":
             # If we extracted the relation, acknowledge it specifically
            if extracted:
                rel = extracted[0]
                template = self.pragmatics.get_template("teaching", ["subject", "object"], intent="acknowledge_learning")
                if template:
                    return self.pragmatics.fill(template, {"subject": rel.subject, "object": rel.object})
            # Even if extraction failed/wasn't perfect, acknowledge the attempt
            return "I am updating my understanding based on what you said."

        # Strategy 1: Verbalize Inference (Explicit thought sharing)
        # If we found connections in the graph, speak them using learned patterns.
        if inference:
            # Pick the highest confidence path
            top_result = inference[0] # {'path_ids': ['A', 'B'], 'confidence': 0.9}
            
            # Use graph to resolve IDs to terms
            # This is a bit hacky, normally we'd have full node objects.
            # Assuming we can verbalize the edge between Path[0] and Path[1]
            if "path_ids" in top_result and len(top_result["path_ids"]) >= 2:
                src_id = top_result["path_ids"][0]
                tgt_id = top_result["path_ids"][1]
                
                # SPECIAL CASE: Pattern Trigger -> Response
                # If we traversed a pattern edge, the result IS the response node.
                # We need to peek at the graph to see what the node is.
                # Since we don't have direct graph access easily here for node content without ID lookup...
                # Actually we DO have self.graph
                try:
                    tgt_node = self.graph.get_node(tgt_id)
                    # Use MultiTenant support by checking if manager
                    if hasattr(self.graph, "get_node") and "tenant_id" in self.graph.get_node.__code__.co_varnames:
                         tgt_node = self.graph.get_node(tgt_id, tenant_id=thought_context.get("affect", {}).get("tenant_id")) # Tenant ID not directly in affect, but usually in ctx.
                    
                    if tgt_node and tgt_node.get("term"):
                        # Heuristic: If it's a long utterance, it's likely a response.
                        return tgt_node["term"]
                except Exception as e:
                    logger.debug(f"Failed to resolve response node: {e}")
                
            elif "subject" in top_result and "object" in top_result:
                # Direct traversal result from simpler graph stores
                # {'subject': 'ruby', 'predicate': 'is_a', 'object': 'red_gemstone'}
                
                # Use definition template
                tpl = self.pragmatics.get_template("definition", ["concept", "property"])
                if tpl:
                    return self.pragmatics.fill(tpl, {"concept": top_result["subject"], "property": top_result["object"]})
                
                # Fallback
                return f"{top_result['subject']} is related to {top_result['object']}."

        extracted = thought_context.get("extracted_knowledge", [])
        if extracted:
            # Reflection: "I understand that X is Y."
            rel = extracted[0]
            
            # Use Pragmatics for clearer "Teaching Acknowledgment"
            template = self.pragmatics.get_template("teaching", ["subject", "object"])
            if template:
                return self.pragmatics.fill(template, {"subject": rel.subject, "object": rel.object})

            # Fallback to grammar template
            template_str = self._select_template(rel.predicate, mood)
            return self._fill_template(template_str, rel.subject, rel.object)

        # Strategy 2: Verbalize External Knowledge (Augmentation)
        external_context = thought_context.get("external_knowledge", [])
        if external_context:
            # We found something externally (e.g. Wiki summary)
            # Summarize or repeat it.
            # Assuming external_context is a list of strings.
            return external_context[0]

        # Strategy 3: Fallback / Chit-chat
        # If we have absolutely nothing (no inference, no extraction), we are effectively "listening" or "confused".
        # We should NOT just greet, because that implies we didn't hear the question.
        
        # Check if input was a question? (Naive check)
        # For now, just return a fallback that signals we are waiting/listening.
        return "I am listening. Please provide more context."

    def _select_template(self, predicate: str, mood: Any) -> str:
        # Check cache or graph
        if predicate not in self._template_cache:
            # Fallback patterns
            defaults = {
                "is_a": "{subject} is {object}",
                "has_a": "{subject} has {object}",
                "part_of": "{subject} is part of {object}"
            }
            return defaults.get(predicate, "{subject} relates to {object}")
        
        options = self._template_cache[predicate]
        return random.choice(options)

    def _fill_template(self, template: str, subject: str, obj: str) -> str:
        # Formatting
        # Auto-capitalize first letter
        res = template.format(subject=subject, object=obj)
        res = res[0].upper() + res[1:]
        return res
