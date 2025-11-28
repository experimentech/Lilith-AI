"""
Reasoning Stage - BNN-based Deliberation and Inference

A cognitive layer that uses the PMFlow latent space for "thinking":
- Activates relevant concepts in working memory
- Lets the BNN evolve these concepts to find connections
- Generates inferences from concept convergence/divergence
- Resolves ambiguity through multi-step deliberation

This is the "missing layer" between retrieval and composition that enables
the system to THINK rather than just pattern-match.

Architecture:
    Query → Activate Concepts → Deliberation Steps → Inferences → Response

The key insight is that the PMFlow field has learned structure - semantically
related concepts will flow toward shared attractors. By running multiple
deliberation steps, we can:
1. Find implicit connections between concepts
2. Generate novel inferences
3. Resolve ambiguous queries
4. Combine partial information

Phase 3: Deliberative Reasoning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ActivatedConcept:
    """A concept activated in working memory during reasoning."""
    concept_id: str
    term: str
    embedding: torch.Tensor          # BNN embedding in latent space
    activation: float                # How strongly activated (0-1)
    source: str                      # "query", "retrieved", "inferred"
    properties: List[str] = field(default_factory=list)
    
    
@dataclass
class Inference:
    """An inference generated during deliberation."""
    inference_type: str              # "connection", "implication", "contradiction", "elaboration"
    source_concepts: List[str]       # Concepts that led to this inference
    conclusion: str                  # The inferred conclusion
    confidence: float                # How confident we are (0-1)
    reasoning_path: List[str]        # Steps that led here
    

@dataclass 
class DeliberationResult:
    """Result of a deliberation cycle."""
    activated_concepts: List[ActivatedConcept]
    inferences: List[Inference]
    focus_concept: Optional[str]     # Main concept to respond about
    resolved_intent: Optional[str]   # Clarified intent after reasoning
    deliberation_steps: int          # How many steps of "thought"
    confidence: float                # Overall confidence in reasoning
    cleaned_query: Optional[str] = None  # Query with filler phrases removed


class ReasoningStage:
    """
    BNN-based reasoning through latent space evolution.
    
    Key mechanisms:
    1. Working Memory: Active concepts with attention weights
    2. Deliberation: Multi-step PMFlow evolution to find connections
    3. Inference: Detect convergence (implies) and divergence (contradicts)
    4. Focus: Attention mechanism to select most relevant concepts
    
    This creates a form of "thinking" where:
    - Concepts interact in latent space
    - Related concepts flow toward each other
    - Novel connections emerge from evolution
    """
    
    def __init__(
        self,
        encoder,
        concept_store=None,
        max_working_memory: int = 7,  # Miller's magic number
        deliberation_steps: int = 5,
        convergence_threshold: float = 0.75,
        enable_chaining: bool = True
    ):
        """
        Initialize reasoning stage.
        
        Args:
            encoder: PMFlow encoder with latent space
            concept_store: Optional concept store for retrieval
            max_working_memory: Maximum concepts to hold active
            deliberation_steps: Number of evolution steps per deliberation
            convergence_threshold: Similarity threshold to detect connection
            enable_chaining: Enable inference chaining (A→B, B→C ⊢ A→C)
        """
        self.encoder = encoder
        self.concept_store = concept_store
        self.max_working_memory = max_working_memory
        self.deliberation_steps = deliberation_steps
        self.convergence_threshold = convergence_threshold
        self.enable_chaining = enable_chaining
        
        # Working memory - currently activated concepts
        self.working_memory: Dict[str, ActivatedConcept] = {}
        
        # Inference cache - recently generated inferences
        self.inference_cache: List[Inference] = []
        
        # Attention weights for focus
        self.attention_weights: Dict[str, float] = {}
        
        # Filler phrases to strip from queries
        self.filler_phrases = [
            "i mean", "i meant", "well", "so", "like", "um", "uh",
            "you know", "actually", "basically", "honestly", "literally",
            "just", "okay so", "ok so", "alright so", "anyway",
            "to be honest", "in fact", "the thing is"
        ]
        
        # Cleaned query cache (for retrieval)
        self.last_cleaned_query = None
        
        print("  🧠 Reasoning stage enabled (deliberative thinking)!")
    
    def clean_query(self, query: str) -> str:
        """
        Clean a query by stripping filler phrases.
        
        This resolves cases like "I mean, what can you do?" → "what can you do?"
        
        Args:
            query: Raw user query
            
        Returns:
            Cleaned query with filler phrases removed
        """
        cleaned = query.lower().strip()
        
        # Strip filler phrases from the beginning
        for filler in self.filler_phrases:
            if cleaned.startswith(filler):
                # Remove filler and any following comma/space
                cleaned = cleaned[len(filler):].lstrip(", ")
                
        # Also strip from middle (e.g., "what, like, can you do")
        for filler in self.filler_phrases:
            cleaned = cleaned.replace(f", {filler},", ",")
            cleaned = cleaned.replace(f", {filler} ", " ")
            
        # Restore original case pattern by using cleaned words
        # but keeping original capitalization where possible
        original_words = query.split()
        cleaned_words = cleaned.split()
        
        # Simple approach: if cleaned query is shorter, return it title-cased
        if len(cleaned_words) < len(original_words):
            # Capitalize first letter
            cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else cleaned
            
        # Cache for retrieval
        self.last_cleaned_query = cleaned
            
        return cleaned
        
    def clear_working_memory(self):
        """Clear working memory for new reasoning session."""
        self.working_memory.clear()
        self.attention_weights.clear()
        
    def activate_concept(
        self,
        term: str,
        embedding: torch.Tensor,
        activation: float = 1.0,
        source: str = "retrieved",
        properties: Optional[List[str]] = None
    ) -> ActivatedConcept:
        """
        Activate a concept in working memory.
        
        Args:
            term: Concept term
            embedding: BNN embedding
            activation: Initial activation strength
            source: Where this concept came from
            properties: Optional properties list
            
        Returns:
            ActivatedConcept instance
        """
        concept_id = f"active_{term.replace(' ', '_')}"
        
        # Ensure embedding is properly shaped
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
            
        concept = ActivatedConcept(
            concept_id=concept_id,
            term=term,
            embedding=embedding,
            activation=activation,
            source=source,
            properties=properties or []
        )
        
        # Add to working memory (with capacity limit)
        if len(self.working_memory) >= self.max_working_memory:
            # Remove least activated concept
            min_key = min(self.working_memory.keys(), 
                         key=lambda k: self.working_memory[k].activation)
            del self.working_memory[min_key]
            
        self.working_memory[concept_id] = concept
        self.attention_weights[concept_id] = activation
        
        return concept
        
    def activate_from_query(self, query: str) -> List[ActivatedConcept]:
        """
        Parse query and activate relevant concepts.
        
        This is the first step of reasoning - loading concepts into
        working memory based on the query.
        """
        # Clean the query first
        cleaned_query = self.clean_query(query)
        
        # Encode cleaned query
        query_embedding = self.encoder.encode(cleaned_query.split())
        if hasattr(query_embedding, 'numpy'):
            query_embedding = query_embedding
        
        activated = []
        
        # Activate the cleaned query as a concept
        query_concept = self.activate_concept(
            term=cleaned_query,
            embedding=query_embedding,
            activation=1.0,
            source="query"
        )
        activated.append(query_concept)
        
        # If we have a concept store, retrieve related concepts
        if self.concept_store is not None:
            try:
                # Retrieve related concepts
                related = self.concept_store.retrieve_concepts(cleaned_query, topk=5)
                for concept, score in related:
                    if score > 0.3:  # Reasonable relevance
                        # Get embedding for concept
                        concept_embedding = self.encoder.encode(concept.term.split())
                        act = self.activate_concept(
                            term=concept.term,
                            embedding=concept_embedding,
                            activation=score,
                            source="retrieved",
                            properties=concept.properties if hasattr(concept, 'properties') else []
                        )
                        activated.append(act)
            except Exception as e:
                logger.debug(f"Could not retrieve concepts: {e}")
                
        return activated
        
    def deliberate(
        self,
        query: str,
        context: Optional[str] = None,
        max_steps: Optional[int] = None
    ) -> DeliberationResult:
        """
        Main reasoning method - deliberate on a query.
        
        This runs the full reasoning cycle:
        1. Activate concepts from query
        2. Run deliberation steps (PMFlow evolution)
        3. Detect inferences from concept interactions
        4. Determine focus and resolved intent
        
        Args:
            query: User query to reason about
            context: Optional conversation context
            max_steps: Override deliberation steps
            
        Returns:
            DeliberationResult with inferences and focus
        """
        steps = max_steps or self.deliberation_steps
        
        # Step 1: Clear and activate concepts
        self.clear_working_memory()
        activated = self.activate_from_query(query)
        
        # Also activate from context if provided
        if context and context != query:
            context_embedding = self.encoder.encode(context.split())
            self.activate_concept(
                term="context:" + context[:50],
                embedding=context_embedding,
                activation=0.5,
                source="context"
            )
            
        # Step 2: Run deliberation steps
        inferences = []
        for step in range(steps):
            step_inferences = self._deliberation_step(step)
            inferences.extend(step_inferences)
            
            # Update attention based on new inferences
            self._update_attention(step_inferences)
            
        # Step 3: Determine focus concept
        focus_concept = self._determine_focus()
        
        # Step 4: Resolve intent from reasoning
        resolved_intent = self._resolve_intent(query, inferences)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(inferences)
        
        return DeliberationResult(
            activated_concepts=list(self.working_memory.values()),
            inferences=inferences,
            focus_concept=focus_concept,
            resolved_intent=resolved_intent,
            deliberation_steps=steps,
            confidence=confidence,
            cleaned_query=self.last_cleaned_query
        )
        
    def _deliberation_step(self, step_num: int) -> List[Inference]:
        """
        Single deliberation step - evolve concepts and detect interactions.
        
        Uses PMFlow evolution to let concepts "flow" in latent space,
        then detects when concepts converge (connection) or diverge.
        """
        inferences = []
        
        if len(self.working_memory) < 2:
            return inferences
            
        concepts = list(self.working_memory.values())
        
        # Stack embeddings for batch processing
        embeddings = torch.cat([c.embedding for c in concepts], dim=0)
        
        # Evolve through PMFlow
        try:
            if hasattr(self.encoder, 'pm_field'):
                with torch.no_grad():
                    # Project to latent space if needed
                    if hasattr(self.encoder, '_projection'):
                        latent = embeddings @ self.encoder._projection.to(embeddings.device)
                    else:
                        latent = embeddings
                        
                    # Single evolution step
                    evolved = self.encoder.pm_field(latent)
                    
                    # Normalize
                    evolved = F.normalize(evolved, dim=-1)
                    
        except Exception as e:
            logger.debug(f"PMFlow evolution failed: {e}")
            return inferences
            
        # Detect concept interactions
        for i, concept_a in enumerate(concepts):
            for j, concept_b in enumerate(concepts):
                if i >= j:
                    continue
                    
                # Compute similarity after evolution
                sim = F.cosine_similarity(
                    evolved[i:i+1], 
                    evolved[j:j+1]
                ).item()
                
                # Compute similarity before evolution
                orig_sim = F.cosine_similarity(
                    concept_a.embedding,
                    concept_b.embedding
                ).item()
                
                # Convergence: concepts moved closer together
                if sim > self.convergence_threshold and sim > orig_sim + 0.1:
                    inference = Inference(
                        inference_type="connection",
                        source_concepts=[concept_a.term, concept_b.term],
                        conclusion=f"{concept_a.term} is related to {concept_b.term}",
                        confidence=sim,
                        reasoning_path=[f"step_{step_num}: convergence detected"]
                    )
                    inferences.append(inference)
                    
                # Strong convergence: possible implication
                if sim > 0.9:
                    inference = Inference(
                        inference_type="implication",
                        source_concepts=[concept_a.term, concept_b.term],
                        conclusion=f"{concept_a.term} implies {concept_b.term}",
                        confidence=sim,
                        reasoning_path=[f"step_{step_num}: strong convergence"]
                    )
                    inferences.append(inference)
                    
                # Divergence: concepts moved apart
                if sim < 0.3 and sim < orig_sim - 0.1:
                    inference = Inference(
                        inference_type="contradiction",
                        source_concepts=[concept_a.term, concept_b.term],
                        conclusion=f"{concept_a.term} conflicts with {concept_b.term}",
                        confidence=1 - sim,
                        reasoning_path=[f"step_{step_num}: divergence detected"]
                    )
                    inferences.append(inference)
                    
        # Update embeddings with evolved versions
        for i, concept in enumerate(concepts):
            concept.embedding = evolved[i:i+1]
            
        return inferences
        
    def _update_attention(self, inferences: List[Inference]):
        """Update attention weights based on new inferences."""
        for inference in inferences:
            for concept_term in inference.source_concepts:
                # Find matching concept in working memory
                for concept_id, concept in self.working_memory.items():
                    if concept.term == concept_term:
                        # Boost attention for concepts involved in inferences
                        current = self.attention_weights.get(concept_id, 0.5)
                        self.attention_weights[concept_id] = min(1.0, current + 0.1)
                        
    def _determine_focus(self) -> Optional[str]:
        """Determine the main concept to focus response on."""
        if not self.working_memory:
            return None
            
        # Find concept with highest attention that's not the query itself
        best_concept = None
        best_attention = -1
        
        for concept_id, concept in self.working_memory.items():
            if concept.source == "query":
                continue  # Skip the query itself
            attention = self.attention_weights.get(concept_id, 0.5)
            if attention > best_attention:
                best_attention = attention
                best_concept = concept.term
                
        return best_concept
        
    def _resolve_intent(self, query: str, inferences: List[Inference]) -> Optional[str]:
        """
        Resolve the true intent of the query based on reasoning.
        
        This handles cases like "I mean, what can you do?" where the
        surface form doesn't match the true intent.
        """
        # Look for connection inferences that clarify intent
        for inference in inferences:
            if inference.inference_type == "connection":
                # Check if any source concept is an intent keyword
                intent_keywords = {
                    "capability": ["can", "able", "do", "help", "capabilities"],
                    "identity": ["name", "who", "are you", "yourself"],
                    "definition": ["what is", "define", "meaning"],
                    "explanation": ["how", "why", "explain"]
                }
                
                for intent, keywords in intent_keywords.items():
                    for concept in inference.source_concepts:
                        if any(kw in concept.lower() for kw in keywords):
                            return intent
                            
        # Default: extract from query
        query_lower = query.lower()
        if "can you" in query_lower or "what can" in query_lower or "capabilities" in query_lower:
            return "capability"
        elif "your name" in query_lower or "who are you" in query_lower:
            return "identity"
        elif "what is" in query_lower:
            return "definition"
        elif "how" in query_lower:
            return "explanation"
            
        return None
        
    def _calculate_confidence(self, inferences: List[Inference]) -> float:
        """Calculate overall confidence in the reasoning."""
        if not inferences:
            return 0.5  # Neutral confidence if no inferences
            
        # Average confidence of inferences
        avg_conf = sum(i.confidence for i in inferences) / len(inferences)
        
        # Boost for more inferences (more thinking = more confident)
        inference_boost = min(0.2, len(inferences) * 0.05)
        
        return min(1.0, avg_conf + inference_boost)
        
    def reason_about(
        self,
        query: str,
        retrieved_patterns: List[Tuple] = None,
        context: Optional[str] = None
    ) -> DeliberationResult:
        """
        Convenience method to reason about a query with retrieved patterns.
        
        This is the main entry point for the response composer to invoke
        reasoning before composition.
        
        Args:
            query: User query
            retrieved_patterns: Patterns already retrieved
            context: Conversation context
            
        Returns:
            DeliberationResult to guide composition
        """
        # Start deliberation
        result = self.deliberate(query, context)
        
        # If we have retrieved patterns, activate them too
        if retrieved_patterns:
            for pattern, score in retrieved_patterns[:3]:  # Top 3
                pattern_text = pattern.trigger_context if hasattr(pattern, 'trigger_context') else str(pattern)
                pattern_embedding = self.encoder.encode(pattern_text.split())
                self.activate_concept(
                    term=f"pattern:{pattern_text[:30]}",
                    embedding=pattern_embedding,
                    activation=score,
                    source="pattern"
                )
                
            # Run additional deliberation with patterns
            extra_inferences = []
            for _ in range(2):  # 2 more steps
                step_infs = self._deliberation_step(self.deliberation_steps)
                extra_inferences.extend(step_infs)
                
            result.inferences.extend(extra_inferences)
            result.confidence = self._calculate_confidence(result.inferences)
            
        return result
        
    def get_reasoning_summary(self, result: DeliberationResult) -> str:
        """Generate a human-readable summary of reasoning."""
        lines = []
        
        if result.inferences:
            lines.append(f"🧠 Deliberated for {result.deliberation_steps} steps")
            
            connections = [i for i in result.inferences if i.inference_type == "connection"]
            if connections:
                lines.append(f"  📎 Found {len(connections)} connections:")
                for conn in connections[:3]:
                    lines.append(f"     • {conn.conclusion}")
                    
            implications = [i for i in result.inferences if i.inference_type == "implication"]
            if implications:
                lines.append(f"  ➡️  Found {len(implications)} implications")
                
        if result.focus_concept:
            lines.append(f"  🎯 Focus: {result.focus_concept}")
            
        if result.resolved_intent:
            lines.append(f"  💡 Intent: {result.resolved_intent}")
            
        return "\n".join(lines) if lines else ""


# Demo / test
def demo():
    """Demo the reasoning stage."""
    print("=" * 60)
    print("Reasoning Stage Demo")
    print("=" * 60)
    print()
    
    try:
        from .embedding import PMFlowEmbeddingEncoder
        encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    except ImportError:
        print("Could not import encoder, skipping demo")
        return
        
    reasoning = ReasoningStage(encoder, deliberation_steps=3)
    
    # Test queries
    queries = [
        "What can you do?",
        "I mean, what are your capabilities?",
        "Tell me about machine learning",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        result = reasoning.deliberate(query)
        
        summary = reasoning.get_reasoning_summary(result)
        if summary:
            print(summary)
        else:
            print("  (no significant inferences)")
            
        print(f"  Confidence: {result.confidence:.2f}")
        

if __name__ == "__main__":
    demo()
