"""
Reasoning Stage - Physics-Based Deliberation for V2

Implements "thinking" as a physical process using PMFlow's agentic physics:
- Working memory with concept activation
- Deliberation via trajectory tracing
- Inference from concept convergence/divergence
- Intent injection for goal-directed reasoning

This is the layer between perception and response that enables Lilith to
THINK rather than just pattern-match.

Architecture:
    Input → Activate Concepts → Deliberate (trace trajectories) → Extract Inferences → Output

The key insight: PMFlow's gravitational field creates semantic structure.
Related concepts flow toward shared attractors. By tracing trajectories
and detecting convergence, we can:
1. Find implicit connections between concepts
2. Generate novel inferences
3. Resolve ambiguous queries
4. Combine partial information
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ActivatedConcept:
    """A concept held in working memory during reasoning."""
    concept_id: str
    term: str
    embedding: torch.Tensor
    activation: float  # 0.0 to 1.0
    source: str  # "query", "retrieved", "inferred", "context"
    properties: List[str] = field(default_factory=list)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)  # (relation_type, target_id, target_term)


@dataclass
class Inference:
    """An inference generated during deliberation."""
    inference_type: str  # "connection", "implication", "contradiction", "elaboration"
    source_concepts: List[str]  # Concept terms that led to this
    conclusion: str  # The inferred content
    confidence: float  # 0.0 to 1.0
    reasoning_path: List[str]  # Steps that led here


@dataclass
class DeliberationResult:
    """Result of a deliberation cycle."""
    activated_concepts: List[ActivatedConcept]
    inferences: List[Inference]
    focus_concept: Optional[str]  # Main concept to respond about
    resolved_intent: Optional[str]  # Clarified intent after reasoning
    deliberation_steps: int
    confidence: float  # Overall reasoning confidence
    trajectory_efficiency: float  # How direct was the path (1.0 = straight)
    mental_effort: float  # Path length (higher = more work)


class ReasoningStage:
    """
    Physics-based reasoning using PMFlow agentic capabilities.
    
    Key mechanisms:
    1. Working Memory: Activated concepts with attention decay
    2. Deliberation: Trace trajectories through concept space
    3. Inference: Detect convergence (connection) and divergence (contradiction)
    4. Intent: Inject goals to bias reasoning direction
    
    This creates genuine "thinking" where:
    - Concepts interact via gravitational attraction
    - Intent creates frame-dragging vortices
    - Trajectories represent chains of thought
    - Efficiency indicates confidence
    """
    
    def __init__(
        self,
        encoder: Any,  # PMFlowEmbeddingEncoder with enable_flow=True
        graph_store: Any = None,  # Optional RelationalGraphStore for concept retrieval
        max_working_memory: int = 7,  # Miller's magic number
        deliberation_steps: int = 10,
        convergence_threshold: float = 0.5,
        enable_intent_injection: bool = True,
    ):
        self.encoder = encoder
        self.graph = graph_store
        self.max_working_memory = max_working_memory
        self.deliberation_steps = deliberation_steps
        self.convergence_threshold = convergence_threshold
        self.enable_intent = enable_intent_injection
        
        # Check if encoder supports agentic physics
        self._has_flow = hasattr(encoder, 'enable_flow') and encoder.enable_flow
        if not self._has_flow:
            logger.warning("Encoder doesn't have enable_flow=True; agentic features limited")
        
        # Working memory: concept_id -> ActivatedConcept
        self._working_memory: Dict[str, ActivatedConcept] = {}
        
        # Attention weights for focus mechanism
        self._attention: Dict[str, float] = {}
        
        # Last deliberation result for context
        self._last_result: Optional[DeliberationResult] = None
        
        # Metrics
        self._total_deliberations = 0
        self._total_inferences = 0
    
    def clear_working_memory(self) -> None:
        """Clear working memory for a new reasoning session."""
        self._working_memory.clear()
        self._attention.clear()
    
    def decay_working_memory(self, decay_rate: float = 0.9) -> None:
        """Apply decay to working memory activations."""
        to_remove = []
        for cid, concept in self._working_memory.items():
            concept.activation *= decay_rate
            self._attention[cid] = concept.activation
            if concept.activation < 0.1:
                to_remove.append(cid)
        for cid in to_remove:
            del self._working_memory[cid]
            del self._attention[cid]
    
    def activate_concept(
        self,
        term: str,
        embedding: torch.Tensor,
        activation: float = 1.0,
        source: str = "retrieved",
        properties: Optional[List[str]] = None,
        relations: Optional[List[Tuple[str, str, str]]] = None,
        concept_id: Optional[str] = None,
    ) -> ActivatedConcept:
        """
        Activate a concept in working memory.
        
        Manages capacity by removing least-activated concepts when full.
        """
        cid = concept_id or f"active_{term.replace(' ', '_')}"
        
        # Ensure proper tensor shape
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        concept = ActivatedConcept(
            concept_id=cid,
            term=term,
            embedding=embedding,
            activation=activation,
            source=source,
            properties=properties or [],
            relations=relations or [],
        )
        
        # Capacity management
        if len(self._working_memory) >= self.max_working_memory and cid not in self._working_memory:
            # Remove least activated
            min_key = min(self._working_memory.keys(), key=lambda k: self._working_memory[k].activation)
            del self._working_memory[min_key]
            if min_key in self._attention:
                del self._attention[min_key]
        
        self._working_memory[cid] = concept
        self._attention[cid] = activation
        
        return concept
    
    def activate_from_query(self, query: str) -> List[ActivatedConcept]:
        """
        Parse query and activate relevant concepts in working memory.
        
        Uses the encoder to embed the query and find similar concepts
        in the graph store (if available).
        """
        activated = []
        
        # Encode and activate the query itself
        query_tokens = query.split()
        query_embedding = self.encoder.encode(query_tokens)
        
        query_concept = self.activate_concept(
            term=query,
            embedding=query_embedding,
            activation=1.0,
            source="query",
        )
        activated.append(query_concept)
        
        # Retrieve related concepts from graph if available
        if self.graph is not None:
            activated.extend(self._retrieve_related_concepts(query, query_embedding))
        
        return activated
    
    def _retrieve_related_concepts(
        self, 
        query: str, 
        query_embedding: torch.Tensor
    ) -> List[ActivatedConcept]:
        """Retrieve concepts related to query from graph store."""
        activated = []
        
        # Extract key terms from query
        key_terms = self._extract_key_terms(query)
        
        try:
            # Use graph's concept search if available
            if hasattr(self.graph, 'search_nodes'):
                for term in key_terms[:3]:  # Limit to avoid overload
                    results = self.graph.search_nodes(term, limit=2)
                    for node in results:
                        node_id = node.get('id', '')
                        node_term = node.get('term', term)
                        
                        # Encode the node term for embedding
                        node_embedding = self.encoder.encode(node_term.split())
                        
                        # Calculate similarity for activation strength
                        similarity = F.cosine_similarity(
                            query_embedding.flatten().unsqueeze(0),
                            node_embedding.flatten().unsqueeze(0)
                        ).item()
                        
                        if similarity > 0.3:
                            concept = self.activate_concept(
                                term=node_term,
                                embedding=node_embedding,
                                activation=max(0.3, min(1.0, similarity)),
                                source="retrieved",
                                concept_id=node_id,
                            )
                            activated.append(concept)
        except Exception as e:
            logger.debug(f"Failed to retrieve related concepts: {e}")
        
        return activated
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from a query for concept matching."""
        stop_words = {
            "what", "is", "a", "an", "the", "of", "to", "in", "for", "on", 
            "with", "it", "be", "are", "was", "were", "how", "why", "when",
            "who", "which", "that", "this", "do", "does", "did", "can", "could",
            "would", "should", "will", "tell", "me", "about", "please", "explain",
        }
        
        # Clean and split
        query_clean = query.lower().replace("?", "").replace(".", "").replace(",", "")
        words = query_clean.split()
        
        # Filter
        terms = [w for w in words if len(w) > 2 and w not in stop_words]
        
        # Add multi-word combinations for compound terms
        if len(terms) > 1:
            terms.append(" ".join(terms))
        
        return terms
    
    def deliberate(
        self,
        query: str,
        context: Optional[str] = None,
        goal: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> DeliberationResult:
        """
        Main reasoning method - deliberate on a query.
        
        This runs the full reasoning cycle:
        1. Activate concepts from query
        2. Optionally inject intent toward a goal
        3. Trace trajectories through concept space
        4. Detect inferences from concept interactions
        5. Determine focus and resolved intent
        
        Returns:
            DeliberationResult with inferences, focus, and confidence metrics
        """
        steps = max_steps or self.deliberation_steps
        self._total_deliberations += 1
        
        # Step 1: Activate concepts
        self.clear_working_memory()
        activated = self.activate_from_query(query)
        
        # Also activate context if provided
        if context:
            context_embedding = self.encoder.encode(context.split())
            self.activate_concept(
                term=f"context: {context[:30]}...",
                embedding=context_embedding,
                activation=0.6,
                source="context",
            )
        
        # Step 2: Inject intent if goal provided and flow enabled
        if goal and self._has_flow and self.enable_intent:
            try:
                self.encoder.inject_intent(goal.split(), strength=0.4)
            except Exception as e:
                logger.debug(f"Intent injection failed: {e}")
        
        # Step 3: Trace trajectory and compute metrics
        trajectory_efficiency = 0.5
        mental_effort = 0.0
        
        if self._has_flow:
            try:
                trajectory, metrics = self.encoder.trace_trajectory(query.split(), steps=steps)
                trajectory_efficiency = metrics.get("efficiency", 0.5)
                mental_effort = metrics.get("path_length", 0.0)
            except Exception as e:
                logger.debug(f"Trajectory tracing failed: {e}")
        
        # Step 4: Run deliberation steps to detect inferences
        inferences = []
        for step in range(steps):
            step_inferences = self._deliberation_step(step)
            inferences.extend(step_inferences)
        
        self._total_inferences += len(inferences)
        
        # Step 5: Determine focus concept
        focus_concept = self._determine_focus()
        
        # Step 6: Resolve intent from reasoning
        resolved_intent = self._resolve_intent(query, inferences)
        
        # Step 7: Calculate overall confidence
        confidence = self._calculate_confidence(inferences, trajectory_efficiency)
        
        # Clear intent after reasoning
        if goal and self._has_flow:
            try:
                self.encoder.clear_intent()
            except Exception:
                pass
        
        result = DeliberationResult(
            activated_concepts=list(self._working_memory.values()),
            inferences=inferences,
            focus_concept=focus_concept,
            resolved_intent=resolved_intent,
            deliberation_steps=steps,
            confidence=confidence,
            trajectory_efficiency=trajectory_efficiency,
            mental_effort=mental_effort,
        )
        
        self._last_result = result
        return result
    
    def _deliberation_step(self, step_num: int) -> List[Inference]:
        """
        Single deliberation step - detect concept interactions.
        
        Compares concept embeddings to find connections and contradictions.
        """
        inferences = []
        
        if len(self._working_memory) < 2:
            return inferences
        
        concepts = list(self._working_memory.values())
        
        # Compare all pairs
        for i, concept_a in enumerate(concepts):
            for j, concept_b in enumerate(concepts):
                if i >= j:
                    continue
                
                # Compute similarity
                emb_a = concept_a.embedding.flatten()
                emb_b = concept_b.embedding.flatten()
                
                # Match dimensions
                min_dim = min(emb_a.shape[0], emb_b.shape[0])
                emb_a = emb_a[:min_dim]
                emb_b = emb_b[:min_dim]
                
                similarity = F.cosine_similarity(
                    emb_a.unsqueeze(0),
                    emb_b.unsqueeze(0)
                ).item()
                
                # Connection: concepts are semantically related
                if similarity > self.convergence_threshold:
                    inference = Inference(
                        inference_type="connection",
                        source_concepts=[concept_a.term, concept_b.term],
                        conclusion=f"{concept_a.term} is related to {concept_b.term}",
                        confidence=similarity,
                        reasoning_path=[f"step_{step_num}: convergence detected ({similarity:.2f})"],
                    )
                    inferences.append(inference)
                
                # Strong connection: implication
                if similarity > 0.7:
                    inference = Inference(
                        inference_type="implication",
                        source_concepts=[concept_a.term, concept_b.term],
                        conclusion=f"{concept_a.term} implies {concept_b.term}",
                        confidence=similarity,
                        reasoning_path=[f"step_{step_num}: strong convergence"],
                    )
                    inferences.append(inference)
                
                # Contradiction: opposite meanings
                if similarity < -0.1:
                    inference = Inference(
                        inference_type="contradiction",
                        source_concepts=[concept_a.term, concept_b.term],
                        conclusion=f"{concept_a.term} contradicts {concept_b.term}",
                        confidence=abs(similarity),
                        reasoning_path=[f"step_{step_num}: divergence detected"],
                    )
                    inferences.append(inference)
        
        return inferences
    
    def _determine_focus(self) -> Optional[str]:
        """Determine the main concept to focus response on."""
        if not self._attention:
            return None
        
        # Focus on highest attention non-query concept
        candidates = [
            (cid, weight) 
            for cid, weight in self._attention.items()
            if self._working_memory.get(cid, ActivatedConcept("","",torch.zeros(1),0,"")).source != "query"
        ]
        
        if not candidates:
            # Fall back to query concept
            return list(self._working_memory.values())[0].term if self._working_memory else None
        
        best_cid = max(candidates, key=lambda x: x[1])[0]
        return self._working_memory[best_cid].term
    
    def _resolve_intent(self, query: str, inferences: List[Inference]) -> Optional[str]:
        """Resolve the user's intent from query and inferences."""
        query_lower = query.lower()
        
        # Simple intent classification
        if any(q in query_lower for q in ["what is", "what are", "define", "explain"]):
            return "definition"
        elif any(q in query_lower for q in ["how do", "how to", "how can"]):
            return "procedure"
        elif any(q in query_lower for q in ["why", "reason"]):
            return "explanation"
        elif any(q in query_lower for q in ["compare", "difference", "vs"]):
            return "comparison"
        elif "?" in query:
            return "question"
        
        # Check inferences for clues
        for inf in inferences:
            if inf.inference_type == "implication" and inf.confidence > 0.8:
                return "elaboration"
        
        return "general"
    
    def _calculate_confidence(
        self, 
        inferences: List[Inference],
        trajectory_efficiency: float
    ) -> float:
        """Calculate overall reasoning confidence."""
        if not inferences:
            return trajectory_efficiency * 0.5
        
        # Average inference confidence
        avg_inference_conf = sum(inf.confidence for inf in inferences) / len(inferences)
        
        # Blend with trajectory efficiency
        confidence = 0.4 * avg_inference_conf + 0.6 * trajectory_efficiency
        
        return min(1.0, max(0.0, confidence))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reasoning statistics."""
        return {
            "total_deliberations": self._total_deliberations,
            "total_inferences": self._total_inferences,
            "working_memory_size": len(self._working_memory),
            "has_flow": self._has_flow,
        }
    
    def get_last_result(self) -> Optional[DeliberationResult]:
        """Get the last deliberation result."""
        return self._last_result
