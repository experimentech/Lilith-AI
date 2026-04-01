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
class ConceptChain:
    """A chain of concepts discovered via graph traversal."""
    path: List[str]  # Concept IDs in order
    terms: List[str]  # Human-readable terms
    relations: List[str]  # Relation types between consecutive nodes
    confidence: float  # Overall chain confidence (weakest link)
    depth: int  # Chain length


@dataclass
class Inference:
    """An inference generated during deliberation."""
    inference_type: str  # "connection", "implication", "contradiction", "elaboration", "chain"
    source_concepts: List[str]  # Concept terms that led to this
    conclusion: str  # The inferred content
    confidence: float  # 0.0 to 1.0
    reasoning_path: List[str]  # Steps that led here
    chain: Optional[ConceptChain] = None  # For chain inferences


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
    concept_chains: List[ConceptChain] = field(default_factory=list)  # Discovered chains
    abstractions: List['Abstraction'] = field(default_factory=list)  # Formed abstractions


@dataclass
class Abstraction:
    """A higher-order concept formed from patterns in concept chains.
    
    Abstractions represent reusable knowledge structures:
    - Pattern: The relation sequence that defines this abstraction (e.g., "X causes Y, Y affects Z")
    - Instances: Specific concept chains that match this pattern
    - Name: A generated label for this abstraction
    
    Example: Repeated observations of "disease causes symptom, symptom indicates treatment"
    might form an abstraction called "diagnostic_pathway".
    """
    abstraction_id: str
    name: str  # Human-readable label
    pattern: List[str]  # Relation types that define this pattern
    instances: List[ConceptChain]  # Chains that match this pattern
    confidence: float  # How reliable is this abstraction
    occurrence_count: int  # How many times seen
    exemplar_terms: List[str]  # Representative concept terms
    created_from: str  # "deliberation", "consolidation", "teaching"


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
        
        # Current tenant context (set during deliberate())
        self._current_tenant: Optional[str] = None
        
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
        tenant_id = getattr(self, '_current_tenant', None)
        
        try:
            # Try to find matching nodes using available methods
            found_nodes = []
            
            # Method 1: Use find_nodes_by_term for substring matching
            if hasattr(self.graph, 'find_nodes_by_term'):
                for term in key_terms[:3]:  # Limit to avoid overload
                    results = self.graph.find_nodes_by_term(term, tenant_id=tenant_id)
                    for node in results:
                        found_nodes.append(node)
            
            # Method 2: Try exact node lookup by normalized term
            if hasattr(self.graph, 'get_node'):
                for term in key_terms[:3]:
                    normalized_id = term.lower().replace(" ", "_")
                    node = self.graph.get_node(normalized_id, tenant_id=tenant_id)
                    if node:
                        found_nodes.append(node)
            
            # Deduplicate by ID
            seen_ids = set()
            for node in found_nodes:
                node_id = node.get('id', '')
                if node_id in seen_ids:
                    continue
                seen_ids.add(node_id)
                
                node_term = node.get('term', node_id)
                
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
    
    def chain_concepts(
        self,
        start_concept_id: str,
        max_depth: int = 3,
        edge_types: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> List[ConceptChain]:
        """
        Discover concept chains by traversing the knowledge graph.
        
        Uses BFS from a starting concept to find paths to related concepts.
        This enables multi-hop reasoning: A -> B -> C discoveries.
        
        Args:
            start_concept_id: Node ID to start traversal from
            max_depth: Maximum chain length (default 3)
            edge_types: Optional filter for relation types (e.g., ["is_a", "causes"])
            tenant_id: Optional tenant context for multi-tenant graphs
            
        Returns:
            List of ConceptChains ordered by confidence
        """
        if self.graph is None:
            return []
        
        chains = []
        
        try:
            # Use graph's BFS traversal
            if hasattr(self.graph, 'traverse_bfs'):
                paths = self.graph.traverse_bfs(
                    start_concept_id, 
                    max_depth=max_depth,
                    edge_types=edge_types,
                    tenant_id=tenant_id,
                )
                
                for path_data in paths:
                    path_ids = path_data.get("path_ids", [])
                    confidence = path_data.get("confidence", 0.5)
                    
                    if len(path_ids) < 2:
                        continue
                    
                    # Resolve terms and relations
                    terms = []
                    relations = []
                    
                    for i, node_id in enumerate(path_ids):
                        node = self.graph.get_node(node_id, tenant_id=tenant_id)
                        term = node.get("term", node_id) if node else node_id
                        terms.append(term)
                        
                        # Get relation to next node
                        if i < len(path_ids) - 1:
                            related = self.graph.get_related(node_id, tenant_id=tenant_id)
                            for rel in related:
                                if rel.get("target", {}).get("id") == path_ids[i + 1]:
                                    relations.append(rel.get("relation", "related_to"))
                                    break
                            else:
                                relations.append("related_to")
                    
                    chain = ConceptChain(
                        path=path_ids,
                        terms=terms,
                        relations=relations,
                        confidence=confidence,
                        depth=len(path_ids),
                    )
                    chains.append(chain)
            
            # Also add direct neighbors as depth-2 chains
            if hasattr(self.graph, 'get_related'):
                related = self.graph.get_related(start_concept_id, tenant_id=tenant_id)
                start_node = self.graph.get_node(start_concept_id, tenant_id=tenant_id)
                start_term = start_node.get("term", start_concept_id) if start_node else start_concept_id
                
                for rel in related:
                    target = rel.get("target", {})
                    chain = ConceptChain(
                        path=[start_concept_id, target.get("id", "")],
                        terms=[start_term, target.get("term", "")],
                        relations=[rel.get("relation", "related_to")],
                        confidence=rel.get("confidence", 0.5),
                        depth=2,
                    )
                    chains.append(chain)
                    
        except Exception as e:
            logger.debug(f"Concept chaining failed: {e}")
        
        # Sort by confidence, then depth (prefer shorter high-confidence chains)
        chains.sort(key=lambda c: (c.confidence, -c.depth), reverse=True)
        return chains[:10]  # Limit to top 10 chains
    
    def _activate_chain_concepts(self, chains: List[ConceptChain]) -> List[ActivatedConcept]:
        """
        Activate intermediate concepts discovered through chaining.
        
        This enriches working memory with concepts found via graph traversal,
        enabling reasoning about indirect connections.
        """
        activated = []
        seen_terms = {c.term for c in self._working_memory.values()}
        
        for chain in chains:
            for i, (node_id, term) in enumerate(zip(chain.path, chain.terms)):
                if term in seen_terms:
                    continue
                    
                seen_terms.add(term)
                
                # Activation decays with chain depth
                base_activation = chain.confidence * (0.8 ** i)
                
                try:
                    embedding = self.encoder.encode(term.split())
                    concept = self.activate_concept(
                        term=term,
                        embedding=embedding,
                        activation=max(0.2, base_activation),
                        source="chained",
                        concept_id=node_id,
                        relations=[(chain.relations[j] if j < len(chain.relations) else "related_to", 
                                   chain.path[j+1] if j+1 < len(chain.path) else "",
                                   chain.terms[j+1] if j+1 < len(chain.terms) else "")
                                  for j in range(i, min(i+1, len(chain.relations)))],
                    )
                    activated.append(concept)
                except Exception as e:
                    logger.debug(f"Failed to activate chained concept {term}: {e}")
        
        return activated
    
    def _infer_from_chains(self, chains: List[ConceptChain]) -> List[Inference]:
        """
        Generate inferences from discovered concept chains.
        
        Chain-based inferences capture multi-hop reasoning:
        - A is_a B, B causes C -> A may cause C (transitivity)
        - A relates_to B, B relates_to C -> A indirectly connects to C
        """
        inferences = []
        
        for chain in chains:
            if len(chain.terms) < 2:
                continue
            
            # Build conclusion from chain
            start_term = chain.terms[0]
            end_term = chain.terms[-1]
            
            # Describe the reasoning path
            path_desc = []
            for i in range(len(chain.relations)):
                if i + 1 < len(chain.terms):
                    path_desc.append(f"{chain.terms[i]} --{chain.relations[i]}--> {chain.terms[i+1]}")
            
            # Generate chain inference
            if len(chain.terms) == 2:
                # Direct relationship
                conclusion = f"{start_term} {chain.relations[0]} {end_term}"
            else:
                # Multi-hop: summarize the connection
                conclusion = f"{start_term} connects to {end_term} via {' -> '.join(chain.terms[1:-1])}"
            
            inference = Inference(
                inference_type="chain",
                source_concepts=[start_term, end_term],
                conclusion=conclusion,
                confidence=chain.confidence,
                reasoning_path=path_desc,
                chain=chain,
            )
            inferences.append(inference)
            
            # Check for transitivity patterns
            if len(chain.terms) >= 3:
                # A is_a B, B is_a C -> A is_a C
                if all(r == "is_a" for r in chain.relations):
                    inference = Inference(
                        inference_type="implication",
                        source_concepts=chain.terms,
                        conclusion=f"{start_term} is_a {end_term} (transitive)",
                        confidence=chain.confidence * 0.9,
                        reasoning_path=path_desc + ["transitivity applied"],
                        chain=chain,
                    )
                    inferences.append(inference)
                
                # A causes B, B causes C -> A may cause C
                if all(r == "causes" for r in chain.relations):
                    inference = Inference(
                        inference_type="elaboration",
                        source_concepts=chain.terms,
                        conclusion=f"{start_term} may cause {end_term} (causal chain)",
                        confidence=chain.confidence * 0.8,
                        reasoning_path=path_desc + ["causal transitivity"],
                        chain=chain,
                    )
                    inferences.append(inference)
        
        return inferences

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from a query for concept matching."""
        stop_words = {
            # Question words/determiners
            "what", "is", "a", "an", "the", "of", "to", "in", "for", "on", 
            "with", "it", "be", "are", "was", "were", "how", "why", "when",
            "who", "which", "that", "this", "do", "does", "did", "can", "could",
            "would", "should", "will", "tell", "me", "about", "please", "explain",
            # Common verbs that don't make good concepts
            "know", "find", "think", "have", "has", "had", "make", "take",
            "said", "says", "went", "come", "came", "been", "going", "done",
            # Pronouns
            "you", "your", "i", "my", "we", "our", "they", "them", "their",
            # Other function words
            "yes", "yeah", "okay", "sure", "from", "here", "there",
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
        tenant_id: Optional[str] = None,
        consolidate: bool = False,
    ) -> DeliberationResult:
        """
        Main reasoning method - deliberate on a query.
        
        This runs the full reasoning cycle:
        1. Activate concepts from query
        2. Optionally inject intent toward a goal
        3. Trace trajectories through concept space
        4. Detect inferences from concept interactions
        5. Determine focus and resolved intent
        6. Optionally consolidate memory
        
        Args:
            query: The query to deliberate on
            context: Optional additional context
            goal: Optional goal for intent injection
            max_steps: Override default deliberation steps
            tenant_id: Tenant context for multi-tenant graphs
            consolidate: If True, run memory consolidation after deliberation
        
        Returns:
            DeliberationResult with inferences, focus, and confidence metrics
        """
        steps = max_steps or self.deliberation_steps
        self._total_deliberations += 1
        self._current_tenant = tenant_id  # Store for nested methods
        
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
        
        # Step 3b: Concept Chaining - traverse graph from activated concepts
        chain_inferences = []
        all_chains = []
        if self.graph is not None:
            for concept in list(self._working_memory.values()):
                if concept.source in ("query", "retrieved"):
                    chains = self.chain_concepts(
                        concept.concept_id, 
                        max_depth=3, 
                        tenant_id=self._current_tenant
                    )
                    if chains:
                        all_chains.extend(chains)
                        # Activate intermediate concepts
                        self._activate_chain_concepts(chains)
                        # Generate chain-based inferences
                        chain_inferences.extend(self._infer_from_chains(chains))
        
        # Step 4: Run deliberation steps to detect inferences
        inferences = []
        for step in range(steps):
            step_inferences = self._deliberation_step(step)
            inferences.extend(step_inferences)
        
        # Merge chain inferences
        inferences.extend(chain_inferences)
        
        self._total_inferences += len(inferences)
        
        # Step 5: Determine focus concept
        focus_concept = self._determine_focus()
        
        # Step 6: Resolve intent from reasoning
        resolved_intent = self._resolve_intent(query, inferences)
        
        # Step 7: Calculate overall confidence
        confidence = self._calculate_confidence(inferences, trajectory_efficiency)
        
        # Step 8: Form abstractions from discovered patterns
        abstractions = []
        if all_chains and len(all_chains) >= 2:
            abstractions = self.form_abstractions(
                all_chains, 
                min_occurrences=2,
                tenant_id=self._current_tenant
            )
            
            # Apply known abstractions to generate additional inferences
            for abstraction in abstractions:
                for concept in list(self._working_memory.values())[:3]:
                    applied_inferences = self.apply_abstraction(abstraction, concept)
                    inferences.extend(applied_inferences)
        
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
            concept_chains=all_chains,
            abstractions=abstractions,
        )
        
        self._last_result = result
        
        # Step 9: Memory consolidation (optional)
        if consolidate:
            consolidation_stats = self.consolidate_memory(result, tenant_id)
            logger.debug(f"Memory consolidated: {consolidation_stats}")
        
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
            "abstractions_formed": getattr(self, '_total_abstractions', 0),
        }
    
    def get_last_result(self) -> Optional[DeliberationResult]:
        """Get the last deliberation result."""
        return self._last_result
    
    # ============ Abstraction Layer ============
    
    def detect_patterns(
        self, 
        chains: List[ConceptChain],
        min_occurrences: int = 2,
    ) -> Dict[str, List[ConceptChain]]:
        """
        Detect recurring relation patterns in concept chains.
        
        A pattern is a sequence of relation types (e.g., ["causes", "affects"]).
        When the same pattern appears across multiple chains, it suggests
        a reusable knowledge structure.
        
        Args:
            chains: List of concept chains to analyze
            min_occurrences: Minimum times a pattern must appear to be considered
            
        Returns:
            Dict mapping pattern signature to list of matching chains
        """
        if not chains:
            return {}
        
        # Group chains by their relation pattern
        pattern_groups: Dict[str, List[ConceptChain]] = {}
        
        for chain in chains:
            if not chain.relations:
                continue
            
            # Create pattern signature from relations
            pattern_sig = "|".join(chain.relations)
            
            if pattern_sig not in pattern_groups:
                pattern_groups[pattern_sig] = []
            pattern_groups[pattern_sig].append(chain)
        
        # Filter to patterns with enough occurrences
        significant_patterns = {
            sig: chains 
            for sig, chains in pattern_groups.items() 
            if len(chains) >= min_occurrences
        }
        
        return significant_patterns
    
    def form_abstractions(
        self,
        chains: List[ConceptChain],
        min_occurrences: int = 2,
        tenant_id: Optional[str] = None,
    ) -> List[Abstraction]:
        """
        Form higher-order abstractions from patterns in concept chains.
        
        This is the core of the abstraction layer - it identifies recurring
        patterns and creates named abstractions that can be reused in future
        reasoning.
        
        Args:
            chains: Concept chains from deliberation
            min_occurrences: Minimum pattern frequency to form abstraction
            tenant_id: Tenant context for persistence
            
        Returns:
            List of newly formed abstractions
        """
        abstractions = []
        
        # Detect patterns
        patterns = self.detect_patterns(chains, min_occurrences)
        
        for pattern_sig, matching_chains in patterns.items():
            relations = pattern_sig.split("|")
            
            # Generate a readable name for the abstraction
            name = self._generate_abstraction_name(relations, matching_chains)
            
            # Extract exemplar terms from the chains
            exemplar_terms = []
            for chain in matching_chains[:3]:  # Take up to 3 exemplars
                if chain.terms:
                    exemplar_terms.extend(chain.terms[:2])
            exemplar_terms = list(dict.fromkeys(exemplar_terms))[:5]  # Dedupe, limit to 5
            
            # Calculate confidence based on occurrence and chain confidence
            avg_chain_confidence = sum(c.confidence for c in matching_chains) / len(matching_chains)
            occurrence_boost = min(0.3, len(matching_chains) * 0.05)  # More occurrences = more confident
            confidence = min(1.0, avg_chain_confidence + occurrence_boost)
            
            # Create abstraction
            abstraction = Abstraction(
                abstraction_id=f"abs_{hash(pattern_sig) % 10**8:08x}",
                name=name,
                pattern=relations,
                instances=matching_chains,
                confidence=confidence,
                occurrence_count=len(matching_chains),
                exemplar_terms=exemplar_terms,
                created_from="deliberation",
            )
            abstractions.append(abstraction)
            
            # Persist to graph if available
            if self.graph is not None:
                self._persist_abstraction(abstraction, tenant_id)
        
        # Update stats
        if not hasattr(self, '_total_abstractions'):
            self._total_abstractions = 0
        self._total_abstractions += len(abstractions)
        
        return abstractions
    
    def _generate_abstraction_name(
        self, 
        relations: List[str], 
        chains: List[ConceptChain]
    ) -> str:
        """Generate a human-readable name for an abstraction."""
        # Common pattern names
        pattern_names = {
            ("is_a",): "type_hierarchy",
            ("causes",): "causal_link",
            ("requires",): "dependency",
            ("contains",): "composition",
            ("produces",): "production",
            ("enables",): "enablement",
            ("causes", "affects"): "causal_chain",
            ("requires", "produces"): "process_flow",
            ("is_a", "is_a"): "inheritance_chain",
            ("contains", "contains"): "nested_composition",
            ("produces", "enables"): "capability_chain",
            ("occurs_in", "contains"): "location_structure",
        }
        
        rel_tuple = tuple(relations)
        if rel_tuple in pattern_names:
            return pattern_names[rel_tuple]
        
        # Generate from relations
        if len(relations) == 1:
            return f"{relations[0]}_pattern"
        else:
            return f"{'_to_'.join(relations[:2])}_pathway"
    
    def _persist_abstraction(
        self, 
        abstraction: Abstraction,
        tenant_id: Optional[str] = None
    ) -> None:
        """Persist an abstraction to the graph store."""
        if self.graph is None:
            return
        
        try:
            # Store as a special node type
            data = {
                "pattern": abstraction.pattern,
                "exemplar_terms": abstraction.exemplar_terms,
                "occurrence_count": abstraction.occurrence_count,
                "created_from": abstraction.created_from,
            }
            
            self.graph.add_node(
                abstraction.abstraction_id,
                "abstraction",
                abstraction.name,
                confidence=abstraction.confidence,
                data=data,
                tenant_id=tenant_id,
            )
            
            # Link to exemplar concepts
            for term in abstraction.exemplar_terms[:3]:
                normalized_id = term.lower().replace(" ", "_")
                try:
                    self.graph.add_edge(
                        abstraction.abstraction_id,
                        normalized_id,
                        "abstracts",
                        confidence=abstraction.confidence * 0.8,
                        tenant_id=tenant_id,
                    )
                except Exception:
                    pass  # Exemplar node might not exist
            
            logger.debug(f"Persisted abstraction: {abstraction.name} ({abstraction.abstraction_id})")
            
        except Exception as e:
            logger.debug(f"Failed to persist abstraction: {e}")
    
    def retrieve_abstractions(
        self,
        pattern: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> List[Abstraction]:
        """
        Retrieve previously formed abstractions from the graph.
        
        Args:
            pattern: Optional relation pattern to filter by
            tenant_id: Tenant context
            
        Returns:
            List of matching abstractions
        """
        if self.graph is None:
            return []
        
        abstractions = []
        
        try:
            # Search for abstraction nodes
            if hasattr(self.graph, 'find_nodes_by_term'):
                # Get all abstraction-type nodes
                # (We'd need a more specific query, but this is a reasonable fallback)
                results = self.graph.find_nodes_by_term("pattern", tenant_id=tenant_id)
                
                for node in results:
                    if node.get("type") != "abstraction":
                        continue
                    
                    data = node.get("data", {})
                    if isinstance(data, str):
                        import json
                        data = json.loads(data)
                    
                    node_pattern = data.get("pattern", [])
                    
                    # Filter by pattern if specified
                    if pattern and node_pattern != pattern:
                        continue
                    
                    abstraction = Abstraction(
                        abstraction_id=node.get("id", ""),
                        name=node.get("term", ""),
                        pattern=node_pattern,
                        instances=[],  # Not stored per instance
                        confidence=node.get("confidence", 0.5),
                        occurrence_count=data.get("occurrence_count", 1),
                        exemplar_terms=data.get("exemplar_terms", []),
                        created_from=data.get("created_from", "unknown"),
                    )
                    abstractions.append(abstraction)
                    
        except Exception as e:
            logger.debug(f"Failed to retrieve abstractions: {e}")
        
        return abstractions
    
    def apply_abstraction(
        self,
        abstraction: Abstraction,
        seed_concept: ActivatedConcept,
    ) -> List[Inference]:
        """
        Apply an abstraction to generate inferences about a concept.
        
        When we have an abstraction like "X causes Y, Y affects Z" and we
        encounter a new X, we can infer the pattern might apply.
        
        Args:
            abstraction: The abstraction to apply
            seed_concept: The starting concept
            
        Returns:
            List of inferences generated from applying the abstraction
        """
        inferences = []
        
        # Generate an inference about the expected pattern
        if abstraction.pattern:
            expected_relations = " -> ".join(abstraction.pattern)
            
            inference = Inference(
                inference_type="elaboration",
                source_concepts=[seed_concept.term, abstraction.name],
                conclusion=f"Based on {abstraction.name} pattern, {seed_concept.term} may follow: {expected_relations}",
                confidence=abstraction.confidence * 0.7,  # Slightly lower for predictions
                reasoning_path=[
                    f"Applied abstraction: {abstraction.name}",
                    f"Pattern: {expected_relations}",
                    f"Seen {abstraction.occurrence_count} times before",
                ],
            )
            inferences.append(inference)
        
        return inferences
    
    # ============ Semantic Memory Consolidation ============
    
    def consolidate_memory(
        self,
        deliberation_result: Optional[DeliberationResult] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Consolidate short-term working memory into long-term storage.
        
        This is inspired by biological memory consolidation during sleep:
        - Compress frequently-activated concepts into abstract forms
        - Strengthen high-confidence connections
        - Decay weak connections
        - Merge redundant abstractions
        
        Args:
            deliberation_result: Optional result to consolidate (uses last if None)
            tenant_id: Tenant context
            
        Returns:
            Dict with consolidation statistics
        """
        result = deliberation_result or self._last_result
        if result is None:
            return {"status": "no_result", "consolidated": 0}
        
        stats = {
            "strengthened_connections": 0,
            "decayed_connections": 0,
            "merged_abstractions": 0,
            "new_attractors": 0,
        }
        
        # 1. Strengthen high-activation concepts
        for concept in result.activated_concepts:
            if concept.activation > 0.7 and self.graph is not None:
                self._strengthen_concept(concept, tenant_id)
                stats["strengthened_connections"] += 1
        
        # 2. Create attractors from high-confidence inferences
        for inference in result.inferences:
            if inference.confidence > 0.8:
                self._create_attractor_from_inference(inference, tenant_id)
                stats["new_attractors"] += 1
        
        # 3. Merge similar abstractions
        if result.abstractions and len(result.abstractions) > 1:
            merged = self._merge_similar_abstractions(result.abstractions, tenant_id)
            stats["merged_abstractions"] = merged
        
        # 4. Decay working memory for next session
        self.decay_working_memory(decay_rate=0.5)  # Aggressive decay after consolidation
        stats["decayed_connections"] = len(self._working_memory)
        
        return stats
    
    def _strengthen_concept(
        self,
        concept: ActivatedConcept,
        tenant_id: Optional[str] = None
    ) -> None:
        """
        Strengthen a concept's connections in long-term storage.
        
        When a concept has high activation, we boost its confidence
        and strengthen edges to related concepts.
        """
        if self.graph is None:
            return
        
        try:
            # Boost concept's own confidence
            node = self.graph.get_node(concept.concept_id, tenant_id=tenant_id)
            if node:
                new_confidence = min(1.0, node.get("confidence", 0.5) + 0.1)
                self.graph.add_node(
                    concept.concept_id,
                    node.get("type", "concept"),
                    node.get("term", concept.term),
                    confidence=new_confidence,
                    data=node.get("data", {}),
                    tenant_id=tenant_id,
                )
            
            # Strengthen edges from this concept
            if concept.relations:
                for rel_type, target_id, _ in concept.relations:
                    try:
                        # Re-add edge with boosted confidence
                        self.graph.add_edge(
                            concept.concept_id,
                            target_id,
                            rel_type,
                            confidence=0.9,  # High confidence for strengthened connections
                            tenant_id=tenant_id,
                        )
                    except Exception:
                        pass
                        
        except Exception as e:
            logger.debug(f"Failed to strengthen concept {concept.term}: {e}")
    
    def _create_attractor_from_inference(
        self,
        inference: Inference,
        tenant_id: Optional[str] = None
    ) -> None:
        """
        Create a physics attractor from a high-confidence inference.
        
        This makes the inference's conclusion a "natural resting point"
        in the thought space, so similar future queries will gravitate
        toward this conclusion.
        """
        if self.graph is None:
            return
        
        try:
            # Create an inference node
            inference_id = f"infer_{hash(inference.conclusion) % 10**8:08x}"
            
            self.graph.add_node(
                inference_id,
                "inference",
                inference.conclusion[:100],  # Truncate for term
                confidence=inference.confidence,
                data={
                    "inference_type": inference.inference_type,
                    "source_concepts": inference.source_concepts,
                    "reasoning_path": inference.reasoning_path,
                },
                tenant_id=tenant_id,
            )
            
            # Link to source concepts
            for source in inference.source_concepts[:2]:
                source_id = source.lower().replace(" ", "_")
                try:
                    self.graph.add_edge(
                        source_id,
                        inference_id,
                        "infers",
                        confidence=inference.confidence * 0.8,
                        tenant_id=tenant_id,
                    )
                except Exception:
                    pass
                    
        except Exception as e:
            logger.debug(f"Failed to create attractor from inference: {e}")
    
    def _merge_similar_abstractions(
        self,
        abstractions: List[Abstraction],
        tenant_id: Optional[str] = None
    ) -> int:
        """
        Merge abstractions with overlapping patterns.
        
        If two abstractions have similar patterns (e.g., "is_a" and "is_a|is_a"),
        we might merge the shorter into the longer as a sub-pattern.
        
        Returns:
            Number of merges performed
        """
        merged_count = 0
        
        # Group by pattern length
        by_length: Dict[int, List[Abstraction]] = {}
        for abs in abstractions:
            length = len(abs.pattern)
            if length not in by_length:
                by_length[length] = []
            by_length[length].append(abs)
        
        # Find short patterns that are subsets of longer patterns
        lengths = sorted(by_length.keys())
        for i, short_len in enumerate(lengths[:-1]):
            for long_len in lengths[i+1:]:
                for short_abs in by_length[short_len]:
                    for long_abs in by_length[long_len]:
                        # Check if short pattern is a prefix of long pattern
                        if long_abs.pattern[:short_len] == short_abs.pattern:
                            # Merge: boost long pattern's occurrence count
                            long_abs.occurrence_count += short_abs.occurrence_count
                            long_abs.confidence = max(long_abs.confidence, short_abs.confidence)
                            merged_count += 1
                            
                            # Update in graph if available
                            if self.graph is not None:
                                self._persist_abstraction(long_abs, tenant_id)
        
        return merged_count
    
    def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get memory consolidation statistics."""
        return {
            "working_memory_size": len(self._working_memory),
            "total_abstractions": getattr(self, '_total_abstractions', 0),
            "last_deliberation": self._last_result is not None,
        }
    
    # ============ Explanation Generation ============
    
    def explain(
        self,
        deliberation_result: Optional[DeliberationResult] = None,
        style: str = "narrative",  # narrative, step_by_step, technical
        max_length: int = 500,
    ) -> str:
        """
        Generate a human-readable explanation of the reasoning process.
        
        This transforms the internal reasoning structures (inferences, chains,
        abstractions) into natural language that explains how a conclusion
        was reached.
        
        Args:
            deliberation_result: Result to explain (uses last if None)
            style: Explanation style - narrative, step_by_step, or technical
            max_length: Approximate max length of explanation
            
        Returns:
            Human-readable explanation string
        """
        result = deliberation_result or self._last_result
        if result is None:
            return "No reasoning occurred yet."
        
        if style == "step_by_step":
            return self._explain_step_by_step(result, max_length)
        elif style == "technical":
            return self._explain_technical(result, max_length)
        else:
            return self._explain_narrative(result, max_length)
    
    def _explain_narrative(
        self,
        result: DeliberationResult,
        max_length: int = 500,
    ) -> str:
        """Generate a narrative-style explanation."""
        parts = []
        
        # Opening with focus
        if result.focus_concept:
            parts.append(f"The key focus is on '{result.focus_concept}'.")
        
        # Intent resolution
        if result.resolved_intent:
            intent_phrases = {
                "definition": "This appears to be asking for a definition",
                "explanation": "This is seeking an explanation",
                "comparison": "This involves comparing concepts",
                "cause": "This is exploring causes or reasons",
                "example": "This is looking for examples",
                "general": "This is a general inquiry",
            }
            phrase = intent_phrases.get(result.resolved_intent, f"The intent is {result.resolved_intent}")
            parts.append(f"{phrase}.")
        
        # Key reasoning path
        if result.concept_chains:
            best_chain = max(result.concept_chains, key=lambda c: c.confidence)
            chain_description = self._describe_chain(best_chain)
            parts.append(f"The reasoning follows: {chain_description}")
        
        # Main conclusion from inferences
        if result.inferences:
            # Find the strongest inference
            strongest = max(result.inferences, key=lambda i: i.confidence)
            parts.append(f"This leads to the conclusion: {strongest.conclusion[:100]}.")
        
        # Pattern recognition
        if result.abstractions:
            abs_names = [a.name for a in result.abstractions[:2]]
            parts.append(f"This reasoning pattern resembles: {', '.join(abs_names)}.")
        
        # Confidence statement
        if result.confidence > 0.8:
            parts.append("The reasoning confidence is high.")
        elif result.confidence < 0.4:
            parts.append("There is some uncertainty in this reasoning.")
        
        explanation = " ".join(parts)
        return explanation[:max_length] if len(explanation) > max_length else explanation
    
    def _explain_step_by_step(
        self,
        result: DeliberationResult,
        max_length: int = 500,
    ) -> str:
        """Generate a step-by-step explanation."""
        steps = []
        
        # Step 1: What was activated
        activated_terms = [c.term for c in result.activated_concepts[:3]]
        steps.append(f"1. Started with concepts: {', '.join(activated_terms)}")
        
        # Step 2: Chains followed
        if result.concept_chains:
            best_chain = max(result.concept_chains, key=lambda c: c.confidence)
            chain_desc = " → ".join(best_chain.terms[:4])
            steps.append(f"2. Traced path: {chain_desc}")
        
        # Step 3: Pattern recognized
        if result.abstractions:
            abs_name = result.abstractions[0].name
            steps.append(f"3. Recognized pattern: {abs_name}")
        
        # Step 4: Key inferences
        chain_inferences = [i for i in result.inferences if i.inference_type == "chain"]
        if chain_inferences:
            best = max(chain_inferences, key=lambda i: i.confidence)
            steps.append(f"4. Inferred: {best.conclusion[:60]}...")
        elif result.inferences:
            best = max(result.inferences, key=lambda i: i.confidence)
            steps.append(f"4. Concluded: {best.conclusion[:60]}...")
        
        # Step 5: Focus determined
        if result.focus_concept:
            steps.append(f"5. Focus: {result.focus_concept}")
        
        explanation = "\n".join(steps)
        return explanation[:max_length] if len(explanation) > max_length else explanation
    
    def _explain_technical(
        self,
        result: DeliberationResult,
        max_length: int = 500,
    ) -> str:
        """Generate a technical explanation with metrics."""
        lines = []
        
        lines.append(f"Deliberation({result.deliberation_steps} steps):")
        lines.append(f"  Activation: {len(result.activated_concepts)} concepts")
        lines.append(f"  Chains: {len(result.concept_chains)}, Inferences: {len(result.inferences)}")
        lines.append(f"  Confidence: {result.confidence:.3f}")
        lines.append(f"  Efficiency: {result.trajectory_efficiency:.3f}")
        lines.append(f"  Mental Effort: {result.mental_effort:.4f}")
        
        if result.abstractions:
            lines.append(f"  Patterns: {[a.name for a in result.abstractions]}")
        
        if result.focus_concept:
            lines.append(f"  Focus: {result.focus_concept}")
        
        if result.resolved_intent:
            lines.append(f"  Intent: {result.resolved_intent}")
        
        explanation = "\n".join(lines)
        return explanation[:max_length] if len(explanation) > max_length else explanation
    
    def _describe_chain(self, chain: ConceptChain) -> str:
        """Create a natural language description of a concept chain."""
        if not chain.terms:
            return "empty chain"
        
        if len(chain.terms) == 1:
            return chain.terms[0]
        
        # Build description with relations
        parts = []
        for i, (term, relation) in enumerate(zip(chain.terms[:-1], chain.relations)):
            # Convert relation type to natural language
            rel_phrases = {
                "is_a": "is a type of",
                "has_a": "has",
                "part_of": "is part of",
                "causes": "causes",
                "affects": "affects",
                "related_to": "relates to",
                "contains": "contains",
                "has_frame": "has the structure",
            }
            natural_rel = rel_phrases.get(relation, relation)
            parts.append(f"'{term}' {natural_rel}")
        
        parts.append(f"'{chain.terms[-1]}'")
        return " ".join(parts)
    
    def explain_inference(self, inference: Inference) -> str:
        """Explain a single inference in natural language."""
        type_phrases = {
            "implication": "This implies",
            "elaboration": "Expanding on this",
            "connection": "There is a connection where",
            "causal": "This suggests that",
            "chain": "Following the reasoning chain",
        }
        
        prefix = type_phrases.get(inference.inference_type, "It follows that")
        
        if inference.reasoning_path:
            path_str = " → ".join(inference.reasoning_path[:3])
            return f"{prefix}: {inference.conclusion} (via: {path_str})"
        
        return f"{prefix}: {inference.conclusion}"
