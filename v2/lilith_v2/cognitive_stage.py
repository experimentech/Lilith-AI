import logging
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np

# Abstractions for the specific layer implementations (to be injected)
# This keeps the stage generic across modalities
from .stage import Stage
from .pmflow_sqlite import PMFlowStateStore
from .relational_graph_store import RelationalGraphStore
from .knowledge_service import KnowledgeService, KnowledgeFragment
from .affective_system import AffectiveSystem
from .feedback_system import FeedbackDetector, FeedbackResult
from .semantic_extractor import SemanticExtractor, ExtractedRelation
from .generative_system import GenerativeSystem
from .concept_grounding import ConceptGrounder, GroundedConcept
from .world_model import WorldModel
from .topic_extractor import TopicExtractor
from .pmflow_physics import PMField, ParallelPMField
from .linguistic_processor import LinguisticProcessor, LinguisticArtifact
from .response_composer import ResponseComposer, ComposedResponse
from .pattern_store_adapter import PatternStoreAdapter
from .store import Store
from .reasoning_stage import ReasoningStage, DeliberationResult, Inference

logger = logging.getLogger(__name__)

class CognitiveStage: # We don't inherit from Stage Protocol directly as class, we implement it
    """
    Unified Cognitive Stage for Lilith v2 (The integrated "Brain" and "Textbook").

    This stage orchestrates the interaction between implicit intuition (BioNN/PMFlow)
    and explicit knowledge (Relational Graph).

    Design Philosophy:
    - Multi-modal agnostic: Operates on Embeddings (The lingua franca of the brain).
    - Multi-tenant capable: State and Graph are instance-specific (can be bound per tenant).
    - No split-brain: A single source of truth for "thinking".

    Process:
    1. Perception: Input -> Embedding (via modality-specific Encoder)
    2. Affect: Modulate processing based on internal state (Mood/Personality)
    3. Intuition: Embedding -> Evolved Embedding (via PMFlow)
    4. Grounding: Evolved Embedding -> Concept IDs (via Vector Similarity)
    5. Gap Filling: Low Confidence -> External Knowledge Retrieval (Wikipedia/Web)
    6. Reasoning: Concept IDs -> Connected Subgraph (via RelationalGraphStore)
    7. Synthesis: Subgraph + Evolved Embedding -> Output Context
    """

    def __init__(
        self,
        node_id: str,
        pmflow_store: PMFlowStateStore,
        graph_store: RelationalGraphStore,
        encoder: Any, # Duck-typed: must implement encode(data) -> Tensor
        config: Optional[Dict[str, Any]] = None,
        knowledge_service: Optional[KnowledgeService] = None,
        affective_system: Optional[AffectiveSystem] = None
    ) -> None:
        self.id = node_id # Protocol field name is 'id'
        self.pmflow = pmflow_store
        self.graph = graph_store
        self.encoder = encoder
        self.config = config or {}
        
        # Subsystems
        self.knowledge_service = knowledge_service or KnowledgeService(enabled=self.config.get("knowledge_enabled", True))
        self.affective = affective_system or AffectiveSystem(self.config.get("affective_config"))
        self.feedback_detector = FeedbackDetector()
        self.semantic_extractor = SemanticExtractor()
        self.generative = GenerativeSystem(self.graph)
        self.grounder = ConceptGrounder(self.graph)
        self.world_model = WorldModel()
        self.topic_extractor = TopicExtractor(self.encoder)
        
        # Response Composer (Pattern-based response generation with learning)
        # Initialize pattern store for response patterns
        self._response_store = None
        self._response_composer = None
        response_store = config.get("response_store") if config else None
        # Duck-type check for Store interface (has put/get/list methods)
        if response_store and hasattr(response_store, "put") and hasattr(response_store, "get"):
            self._response_store = PatternStoreAdapter(response_store, f"{node_id}_responses")
            self._response_composer = ResponseComposer(
                pattern_store=self._response_store,
                graph_store=self.graph,
                encoder=self.encoder,
                composition_mode=config.get("composition_mode", "adaptive") if config else "adaptive",
                pragmatic_system=self.generative.pragmatics,
                enable_blending=config.get("enable_blending", True) if config else True,
                enable_learning=config.get("enable_learning", True) if config else True,
            )
            logger.info(f"[{node_id}] ResponseComposer initialized with pattern store")
        
        # Linguistic Processor (multi-stage BioNN/Database processing)
        self.linguistic = LinguisticProcessor(
            graph_store=self.graph,
            encoder=self.encoder
        )
        
        # Reasoning Stage (Agentic Physics-based deliberation)
        # Only enabled if encoder has enable_flow=True
        self._reasoning_enabled = config.get("enable_reasoning", True) if config else True
        self._reasoning = None
        if self._reasoning_enabled:
            self._reasoning = ReasoningStage(
                encoder=self.encoder,
                graph_store=self.graph,
                max_working_memory=config.get("max_working_memory", 7) if config else 7,
                deliberation_steps=config.get("deliberation_steps", 10) if config else 10,
            )
            has_flow = hasattr(encoder, 'enable_flow') and encoder.enable_flow
            logger.info(f"[{node_id}] ReasoningStage initialized (agentic_flow={has_flow})")
        
        # Physics Engine (BioNN / Intuition)
        # Determine encoder output dimension by running a test encode
        # The physics engine needs to match this for evolution to work
        try:
            test_emb = self.encoder.encode(["test"])
            if isinstance(test_emb, torch.Tensor):
                dim = test_emb.flatten().shape[0]
            else:
                dim = 64
        except Exception:
            dim = getattr(self.encoder, 'dimension', 64)
        
        # Use ParallelPMField to hold the "Mind Landscape" (Attractors)
        self.physics = ParallelPMField(d_latent=dim, n_centers=8, steps=10) # Default 8 thoughts

        # In-memory interaction cache for "Short Term Memory" / Working Set
        # Concept ID -> Embedding Vector
        self._concept_vectors: Dict[str, torch.Tensor] = {}
        # HACK: Cold-start population. In production, use a vector DB or ANN index.
        self._hydrate_concept_vectors()
        
        # State tracking
        self.last_thought: Dict[str, Any] = {}
        self.last_interaction: Optional[Dict[str, str]] = None
        
        # Sync physics with store
        self._sync_physics_landscape()

    def _sync_physics_landscape(self, tenant_id: str = None):
        """
        Load the 'Mind Landscape' (Attractors/Repulsors) from the PMFlow State Store.
        This gives Lilith her 'Intuition' - specific topics will naturally attract/repel thoughts.
        """
        try:
            # Load from SQLite JSON blob
            state = self.pmflow.load_state("main", tenant_id=tenant_id) or {}
            attractors = state.get("attractors", [])
            
            if not attractors:
                # Default seed: Center is home (0,0) with gentle gravity
                self.physics.centers = torch.zeros(8, self.physics.latent_dim)
                self.physics.mus = torch.full((8,), 0.05)
                self.physics.omegas = torch.zeros(8)
                return

            # Map attractors to physics engine
            # attractors structure: [{'vector': [...], 'weight': 1.0, 'spin': 0.1}]
            loaded_count = 0
            for i, attr in enumerate(attractors[:8]):
                vec = torch.tensor(attr.get("vector", [0.0]*self.physics.latent_dim), dtype=torch.float32)
                
                # Handle dimension mismatch - stored attractors may have wrong size
                if vec.shape[0] != self.physics.latent_dim:
                    # Skip mismatched attractors - they'll be regenerated on next learning
                    logger.debug(f"[{self.id}] Skipping attractor with dim {vec.shape[0]} (expected {self.physics.latent_dim})")
                    continue
                    
                self.physics.centers[i] = vec
                self.physics.mus[i] = attr.get("weight", 0.1)
                self.physics.omegas[i] = attr.get("spin", 0.0)
                loaded_count += 1
            
            if loaded_count > 0:
                logger.info(f"[{self.id}] Synchronized Physics Landscape with {loaded_count} attractors.")
        except Exception as e:
            logger.warning(f"[{self.id}] Failed to sync physics landscape: {e}")

    def _hydrate_concept_vectors(self):
        """Preload vector cache from graph store data.
        
        Assumes 'data' field in nodes might contain a 'vector' or that we
        re-encode terms on startup. For now, we'll lazily rely on terms.
        """
        # This is optimization for the 'Grounding' phase. 
        # In a real multi-modal system, nodes in the graph store might reference 
        # a vector stored in the PMFlow store or an external vector store.
        pass

    def learn(self, payload: Any, ctx: Dict[str, Any] = None) -> None:
        """
        Main cognitive loop: Detect -> Extract -> Perceive -> Feel -> Think -> Connect -> Augment.
        
        Multi-stage processing:
        1. Linguistic: Tokenize, POS tag, build symbolic frame (BioNN + Graph)
        2. Feedback: Detect implicit/explicit feedback
        3. Semantic: Extract relations for auto-learning
        4. Perception: Encode to embedding
        5. Intuition: Evolve embedding via PMFlow
        6. Grounding: Map to known concepts
        7. Reasoning: Graph traversal from grounded concepts
        8. Synthesis: Generate response
        """
        ctx = ctx or {}
        tenant_id = ctx.get("tenant")
        if isinstance(payload, str):
            ctx["original_text"] = payload
        
        # 0. LINGUISTIC PROCESSING (Multi-stage BioNN/Database)
        # This is the core V1 pipeline: Intake -> Parse -> Frame -> Store
        linguistic_artifact = None
        grounding_terms = []
        
        if isinstance(payload, str):
            try:
                # Run full linguistic pipeline
                linguistic_artifact = self.linguistic.process(payload, tenant_id=tenant_id)
                ctx["linguistic"] = linguistic_artifact
                
                # Extract content words for enhanced grounding
                grounding_terms = self.linguistic.get_tokens_for_grounding(linguistic_artifact)
                ctx["grounding_terms"] = grounding_terms
                
                # Use symbolic frame for reasoning hints
                frame_intent = linguistic_artifact.frame.attributes.get("intent")
                if frame_intent:
                    ctx["intent"] = frame_intent
                if linguistic_artifact.frame.actor:
                    ctx["actor"] = linguistic_artifact.frame.actor
                if linguistic_artifact.frame.action:
                    ctx["action"] = linguistic_artifact.frame.action
                if linguistic_artifact.frame.target:
                    ctx["target"] = linguistic_artifact.frame.target
                    
                logger.debug(f"[{self.id}] Linguistic: {len(linguistic_artifact.parsed.tokens)} tokens, "
                            f"POS: {linguistic_artifact.parsed.pos_sequence[:5]}, "
                            f"Frame: {linguistic_artifact.frame.actor} {linguistic_artifact.frame.action} {linguistic_artifact.frame.target}")
            except Exception as e:
                logger.warning(f"[{self.id}] Linguistic processing failed: {e}")
        
        # 1. Implicit Feedback Detection & Extraction (Learning from Input)
        extracted_knowledge = []
        is_teaching_intent = False

        if isinstance(payload, str):
            # PRAGMATICS: Learn from previous turn
            if self.last_interaction:
                 prev_out = self.last_interaction.get("response", "")
                 # Detect if user is teaching us something based on our failure
                 if self.generative.pragmatics.detect_teaching_intent(prev_out, payload):
                     is_teaching_intent = True
                     # Boost mood as we are learning!
                     self.affective.update(0.5, ctx)
                     logger.info(f"[{self.id}] Pragmatic: Teaching Intent Detected")
                 
                 # Record interaction for patterns
                 current_topic = self.topic_extractor.extract_topic(payload) or "general"
                 self.generative.pragmatics.learn_from_interaction(
                     payload, prev_out, current_topic
                 )

            # Feedback Detection & Response Learning
            fb_result = self.feedback_detector.detect(payload)
            if fb_result.score != 0.0:
                 self.affective.update(fb_result.score, ctx)
                 logger.info(f"[{self.id}] Implicit Feedback: {fb_result.signal} ({fb_result.score})")
                 
                 # LEARNING LOOP: Propagate feedback to response composer
                 # This learns which response patterns work and which don't
                 if self._response_composer and self.last_interaction:
                     # Convert feedback score to success boolean
                     # Positive signals (thanks, great, yes) = success
                     # Negative signals (what?, confused, wrong) = failure
                     success = fb_result.score > 0
                     self._response_composer.record_outcome(success)
                     outcome_symbol = '✓' if success else '✗'
                     logger.debug(f"[{self.id}] Response learning: {outcome_symbol} from implicit feedback")
            
            # World Modeling (Grounded situational awareness)
            self.world_model.process_utterance(payload)
            
            # Semantic Extraction (The "Auto-Learner")
            extracted = self.semantic_extractor.extract(payload)
            for rel in extracted:
                # If we detected a teaching intent, we lower the threshold significantly
                # because the user is EXPLICITLY trying to teach us.
                threshold = 0.6 if is_teaching_intent else 0.8
                
                if rel.confidence > threshold:
                    try:
                        # Ensure nodes exist (Upsert logic)
                        # We use the term itself as the ID for this simple phase
                        subj_id = rel.subject.lower().replace(" ", "_")
                        obj_id = rel.object.lower().replace(" ", "_")
                        
                        self.graph.add_node(subj_id, "learned_concept", rel.subject, confidence=rel.confidence, tenant_id=tenant_id)
                        self.graph.add_node(obj_id, "learned_concept", rel.object, confidence=rel.confidence, tenant_id=tenant_id)
                        
                        self.graph.add_edge(subj_id, obj_id, rel.predicate, confidence=rel.confidence, tenant_id=tenant_id)
                        
                        extracted_knowledge.append(rel)
                        
                        # Topic Learning: If we learned a definitive Subject, treat it as a Topic
                        # e.g. "Python is a language" -> Learn Topic "Python"
                        self.topic_extractor.learn_topic(rel.subject, payload)
                        if rel.predicate == "is_a":
                             # "Python is a language" -> Learn Topic "Language" too
                             self.topic_extractor.learn_topic(rel.object, payload)

                        logger.info(f"[{self.id}] Learned & Persisted: {rel.subject} --{rel.predicate}--> {rel.object}")
                    except Exception as e:
                        logger.error(f"Failed to persist knowledge: {e}")
            
            # Syntax Learning (Novelty: Learn HOW it was said)
            self.generative.learn_grammar(payload, extracted_knowledge)

        # 0b. Explicit Feedback (if provided in ctx)
        start_feedback = ctx.get("feedback_score", 0.0)
        if start_feedback != 0.0:
            self.affective.update(start_feedback, ctx)

        affective_state = self.affective.get_state_vector()

        # 1. Perception
        # The encoder abstracts the modality (Text, Image, Audio)
        # It just returns a pytorch tensor.
        try:
            perception_vector = self.encoder.encode(payload)
            if isinstance(perception_vector, np.ndarray):
                perception_vector = torch.from_numpy(perception_vector)
        except Exception as e:
            logger.error(f"[{self.id}] Perception failed: {e}")
            return

        # 2. Intuition (The "Thinking" Step)
        # Evolve the vector using the BioNN's current state (PMField).
        # This represents "intuition" - where does this thought naturally flow?
        # We modulate intuition with affective state (e.g., curiosity alters the vector)
        
        # Now active: The 'Mind Landscape' (attractors) bends the thought vector.
        # This means concepts trigger related concepts naturally via physics.
        try:
            # Physics engine expects [Batch, Dim]
            # .forward(x) runs the simulation for N steps
            # Note: encoder.encode() already returns [1, dim], so we ensure 2D here
            if perception_vector.dim() == 1:
                physics_input = perception_vector.unsqueeze(0)
            else:
                physics_input = perception_vector
            evolved_vector = self.physics.forward(physics_input).squeeze(0)
            
            # TODO: If affective['energy'] is low, maybe take fewer steps?
        except Exception as e:
            logger.error(f"[{self.id}] Physics simulation failed: {e}")
            evolved_vector = perception_vector 
        
        # 3. Grounding (Vector -> Symbol)
        # Which known concepts does this thought resemble?
        # In a multi-modal system, "Dog" (text) and [Image of Dog] collapse to same ID.
        active_concepts = self._ground_vector_to_concepts(evolved_vector, ctx)
        
        # 4. Gap Filling (Knowledge Augmentation)
        # If we have low grounding confidence (few concepts found) OR high curiosity,
        # we try to fetch external knowledge.
        external_context: List[KnowledgeFragment] = []
        should_augment = (
            len(active_concepts) == 0 or 
            (affective_state["curiosity_drive"] > 0.7 and len(active_concepts) < 3)
        )
        
        if should_augment and isinstance(payload, str): # Currently only text lookup supported
            try:
                fragments = self.knowledge_service.search(payload, ctx)
                if fragments:
                    external_context = fragments
                    # In a full system, we would immediately encode these fragments
                    # and re-run Grounding to find new concept IDs they point to.
                    logger.info(f"[{self.id}] Knowledge Augmentation found {len(fragments)} items.")
            except Exception as e:
                logger.warning(f"[{self.id}] Knowledge Augmentation error: {e}")

        # 4b. Reasoning (Symbolic Traversal)
        # Use the explicit graph to find logical connections, starting from active concepts
        inferred_knowledge = []
        for concept_id, confidence in active_concepts:
            # BFS logic from the Graph Store
            # "What implies this?" "What does this imply?"
            subgraph = self.graph.traverse_bfs(concept_id, max_depth=2, tenant_id=tenant_id)
            inferred_knowledge.extend(subgraph)

        # 4c. Deliberative Reasoning (Physics-based "Thinking")
        # Uses PMFlow agentic physics to trace reasoning trajectories
        deliberation_result = None
        if self._reasoning and isinstance(payload, str):
            try:
                # Determine goal from context (if available)
                goal = ctx.get("intent") or ctx.get("target")
                
                deliberation_result = self._reasoning.deliberate(
                    query=payload,
                    context=ctx.get("original_text"),
                    goal=goal,
                    max_steps=self.config.get("deliberation_steps", 10),
                )
                
                # Extract additional inferences
                for inf in deliberation_result.inferences:
                    if inf.confidence > 0.5:
                        inferred_knowledge.append({
                            "type": inf.inference_type,
                            "conclusion": inf.conclusion,
                            "confidence": inf.confidence,
                            "path": inf.reasoning_path,
                        })
                
                # Update intent if reasoning resolved it
                if deliberation_result.resolved_intent:
                    ctx["resolved_intent"] = deliberation_result.resolved_intent
                
                logger.debug(f"[{self.id}] Deliberation: {len(deliberation_result.inferences)} inferences, "
                            f"efficiency={deliberation_result.trajectory_efficiency:.2f}")
            except Exception as e:
                logger.warning(f"[{self.id}] Deliberative reasoning failed: {e}")

        self.last_thought = {
             "perception": perception_vector,
             "intuition": evolved_vector,
             "affect": affective_state,
             "grounding": active_concepts,
             "world_state": self.world_model.get_context(),
             "extracted_knowledge": extracted_knowledge,
             "external_knowledge": [f.content for f in external_context],
             "inference": inferred_knowledge,
             "pragmatic_intent": "teaching" if is_teaching_intent else None,
             "deliberation": deliberation_result,
             "reasoning_confidence": deliberation_result.confidence if deliberation_result else None,
             "trajectory_efficiency": deliberation_result.trajectory_efficiency if deliberation_result else None,
        }
        
        # 5. Synthesis / Generation
        # Try ResponseComposer first (pattern-based with learning), fallback to GenerativeSystem
        response_text = ""
        composed_response = None
        
        if self._response_composer:
            try:
                composed_response = self._response_composer.compose(
                    thought_context=self.last_thought,
                    user_input=ctx.get("original_text", ""),
                    topk=5,
                )
                if composed_response and not composed_response.is_fallback:
                    response_text = composed_response.text
                    self.last_thought["composition_source"] = composed_response.source
                    if composed_response.is_blended:
                        logger.debug(f"[{self.id}] Blended response from {len(composed_response.fragment_ids)} patterns")
            except Exception as e:
                logger.warning(f"[{self.id}] ResponseComposer failed: {e}")
        
        # Fallback to GenerativeSystem
        if not response_text:
            response_text = self.generative.compose(self.last_thought)
            self.last_thought["composition_source"] = "generative"
            
        self.last_thought["response"] = response_text
        self.last_thought["composed_response"] = composed_response
        
        # Update interaction history
        if isinstance(payload, str):
            self.last_interaction = {
                "input": payload,
                "response": response_text
            }
            
        # 6. Plasticity (Structural Learning)
        # Update the intuitive landscape based on what we just activated.
        # "Neuronal" connections that fired together (Concepts in this thought) get reinforced.
        self._update_intuition(active_concepts, ctx)
        
        logger.info(f"[{self.id}] Thought process complete. Concepts: {len(active_concepts)}, Response len: {len(response_text)}")

    def _update_intuition(self, activated_concepts: List[Tuple[str, float]], ctx: Dict):
        """
        Plasticity: Modify the PMField landscape (Intuition) based on experience.
        Frequent concepts become stronger attractors (Gravity wells).
        
        Mechanism: Hebbian Learning / Vector Quantization.
        - If a concept is grounded with high confidence (> 0.8):
        - Find or Create its Attractor in PMFlow State.
        - Reinforce its mass (mu) and pull its center towards current intuition.
        """
        if not activated_concepts:
            return

        tenant_id = ctx.get("tenant")

        # 1. Get current "Thought Location" (The intuitive vector)
        # We want to reinforce the location that the brain *converged* to.
        thought_vector = self.last_thought.get("intuition")
        if thought_vector is None:
            return
            
        # Ensure tensor is CPU/detached and FLATTENED for storage
        if isinstance(thought_vector, torch.Tensor):
            thought_vector = thought_vector.detach().cpu().flatten().tolist()
        elif isinstance(thought_vector, list) and thought_vector and isinstance(thought_vector[0], list):
            # Already a nested list - flatten it
            thought_vector = thought_vector[0]

        try:
            # 2. Load State
            state = self.pmflow.load_state("main", tenant_id=tenant_id) or {}
            attractors = state.get("attractors", [])
            state_changed = False
            
            # Map existing attractors by concept_id for O(1) lookup
            attr_map = {a.get("concept_id"): i for i, a in enumerate(attractors) if a.get("concept_id")}

            # 3. Hebbian Update Loop
            for concept_id, confidence in activated_concepts:
                if confidence < 0.8: # Only learn from strong signals
                    continue
                
                learning_rate = 0.05 * confidence
                
                if concept_id in attr_map:
                    # Reinforce existing
                    idx = attr_map[concept_id]
                    attr = attractors[idx]
                    
                    # Mass increase (Hebb: Firing -> Growth)
                    # Cap mass at 0.5 to prevent black holes
                    attr["weight"] = min(attr["weight"] + learning_rate, 0.5)
                    
                    # Vector Centering (Drift towards new observation)
                    # new_pos = old_pos + alpha * (observed - old_pos)
                    old_vec = attr["vector"]
                    new_vec = [o + learning_rate * (t - o) for o, t in zip(old_vec, thought_vector)]
                    attr["vector"] = new_vec
                    
                    state_changed = True
                    logger.debug(f"[{self.id}] Plasticity: Reinforced attractor for '{concept_id}' (mass={attr['weight']:.2f})")
                
                else:
                    # Genesis: New Attractor
                    # Only if we have space (Max 50 concepts in short-term intuition)
                    if len(attractors) < 50:
                        new_attr = {
                            "concept_id": concept_id,
                            "vector": thought_vector,
                            "weight": 0.1, # Start small
                            "spin": 0.0
                        }
                        attractors.append(new_attr)
                        attr_map[concept_id] = len(attractors) - 1
                        state_changed = True
                        logger.info(f"[{self.id}] Plasticity: Created new intuitive attractor for '{concept_id}'")

            # 4. Save & Sync
            if state_changed:
                state["attractors"] = attractors
                self.pmflow.save_state("main", state, version=1, tenant_id=tenant_id) # Version handling simplified for now
                
                # Live Update (so we don't need restart to feel effect)
                self._sync_physics_landscape(tenant_id=tenant_id)
                
        except Exception as e:
            logger.error(f"[{self.id}] Plasticity update failed: {e}")

    def _ground_vector_to_concepts(
        self, 
        vector: torch.Tensor, 
        ctx: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """
        Finds the closest symbolic nodes to a given vector.
        This bridge allows "thinking" in vectors but "remembering" in facts.
        """
        matches = []
        
        # 1. Text-based Grounding (The "Breadcrumb" Implementation)
        # If we have the raw text payload available, we can use fuzzy/synonym matching
        # to find relevant concepts in the graph.
        original_text = ctx.get("original_text", "")
        tenant_id = ctx.get("tenant")
        
        if original_text:
            # A: Full-phrase grounding (for QA matching)
            grounded = self.grounder.ground(original_text, tenant_id=tenant_id)
            for g in grounded:
                matches.append((g.concept_id, g.confidence))
            
            # B: Linguistic token grounding (content words from parser)
            # This gives us lemmatized nouns, verbs, adjectives
            grounding_terms = ctx.get("grounding_terms", [])
            for term in grounding_terms:
                term_grounded = self.grounder.ground(term, threshold=85.0, tenant_id=tenant_id)
                for g in term_grounded:
                    # Slightly lower confidence since these are individual words
                    matches.append((g.concept_id, g.confidence * 0.9))
            
            # C: Topic Extraction (Neural Grounding)
            # Find implicit topics ("Tell me about snakes" -> snakes)
            topic = self.topic_extractor.extract_topic(original_text)
            if topic:
                # Add topic as a concept match with high confidence
                t_id = topic.lower().replace(" ", "_")
                matches.append((t_id, 0.95))
                logger.info(f"[{self.id}] Neural Topic Grounding found: {topic}")

        # 2. Vector-based Grounding (Future)
        # In production v2, this would be an ANN search (FAISS/HNSW).
        # matches.extend(self.vector_index.search(vector))
        
        # Deduplicate by concept_id, keeping highest confidence
        seen = {}
        for concept_id, conf in matches:
            if concept_id not in seen or conf > seen[concept_id]:
                seen[concept_id] = conf
        
        return sorted(seen.items(), key=lambda x: x[1], reverse=True)

    def encode(self, item: Any, ctx: Dict[str, Any]) -> Any:
        # Pass-through to encoder for interface compliance
        return self.encoder.encode(item)

    def retrieve(self, query: Any, ctx: Dict[str, Any]) -> Any:
        # Implementation required by protocol (could return last_thought or synthesize answer)
        return self.last_thought.get("inference", [])

    def update_success(self, feedback: Any, ctx: Dict[str, Any]) -> None:
        """Process explicit feedback (used for training pulse)."""
        score = 0.0
        if isinstance(feedback, bool):
            score = 1.0 if feedback else -0.5
        elif isinstance(feedback, (float, int)):
            score = float(feedback)
        
        self.affective.update(score, ctx)
        
        # Propagate to response composer for pattern learning
        if self._response_composer:
            success = score > 0
            self._response_composer.record_outcome(success)
            logger.debug(f"[{self.id}] Response outcome recorded: {'success' if success else 'failure'}")
    
    def record_response_outcome(self, success: bool) -> None:
        """
        Record the outcome of the last response for learning.
        
        Call this after each response to enable success-based learning.
        The system learns which response patterns work for which queries.
        
        Args:
            success: True if conversation continued well, False if it broke down
                     
        Signals of success:
            - User continues the topic → True
            - User asks follow-up question → True  
            - User changes topic abruptly → False
            - User says "what?" or "huh?" → False
        """
        if self._response_composer:
            self._response_composer.record_outcome(success)
        
        # Also update affective state
        feedback_score = 0.3 if success else -0.2
        self.affective.update(feedback_score, {})

    def stats(self) -> Dict[str, Any]:
        stats = {
            "mood": self.affective.mood.label,
            "knowledge_enabled": self.knowledge_service.enabled,
        }
        
        # Add response composer stats if available
        if self._response_composer:
            composer_stats = self._response_composer.get_stats()
            stats["response_composer"] = composer_stats
            stats["composition_mode"] = self._response_composer.mode.value
            stats["adaptive_threshold"] = self._response_composer.adaptive.adaptive_threshold
        
        # Add reasoning stage stats if available
        if self._reasoning:
            reasoning_stats = self._reasoning.get_stats()
            stats["reasoning"] = reasoning_stats
            stats["reasoning_enabled"] = True
        else:
            stats["reasoning_enabled"] = False
        
        return stats

    def relational_sidecar(self, sql_ctx: Any) -> Optional[Any]:
        return None

__all__ = ["CognitiveStage"]
