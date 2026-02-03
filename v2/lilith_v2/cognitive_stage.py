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
from .action_planner import ActionPlanner, ExecutionPlan, PlannedStep

# Conversational Architecture Stages (v2 enhancement)
from .discourse_manager import DiscourseManager, DialogueState, DialogueAct
from .communication_planner import CommunicationPlanner, CommunicationPlan
from .compositional_realizer import CompositionalRealizer, RealizationResult
from .self_monitor import SelfMonitor, MonitoringResult

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
        
        # Linguistic Processor (multi-stage BioNN/Database processing)
        # Created early so TopicExtractor can use it for POS tagging
        self.linguistic = LinguisticProcessor(
            graph_store=self.graph,
            encoder=self.encoder
        )
        
        # Topic Extractor (uses linguistic processor for noun extraction)
        self.topic_extractor = TopicExtractor(
            encoder=self.encoder,
            linguistic_processor=self.linguistic
        )
        
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
        
        # ===== Conversational Architecture Stages =====
        # These stages enable motivated, coherent conversation by:
        # 1. Tracking dialogue state and obligations
        # 2. Planning what to communicate based on drives
        # 3. Realizing plans as language (using learned frames)
        # 4. Monitoring output quality before sending
        
        self._discourse_manager = DiscourseManager(max_history=10)
        self._communication_planner = CommunicationPlanner(
            curiosity_threshold=config.get("curiosity_threshold", 0.6) if config else 0.6,
            confidence_threshold=config.get("confidence_threshold", 0.5) if config else 0.5,
            warmth_threshold=config.get("warmth_threshold", 0.5) if config else 0.5,
        )
        # CompositionalRealizer with pattern store for learning-driven responses
        self._compositional_realizer = CompositionalRealizer(
            graph_store=self.graph,
            pattern_store=self._response_store  # Enables learned pattern retrieval
        )
        self._self_monitor = SelfMonitor(min_score=0.5)
        
        # Enable/disable conversational architecture
        self._conversational_mode = config.get("conversational_mode", True) if config else True
        if self._conversational_mode:
            logger.info(f"[{node_id}] Conversational Architecture enabled (discourse, planning, composition, monitoring)")
        
        # ===== Action Planner (Physics-based action sequencing) =====
        # Uses trajectory tracing to plan multi-step actions
        self._action_planning_enabled = config.get("enable_action_planning", True) if config else True
        self._action_planner: Optional[ActionPlanner] = None
        if self._action_planning_enabled:
            self._action_planner = ActionPlanner(
                encoder=self.encoder,
                graph=self.graph,
                trajectory_steps=config.get("action_trajectory_steps", 10) if config else 10,
                grounding_threshold=config.get("action_grounding_threshold", 0.3) if config else 0.3,
            )
            has_flow = hasattr(encoder, 'enable_flow') and encoder.enable_flow
            logger.info(f"[{node_id}] ActionPlanner initialized (agentic_flow={has_flow})")
        
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
        
        # Bootstrap contrastive learning for semantic separation
        self._bootstrap_contrastive_pairs()

    def _bootstrap_contrastive_pairs(self):
        """
        Bootstrap core semantic relationships via contrastive learning.
        
        This teaches the embedding space to separate opposite concepts
        and cluster similar concepts, even before interaction data.
        """
        try:
            from lilith.contrastive_learner import ContrastiveLearner
            CONTRASTIVE_AVAILABLE = True
        except ImportError:
            CONTRASTIVE_AVAILABLE = False
            
        if not CONTRASTIVE_AVAILABLE:
            logger.debug(f"[{self.id}] Contrastive learning not available")
            return
            
        if not hasattr(self.encoder, 'pm_field'):
            logger.debug(f"[{self.id}] Encoder lacks pm_field, skipping contrastive bootstrap")
            return
            
        try:
            learner = ContrastiveLearner(
                encoder=self.encoder,
                margin=0.3,
                temperature=0.07,
                learning_rate=1e-3,
            )
            
            # Core semantic opposites (critical for dialog)
            opposites = [
                ("agree", "disagree"), ("yes", "no"), ("like", "dislike"),
                ("good", "bad"), ("happy", "sad"), ("right", "wrong"),
                ("true", "false"), ("positive", "negative"),
                ("accept", "reject"), ("approve", "disapprove"),
            ]
            
            # Similar intent patterns
            similar_intents = [
                ("how are you", "what's up"),
                ("how are you", "how's it going"),
                ("hello", "hi"),
                ("goodbye", "bye"),
                ("thanks", "thank you"),
                ("sorry", "my apologies"),
                ("I agree", "that's right"),
                ("I disagree", "that's wrong"),
            ]
            
            count = 0
            for a, b in opposites:
                learner.add_symmetric_pair(a, b, "hard_negative", 1.0, "bootstrap")
                count += 2
            for a, b in similar_intents:
                learner.add_symmetric_pair(a, b, "positive", 0.9, "bootstrap")
                count += 2
                
            # Quick training pass (just a few epochs to seed)
            if count > 0:
                learner.train(epochs=3, batch_size=16, verbose=False)
                logger.debug(f"[{self.id}] Bootstrapped {count} contrastive pairs")
                
        except Exception as e:
            logger.debug(f"[{self.id}] Contrastive bootstrap failed: {e}")

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
                 topic_result = self.topic_extractor.extract_topic(payload)
                 current_topic = topic_result[0] if topic_result[0] else "general"
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
                     
                     # PATTERN EXTRACTION: If positive feedback, store response as pattern
                     # This is where the system grows its repertoire by learning
                     # what responses work in what contexts
                     if success and fb_result.score > 0.3:
                         try:
                             last_input = self.last_interaction.get("input", "")
                             last_response = self.last_interaction.get("response", "")
                             if last_input and last_response and len(last_response) > 5:
                                 # Determine intent from discourse state
                                 intent = self._discourse_manager.state.last_user_intent or "general"
                                 self._response_composer.add_pattern(
                                     trigger=last_input,
                                     response=last_response,
                                     intent=intent,
                                     initial_score=0.6 + (fb_result.score * 0.2)
                                 )
                                 logger.info(f"[{self.id}] Learned pattern: '{last_input[:30]}...' → '{last_response[:30]}...'")
                         except Exception as e:
                             logger.debug(f"[{self.id}] Pattern extraction failed: {e}")
            
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
            # This feeds both GenerativeSystem AND CompositionalRealizer
            self.generative.learn_grammar(payload, extracted_knowledge)
            
            # Learn syntactic frames for compositional output
            # Extract the pattern from the input text and the relations we found
            for rel in extracted_knowledge:
                self._learn_syntactic_frame_from_input(payload, rel)

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
        external_context: List[KnowledgeFragment] = []
        
        # First check if we already have knowledge in the graph about this topic
        has_graph_knowledge = False
        if isinstance(payload, str):
            # Extract likely subject and check graph
            subject = self._extract_subject(payload, "")
            if subject:
                subject_id = subject.lower().replace(" ", "_")
                existing = self.graph.get_node(subject_id, tenant_id=tenant_id)
                if existing and existing.get("data", {}).get("definition"):
                    has_graph_knowledge = True
                    # Use the stored knowledge
                    definition = existing["data"]["definition"]
                    external_context = [KnowledgeFragment(
                        source="graph_recall",
                        content=definition,
                        identifier=subject,
                        confidence=existing.get("confidence", 0.8),
                    )]
                    logger.info(f"[{self.id}] Recalled knowledge from graph: {subject}")
        
        # Trigger external lookup when:
        # - No concepts grounded, OR
        # - Few concepts (<3) with high curiosity, OR
        # - Query is a question and we have no strong inference confidence
        # - Statement contains unknown nouns (proactive learning)
        # BUT only if we don't already have graph knowledge
        # AND only for genuine knowledge queries, not social/phatic phrases
        
        # Social/phatic utterances that should NOT trigger knowledge lookup
        SOCIAL_PATTERNS = [
            "how are you", "how's it going", "what's up", "how do you do",
            "hi", "hello", "hey", "bye", "goodbye", "thanks", "thank you",
            "good morning", "good afternoon", "good evening", "good night",
            "nice to meet you", "pleased to meet you", "you're welcome",
        ]
        is_social = isinstance(payload, str) and any(
            payload.lower().strip().startswith(p) or payload.lower().strip() == p
            for p in SOCIAL_PATTERNS
        )
        
        is_question = isinstance(payload, str) and not is_social and any(
            payload.lower().startswith(q) for q in 
            ["what ", "who ", "where ", "when ", "why ", "how ", "do you know", "tell me about", "define ", "can you tell"]
        )
        
        # Check for unknown nouns in statements (proactive learning)
        # Skip for social phrases
        unknown_nouns = []
        if isinstance(payload, str) and not is_question and not is_social:
            unknown_nouns = self._find_unknown_nouns(payload, tenant_id)
        
        # For questions about specific topics without graph knowledge, always try external lookup
        # For general queries, only augment with sparse grounding
        # For statements with unknown nouns, learn about them proactively
        # Skip for social/phatic phrases - they don't need external knowledge
        should_augment = (
            not is_social and 
            not has_graph_knowledge and (
                len(active_concepts) == 0 or 
                (affective_state["curiosity_drive"] > 0.7 and len(active_concepts) < 3) or
                is_question  # Questions about unknown topics should always try external lookup
            )
        )
        
        if should_augment and isinstance(payload, str): # Currently only text lookup supported
            try:
                fragments = self.knowledge_service.search(payload, ctx)
                if fragments:
                    external_context = fragments
                    
                    # Learn from the fetched knowledge
                    for fragment in fragments:
                        self._learn_from_fragment(fragment, payload, tenant_id)
                    
                    logger.info(f"[{self.id}] Knowledge Augmentation found {len(fragments)} items.")
            except Exception as e:
                logger.warning(f"[{self.id}] Knowledge Augmentation error: {e}")
        
        # Proactive learning: look up unknown nouns from statements
        if unknown_nouns and isinstance(payload, str):
            for noun in unknown_nouns[:2]:  # Limit to 2 nouns per statement
                try:
                    # Check if already in graph
                    noun_id = noun.lower().replace(" ", "_")
                    if self.graph.get_node(noun_id, tenant_id=tenant_id):
                        continue
                    
                    # Look up the noun
                    fragments = self.knowledge_service.search(f"What is a {noun}?", ctx)
                    if fragments:
                        for fragment in fragments:
                            self._learn_from_fragment(fragment, f"What is a {noun}?", tenant_id)
                        logger.info(f"[{self.id}] Proactive learning: {noun}")
                except Exception as e:
                    logger.debug(f"[{self.id}] Proactive learning failed for {noun}: {e}")

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
                    tenant_id=tenant_id,
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
        # Use the Conversational Architecture for motivated, coherent response generation
        response_text = ""
        composed_response = None
        
        if self._conversational_mode:
            try:
                response_text = self._generate_conversational_response(
                    payload=payload,
                    ctx=ctx,
                    active_concepts=active_concepts,
                    inferred_knowledge=inferred_knowledge,
                    external_context=external_context,
                    deliberation_result=deliberation_result,
                    tenant_id=tenant_id,
                )
                self.last_thought["composition_source"] = "conversational"
            except Exception as e:
                logger.warning(f"[{self.id}] Conversational generation failed: {e}")
                response_text = ""
        
        # Fallback: Try ResponseComposer (pattern-based with learning)
        if not response_text and self._response_composer:
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
        
        # Ultimate fallback to GenerativeSystem
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

    def _generate_conversational_response(
        self,
        payload: Any,
        ctx: Dict[str, Any],
        active_concepts: List[Tuple[str, float]],
        inferred_knowledge: List[Dict],
        external_context: List,
        deliberation_result: Optional[DeliberationResult],
        tenant_id: Optional[str] = None,
    ) -> str:
        """
        Generate a response using the full Conversational Architecture.
        
        Four-stage pipeline:
        1. Discourse Management: Track dialogue state, identify intent, obligations
        2. Communication Planning: Decide what to communicate based on drives
        3. Compositional Realization: Build language from concepts/relations
        4. Self-Monitoring: Check quality before output
        
        This is where learning manifests as behavior - learned frames, discourse
        markers, and syntactic patterns are used to generate output.
        """
        user_input = ctx.get("original_text", str(payload))
        
        # ===== Stage 1: Discourse Management =====
        # Track where we are in the conversation and what's expected
        
        # Extract entities and topic from grounding
        entities = []
        for concept_id, confidence in active_concepts:
            if confidence > 0.5:
                # Get term from graph
                try:
                    node = self.graph.get_node(concept_id, tenant_id=tenant_id)
                    if node and node.get("term"):
                        entities.append(node["term"])
                except Exception:
                    entities.append(concept_id.replace("_", " "))
        
        # Get topic - simple approach, let learning discover what works
        # Priority: entities from grounding > previous topic > first content word
        topic = None
        
        # 1. First entity from grounding (already semantically relevant)
        if entities:
            for entity in entities:
                if len(entity) > 2 and not entity.startswith("I "):
                    topic = entity
                    break
        
        # 2. Fallback to topic from previous turn (continuity)
        if not topic:
            prev_context = self._discourse_manager.get_topic_context()
            if prev_context.get("current_topic"):
                topic = prev_context["current_topic"]
        
        # 3. Fallback to extracted knowledge subject
        if not topic and self.last_thought.get("extracted_knowledge"):
            extracted = self.last_thought["extracted_knowledge"]
            if extracted:
                topic = extracted[0].subject
        
        # 4. Last resort: first meaningful word from input
        if not topic and isinstance(payload, str):
            words = payload.split()
            for word in words:
                # Skip very short words and common function words
                if len(word) > 3 and word.lower() not in {"what", "when", "where", "which", "that", "this", "have", "been", "will", "would", "could"}:
                    topic = word.rstrip("?!.,")
                    break
        
        # Update discourse state
        dialogue_state = self._discourse_manager.update(
            user_input=user_input,
            extracted_entities=entities,
            extracted_topic=topic,
            bot_response=self.last_interaction.get("response") if self.last_interaction else None,
        )
        
        # Get obligations (what must we address?)
        obligations = self._discourse_manager.get_obligations()
        topic_context = self._discourse_manager.get_topic_context()
        
        logger.debug(f"[{self.id}] Discourse: phase={dialogue_state.phase}, "
                    f"intent={dialogue_state.last_user_intent}, obligations={obligations}")
        
        # ===== Stage 2: Communication Planning =====
        # Decide what to communicate based on drives and content
        
        # Get affective state (the drives)
        affective_state = self.affective.get_state_vector()
        
        # Build working memory representation for planner
        working_memory = {}
        if deliberation_result:
            for cid, concept in self._reasoning._working_memory.items():
                working_memory[cid] = concept
        
        # Prepare inferences for planner
        plan_inferences = []
        for inf in inferred_knowledge:
            plan_inferences.append(inf)
        
        # Create communication plan
        plan = self._communication_planner.plan(
            obligations=obligations,
            working_memory=working_memory,
            affective_state=affective_state,
            inferences=plan_inferences,
            external_knowledge=external_context,
            current_topic=topic,
            topic_context=topic_context,
            user_intent=dialogue_state.last_user_intent,
        )
        
        logger.debug(f"[{self.id}] Plan: goal={plan.primary_goal}, stance={plan.stance}, "
                    f"content={len(plan.content_concepts)}, follow_up={plan.follow_up_goal}")
        
        # ===== Stage 3: Compositional Realization =====
        # Build language from the plan using learned frames
        
        # Add relations from extracted knowledge if not already in plan
        if self.last_thought.get("extracted_knowledge"):
            for rel in self.last_thought["extracted_knowledge"]:
                relation_tuple = (rel.subject, rel.predicate, rel.object)
                if relation_tuple not in plan.content_relations:
                    plan.content_relations.append(relation_tuple)
        
        # Add relations from external knowledge if it's a definition
        if external_context and plan.primary_goal in ("inform", "elaborate"):
            for frag in external_context[:1]:
                content = frag.content if hasattr(frag, 'content') else str(frag)
                if content and topic:
                    # Add as a fact
                    if content not in plan.content_facts:
                        plan.content_facts.append(content)
        
        # Realize the plan as language
        result = self._compositional_realizer.realize(
            plan=plan,
            topic_context=topic_context,
        )
        
        draft_response = result.text
        
        logger.debug(f"[{self.id}] Realization: {len(result.frames_used)} frames, "
                    f"confidence={result.confidence:.2f}")
        
        # ===== Stage 4: Self-Monitoring =====
        # Check quality before output
        
        monitoring = self._self_monitor.evaluate(
            draft=draft_response,
            plan=plan,
            dialogue_state=dialogue_state,
            user_input=user_input,
        )
        
        if not monitoring.passed:
            logger.debug(f"[{self.id}] Self-monitor issues: {monitoring.issues}")
            # Try revision
            revised = self._self_monitor.suggest_revision(
                draft=draft_response,
                result=monitoring,
                plan=plan,
            )
            if revised and revised != draft_response:
                draft_response = revised
                logger.debug(f"[{self.id}] Self-monitor revised response")
        
        # Record response for future repetition detection
        self._self_monitor.record_response(draft_response, topic)
        
        # Update discourse manager with our response
        self._discourse_manager.state.last_bot_response = draft_response
        
        # ===== Learning from Output =====
        # If we used learned frames successfully, reinforce them
        for frame_id in result.frames_used:
            if "learned_" in frame_id:
                self._compositional_realizer.update_frame_success(frame_id, True)
        
        return draft_response

    def _learn_syntactic_frame_from_input(self, text: str, relation: ExtractedRelation) -> None:
        """
        Learn a syntactic frame from observed input.
        
        This is crucial for compositional generation - we learn HOW things
        are expressed so we can express things the same way.
        
        Example:
            Input: "A dolphin is a marine mammal"
            Relation: (dolphin, is_a, marine mammal)
            Learned Frame: "A {subject} is a {object}"
        """
        import re
        
        if not relation.subject or not relation.object:
            return
        
        # Create template by replacing subject/object with placeholders
        template = text
        
        # Case-insensitive replacement
        subj_pattern = re.escape(relation.subject)
        obj_pattern = re.escape(relation.object)
        
        # Replace subject first (usually longer), then object
        # Sort by length to avoid partial replacements
        replacements = sorted([
            (subj_pattern, "{subject}", relation.subject),
            (obj_pattern, "{object}", relation.object),
        ], key=lambda x: len(x[2]), reverse=True)
        
        for pattern, placeholder, original in replacements:
            template = re.sub(f"(?i)\\b{pattern}\\b", placeholder, template, count=1)
        
        # Only learn if we successfully created a template with both placeholders
        if "{subject}" in template and "{object}" in template:
            # Clean up the template
            template = template.strip()
            
            # Normalize predicate
            predicate = relation.predicate.lower().replace(" ", "_")
            
            # Teach the compositional realizer
            self._compositional_realizer.learn_frame(
                predicate=predicate,
                template=template,
            )
            
            logger.debug(f"[{self.id}] Learned syntactic frame for '{predicate}': {template}")


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
        
        # Robust flattening: unwrap nested lists until we have a flat list of numbers
        while isinstance(thought_vector, list) and thought_vector:
            if isinstance(thought_vector[0], (list, tuple)):
                thought_vector = thought_vector[0]
            elif hasattr(thought_vector[0], 'tolist'):
                # Element is a tensor/ndarray
                thought_vector = thought_vector[0].tolist()
                if not isinstance(thought_vector, list):
                    thought_vector = [thought_vector]
            else:
                # First element is a scalar - we're flat
                break
        
        # Final validation: ensure all elements are numeric
        if not thought_vector or not all(isinstance(x, (int, float)) for x in thought_vector):
            logger.warning(f"[{self.id}] Plasticity: Invalid thought_vector type: {type(thought_vector)}")
            return

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
            topic, topic_confidence = self.topic_extractor.extract_topic(original_text)
            if topic:
                # Add topic as a concept match with confidence from extractor
                t_id = topic.lower().replace(" ", "_")
                # Use topic_confidence but cap at 0.95
                matches.append((t_id, min(topic_confidence, 0.95)))
                logger.info(f"[{self.id}] Neural Topic Grounding found: {topic} (conf: {topic_confidence:.2f})")

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
    
    def _learn_from_fragment(
        self,
        fragment: KnowledgeFragment,
        original_query: str,
        tenant_id: Optional[str] = None
    ) -> int:
        """
        Learn from an external knowledge fragment.
        
        This extracts semantic relations, concepts, and vocabulary from
        external knowledge (Wikipedia, Dictionary) and adds them to the
        graph and pattern stores.
        
        Args:
            fragment: The knowledge fragment to learn from
            original_query: The query that triggered the lookup
            tenant_id: Tenant context for storage
            
        Returns:
            Number of items learned (relations + concepts)
        """
        learned_count = 0
        content = fragment.content
        identifier = fragment.identifier
        source = fragment.source
        
        if not content:
            return 0
        
        logger.debug(f"[{self.id}] Learning from {source}: {identifier[:50]}...")
        
        # 1. Extract the main subject from the query
        subject = self._extract_subject(original_query, identifier)
        
        # 2. Extract semantic relations from the content
        extractor = SemanticExtractor()
        relations = extractor.extract(content)
        
        # 3. Add the main concept to the graph
        try:
            subject_id = subject.lower().replace(" ", "_")
            
            # Create concept node with embedding
            subject_embedding = self.encoder.encode(subject.split())
            embedding_list = subject_embedding.flatten().tolist()[:64]  # Truncate for storage
            
            self.graph.add_node(
                node_id=subject_id,
                node_type="concept",
                term=subject,
                confidence=fragment.confidence,
                data={
                    "source": source,
                    "definition": content[:500],  # Store first 500 chars
                    "embedding": embedding_list,
                },
                tenant_id=tenant_id,
            )
            learned_count += 1
            logger.debug(f"[{self.id}] Added concept node: {subject}")
        except Exception as e:
            logger.debug(f"[{self.id}] Failed to add concept {subject}: {e}")
        
        # 4. Add extracted relations to the graph
        for rel in relations:
            try:
                # Normalize IDs
                subj_id = rel.subject.lower().replace(" ", "_")
                obj_id = rel.object.lower().replace(" ", "_")
                
                # Add object node if it doesn't exist
                if not self.graph.get_node(obj_id, tenant_id=tenant_id):
                    self.graph.add_node(
                        node_id=obj_id,
                        node_type="concept",
                        term=rel.object,
                        confidence=rel.confidence * 0.8,  # Slightly lower for inferred
                        data={"source": f"extracted_from_{source}"},
                        tenant_id=tenant_id,
                    )
                
                # Add the relationship edge
                self.graph.add_edge(
                    from_id=subj_id,
                    to_id=obj_id,
                    relation_type=rel.predicate,
                    confidence=rel.confidence,
                    tenant_id=tenant_id,
                )
                learned_count += 1
                logger.debug(f"[{self.id}] Added relation: {rel.subject} -[{rel.predicate}]-> {rel.object}")
            except Exception as e:
                logger.debug(f"[{self.id}] Failed to add relation: {e}")
        
        # 5. Add to response pattern store for future retrieval
        if self._response_composer and self._response_composer.patterns:
            try:
                # Create a concise definition response
                first_sentence = content.split('.')[0] + '.'
                self._response_composer.patterns.add_pattern(
                    trigger_context=original_query,
                    response_text=first_sentence,
                    success_score=fragment.confidence,
                    intent="learned_definition",
                )
                learned_count += 1
                logger.debug(f"[{self.id}] Added response pattern for: {original_query[:30]}")
            except Exception as e:
                logger.debug(f"[{self.id}] Failed to add response pattern: {e}")
        
        # 6. Extract and learn vocabulary (key terms)
        key_terms = self._extract_key_terms_from_content(content)
        for term in key_terms[:5]:  # Limit to top 5 terms
            try:
                term_id = term.lower().replace(" ", "_")
                if not self.graph.get_node(term_id, tenant_id=tenant_id):
                    term_embedding = self.encoder.encode(term.split())
                    self.graph.add_node(
                        node_id=term_id,
                        node_type="term",
                        term=term,
                        confidence=0.6,  # Lower confidence for extracted terms
                        data={
                            "source": f"vocabulary_{source}",
                            "embedding": term_embedding.flatten().tolist()[:64],
                        },
                        tenant_id=tenant_id,
                    )
                    learned_count += 1
            except Exception as e:
                logger.debug(f"[{self.id}] Failed to add term {term}: {e}")
        
        logger.info(f"[{self.id}] Learned {learned_count} items from {source}: {identifier}")
        return learned_count
    
    def _extract_subject(self, query: str, identifier: str) -> str:
        """Extract the main subject from a query or use the identifier."""
        q = query.lower()
        # Sorted by length (longest first) like knowledge_service
        prefixes = [
            "can you tell me about ", "can you explain ",
            "do you know about ", "do you know what ",
            "tell me about ", "tell me what ",
            "what is the ", "what is an ", "what is a ", "what is ",
            "what are the ", "what are ",
            "definition of ",
            "who was ", "who are ", "who is ",
            "where was ", "where is ",
            "when was ", "when did ",
            "why does ", "why do ", "why is ",
            "how does ", "how do ", "how is ",
            "describe ", "explain ", "define ",
        ]
        for prefix in prefixes:
            if q.startswith(prefix):
                result = query[len(prefix):].strip("?.! ")
                # Strip leading articles
                for article in ["a ", "an ", "the "]:
                    if result.lower().startswith(article):
                        result = result[len(article):]
                return result.strip()
        return identifier
    
    def _extract_key_terms_from_content(self, content: str) -> List[str]:
        """Extract key terms (proper nouns, technical terms) from content."""
        import re
        terms = []
        
        # Find proper nouns (capitalized words not at sentence start)
        words = content.split()
        for i, word in enumerate(words[1:], 1):  # Skip first word
            # Check if capitalized and not common word
            if word[0].isupper() and len(word) > 2:
                clean = re.sub(r'[^\w]', '', word)
                if clean and clean.lower() not in {'the', 'a', 'an', 'is', 'are', 'was', 'were'}:
                    terms.append(clean)
        
        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
        
        return unique_terms
    
    def _find_unknown_nouns(
        self,
        text: str,
        tenant_id: Optional[str] = None
    ) -> List[str]:
        """
        Find nouns in text that are not yet known in the graph.
        
        Uses simple heuristics to identify likely nouns:
        - Capitalized words (proper nouns)
        - Words after articles (a, an, the)
        - Common noun patterns
        
        Args:
            text: Input text to analyze
            tenant_id: Tenant context
            
        Returns:
            List of unknown noun terms
        """
        import re
        
        unknown = []
        words = text.split()
        
        # Common function words to skip
        skip_words = {
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'this', 'that', 'these', 'those',
            'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'into', 'through', 'during',
            'very', 'really', 'quite', 'just', 'also', 'too', 'so', 'today', 'yesterday',
            'saw', 'see', 'seen', 'look', 'looked', 'pretty', 'beautiful', 'nice', 'good',
        }
        
        for i, word in enumerate(words):
            # Clean the word
            clean = re.sub(r'[^\w]', '', word).lower()
            if not clean or len(clean) < 3 or clean in skip_words:
                continue
            
            # Check if it's after an article (likely a noun)
            is_after_article = i > 0 and words[i-1].lower() in ('a', 'an', 'the')
            
            # Check if capitalized (proper noun) - not at sentence start
            is_proper_noun = word[0].isupper() and i > 0 and not words[i-1].endswith('.')
            
            if is_after_article or is_proper_noun:
                # Check if known in graph
                node_id = clean.replace(" ", "_")
                existing = self.graph.get_node(node_id, tenant_id=tenant_id)
                
                if not existing or not existing.get("data", {}).get("definition"):
                    unknown.append(clean)
        
        # Deduplicate
        seen = set()
        unique = []
        for noun in unknown:
            if noun not in seen:
                seen.add(noun)
                unique.append(noun)
        
        return unique

    # ===== Action Planning API =====
    
    def plan_actions(
        self,
        goal: str,
        current_state: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> Optional[ExecutionPlan]:
        """
        Plan a sequence of actions to achieve a goal.
        
        Uses PMFlow's agentic physics to trace a trajectory from current state
        toward the goal, grounding waypoints to registered actions.
        
        Args:
            goal: Natural language goal description (e.g., "sign up to Moltbook")
            current_state: Optional description of current state
            context: Additional context for filling action arguments
            tenant_id: Optional tenant ID for multi-tenant
            
        Returns:
            ExecutionPlan with steps, or None if planning fails
            
        Example:
            plan = brain.plan_actions("copy file.txt to backup/")
            for step in plan.steps:
                print(f"{step.step_num}: {step.action.name} -> {step.action.tool_binding}")
        """
        if not self._action_planner:
            logger.warning("Action planning not enabled")
            return None
        
        return self._action_planner.plan(
            goal=goal,
            current_state=current_state,
            context=context or {},
            tenant_id=tenant_id,
        )
    
    def register_action(
        self,
        name: str,
        description: str,
        tool_binding: str,
        arg_template: Optional[Dict[str, Any]] = None,
        preconditions: Optional[List[str]] = None,
        effects: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Register an action for planning.
        
        Actions are stored as nodes in the knowledge graph with embeddings
        computed from their description, enabling physics-based grounding.
        
        Args:
            name: Action name (e.g., "navigate_to_url")
            description: What this action does (used for semantic embedding)
            tool_binding: Tool name in LocalToolsTransport to call
            arg_template: Default/required arguments template
            preconditions: State requirements before action can run
            effects: State changes after action completes
            tenant_id: Optional tenant ID
            
        Returns:
            action_id if successful, None if planning not enabled
        """
        if not self._action_planner:
            logger.warning("Action planning not enabled")
            return None
        
        return self._action_planner.register_action(
            name=name,
            description=description,
            tool_binding=tool_binding,
            arg_template=arg_template,
            preconditions=preconditions,
            effects=effects,
            tenant_id=tenant_id,
        )
    
    def get_execution_commands(self, plan: ExecutionPlan) -> List[Dict[str, Any]]:
        """
        Convert a plan to execution format for LocalToolsTransport.
        
        Returns list of {"action": tool_name, "args": {...}} dicts
        ready for somatic layer execution.
        """
        if not self._action_planner:
            return []
        return self._action_planner.to_execution_format(plan)
    
    def execute_plan(
        self,
        plan: ExecutionPlan,
        transport,  # LocalToolsTransport or compatible
        stop_on_error: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Execute a planned action sequence.
        
        Args:
            plan: ExecutionPlan from plan_actions()
            transport: Tool transport with call() method
            stop_on_error: Stop on first error if True
            
        Returns:
            List of step results
            
        Example:
            plan = brain.plan_actions("backup important files")
            results = brain.execute_plan(plan, tools_transport)
        """
        if not self._action_planner:
            logger.warning("Action planning not enabled")
            return []
        
        return self._action_planner.execute_plan(plan, transport, stop_on_error)

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
        
        # Add action planner stats if available
        if self._action_planner:
            stats["action_planning_enabled"] = True
            stats["registered_actions"] = len(self._action_planner._action_cache)
        else:
            stats["action_planning_enabled"] = False
        
        return stats

    def relational_sidecar(self, sql_ctx: Any) -> Optional[Any]:
        return None

__all__ = ["CognitiveStage"]
