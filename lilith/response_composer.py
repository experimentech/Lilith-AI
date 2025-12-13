"""
Response Composer - PMFlow-Guided Response Assembly

Composes responses using learned patterns weighted by PMFlow activations.
Maintains coherence using working memory state.

This is the symmetric counterpart to semantic understanding:
  Understanding: Text → Embedding → Retrieve context
  Generation: Context → Retrieve patterns → Compose text

Pure neuro-symbolic - no LLM!
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Union, TYPE_CHECKING
import numpy as np

from .response_fragments import ResponseFragmentStore, ResponsePattern
from .conversation_state import ConversationState
from .conversation_history import ConversationHistory
from .bnn_intent_classifier import BNNIntentClassifier
from .intake import NoiseNormalizer  # For query cleaning at intake layer

# Import MultiTenantFragmentStore for type hints
if TYPE_CHECKING:
    from .multi_tenant_store import MultiTenantFragmentStore

# Optional: Import query pattern matcher for query understanding
try:
    from .query_pattern_matcher import QueryPatternMatcher, QueryMatch
    QUERY_PATTERN_MATCHING_AVAILABLE = True
except ImportError:
    QUERY_PATTERN_MATCHING_AVAILABLE = False

# Optional: Import contrastive learner for semantic embeddings
try:
    from .contrastive_learner import ContrastiveLearner
    CONTRASTIVE_LEARNING_AVAILABLE = True
except ImportError:
    CONTRASTIVE_LEARNING_AVAILABLE = False

# Optional: Import concept store and template composer for compositional responses
try:
    from .production_concept_store import ProductionConceptStore
    from .template_composer import TemplateComposer
    COMPOSITIONAL_AVAILABLE = True
except ImportError:
    COMPOSITIONAL_AVAILABLE = False

# Optional: Import grammar stage for sophisticated composition
try:
    from .syntax_stage_bnn import SyntaxStage
    GRAMMAR_AVAILABLE = True
except ImportError:
    GRAMMAR_AVAILABLE = False

# Optional: Import knowledge augmentation for external lookups
try:
    from .knowledge_augmenter import KnowledgeAugmenter
    KNOWLEDGE_AUGMENTATION_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AUGMENTATION_AVAILABLE = False

# Optional: Import modal routing for multi-modal queries
try:
    from .modal_classifier import ModalClassifier, Modality
    from .math_backend import MathBackend
    MODAL_ROUTING_AVAILABLE = True
except ImportError:
    MODAL_ROUTING_AVAILABLE = False
    Modality = None  # type: ignore

# Optional: Import reasoning stage for deliberative thinking
try:
    from .reasoning_stage import ReasoningStage, DeliberationResult
    REASONING_AVAILABLE = True
except ImportError:
    REASONING_AVAILABLE = False

# Optional: Import pragmatic templates for conversational patterns
try:
    from .pragmatic_templates import PragmaticTemplateStore
    PRAGMATIC_TEMPLATES_AVAILABLE = True
except ImportError:
    PRAGMATIC_TEMPLATES_AVAILABLE = False


@dataclass
class ComposedResponse:
    """A response composed from learned patterns"""
    text: str                           # Final response text
    fragment_ids: List[str]            # Which patterns were used
    composition_weights: List[float]   # How much each pattern contributed
    coherence_score: float             # How well it fits working memory
    primary_pattern: Optional[ResponsePattern] = None   # Main pattern used (can be None for fallback)
    confidence: float = 1.0            # Retrieval confidence (for teaching detection)
    is_fallback: bool = False          # Whether this was a fallback response
    is_low_confidence: bool = False    # Whether best pattern was below threshold
    modality: Optional['Modality'] = None  # Query modality (LINGUISTIC, MATH, CODE, etc.)
    

class ResponseComposer:
    """
    Composes responses using PMFlow-guided assembly.
    
    Key insight: Response generation uses the SAME mechanisms
    as semantic understanding - retrieve, weight, compose.
    
    No templates, no LLM - pure learned behavior!
    """
    
    def __init__(
        self, 
        fragment_store: Union[ResponseFragmentStore, 'MultiTenantFragmentStore'],
        conversation_state: ConversationState,
        conversation_history: Optional[ConversationHistory] = None,
        composition_mode: str = "weighted_blend",
        use_grammar: bool = False,
        semantic_encoder = None,
        enable_knowledge_augmentation: bool = True,
        concept_store: Optional['ProductionConceptStore'] = None,
        enable_compositional: bool = True,
        enable_modal_routing: bool = True,
        pragmatic_templates: Optional['PragmaticTemplateStore'] = None,
        enable_pragmatic_templates: bool = True,
        knowledge_timeout_seconds: float = 3.0
    ):
        """
        Initialize response composer.
        
        Args:
            fragment_store: Store of learned response patterns
            conversation_state: Working memory state
            conversation_history: Short-term memory (recent turns, sliding window)
            composition_mode: How to compose responses
                - "best_match": Use highest-weighted pattern only
                - "weighted_blend": Blend multiple patterns
                - "grammar_guided": Use grammatical templates (requires syntax stage)
                - "adaptive": Choose based on confidence
                - "parallel": Try both patterns AND concepts, use best
                - "pragmatic": Use pragmatic templates + concepts (Layer 4 restructured)
            use_grammar: Enable grammar-guided composition
            semantic_encoder: BioNN encoder for intent clustering (optional)
            enable_knowledge_augmentation: Enable external knowledge lookup (Wikipedia, etc.)
            concept_store: Optional ConceptStore for compositional responses
            enable_compositional: Enable compositional response generation
            enable_modal_routing: Enable modal routing (math, code, etc.)
            pragmatic_templates: Optional PragmaticTemplateStore for conversational patterns
            enable_pragmatic_templates: Enable pragmatic template-based composition
        """
        self.fragments = fragment_store
        self.state = conversation_state
        self.conversation_history = conversation_history
        self.composition_mode = composition_mode
        
        # Store semantic encoder for later use (relevance validation, etc.)
        self.semantic_encoder = semantic_encoder
        
        # Initialize normalizer for query cleaning (INTAKE layer)
        self.normalizer = NoiseNormalizer()
        
        # Initialize BioNN intent classifier if encoder provided
        self.intent_classifier = None
        if semantic_encoder is not None:
            self.intent_classifier = BNNIntentClassifier(semantic_encoder)
            print("  🎯 BioNN intent clustering enabled!")
        
        # Initialize syntax stage if available and requested
        self.syntax_stage = None
        if use_grammar and GRAMMAR_AVAILABLE:
            self.syntax_stage = SyntaxStage()
            print("  📝 BioNN-based syntax stage enabled!")
        elif use_grammar and not GRAMMAR_AVAILABLE:
            print("  ⚠️  Syntax stage not available, falling back to standard composition")
        
        # Initialize knowledge augmentation lazily to avoid heavy startup
        self.enable_knowledge_augmentation = enable_knowledge_augmentation
        self.knowledge_augmenter = None
        self._knowledge_augmenter_factory = None
        if enable_knowledge_augmentation and KNOWLEDGE_AUGMENTATION_AVAILABLE:
            self._knowledge_augmenter_factory = lambda: KnowledgeAugmenter(enabled=True, timeout_seconds=knowledge_timeout_seconds)
            print("  🌐 External knowledge augmentation enabled (lazy load)")
        elif enable_knowledge_augmentation and not KNOWLEDGE_AUGMENTATION_AVAILABLE:
            print("  ⚠️  Knowledge augmentation not available")
        
        # Initialize compositional architecture (concepts + templates)
        self.concept_store = concept_store
        self.template_composer = None
        if enable_compositional and COMPOSITIONAL_AVAILABLE and concept_store:
            self.template_composer = TemplateComposer()
            print("  🧩 Compositional architecture enabled (concepts + templates)!")
        elif enable_compositional and not COMPOSITIONAL_AVAILABLE:
            print("  ⚠️  Compositional architecture not available")
        
        # Initialize pragmatic templates for conversational patterns (Layer 4 restructuring)
        self.pragmatic_templates = pragmatic_templates
        if self.pragmatic_templates is None and enable_pragmatic_templates and PRAGMATIC_TEMPLATES_AVAILABLE:
            # Auto-create with defaults if not provided
            self.pragmatic_templates = PragmaticTemplateStore()
            print("  💬 Pragmatic templates enabled (conversational patterns)!")
        elif enable_pragmatic_templates and not PRAGMATIC_TEMPLATES_AVAILABLE:
            print("  ⚠️  Pragmatic templates not available")
        
        # Initialize modal routing (math, code, etc.)
        self.modal_classifier = None
        self.math_backend = None
        if enable_modal_routing and MODAL_ROUTING_AVAILABLE:
            self.modal_classifier = ModalClassifier()
            try:
                self.math_backend = MathBackend()
                print("  🔢 Math backend enabled (symbolic computation)!")
            except RuntimeError as e:
                print(f"  ⚠️  Math backend not available: {e}")
        elif enable_modal_routing and not MODAL_ROUTING_AVAILABLE:
            print("  ⚠️  Modal routing not available")
        
        # Initialize query pattern matcher for query understanding
        self.query_matcher = None
        if QUERY_PATTERN_MATCHING_AVAILABLE:
            self.query_matcher = QueryPatternMatcher()
            print("  🔍 Query pattern matching enabled!")
        
        # Initialize reasoning stage for deliberative thinking
        self.reasoning_stage = None
        if REASONING_AVAILABLE and semantic_encoder is not None:
            try:
                self.reasoning_stage = ReasoningStage(
                    encoder=semantic_encoder,
                    concept_store=concept_store,
                    deliberation_steps=3,  # Quick deliberation by default
                    convergence_threshold=0.75
                )
            except Exception as e:
                print(f"  ⚠️  Reasoning stage not available: {e}")
        
        # Contrastive learner for online semantic training
        self.contrastive_learner = None
        
        # Track metrics for pattern vs concept approaches
        self.metrics = {
            'pattern_count': 0,
            'concept_count': 0,
            'pattern_success': 0,
            'concept_success': 0,
            'parallel_uses': 0,
            'math_count': 0,  # Track math backend usage
            'pragmatic_count': 0  # NEW: Track pragmatic template usage
        }
        
        # Track last query and response for success learning
        self.last_query = None
        self.last_response = None
        self.last_approach = None  # 'pattern', 'concept', 'parallel', or 'math'

    def _ensure_knowledge_augmenter(self):
        """Lazily instantiate knowledge augmenter to avoid startup cost."""
        if not self.enable_knowledge_augmentation or not KNOWLEDGE_AUGMENTATION_AVAILABLE:
            return None
        if self.knowledge_augmenter is None and self._knowledge_augmenter_factory:
            try:
                self.knowledge_augmenter = self._knowledge_augmenter_factory()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"  ⚠️  Knowledge augmentation unavailable: {exc}")
                self.enable_knowledge_augmentation = False
        return self.knowledge_augmenter
    
    def load_contrastive_weights(self, path: str) -> bool:
        """
        Load pre-trained contrastive weights into the semantic encoder.
        
        This improves semantic similarity understanding by loading weights
        trained via contrastive learning (see tools/train_contrastive.py).
        
        Args:
            path: Path to saved contrastive learner state (without extension)
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not CONTRASTIVE_LEARNING_AVAILABLE:
            print("  ⚠️  Contrastive learning not available")
            return False
        
        if self.intent_classifier is None:
            print("  ⚠️  No semantic encoder - can't load contrastive weights")
            return False
        
        from pathlib import Path as PathLib
        path_obj = PathLib(path)
        
        if not path_obj.with_suffix('.json').exists():
            print(f"  ⚠️  Contrastive weights not found at {path}")
            return False
        
        try:
            self.contrastive_learner = ContrastiveLearner(
                self.intent_classifier.encoder
            )
            self.contrastive_learner.load(path_obj)
            print(f"  🎓 Loaded contrastive weights: {len(self.contrastive_learner.pairs)} pairs, "
                  f"{len(self.contrastive_learner.metrics_history)} epochs trained")
            return True
        except Exception as e:
            print(f"  ⚠️  Failed to load contrastive weights: {e}")
            return False
    
    def add_semantic_correction(self, concept_a: str, concept_b: str, should_be_similar: bool):
        """
        Add a user correction to teach semantic relationships.
        
        This enables online learning from user feedback without full retraining.
        
        Args:
            concept_a: First concept
            concept_b: Second concept  
            should_be_similar: True if concepts should be similar, False if different
        """
        if self.contrastive_learner is None:
            if not CONTRASTIVE_LEARNING_AVAILABLE or self.intent_classifier is None:
                return
            # Initialize learner on first correction
            self.contrastive_learner = ContrastiveLearner(self.intent_classifier.encoder)
        
        self.contrastive_learner.add_user_correction(concept_a, concept_b, should_be_similar)
        
        # Do incremental update
        relationship = "positive" if should_be_similar else "hard_negative"
        self.contrastive_learner.incremental_update(
            [(concept_a, concept_b, relationship)],
            steps=5
        )
        print(f"  📚 Learned: '{concept_a}' {'≈' if should_be_similar else '≠'} '{concept_b}'")
    
    def record_conversation_outcome(self, success: bool):
        """
        Record the outcome of the last conversation turn.
        
        Call this after each response to enable success-based learning.
        The system learns which patterns work for which queries.
        
        Args:
            success: True if conversation continued well, False if it broke down
                     
        Examples of success signals:
            - User continues the topic → True
            - User asks follow-up question → True
            - User changes topic abruptly → False
            - User says "what?" or "huh?" → False
        """
        # Don't learn math/code responses (prevents database pollution)
        if self.last_approach == 'math':
            # Math is exact computation, not linguistic learning
            return
        
        # Track success for metrics
        if self.last_approach == 'pattern':
            self.metrics['pattern_success'] += 1 if success else 0
        elif self.last_approach == 'concept':
            self.metrics['concept_success'] += 1 if success else 0
        
        # Record pattern-based learning if applicable
        if self.last_query and self.last_response and hasattr(self.fragments, 'record_conversation_outcome'):
            # Get the primary pattern ID that was used
            if self.last_response.primary_pattern:
                pattern_id = self.last_response.primary_pattern.fragment_id
                self.fragments.record_conversation_outcome(
                    self.last_query,
                    pattern_id,
                    success
                )
    
    def cluster_patterns(self):
        """Build intent clusters from learned patterns using BioNN embeddings."""
        if self.intent_classifier is None:
            print("  ⚠️  Intent classifier not available - skipping clustering")
            return
        
        print("  🎯 Clustering patterns by semantic intent...")
        clusters = self.intent_classifier.cluster_patterns(self.fragments.patterns)
        
        stats = self.intent_classifier.get_stats()
        print(f"  ✅ Created {stats['total_clusters']} intent clusters")
        print(f"     Avg cluster size: {stats['avg_cluster_size']:.1f}")
        print(f"     Avg coherence: {stats['avg_coherence']:.3f}")
        
        return clusters
        
    def compose_response(
        self, 
        context: str = "",
        user_input: str = "",
        topk: int = 5,
        use_intent_filtering: bool = False,  # Disabled: BioNN intent classification unreliable on user inputs
        use_semantic_retrieval: bool = True,  # ENABLED: BioNN + success-based learning (OPEN BOOK EXAM)
        semantic_weight: float = 0.5  # Balanced: 50% BioNN semantics, 50% keywords
    ) -> ComposedResponse:
        """
        Generate response through learned composition.
        
        Args:
            context: Current conversation context (from semantic stage)
            user_input: Raw user input (for direct references)
            topk: Number of patterns to consider
            use_intent_filtering: Use BioNN intent classification to filter patterns
            use_semantic_retrieval: Use BioNN embeddings for similarity (OPEN BOOK EXAM)
            semantic_weight: Weight for semantic similarity (0.0=keywords only, 1.0=semantic only)
            
        Returns:
            ComposedResponse with text and metadata
        """
        # Track query for success learning
        self.last_query = user_input if user_input else context
        
        # FAST-TRACK: Handle simple greetings immediately without heavy processing
        if user_input:
            user_lower = user_input.lower().strip()
            simple_greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
            if user_lower in simple_greetings or (len(user_lower.split()) == 1 and user_lower in ["hello", "hi", "hey"]):
                # Use pragmatic template for greeting if available
                if self.pragmatic_templates:
                    greeting_response = self._compose_with_pragmatic_templates(context, user_input)
                    if greeting_response and greeting_response.confidence > 0.70:
                        self.last_response = greeting_response
                        self.last_approach = 'pragmatic'
                        return greeting_response
                # Fallback to simple greeting response
                greeting_texts = ["Hello! How can I help you?", "Hi! What would you like to talk about?", "Hey! What's up?"]
                import random
                return ComposedResponse(
                    text=random.choice(greeting_texts),
                    fragment_ids=["greeting_fasttrack"],
                    composition_weights=[1.0],
                    coherence_score=0.95,
                    primary_pattern=None,
                    confidence=0.95,
                    is_fallback=False,
                    is_low_confidence=False
                )
        
        # MODAL ROUTING: Check if query is mathematical/code/etc.
        if self.modal_classifier and user_input:
            modality, confidence = self.modal_classifier.classify(user_input)
            
            # Route to math backend if mathematical query
            if modality == Modality.MATH and self.math_backend and confidence > 0.60:
                math_response = self._compose_math_response(user_input)
                if math_response:  # Math computation succeeded
                    self.last_response = math_response
                    self.last_approach = 'math'
                    return math_response
                # If math backend failed, fall through to linguistic
        
        # PRAGMATIC MODE: Use template + concept composition (Layer 4 restructured)
        # BUT: Don't immediately return - compare with pattern-based approach
        pragmatic_response = None
        if self.composition_mode == "pragmatic" and self.pragmatic_templates and self.concept_store:
            pragmatic_response = self._compose_with_pragmatic_templates(context, user_input)
            # Don't return yet - compare with pattern-based first
        
        # PARALLEL MODE: Try both pattern-based AND concept-based approaches
        if self.composition_mode == "parallel" and self.concept_store is not None:
            return self._compose_parallel(context, user_input, topk)
        
        # Standard pattern-based composition
        pattern_response = self._compose_from_patterns_internal(
            context, user_input, topk, 
            use_intent_filtering, use_semantic_retrieval, semantic_weight
        )
        
        # PRAGMATIC MODE: Compare pragmatic vs pattern-based
        if pragmatic_response and pragmatic_response.confidence >= 0.70:
            # Debug logging
            print(f"  🔍 Template vs Pattern comparison:")
            print(f"     Template confidence: {pragmatic_response.confidence:.3f}")
            print(f"     Pattern confidence: {pattern_response.confidence:.3f}")
            print(f"     Pattern is_fallback: {pattern_response.is_fallback}")
            print(f"     Pattern is_low_confidence: {pattern_response.is_low_confidence}")
            
            # Pragmatic template has good confidence
            if pattern_response.is_fallback or pattern_response.is_low_confidence:
                # Pattern-based failed or low confidence, use pragmatic
                print(f"  → Using template (pattern failed/low confidence)")
                self.last_response = pragmatic_response
                self.last_approach = 'pragmatic'
                self.metrics['pragmatic_count'] = self.metrics.get('pragmatic_count', 0) + 1
                return pragmatic_response
            elif pattern_response.confidence > 0.85:
                # Pattern has very high confidence (exact/near-exact match) - prefer it
                print(f"  → Using pattern (very high confidence: {pattern_response.confidence:.3f})")
                return pattern_response
            elif pragmatic_response.confidence > pattern_response.confidence + 0.15:
                # Pragmatic is significantly better - use it
                print(f"  → Using template (significantly better)")
                self.last_response = pragmatic_response
                self.last_approach = 'pragmatic'
                self.metrics['pragmatic_count'] = self.metrics.get('pragmatic_count', 0) + 1
                return pragmatic_response
            # Otherwise use pattern-based (comparable confidence)
            print(f"  → Using pattern (comparable/better confidence)")
        
        return pattern_response
    
    def _compose_from_patterns_internal(
        self,
        context: str,
        user_input: str,
        topk: int,
        use_intent_filtering: bool,
        use_semantic_retrieval: bool,
        semantic_weight: float
    ) -> ComposedResponse:
        """
        Internal method for pattern-based composition.
        Separated to avoid recursion with parallel mode.
        """
        
        # 0. INTAKE LAYER: Clean query of filler phrases before any processing
        # This resolves "I mean, what can you do?" → "what can you do?"
        # NOTE: This is LANGUAGE-LEVEL preprocessing, happens at intake
        cleaned_user_input = user_input
        if user_input:
            cleaned_user_input = self.normalizer.clean_query(user_input)
            if cleaned_user_input.lower() != user_input.lower().strip():
                print(f"  🧹 Cleaned query: '{user_input}' → '{cleaned_user_input}'")
        
        # 0.5 CONVERSATION HISTORY: Check for repetition and build conversational context
        # This enables "As you mentioned..." style continuity and prevents repetitive responses
        conversational_context = None
        previous_response = None
        is_repetition = False
        
        if self.conversation_history and cleaned_user_input:
            # Check if user is repeating themselves
            is_repetition = self.conversation_history.detect_repetition(cleaned_user_input)
            
            if is_repetition:
                print(f"  🔁 Detected query repetition - varying response")
                # User is repeating their query - we should acknowledge this
                # and try a different approach or ask for clarification
                conversational_context = "user_repeated_query"
            
            # Get recent turns for context awareness
            recent_turns = self.conversation_history.get_recent_turns(n=3)
            if recent_turns:
                # Build context from recent conversation
                # This enables reference to previous topics like "As you mentioned..."
                previous_response = recent_turns[-1].bot_response
                
                # Extract topics from recent turns for continuity
                recent_topics = []
                for turn in recent_turns:
                    user_msg = turn.user_input.lower()
                    # Simple topic extraction (could be enhanced)
                    if user_msg:
                        recent_topics.append(user_msg)
                
                if recent_topics:
                    conversational_context = f"recent_topics: {'; '.join(recent_topics[-2:])}"
                    print(f"  💬 Conversation context: {len(recent_turns)} recent turns")
        
        # 0.7 PERSONALITY BIAS: Inject limbic-style embedding bias from session
        # This modulates BNN retrieval based on interests/aversions
        personality_bias_fn = None
        if hasattr(self, '_personality_bias_fn'):
            personality_bias_fn = self._personality_bias_fn
        
        # 1. SYMBOLIC REASONING: Deliberate on cleaned query (core thinking layer)
        # This is LANGUAGE-AGNOSTIC - works on PMFlow embeddings and concepts
        deliberation_result = None
        if self.reasoning_stage and cleaned_user_input:
            try:
                deliberation_result = self.reasoning_stage.deliberate(
                    query=cleaned_user_input,
                    context=context,
                    max_steps=3  # Quick deliberation
                )
                
                # Log deliberation insights
                if deliberation_result.inferences:
                    print(f"🧠 Deliberated for {deliberation_result.deliberation_steps} steps")
                    
                    # Show key inferences
                    for inf in deliberation_result.inferences[:3]:  # Top 3
                        if inf.inference_type == "connection":
                            print(f"  📎 Found {len(deliberation_result.inferences)} connections:")
                            for conn in deliberation_result.inferences[:3]:
                                print(f"     • {conn.conclusion}")
                            break
                    
                    # Show implications
                    implications = [i for i in deliberation_result.inferences if i.inference_type == "implication"]
                    if implications:
                        print(f"  ➡️  Found {len(implications)} implications")
                        
            except Exception as e:
                print(f"  ⚠️ Deliberation failed: {e}")
        
        # 2. QUERY PATTERN MATCHING - Extract query structure and intent
        # Use CLEANED query for better pattern matching
        query_match = None
        main_concept = None
        intent_hint = None  # Initialize intent hint
        
        # Use deliberation result if available (symbolic reasoning trumps pattern matching)
        if deliberation_result and deliberation_result.resolved_intent:
            intent_hint = deliberation_result.resolved_intent
            main_concept = deliberation_result.focus_concept
            print(f"  🎯 Reasoning resolved intent: {intent_hint}")
            if main_concept:
                print(f"  🔍 Focus concept: {main_concept}")
        elif self.query_matcher and cleaned_user_input:
            query_match = self.query_matcher.match_query(cleaned_user_input)
            if query_match and query_match.confidence > 0.75:
                # Extracted structural information from query
                main_concept = self.query_matcher.extract_main_concept(query_match)
                
                # Use query intent to override BioNN intent (more reliable)
                if query_match.confidence > 0.85:
                    intent_hint = query_match.intent
                    use_intent_filtering = False  # Skip BioNN, we have better intent
        
        # 3. Classify intent using BioNN if available (if not already extracted from query or reasoning)
        if intent_hint is None and use_intent_filtering and self.intent_classifier is not None and cleaned_user_input:
            # BioNN extracts semantic intent
            intent_scores = self.intent_classifier.classify_intent(cleaned_user_input, topk=1)
            
            if intent_scores and intent_scores[0][1] > 0.5:  # Reasonable confidence
                intent_hint = intent_scores[0][0]  # Top intent label
        
        # 3.5 TRY DELIBERATION-BASED COMPOSITION FIRST
        # If reasoning found activated concepts with properties, compose from those
        # This is the proper cognitive flow: deliberate → compose from concepts
        deliberation_failed_relevance = False  # Track if we rejected concepts
        
        if deliberation_result and deliberation_result.activated_concepts:
            # Detect conversation category for template selection
            category = self._detect_conversation_category(user_input)
            
            deliberation_response = self._compose_from_deliberation(
                deliberation_result,
                user_input,
                category
            )
            
            if deliberation_response and deliberation_response.confidence >= 0.70:
                print(f"  ✨ Composed from reasoning (confidence: {deliberation_response.confidence:.2f})")
                self.last_response = deliberation_response
                self.last_approach = 'deliberation'
                self.metrics['deliberation_count'] = self.metrics.get('deliberation_count', 0) + 1
                return deliberation_response
            elif deliberation_response:
                print(f"  ⚠️ Deliberation composition low confidence ({deliberation_response.confidence:.2f}), trying patterns")
            else:
                # Deliberation returned None - concepts were rejected as irrelevant
                # This means we don't actually know about this topic
                deliberation_failed_relevance = True
                print(f"  ⚠️ Deliberation found no relevant concepts - topic may be unknown")
        
        # 3.6 PROACTIVE KNOWLEDGE AUGMENTATION for unknown topics
        # If deliberation couldn't find relevant concepts, try learning about the topic
        # BEFORE falling back to pattern matching (which might hallucinate)
        aug_for_unknowns = self._ensure_knowledge_augmenter()
        if deliberation_failed_relevance and aug_for_unknowns and user_input:
            print(f"  🔍 Attempting proactive knowledge lookup for unknown topic...")
            filled_response = self._fill_gaps_and_retry(user_input)
            if filled_response and filled_response.confidence >= 0.6:
                print(f"  ✨ Learned about topic on-the-fly!")
                return filled_response
            
            # Also try direct lookup
            conv_context = self._get_conversation_context(max_turns=3)
            external_result = aug_for_unknowns.lookup(user_input, conversation_history=conv_context, min_confidence=0.6)
            if external_result:
                response_text, confidence, source = external_result

                # Ingest the snippet into vocab/concepts/patterns before replying
                self._ingest_external_knowledge(user_input, response_text, confidence, source)

                # Respond with a concise, cleaned snippet
                response_text = self._clean_composed_response(self._first_sentence(response_text))
                print(f"  💡 Found external knowledge from {source} (confidence: {confidence:.2f})")
                
                return ComposedResponse(
                    text=response_text,
                    fragment_ids=[f"external_{source}"],
                    composition_weights=[confidence],
                    coherence_score=confidence,
                    primary_pattern=None,
                    confidence=confidence,
                    is_fallback=True,
                    is_low_confidence=False
                )
        
        # 4. RETRIEVE PATTERNS - Choose method based on configuration  
        # MULTI-TURN COHERENCE: Use enriched context (includes history + topics)
        # instead of just raw user_input for better topic continuity
        # 
        # Context format: "Previous: X → Y | Earlier: Z | Current: user_input"
        # This helps resolve pronouns and maintain topic threads
        # Use cleaned query for retrieval if available
        retrieval_query = context if context != user_input else cleaned_user_input
        
        # For confirmation questions (is X a Y?), also try statement form (X is a Y)
        # This helps match learned facts like "games are not edible" when asking "are games edible?"
        statement_form = self._confirmation_to_statement(user_input) if user_input else None
        
        if use_semantic_retrieval and hasattr(self.fragments, 'retrieve_patterns_hybrid'):
            # NEW PATH: BioNN embedding + keyword hybrid (OPEN BOOK EXAM)
            # BioNN learns "how to recognize similar contexts" 
            # Database stores "what to respond"
            patterns = self.fragments.retrieve_patterns_hybrid(
                retrieval_query,
                topk=topk * 3,
                min_score=0.0,
                semantic_weight=semantic_weight,
                intent_filter=intent_hint  # Pass extracted intent for filtering
            )
            
            # CONFIRMATION QUESTION ENHANCEMENT: Also search for statement form
            # E.g., "is a parrot a bird?" should also match "a parrot is a bird"
            if statement_form and len(patterns) < topk:
                statement_patterns = self.fragments.retrieve_patterns_hybrid(
                    statement_form,
                    topk=topk,
                    min_score=0.0,
                    semantic_weight=semantic_weight
                )
                # Merge, avoiding duplicates
                existing_ids = {p.pattern_id for p, _ in patterns}
                for pattern, score in statement_patterns:
                    if pattern.pattern_id not in existing_ids:
                        patterns.append((pattern, score))
                        existing_ids.add(pattern.pattern_id)
            
            # MULTI-TURN COHERENCE: Boost patterns that match active conversation topics
            # NOTE: Disabled for now - the real issue is pattern quality, not topic coherence
            # patterns = self._boost_topic_coherent_patterns(patterns, context)
        else:
            # OLD PATH: Pure keyword matching
            patterns = self.fragments.retrieve_patterns(
                retrieval_query,
                topk=topk * 3
            )
            
            # Also try statement form for keyword matching
            if statement_form and len(patterns) < topk:
                statement_patterns = self.fragments.retrieve_patterns(
                    statement_form,
                    topk=topk
                )
                existing_ids = {p.pattern_id for p, _ in patterns}
                for pattern, score in statement_patterns:
                    if pattern.pattern_id not in existing_ids:
                        patterns.append((pattern, score))
        
        if not patterns:
            # Fallback if no patterns found - try external knowledge
            return self._fallback_response(user_input)
        
        # 2b. Score patterns by semantic relevance to user query
        # SKIP if using hybrid retrieval (already scored by BioNN + keywords)
        if user_input and not use_semantic_retrieval:
            # Only re-score for keyword-only retrieval
            patterns = self._score_by_query_relevance(patterns, user_input, topk=topk)
        else:
            # Hybrid retrieval already scored - just limit to topk
            patterns = patterns[:topk]
        
        if not patterns:
            return self._fallback_response(user_input)
        
        # 2c. Filter out assumptive responses (assume prior conversation context)
        # These patterns come from different conversations and don't fit
        patterns = self._filter_assumptive_responses(patterns)
        
        if not patterns:
            return self._fallback_response(user_input)
        
        # 2d. Check if best pattern has sufficient relevance
        # If not, provide a graceful fallback instead of hallucinating
        best_pattern, best_score = patterns[0]
        
        # CONVERSATION HISTORY: Avoid repeating exact same response for repeated queries
        # If user is repeating their question, try a different pattern or add variation
        if is_repetition and previous_response:
            # Check if best pattern would give same response as before
            if best_pattern.response_text.strip().lower() == previous_response.strip().lower():
                print(f"  🔄 Would repeat same response - trying alternative")
                # Try second-best pattern if available
                if len(patterns) > 1:
                    best_pattern, best_score = patterns[1]
                    print(f"  ✨ Using alternative pattern (score: {best_score:.3f})")
                else:
                    # Only one pattern - add conversational variation
                    # This will be handled in pattern adaptation phase
                    pass
        
        # ADAPTIVE CONFIDENCE THRESHOLD
        # Lower threshold for newly learned patterns (teaching scenarios)
        # Higher threshold for established patterns (prevent hallucination)
        if best_pattern.usage_count < 5:
            # Newly learned pattern - use lower threshold to give it a chance
            confidence_threshold = 0.65
        else:
            # Established pattern - use stricter threshold
            confidence_threshold = 0.80
        
        if best_score < confidence_threshold:
            return self._fallback_response_low_confidence(user_input, best_pattern, best_score)
        
        # 2d. Additional check: is user asking about specific topic not in training data?
        # Check for specific technical terms that aren't well-covered
        uncovered_terms = ['transformer', 'cnn', 'rnn', 'gan', 'lstm', 'gru', 'vae', 
                          'resnet', 'bert', 'gpt', 'diffusion', 'reinforcement']
        user_lower = user_input.lower()
        if any(term in user_lower for term in uncovered_terms):
            # Check if response actually addresses that term
            response_lower = best_pattern.response_text.lower()
            query_terms = set(user_lower.split())
            response_terms = set(response_lower.split())
            
            # If the specific term isn't in the response, it's probably not a good match
            uncovered_in_query = [t for t in uncovered_terms if t in user_lower]
            if uncovered_in_query and not any(t in response_lower for t in uncovered_in_query):
                return self._fallback_response_low_confidence(user_input, best_pattern, best_score)
            
        # 3. REASONING STAGE: Deliberate on query before composition
        # This adds a "thinking" step where the BioNN explores concept connections
        deliberation_result = None
        if self.reasoning_stage and user_input:
            try:
                # Run deliberation with retrieved patterns
                deliberation_result = self.reasoning_stage.reason_about(
                    query=user_input,
                    retrieved_patterns=patterns[:5],  # Top 5 patterns
                    context=context
                )
                
                # If reasoning resolved a clearer intent, use it
                if deliberation_result.resolved_intent and not intent_hint:
                    intent_hint = deliberation_result.resolved_intent
                    
                # Print reasoning summary for visibility
                summary = self.reasoning_stage.get_reasoning_summary(deliberation_result)
                if summary and deliberation_result.inferences:
                    print(summary)
                    
            except Exception as e:
                # Reasoning is optional - don't fail if it errors
                import logging
                logging.debug(f"Reasoning stage error: {e}")
            
        # 4. PATTERN ADAPTATION: Use pattern as template, not verbatim
        # Brain-like: retrieved pattern provides structure/intent, BioNN adapts to context
        if user_input and best_score > 0.75:
            # EXACT MATCHES (≥0.90): Use verbatim - don't adapt taught Q&A pairs!
            # These are factual responses that should be returned as-is
            # Includes: 1.0 (perfect), 0.98 (case), 0.95 (normalized), 0.90 (canonical paraphrase)
            if best_score >= 0.90:
                # Perfect/canonical match - use pattern exactly as taught
                response_text = best_pattern.response_text
            else:
                # High confidence but not perfect - adapt pattern to current context
                response_text = self._adapt_pattern_to_context(
                    best_pattern, 
                    user_input, 
                    context,
                    activation_signature=self._get_activation_signature()
                )
            
            # GRAMMAR REFINEMENT: Use syntax stage to fix grammatical errors
            # This is the hybrid approach: adaptation for context + grammar for correctness
            if self.syntax_stage:
                refined_text = self.syntax_stage.check_and_correct(response_text)
                # Learn if correction was made
                if refined_text != response_text:
                    self.syntax_stage.learn_correction(response_text, refined_text)
                response_text = refined_text
            
            response = ComposedResponse(
                text=response_text,
                fragment_ids=[best_pattern.fragment_id],
                composition_weights=[1.0],
                coherence_score=best_score,
                primary_pattern=best_pattern
            )
            self.last_response = response
            return response
            
        # 4. For medium-relevance patterns, apply gentle PMFlow weighting
        activation_signature = self._get_activation_signature()
        
        # 5. Weight patterns by activation + success scores
        weighted_patterns = self._weight_patterns(
            patterns, 
            activation_signature
        )
        
        # 6. Apply coherence constraints from working memory
        coherent_patterns = self._filter_by_coherence(
            weighted_patterns,
            self.state
        )
        
        if not coherent_patterns:
            # Use original patterns if coherence filtering too strict
            coherent_patterns = weighted_patterns
            
        # 7. Compose final response
        response = self._compose_from_patterns(
            coherent_patterns,
            context,
            user_input
        )
        
        # Track response for success learning
        self.last_response = response
        
        return response
    
    def _boost_topic_coherent_patterns(
        self,
        patterns: List[Tuple[ResponsePattern, float]],
        context: str
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Boost patterns that maintain topic coherence with recent conversation.
        
        This is key for multi-turn coherence - patterns that match active topics
        should be preferred over random tangents.
        
        Args:
            patterns: Retrieved patterns with scores
            context: Current context (may include history like "Previous: X | Current: Y")
            
        Returns:
            Patterns with topic-coherent ones boosted
        """
        # The issue with keyword overlap on full context is that it's too noisy
        # ("movies" shares "about"/"learn" with "learn about machine learning")
        # 
        # Better approach: penalize patterns that introduce NEW topics
        # not mentioned anywhere in recent conversation
        
        # Extract ALL words from context (previous responses + current input)
        context_lower = context.lower()
        context_words = set(context_lower.split())
        
        # Common stop words to ignore
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                     'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'should', 'could', 'may', 'might', 'must', 'can', 'previous', 'current',
                     'earlier', 'i', 'you', 'we', 'they', 'he', 'she', 'it', 'my', 'your',
                     'his', 'her', 'their', 'this', 'that', 'these', 'those', '→', '|'}
        
        context_keywords = context_words - stop_words
        
        if not context_keywords or len(patterns) == 0:
            return patterns  # No context or no patterns to boost
        
        print(f"  🎯 Topic coherence check:")
        print(f"     Context: '{context[:60]}...'")
        
        boosted = []
        for pattern, score in patterns:
            # Check if pattern introduces completely new topics
            pattern_text = (pattern.trigger_context + " " + pattern.response_text).lower()
            pattern_words = set(pattern_text.split()) - stop_words
            
            # Calculate what's new vs what's already in conversation
            new_topics = pattern_words - context_keywords
            existing_topics = pattern_words & context_keywords
            
            if len(pattern_words) == 0:
                # Empty pattern, keep as is
                boosted.append((pattern, score))
                continue
            
            # Ratio of new topics to total pattern content
            novelty_ratio = len(new_topics) / len(pattern_words) if len(pattern_words) > 0 else 1.0
            
            # PENALIZE high novelty (random topic jumps)
            # novelty 0.0 = all words in conversation already (good!)
            # novelty 1.0 = completely new topic (bad!)
            coherence_score = 1.0 - (novelty_ratio * 0.5)  # Max 50% penalty
            
            adjusted_score = score * coherence_score
            
            if len(boosted) < 3:  # Show first 3
                print(f"     Pattern: '{pattern.response_text[:35]}...'")
                print(f"       Novelty: {novelty_ratio:.2f} → coherence: {coherence_score:.2f} → {score:.3f} → {adjusted_score:.3f}")
            
            boosted.append((pattern, adjusted_score))
        
        # Re-sort by adjusted scores
        boosted.sort(key=lambda x: x[1], reverse=True)
        
        return boosted
        
    def _get_activation_signature(self) -> Dict[str, float]:
        """
        Get PMFlow activation strengths from working memory.
        
        Returns topic strengths as activation weights.
        """
        snapshot = self.state.snapshot()
        
        # Convert topics to activation signature
        activation_sig = {}
        for topic in snapshot.topics:
            activation_sig[topic.signature] = topic.strength
            
        return activation_sig
    
    def _adapt_pattern_to_context(
        self,
        pattern: ResponsePattern,
        user_input: str,
        context: str,
        activation_signature: Dict[str, float]
    ) -> str:
        """
        Adapt retrieved pattern to current context - brain-like generation.
        
        Instead of using pattern verbatim (canned response), extract its structure
        and generate contextually appropriate content. The pattern provides:
        - Intent (question/statement/greeting)
        - Sentiment/tone
        - General structure
        
        BioNN and working memory help fill in context-appropriate content.
        
        Args:
            pattern: Retrieved pattern (template)
            user_input: Current user query
            context: Conversation context
            activation_signature: Working memory activations
            
        Returns:
            Adapted response text
        """
        # For meta-queries (capability/identity), use pattern verbatim
        # These are factual responses about the system itself
        if hasattr(pattern, 'intent') and pattern.intent in ['capability', 'identity']:
            return pattern.response_text
        
        # Step 1: Analyze pattern structure
        pattern_intent = pattern.intent if hasattr(pattern, 'intent') else "statement"
        is_question = "?" in pattern.response_text
        is_greeting = any(word in pattern.response_text.lower() 
                         for word in ["hello", "hi", "hey", "greetings"])
        
        # Step 2: Extract key concepts from pattern (what it's ABOUT)
        # These are content words, not structure
        pattern_words = set(pattern.response_text.lower().split())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                     'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'should', 'could', 'may', 'might', 'must', 'can', 'i', 'you', 'we', 
                     'they', 'he', 'she', 'it', 'my', 'your', 'his', 'her', 'their',
                     'this', 'that', 'these', 'those', 'how', 'what', 'about'}
        pattern_concepts = pattern_words - stop_words
        
        # Step 3: Extract context topics (what conversation is ABOUT)
        context_words = set(context.lower().split())
        user_words = set(user_input.lower().split())
        
        # Remove context formatting markers
        context_words.discard("previous:")
        context_words.discard("current:")
        context_words.discard("earlier:")
        
        context_concepts = (context_words | user_words) - stop_words
        
        # Step 4: Decide adaptation strategy based on overlap
        concept_overlap = len(pattern_concepts & context_concepts)
        
        if concept_overlap >= 2:
            # Good overlap - pattern is relevant, use with minor tweaks
            return self._adapt_with_substitution(pattern, user_input, context_concepts)
        elif is_greeting:
            # Greeting pattern - keep structure, adapt to context
            return self._adapt_greeting(pattern, user_input, context_concepts)
        elif is_question:
            # Question pattern - keep interrogative structure, adapt content
            return self._adapt_question(pattern, user_input, context_concepts)
        else:
            # Statement pattern - use as inspiration for topic-relevant statement
            return self._adapt_statement(pattern, user_input, context_concepts)
    
    def _adapt_with_substitution(
        self, 
        pattern: ResponsePattern, 
        user_input: str,
        context_concepts: set
    ) -> str:
        """Pattern is relevant - use it with minor concept substitution."""
        # For now, use pattern mostly as-is since it's already relevant
        # Future: Could use BioNN to find similar but context-appropriate phrases
        return pattern.response_text
    
    def _adapt_greeting(
        self,
        pattern: ResponsePattern,
        user_input: str, 
        context_concepts: set
    ) -> str:
        """Adapt greeting pattern to be context-appropriate."""
        # Extract greeting structure
        if "hello" in pattern.response_text.lower():
            greeting_base = "Hello!"
        elif "hi" in pattern.response_text.lower():
            greeting_base = "Hi there!"
        else:
            greeting_base = "Greetings!"
        
        # Add context-appropriate follow-up
        if "help" in context_concepts or "learn" in context_concepts:
            return f"{greeting_base} I'd be happy to help you. What would you like to know?"
        elif "can" in context_concepts or "do" in context_concepts:
            return f"{greeting_base} I can discuss various topics. What interests you?"
        else:
            return f"{greeting_base} How can I assist you today?"
    
    def _adapt_question(
        self,
        pattern: ResponsePattern,
        user_input: str,
        context_concepts: set
    ) -> str:
        """Adapt question pattern to maintain conversational flow."""
        # Check if user is making an acknowledgment first
        user_lower = user_input.lower()
        if any(word in user_lower for word in ["interesting", "cool", "nice", "great", "okay", "ok"]):
            return "I'm glad you found that interesting! What else would you like to know?"
        
        # Extract question type from pattern
        pattern_lower = pattern.response_text.lower()
        
        # Common question starters
        if pattern_lower.startswith("what"):
            # "What about..." type questions
            # Filter to meaningful concepts only
            meaningful_concepts = {c for c in context_concepts 
                                  if len(c) > 3 and 
                                  c not in {"that's", "what's", "it's", "there's", "do?", "can?", "you?"} and
                                  "?" not in c}
            if meaningful_concepts:
                concept = list(meaningful_concepts)[0]  # Pick first meaningful concept
                return f"What would you like to know about {concept}?"
            return "What would you like to know more about?"
        
        elif pattern_lower.startswith("how"):
            return "How can I help you with that?"
        
        elif pattern_lower.startswith("do you") or pattern_lower.startswith("did you"):
            return "Do you have a specific question I can help with?"
        
        else:
            # Generic question - keep conversational
            return "Is there something specific you'd like to know?"
    
    def _adapt_statement(
        self,
        pattern: ResponsePattern,
        user_input: str,
        context_concepts: set
    ) -> str:
        """Adapt statement pattern to address user's actual query."""
        # Check what user is asking about
        user_lower = user_input.lower()
        
        # Capability/identity questions - use pattern verbatim if it's appropriate
        if any(phrase in user_lower for phrase in ["what can you", "what do you", "can you help", 
                                                    "who are you", "what are you"]):
            # If pattern is capability/identity-related, trust it
            if pattern.intent in ["capability", "identity"]:
                return pattern.response_text
            # Otherwise use generic fallback
            return "I can help answer questions and discuss various topics. What would you like to explore?"
        
        # Interest/preference questions  
        if any(phrase in user_lower for phrase in ["what about", "how about", "do you like"]):
            # Extract topic from user's question
            # "Do you like movies?" → movies
            # "What about the weather?" → weather
            user_words = set(user_input.lower().split())
            stop_and_question_words = {'do', 'you', 'like', 'what', 'about', 'how', 'the', 'a', 'an', 
                                       '?', 'that', 'this', 'think', 'weather', 'today'}
            topic_words = user_words - stop_and_question_words
            
            if topic_words:
                topic = list(topic_words)[0]
                return f"That's an interesting topic. I'd be happy to discuss {topic} with you."
            return "That's an interesting question. Could you tell me more?"
        
        # General acknowledgment
        if any(word in user_lower for word in ["interesting", "cool", "nice", "great", "okay", "ok", "thanks"]):
            # User is acknowledging - ask what they want to explore
            if context_concepts:
                # Filter out generic words
                meaningful_concepts = {c for c in context_concepts 
                                      if len(c) > 3 and c not in {"that's", "what's", "it's", "there's"}}
                if meaningful_concepts:
                    concept = list(meaningful_concepts)[0]
                    return f"I'm glad you found that interesting! Would you like to know more about {concept}?"
            return "I'm glad you found that interesting! What else would you like to know?"
        
        # Default: Use pattern's sentiment but make it generic
        if "!" in pattern.response_text:
            # Enthusiastic pattern
            return "I'd be happy to help with that!"
        else:
            # Neutral pattern
            return "I can help with that. What would you like to know?"
    
    def _filter_assumptive_responses(
        self,
        patterns: List[Tuple[ResponsePattern, float]]
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Deprioritize responses that assume prior conversation context.
        
        These patterns come from other conversations and make assumptive statements
        like "i do too" or "have you decided" that don't make sense without context.
        
        Instead of filtering them out completely (which can leave us with no matches),
        we penalize their scores heavily.
        
        Args:
            patterns: Candidate patterns with scores
            
        Returns:
            Patterns with assumptive ones penalized (but not removed)
        """
        # Phrases that assume prior context
        assumptive_starts = [
            'i do too',
            'i did too',
            'i would too',
            'i will too',
            'me too',
            'me neither',
            'i agree',
            'i disagree',
            'have you decided',
            'did you decide',
            'have you thought',
            'did you think',
            'as i said',
            'as i mentioned',
            'as we discussed',
            'like i said',
            'like i told you',
            'remember when',
            'do you recall',
            'you said',
            'you told me',
            'you mentioned',
            "you're right",  # Assumes prior statement
            "that's what i",  # Assumes prior discussion
        ]
        
        # Phrases that are off-topic additions
        off_topic_patterns = [
            'i just got these shoes',
            'i just bought',
            'i recently got',
        ]
        
        reranked = []
        for pattern, score in patterns:
            response_lower = pattern.response_text.lower().strip()
            
            # Check if response starts with assumptive phrase
            penalty = 1.0
            for phrase in assumptive_starts:
                if response_lower.startswith(phrase):
                    penalty = 0.3  # Heavy penalty (70% reduction)
                    break
            
            # Check if response contains off-topic additions - REMOVE these entirely
            has_offtopic = False
            for phrase in off_topic_patterns:
                if phrase in response_lower:
                    has_offtopic = True
                    break
            
            # Remove off-topic, penalize assumptive, keep others
            if not has_offtopic:
                reranked.append((pattern, score * penalty))
        
        # Re-sort by penalized scores
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked
    
    def _score_by_query_relevance(
        self,
        patterns: List[Tuple[ResponsePattern, float]],
        user_query: str,
        topk: int = 5,
        min_relevance: float = 0.55
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Score patterns by semantic relevance between user query and response content.
        
        This is THE KEY to fixing off-topic responses!
        
        Instead of just matching trigger contexts, we check if the RESPONSE CONTENT
        is actually relevant to what the user is asking about.
        
        Args:
            patterns: Candidate patterns with trigger-based scores
            user_query: User's actual input query
            topk: Number of relevant patterns to return
            min_relevance: Minimum semantic similarity threshold (raised to 0.55 to filter garbage)
            
        Returns:
            Patterns scored by query-response semantic relevance
        """
        # Encode user query
        try:
            query_emb = self.fragments.encoder.encode(user_query)
            if hasattr(query_emb, 'cpu'):
                query_emb = query_emb.cpu().numpy()
            query_emb = query_emb.flatten()
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        except Exception:
            # If encoding fails, return original patterns
            return patterns[:topk]
        
        # Score each pattern by semantic similarity between query and response content
        relevance_scored = []
        for pattern, trigger_score in patterns:
            try:
                # Encode the RESPONSE TEXT (not trigger context!)
                response_emb = self.fragments.encoder.encode(pattern.response_text)
                if hasattr(response_emb, 'cpu'):
                    response_emb = response_emb.cpu().numpy()
                response_emb = response_emb.flatten()
                response_norm = response_emb / (np.linalg.norm(response_emb) + 1e-8)
                
                # Semantic similarity between user query and response content
                relevance = float(np.dot(query_norm, response_norm))
                
                # Also encode trigger context for comparison
                trigger_emb = self.fragments.encoder.encode(pattern.trigger_context)
                if hasattr(trigger_emb, 'cpu'):
                    trigger_emb = trigger_emb.cpu().numpy()
                trigger_emb = trigger_emb.flatten()
                trigger_norm = trigger_emb / (np.linalg.norm(trigger_emb) + 1e-8)
                
                trigger_relevance = float(np.dot(query_norm, trigger_norm))
                
                # IMPORTANT: BioNN embeddings are unreliable! Add keyword/topic boost
                # Extract key content words (not stopwords)
                stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                           'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                           'can', 'could', 'may', 'might', 'must', 'i', 'you', 'he', 'she', 'it',
                           'we', 'they', 'them', 'this', 'that', 'these', 'those', 'to', 'of', 'in',
                           'on', 'at', 'by', 'for', 'with', 'about', 'as', 'from'}
                
                query_words = set(w.lower() for w in user_query.split() if w.lower() not in stopwords)
                response_words = set(w.lower() for w in pattern.response_text.split() if w.lower() not in stopwords)
                trigger_words = set(w.lower() for w in pattern.trigger_context.split() if w.lower() not in stopwords)
                
                # Keyword overlap scores
                response_overlap = len(query_words & response_words) / max(len(query_words), 1)
                trigger_overlap = len(query_words & trigger_words) / max(len(query_words), 1)
                
                # Combined score: BioNN similarity + keyword overlap boost
                # Weight keyword overlap VERY heavily since BioNN is unreliable
                # Prioritize patterns with actual keyword matches
                keyword_boost = max(response_overlap, trigger_overlap) * 0.5  # Up to +0.5 boost
                combined_relevance = max(
                    relevance + keyword_boost,
                    trigger_relevance * 0.7 + keyword_boost
                )
                
                # Filter out low-relevance patterns
                if combined_relevance >= min_relevance:
                    relevance_scored.append((pattern, combined_relevance))
                    
            except Exception:
                # If encoding fails, use original score but penalize
                if trigger_score > min_relevance:
                    relevance_scored.append((pattern, trigger_score * 0.5))
        
        # Sort by relevance descending
        relevance_scored.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k most relevant
        result = relevance_scored[:topk]
        
        # Debug: print selected pattern
        if result:
            best_pattern, best_score = result[0]
            print(f"     → Selected: '{best_pattern.response_text[:50]}...' (intent: {best_pattern.intent}, relevance: {best_score:.3f})")
        
        return result
        
    def _weight_patterns(
        self,
        patterns: List[Tuple[ResponsePattern, float]],
        activation_signature: Dict[str, float]
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Weight patterns by PMFlow activations + learned success.
        
        This is where neural (PMFlow) meets symbolic (patterns)!
        
        IMPORTANT: For semantically-scored patterns, we apply gentle boosting
        to preserve relevance order while still rewarding successful patterns.
        
        Args:
            patterns: Retrieved patterns with similarity/relevance scores
            activation_signature: Current working memory activations
            
        Returns:
            Patterns weighted by: relevance + success_boost + activation_boost
        """
        weighted = []
        
        for pattern, base_score in patterns:
            # Start with base score (relevance or similarity)
            weight = base_score
            
            # Apply gentle success boost (additive, not multiplicative)
            # This prevents destroying relevance order
            success_boost = (pattern.success_score - 0.5) * 0.2  # -0.1 to +0.1 range
            weight += success_boost
            
            # Apply gentle activation boost
            activation_boost = 0.0
            if activation_signature:
                # Small boost from working memory
                max_activation = max(activation_signature.values())
                activation_boost = max_activation * 0.1  # Small boost
                
            weight += activation_boost
            
            weighted.append((pattern, weight))
        
        # Sort by weight descending (but order should be mostly preserved)
        weighted.sort(key=lambda x: x[1], reverse=True)
        
        return weighted
        
    def _filter_by_coherence(
        self,
        patterns: List[Tuple[ResponsePattern, float]],
        state: ConversationState
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Filter patterns to maintain conversation coherence.
        
        Uses working memory to ensure response fits current topics.
        
        Args:
            patterns: Weighted patterns
            state: Conversation state with active topics
            
        Returns:
            Coherent patterns only
        """
        snapshot = state.snapshot()
        
        # If no active topics, allow all patterns
        if not snapshot.topics:
            return patterns
            
        # For now, use simple filtering
        # Could enhance with more sophisticated coherence checking
        coherent = []
        
        for pattern, weight in patterns:
            # Calculate coherence score
            coherence = self._calculate_coherence(pattern, snapshot)
            
            # Only keep if reasonably coherent
            if coherence > 0.3:
                coherent.append((pattern, weight * coherence))
                
        return coherent
        
    def _calculate_coherence(
        self, 
        pattern: ResponsePattern,
        snapshot
    ) -> float:
        """
        Calculate how well pattern fits current conversation state.
        
        Args:
            pattern: Response pattern to evaluate
            snapshot: Current conversation state snapshot
            
        Returns:
            Coherence score (0.0-1.0)
        """
        # Start with baseline coherence
        coherence = 0.5
        
        # Boost if pattern matches dominant topic
        if snapshot.topics:
            # Simple heuristic: higher coherence if pattern has been successful
            coherence += 0.3 * pattern.success_score
            
        # Reduce if novelty is very high (pattern might be off-topic)
        if snapshot.novelty > 0.8:
            coherence *= 0.7
            
        return min(coherence, 1.0)
        
    def _compose_from_patterns(
        self,
        weighted_patterns: List[Tuple[ResponsePattern, float]],
        context: str,
        user_input: str
    ) -> ComposedResponse:
        """
        Compose final response from weighted patterns.
        
        Args:
            weighted_patterns: Patterns with composition weights
            context: Conversation context
            user_input: Raw user text
            
        Returns:
            Composed response
        """
        if not weighted_patterns:
            return self._fallback_response("")  # No user input available at this point
            
        if self.composition_mode == "best_match":
            return self._compose_best_match(weighted_patterns)
            
        elif self.composition_mode == "weighted_blend":
            return self._compose_weighted_blend(weighted_patterns, context)
            
        elif self.composition_mode == "adaptive":
            return self._compose_adaptive(weighted_patterns, context)
            
        else:
            # Default to best match
            return self._compose_best_match(weighted_patterns)
            
    def _compose_best_match(
        self,
        weighted_patterns: List[Tuple[ResponsePattern, float]]
    ) -> ComposedResponse:
        """
        Use the single best-matching pattern.
        
        Simple but effective strategy.
        """
        pattern, weight = weighted_patterns[0]
        
        return ComposedResponse(
            text=pattern.response_text,
            fragment_ids=[pattern.fragment_id],
            composition_weights=[1.0],
            coherence_score=weight,
            primary_pattern=pattern
        )
        
    def _compose_weighted_blend(
        self,
        weighted_patterns: List[Tuple[ResponsePattern, float]],
        context: str
    ) -> ComposedResponse:
        """
        Blend multiple patterns to create novel responses.
        
        Strategy: Combine fragments from top patterns when appropriate.
        This creates NEW utterances from learned pieces!
        """
        # Get top patterns
        primary_pattern, primary_weight = weighted_patterns[0]
        
        response_text = primary_pattern.response_text
        fragment_ids = [primary_pattern.fragment_id]
        weights = [primary_weight]
        
        # Factual intents should NOT be blended - they have specific correct answers
        factual_intents = {'identity', 'capability', 'meta'}
        if primary_pattern.intent in factual_intents and primary_weight >= 0.9:
            # High-confidence factual pattern - return as-is, no blending
            return ComposedResponse(
                text=response_text,
                fragment_ids=fragment_ids,
                composition_weights=weights,
                coherence_score=primary_weight,
                primary_pattern=primary_pattern
            )
        
        # Calculate weight ratio between top two patterns
        # If they're close, blending makes sense
        should_blend = False
        if len(weighted_patterns) > 1:
            secondary_pattern, secondary_weight = weighted_patterns[1]
            
            # Blend if:
            # 1. Secondary is at least 60% as good as primary (close competition)
            # 2. Both patterns have reasonable weights
            weight_ratio = secondary_weight / (primary_weight + 1e-6)
            if weight_ratio > 0.6 and secondary_weight > 0.5:
                should_blend = True
                
                # Try blending
                blended = self._blend_patterns(primary_pattern, secondary_pattern)
                if blended:
                    # Validate the blended response
                    if self._validate_response(blended):
                        response_text = blended
                        fragment_ids.append(secondary_pattern.fragment_id)
                        weights.append(secondary_weight)
                        print(f"  🎨 Blended {primary_pattern.intent} + {secondary_pattern.intent}")
                    else:
                        # Blend failed validation, use primary only
                        print(f"  ⚠️  Blend rejected (validation failed), using primary only")
                        should_blend = False
        
        # If all weights are low (no good match), try composing from fragments
        if not should_blend and primary_weight < 1.5 and len(weighted_patterns) >= 3:
            # Create novel composition from multiple weak matches
            composed = self._compose_from_fragments(weighted_patterns[:3])
            if composed:
                response_text = composed
                fragment_ids = [p.fragment_id for p, _ in weighted_patterns[:3]]
                weights = [w for _, w in weighted_patterns[:3]]
                print(f"  🎨 Composed from {len(fragment_ids)} fragments")
        
        return ComposedResponse(
            text=response_text,
            fragment_ids=fragment_ids,
            composition_weights=weights,
            coherence_score=primary_weight,
            primary_pattern=primary_pattern
        )
        
    def _compose_adaptive(
        self,
        weighted_patterns: List[Tuple[ResponsePattern, float]],
        context: str
    ) -> ComposedResponse:
        """
        Adaptively choose composition strategy based on confidence.
        
        High confidence: Use best match
        Low confidence: Blend or ask for clarification
        """
        primary_pattern, primary_weight = weighted_patterns[0]
        
        if primary_weight > 0.7:
            # High confidence: use best match
            return self._compose_best_match(weighted_patterns)
        elif primary_weight > 0.4:
            # Medium confidence: try blending
            return self._compose_weighted_blend(weighted_patterns, context)
        else:
            # Low confidence: ask for clarification
            return ComposedResponse(
                text="Could you clarify what you mean?",
                fragment_ids=["clarification_request"],
                composition_weights=[1.0],
                coherence_score=primary_weight,
                primary_pattern=primary_pattern
            )
    
    def _blend_patterns(
        self,
        primary: ResponsePattern,
        secondary: ResponsePattern
    ) -> Optional[str]:
        """
        Blend two patterns into a novel response.
        
        Strategy: Check compatibility first, then use BioNN syntax stage if available
        for grammatically-guided blending. This creates NEW utterances from learned fragments!
        
        Args:
            primary: Main pattern
            secondary: Supporting pattern
            
        Returns:
            Blended text or None if blending not appropriate
        """
        # Check if patterns are compatible for blending
        if not self._should_blend(primary, secondary):
            return None
        
        # If BioNN syntax stage is available, use it for intelligent composition!
        if self.syntax_stage:
            return self._blend_with_syntax_bnn(primary, secondary)
        
        # Fallback: Heuristic blending (original method)
        # Don't blend if patterns end with questions (creates awkward double-question)
        if primary.response_text.strip().endswith('?') and secondary.response_text.strip().endswith('?'):
            return None
    
    def _should_blend(self, primary: ResponsePattern, secondary: ResponsePattern) -> bool:
        """
        Check if two patterns are compatible for blending.
        
        Prevents nonsensical combinations like "I'm doing well" + "medical diagnosis"
        
        Args:
            primary: Primary pattern
            secondary: Secondary pattern
            
        Returns:
            True if patterns can be blended coherently
        """
        # Define incompatible intent pairs
        incompatible_pairs = {
            ('greeting', 'technical_explain'),  # "Hello" + technical content
            ('greeting', 'explain'),            # "Hi" + explanation
            ('farewell', 'question_info'),      # "Goodbye" + question
            ('farewell', 'technical_explain'),  # "Bye" + technical
            ('agreement', 'greeting'),          # "Yes" + "Hello"
            ('disagreement', 'greeting'),       # "No" + "Hello"
            ('statement', 'greeting'),          # Statement + greeting is awkward
        }
        
        # Check if this pair is explicitly incompatible
        intent_pair = (primary.intent, secondary.intent)
        intent_pair_reverse = (secondary.intent, primary.intent)
        
        if intent_pair in incompatible_pairs or intent_pair_reverse in incompatible_pairs:
            return False
        
        # Don't blend if both patterns are very short (likely fragments)
        if len(primary.response_text.split()) < 3 and len(secondary.response_text.split()) < 3:
            return False
        
        # Don't blend if combined length would be too long (over 25 words)
        total_words = len(primary.response_text.split()) + len(secondary.response_text.split())
        if total_words > 25:
            return False
        
        # Check semantic compatibility using embeddings if available
        if hasattr(primary, 'embedding_cache') and hasattr(secondary, 'embedding_cache'):
            if primary.embedding_cache and secondary.embedding_cache:
                # Calculate cosine similarity between embeddings
                import numpy as np
                emb1 = np.array(primary.embedding_cache)
                emb2 = np.array(secondary.embedding_cache)
                
                # Normalize
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = np.dot(emb1, emb2) / (norm1 * norm2)
                    
                    # If patterns are too dissimilar semantically, don't blend
                    if similarity < 0.3:  # Threshold for semantic compatibility
                        return False
        
        # Define compatible intent pairs that work well together
        compatible_pairs = {
            ('question_info', 'explain'),       # Question + explanation
            ('question_info', 'technical_explain'),  # Question + technical answer
            ('acknowledgment', 'interest'),     # "I see" + "That's interesting"
            ('interest', 'question_info'),      # "Interesting!" + "How does..."
            ('agreement', 'explain'),           # "Yes" + explanation
            ('agreement', 'technical_explain'), # "Yes" + technical detail
            ('explain', 'technical_explain'),   # Explanation + more detail
            ('statement', 'explain'),           # Statement + elaboration
            ('statement', 'technical_explain'), # Statement + technical detail
        }
        
        # If it's a known good pair, allow it
        if intent_pair in compatible_pairs or intent_pair_reverse in compatible_pairs:
            return True
        
        # Same intent types usually blend well
        if primary.intent == secondary.intent:
            return True
        
        # Default: be conservative, don't blend unless we're sure
        return False
    
    def _validate_response(self, response: str) -> bool:
        """
        Validate a composed response for quality and coherence.
        
        Rejects responses that are:
        - Too short or degenerate
        - Have obvious grammar errors
        - Contain repetitive content
        
        Args:
            response: Composed response text
            
        Returns:
            True if response passes validation
        """
        if not response or not response.strip():
            return False
        
        words = response.split()
        
        # Reject if too short (less than 3 words, likely degenerate)
        if len(words) < 3:
            return False
        
        # Reject if too long (over 40 words, likely rambling)
        if len(words) > 40:
            return False
        
        # Check for repetitive content (same word repeated)
        # "Yes, Yes." or "Hello. Hello!" or "I'm doing well, I'm doing well"
        if len(words) >= 2:
            # Check for immediate repetition
            for i in range(len(words) - 1):
                word1 = words[i].strip('.,!?').lower()
                word2 = words[i + 1].strip('.,!?').lower()
                if word1 == word2 and len(word1) > 2:  # Allow "a a" but not "hello hello"
                    return False
            
            # Check for phrase repetition across sentences
            # Split by punctuation and check if phrases repeat
            parts = [p.strip() for p in response.replace('!', '.').replace('?', '.').replace(',', '.').split('.') if p.strip()]
            if len(parts) >= 2:
                # Check if any two parts are identical or very similar
                for i in range(len(parts)):
                    for j in range(i + 1, len(parts)):
                        # Compare normalized versions (lowercase, no punctuation)
                        norm_i = parts[i].lower().strip('.,!?\'\"')
                        norm_j = parts[j].lower().strip('.,!?\'\"')
                        if norm_i == norm_j:
                            return False
                        # Also check if one is a substring of the other (high overlap)
                        if len(norm_i) > 5 and len(norm_j) > 5:
                            if norm_i in norm_j or norm_j in norm_i:
                                return False
        
        # Check for obvious sentence fragments
        # If it starts with a lowercase letter and isn't a continuation word
        if response[0].islower() and not response.startswith(('and', 'but', 'or', 'so')):
            return False
        
        # Check for multiple sentences - first should end with punctuation
        sentences = [s.strip() for s in response.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        if len(sentences) > 1:
            # Each sentence should start with capital
            for sentence in sentences:
                if sentence and not sentence[0].isupper():
                    return False
        
        # Check for obvious grammar errors - multiple punctuation
        if '..' in response or '!!' in response or '??' in response:
            return False
        
        # Check for mismatched quotes or parentheses
        if response.count('"') % 2 != 0:
            return False
        if response.count('(') != response.count(')'):
            return False
        
        # All checks passed
        return True
            
        # Choose connector based on pattern types
        connectors = {
            ('question_info', 'explain'): ' ',  # Question + explanation flows naturally
            ('acknowledgment', 'interest'): ' ',  # "I see. That's interesting!"
            ('interest', 'question_detail'): ' ',  # "Interesting! Tell me more?"
            ('agreement', 'explain'): ' ',  # "Yes. The reason is..."
            ('greeting', 'question_info'): ' ',  # "Hello! What can I help with?"
            ('greeting', 'interest'): ' ',  # "Hello! Wow, that's..."
            ('capability', 'question_detail'): ' ',  # Statement + question
            ('explain', 'meta'): ' ',  # Explanation + meta-comment
            ('learned', 'capability'): ' ',  # Learned phrase + capability
            ('learned', 'meta'): ' ',  # Learned phrase + meta
            ('clarify', 'capability'): ' ',  # Question + answer
        }
        
        # Get connector (default: space for natural flow)
        intent_pair = (primary.intent, secondary.intent)
        connector = connectors.get(intent_pair, ' ')
        
        # Clean up primary text punctuation
        primary_text = primary.response_text.strip()
        
        # If primary doesn't end with strong punctuation and connector is just space,
        # we need to ensure proper sentence boundary
        if connector == ' ' and primary_text[-1] not in '!?':
            # Replace trailing period with nothing if present, we'll handle it
            if primary_text[-1] == '.':
                primary_text = primary_text[:-1]
            # Add appropriate punctuation based on context
            if secondary.response_text.strip()[0].isupper():
                primary_text += '.'  # New sentence
            else:
                primary_text += ','  # Continuation
        elif not primary_text[-1] in '.!?,':
            primary_text += '.'
            
        # Blend
        blended = primary_text + connector + secondary.response_text.strip()
        
        # Don't return if too long (over 150 chars gets verbose)
        if len(blended) > 150:
            return None
            
        return blended
    
    def _compose_from_fragments(
        self,
        patterns: List[Tuple[ResponsePattern, float]]
    ) -> Optional[str]:
        """
        Create novel composition from multiple weak pattern matches.
        
        When no single pattern is strong, combine fragments to create
        a reasonable response. This is pure compositional generation!
        
        Args:
            patterns: Multiple patterns to compose from
            
        Returns:
            Composed text or None if composition fails
        """
        if len(patterns) < 2:
            return None
            
        # Extract key phrases from patterns
        fragments = []
        for pattern, weight in patterns:
            # Take short patterns whole, extract from long ones
            text = pattern.response_text.strip()
            if len(text.split()) <= 6:
                fragments.append(text)
            else:
                # Extract first clause or sentence
                if '.' in text:
                    fragments.append(text.split('.')[0].strip())
                elif ',' in text:
                    fragments.append(text.split(',')[0].strip())
                else:
                    fragments.append(text)
        
        # Try to compose coherently
        # Strategy: Use first fragment as base, add others if they fit
        if not fragments:
            return None
            
        composed = fragments[0]
        
        # Add second fragment if it complements
        if len(fragments) > 1 and len(composed.split()) < 8:
            # Add connector
            if not composed[-1] in '.!?,':
                composed += ','
            composed += ' ' + fragments[1]
            
        # Ensure proper ending
        if not composed[-1] in '.!?':
            composed += '.'
            
        return composed
    
    def _blend_with_syntax_bnn(self, primary: ResponsePattern, secondary: ResponsePattern) -> str:
        """
        Blend two patterns using BioNN-based syntax stage for grammatical composition.
        
        Args:
            primary: Primary pattern to use as base
            secondary: Secondary pattern to blend in
            
        Returns:
            Grammatically composed text using BioNN-learned templates
        """
        if not self.syntax_stage:
            # Fallback if syntax stage not available
            return self._simple_blend(primary.response_text, secondary.response_text)
        
        # Process patterns through syntax stage
        tokens_a = primary.response_text.split()
        tokens_b = secondary.response_text.split()
        
        artifact_a = self.syntax_stage.process(tokens_a)
        artifact_b = self.syntax_stage.process(tokens_b)
        
        # Extract intents and retrieved patterns from artifacts
        intent_a = artifact_a.metadata.get('intent', 'statement')
        intent_b = artifact_b.metadata.get('intent', 'statement')
        
        # Get matched patterns from retrieval_info (list of dicts)
        retrieval_a = artifact_a.metadata.get('retrieval_info', [])
        retrieval_b = artifact_b.metadata.get('retrieval_info', [])
        
        # Select composition strategy based on intents
        template = None
        if intent_a == intent_b:
            # Same intent: blend templates
            if retrieval_a and retrieval_b:
                # Use higher-confidence template as base
                base_info = retrieval_a[0] if retrieval_a[0]['similarity'] > retrieval_b[0]['similarity'] else retrieval_b[0]
                template = base_info.get('template')
            elif retrieval_a:
                template = retrieval_a[0].get('template')
            elif retrieval_b:
                template = retrieval_b[0].get('template')
        else:
            # Different intents: choose dominant
            base_info = retrieval_a[0] if retrieval_a else (retrieval_b[0] if retrieval_b else None)
            template = base_info.get('template') if base_info else None
        
        # Apply template or use heuristic fallback
        if template:
            # Fill template with content from both patterns
            composed = self._apply_syntax_template(template, tokens_a, tokens_b)
        else:
            # Fallback to simpler composition
            composed = self._simple_blend(primary.response_text, secondary.response_text)
        
        return composed
    
    def _apply_syntax_template(self, template: str, tokens_a: list, tokens_b: list) -> str:
        """
        Apply a syntax template by filling slots with tokens.
        
        Args:
            template: Template string with {pos} slots (e.g., "this is {adv} {adj}")
            tokens_a: Tokens from primary pattern
            tokens_b: Tokens from secondary pattern
            
        Returns:
            Composed text with template slots filled
        """
        import re
        
        # Combine token pools
        all_tokens = tokens_a + tokens_b
        
        # Extract slot types from template
        slots = re.findall(r'\{([^}]+)\}', template)
        
        if not slots:
            # No slots, template is literal
            return template
        
        # Simple slot filling: use tokens in order
        filled = template
        token_idx = 0
        
        for slot in slots:
            if token_idx < len(all_tokens):
                filled = filled.replace(f'{{{slot}}}', all_tokens[token_idx], 1)
                token_idx += 1
            else:
                # Out of tokens, use placeholder
                filled = filled.replace(f'{{{slot}}}', slot.lower(), 1)
        
        # Ensure proper capitalization and ending
        if filled:
            filled = filled[0].upper() + filled[1:]
            if not filled[-1] in '.!?':
                filled += '.'
        
        return filled
    
    def _simple_blend(self, text_a: str, text_b: str) -> str:
        """
        Simple fallback blending when no BioNN template available.
        
        Args:
            text_a: Primary text
            text_b: Secondary text
            
        Returns:
            Simple grammatical blend
        """
        # Take first clause from each
        clause_a = text_a.split('.')[0].split(',')[0].strip()
        clause_b = text_b.split('.')[0].split(',')[0].strip()
        
        # Combine with connector
        if len(clause_a.split()) + len(clause_b.split()) < 12:
            return f"{clause_a}, {clause_b}."
        else:
            return f"{clause_a}."
    
    def _compose_from_concepts(
        self,
        context: str,
        user_input: str,
        min_similarity: float = 0.60
    ) -> Optional[ComposedResponse]:
        """
        Generate response using concept store + template composer.
        
        This is the compositional architecture: separate facts (concepts)
        from syntax (templates).
        
        Args:
            context: Current conversation context
            user_input: Raw user input
            min_similarity: Minimum concept similarity
            
        Returns:
            ComposedResponse if concepts found, None otherwise
        """
        if not self.concept_store or not self.template_composer:
            return None
        
        # 1. Retrieve matching concepts
        concepts = self.concept_store.retrieve_by_text(
            user_input,
            top_k=5,
            min_similarity=min_similarity
        )
        
        if not concepts:
            return None
        
        # 2. Get best concept
        best_concept, similarity = concepts[0]
        
        # 3. Compose response using template
        result = self.template_composer.compose_response(
            user_input,
            best_concept.term,
            best_concept.properties,
            confidence=similarity
        )
        
        if not result:
            return None
        
        # 4. Clean up grammar and return ComposedResponse
        cleaned_text = self._clean_composed_response(result['text'])
        
        return ComposedResponse(
            text=cleaned_text,
            fragment_ids=[best_concept.concept_id],
            composition_weights=[similarity],
            coherence_score=result['confidence'],
            primary_pattern=None,  # Compositional, not pattern-based
            confidence=result['confidence'],
            is_fallback=False,
            is_low_confidence=similarity < 0.75
        )
    
    def _compose_from_deliberation(
        self,
        deliberation_result,
        user_input: str,
        category: str
    ) -> Optional[ComposedResponse]:
        """
        Compose response using deliberation results from reasoning stage.
        
        SEMANTIC BRIDGE: This method bridges between the symbolic reasoning layer
        and the linguistic composition layer:
        
        1. Reasoning stage outputs CONCEPT IDs (symbolic, embedding-based)
        2. This bridge translates IDs → actual content via concept store
        3. Composition uses pragmatic templates to frame the content
        
        The BioNN guided activation found which concepts are relevant -
        now we retrieve their actual content for the response.
        
        Args:
            deliberation_result: Result from reasoning_stage.deliberate()
            user_input: Original user input
            category: Detected conversation category (definition, confirmation, etc.)
            
        Returns:
            ComposedResponse if composition succeeded, None otherwise
        """
        if not deliberation_result or not deliberation_result.activated_concepts:
            return None
        
        if not self.concept_store:
            return None
        
        # SEMANTIC BRIDGE: Translate activated concept IDs to actual content
        # The reasoning stage worked symbolically with embeddings
        # Now we retrieve the linguistic content for composition
        activated = deliberation_result.activated_concepts
        focus_term = deliberation_result.focus_concept
        inferences = deliberation_result.inferences or []
        
        # Find concepts activated by semantic similarity (not the query itself)
        main_concept = None
        main_concept_data = None
        concept_relevance = 0.0  # Track how relevant the concept is to the query
        
        for concept in activated:
            if concept.source == "semantic_similarity":
                # This is a concept ID from symbolic activation
                # Use the semantic bridge to get actual content
                concept_id = concept.term  # term holds the concept_id in symbolic mode
                
                try:
                    db_concept = self.concept_store.get_concept_by_id(concept_id)
                    if db_concept and db_concept.properties:
                        # VALIDATE: Check that the concept is actually relevant to the query
                        # This prevents "colours" from matching to "birds" just because
                        # birds was the only concept in the store
                        if self.semantic_encoder and user_input:
                            query_emb = self.semantic_encoder.encode(user_input)
                            concept_emb = self.semantic_encoder.encode(db_concept.term)
                            
                            # Flatten and normalize
                            if hasattr(query_emb, 'cpu'):
                                query_emb = query_emb.cpu().numpy().flatten()
                            if hasattr(concept_emb, 'cpu'):
                                concept_emb = concept_emb.cpu().numpy().flatten()
                            
                            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
                            concept_norm = concept_emb / (np.linalg.norm(concept_emb) + 1e-8)
                            concept_relevance = float(np.dot(query_norm, concept_norm))
                            
                            # Reject if concept isn't actually relevant to the query
                            if concept_relevance < 0.5:
                                print(f"  ⚠️ Rejecting concept '{db_concept.term}' - low relevance to query ({concept_relevance:.3f})")
                                continue
                        
                        print(f"  🔗 Bridge: {concept_id} → {db_concept.term} (relevance: {concept_relevance:.3f})")
                        main_concept = concept
                        main_concept_data = db_concept
                        break
                except Exception as e:
                    print(f"  ⚠️ Bridge failed for {concept_id}: {e}")
        
        # Fallback: try text-based retrieval if no symbolic match
        if not main_concept_data and focus_term:
            try:
                retrieved = self.concept_store.retrieve_by_text(focus_term, top_k=1, min_similarity=0.6)
                if retrieved:
                    candidate_concept, score = retrieved[0]
                    
                    # VALIDATE: Check relevance to actual user query (not just focus_term)
                    if self.semantic_encoder and user_input:
                        query_emb = self.semantic_encoder.encode(user_input)
                        concept_emb = self.semantic_encoder.encode(candidate_concept.term)
                        
                        if hasattr(query_emb, 'cpu'):
                            query_emb = query_emb.cpu().numpy().flatten()
                        if hasattr(concept_emb, 'cpu'):
                            concept_emb = concept_emb.cpu().numpy().flatten()
                        
                        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
                        concept_norm = concept_emb / (np.linalg.norm(concept_emb) + 1e-8)
                        concept_relevance = float(np.dot(query_norm, concept_norm))
                        
                        if concept_relevance < 0.5:
                            print(f"  ⚠️ Rejecting fallback concept '{candidate_concept.term}' - low relevance ({concept_relevance:.3f})")
                        else:
                            main_concept_data = candidate_concept
                            print(f"  🔍 Fallback retrieval: {main_concept_data.term} (score: {score:.2f}, relevance: {concept_relevance:.3f})")
                    else:
                        main_concept_data = candidate_concept
                        print(f"  🔍 Fallback retrieval: {main_concept_data.term} (score: {score:.2f})")
            except Exception as e:
                print(f"  ⚠️ Fallback retrieval failed: {e}")
        
        if not main_concept_data or not main_concept_data.properties:
            return None
        
        # BUILD SLOTS from concept data (linguistic content)
        available_slots = {}
        available_slots["concept"] = main_concept_data.term.capitalize()
        available_slots["topic"] = main_concept_data.term
        
        # Extract property from definition
        if main_concept_data.properties:
            definition = main_concept_data.properties[0]
            # Use the property extraction for cleaner responses
            primary_property = self._extract_property_from_definition(
                definition, main_concept_data.term
            ) or definition
            
            available_slots["primary_property"] = primary_property
            available_slots["definition"] = definition
            
            if len(main_concept_data.properties) > 1:
                available_slots["elaboration"] = main_concept_data.properties[1]
                available_slots["properties"] = ", ".join(main_concept_data.properties[1:3])
        
        # Add inference information if available
        connections = [inf for inf in inferences if inf.inference_type == "connection"]
        if connections:
            related_terms = [inf.source_concepts[1] for inf in connections[:2] 
                           if len(inf.source_concepts) > 1]
            if related_terms:
                available_slots["related"] = ", ".join(related_terms)
        
        # For confirmation questions, extract subject/relationship
        if category == "confirmation":
            subject, relationship = self._extract_confirmation_parts(user_input)
            if subject:
                available_slots["subject"] = subject
            if relationship:
                available_slots["relationship"] = relationship
            # Use first property as elaboration for confirmation
            if main_concept_data.properties:
                available_slots["elaboration"] = main_concept_data.properties[0][:100]
        
        # For teaching statements, acknowledge what was taught
        elif category == "teaching":
            subject, relationship = self._extract_teaching_parts(user_input)
            if subject:
                available_slots["subject"] = subject
            if relationship:
                available_slots["relationship"] = relationship
        
        # Use syntax stage to determine grammatical structure if available
        syntax_intent = "statement"
        if self.syntax_stage:
            try:
                # Process the main property to get syntactic structure
                tokens = main_concept_data.properties[0].split() if main_concept_data.properties else []
                if tokens:
                    syntax_artifact = self.syntax_stage.process(tokens)
                    syntax_intent = syntax_artifact.metadata.get('intent', 'statement')
            except Exception:
                pass
        
        # Get appropriate pragmatic template
        if not self.pragmatic_templates:
            return None
        
        template = self.pragmatic_templates.match_best_template(category, available_slots)
        if not template:
            # Fallback to definition template if category not found
            template = self.pragmatic_templates.match_best_template("definition", available_slots)
        
        if not template:
            return None
        
        # Fill template with slots
        response_text = self.pragmatic_templates.fill_template(template, available_slots)
        
        if not response_text:
            return None
        
        # Clean up grammar issues (double verbs, plural agreement)
        response_text = self._clean_composed_response(response_text)
        
        # Calculate confidence based on concept RELEVANCE (not just activation)
        # This ensures we don't confidently return unrelated concepts
        base_confidence = concept_relevance if concept_relevance > 0 else getattr(main_concept, 'activation', 0.5)
        inference_bonus = min(len(inferences) * 0.03, 0.10)  # Reduced bonus
        confidence = min(base_confidence + inference_bonus, 0.90)
        
        print(f"  🧠 Composed from deliberation: {len(activated)} concepts, {len(inferences)} inferences")
        
        return ComposedResponse(
            text=response_text,
            fragment_ids=[template.template_id, main_concept_data.concept_id],
            composition_weights=[0.6, 0.4],
            coherence_score=confidence,
            primary_pattern=None,
            confidence=confidence,
            is_fallback=False,
            is_low_confidence=confidence < 0.70
        )

    def _compose_math_response(self, query: str) -> Optional[ComposedResponse]:
        """
        Generate response using math backend.
        
        Args:
            query: Mathematical query
            
        Returns:
            ComposedResponse if math computation succeeded, None otherwise
        """
        if not self.math_backend:
            return None
        
        result = self.math_backend.compute(query)
        
        if result:
            # Format mathematical response
            response_text = self._format_math_result(result)
            
            # Track math usage
            self.metrics['math_count'] += 1
            
            return ComposedResponse(
                text=response_text,
                fragment_ids=["math_computed"],
                composition_weights=[1.0],
                coherence_score=1.0,
                confidence=result.confidence,
                is_fallback=False,
                is_low_confidence=False,
                modality=Modality.MATH if MODAL_ROUTING_AVAILABLE else None
            )
        
        # Math backend couldn't compute
        return None
    
    def _format_math_result(self, result) -> str:
        """Format mathematical result as natural language"""
        if result.steps and len(result.steps) > 1:
            # Show work
            steps_text = "\n".join(f"{step}" for step in result.steps)
            return f"{steps_text}"
        elif result.expression and result.result:
            # Simple result
            return f"{result.expression} = {result.result}"
        else:
            # Just the result
            return str(result.result)
    
    def _compose_parallel(
        self,
        context: str,
        user_input: str,
        topk: int = 5
    ) -> ComposedResponse:
        """
        Parallel implementation: Try BOTH pattern and concept approaches.
        
        This allows us to compare approaches and gradually transition.
        
        Args:
            context: Current conversation context
            user_input: Raw user input
            topk: Number of patterns to consider
            
        Returns:
            Best response from either approach
        """
        self.metrics['parallel_uses'] += 1
        
        # 0. Extract query structure if available
        query_match = None
        main_concept = None
        if self.query_matcher and user_input:
            query_match = self.query_matcher.match_query(user_input)
            if query_match and query_match.confidence > 0.75:
                main_concept = self.query_matcher.extract_main_concept(query_match)
        
        # 1. Try pattern-based approach (call internal method to avoid recursion)
        pattern_response = self._compose_from_patterns_internal(
            context,
            user_input,
            topk,
            use_intent_filtering=False,
            use_semantic_retrieval=True,
            semantic_weight=0.5
        )
        
        # 2. Try concept-based approach (use extracted concept if available)
        if main_concept:
            # Use extracted concept for focused retrieval
            concept_response = self._compose_from_concepts(
                context,
                main_concept  # Use extracted concept instead of full query
            )
        else:
            # Fallback to full query
            concept_response = self._compose_from_concepts(
                context,
                user_input
            )
        
        # 3. Choose best response
        if concept_response and not concept_response.is_low_confidence:
            # Concept-based has good confidence
            if pattern_response.is_fallback or pattern_response.is_low_confidence:
                # Pattern-based failed or low confidence, use concept
                self.metrics['concept_count'] += 1
                self.last_approach = 'concept'
                return concept_response
            else:
                # Both are good - compare confidence
                if concept_response.confidence > pattern_response.confidence:
                    self.metrics['concept_count'] += 1
                    self.last_approach = 'concept'
                    return concept_response
        
        # Default: Use pattern-based
        self.metrics['pattern_count'] += 1
        self.last_approach = 'pattern'
        return pattern_response
    
    def _compose_with_pragmatic_templates(
        self,
        context: str,
        user_input: str
    ) -> Optional[ComposedResponse]:
        """
        Compose response using pragmatic templates + concept store.
        
        This is the Layer 4 restructuring: Separate linguistic patterns (templates)
        from semantic knowledge (concepts).
        
        Flow:
        1. BioNN classifies intent (definition, greeting, acknowledgment, etc.)
        2. Select appropriate pragmatic template for that intent
        3. BioNN retrieves relevant concepts from concept store
        4. Fill template slots with concept properties
        5. Return composed response
        
        Args:
            context: Conversation context
            user_input: Raw user input
            
        Returns:
            Composed response or None if can't compose
        """
        if not self.pragmatic_templates or not self.concept_store:
            return None
        
        # Step 1: Detect conversational category from history
        category = self._detect_conversation_category(user_input)
        
        # Step 2: Extract concept from query (if definition/elaboration)
        concept_term = None
        if category in ["definition", "elaboration"]:
            # Simple extraction - in production would use syntax stage
            concept_term = self._extract_concept_from_query(user_input)
        
        # Step 3: Retrieve concept from store (if we have a term)
        concept = None
        if concept_term and self.concept_store:
            # Encode concept term
            embedding = self.encoder.encode(concept_term) if hasattr(self, 'encoder') else None
            
            if embedding is not None:
                # Retrieve similar concepts
                results = self.concept_store.retrieve_similar(embedding, top_k=1)
                if results:
                    concept_id, similarity = results[0]
                    if similarity > 0.60:  # Reasonable threshold
                        concept_data = self.concept_store.get_concept(concept_id)
                        if concept_data:
                            concept = concept_data
        
        # Step 4: Build available slots based on what we have
        available_slots = {}
        
        if category == "greeting":
            available_slots["offer_help"] = "How can I help you?"
            if self.conversation_history:
                recent = self.conversation_history.get_recent_turns(n=1)
                if recent:
                    last_topic = self._extract_topic_from_turn(recent[-1])
                    if last_topic:
                        available_slots["continue_previous_topic"] = f"Want to continue talking about {last_topic}?"
        
        elif category == "acknowledgment":
            # Build acknowledgment from context
            available_slots["elaboration"] = "That's interesting."
            if concept:
                available_slots["related_concept"] = f"That relates to {concept.term}."
        
        elif category == "opinion":
            # Extract topic from opinion question
            topic = self._extract_concept_from_opinion(user_input)
            if topic:
                available_slots["topic"] = topic
                
                # Look up concept to provide informed opinion
                if self.concept_store:
                    embedding = self.encoder.encode(topic) if hasattr(self, 'encoder') else None
                    if embedding is not None:
                        results = self.concept_store.retrieve_similar(embedding, top_k=1)
                        if results:
                            concept_id, similarity = results[0]
                            if similarity > 0.60:
                                concept_data = self.concept_store.get_concept(concept_id)
                                if concept_data and concept_data.properties:
                                    # Use first property as aspect
                                    available_slots["aspect"] = concept_data.properties[0]
                                    # Use second property as elaboration if available
                                    if len(concept_data.properties) > 1:
                                        available_slots["elaboration"] = concept_data.properties[1]
                                    else:
                                        available_slots["elaboration"] = f"There's so much to explore about {topic}."
                                else:
                                    # Generic elaboration
                                    available_slots["elaboration"] = f"There's a lot of interesting aspects to {topic}."
                            else:
                                # Low similarity - generic response
                                available_slots["elaboration"] = f"It's an interesting topic to explore."
                        else:
                            available_slots["elaboration"] = f"It's an interesting subject."
                    else:
                        available_slots["elaboration"] = f"That's a great question about {topic}!"
                else:
                    available_slots["elaboration"] = f"That's a thoughtful question."
        
        elif category == "definition" and concept:
            # Fill from concept properties
            available_slots["concept"] = concept.term
            if concept.properties:
                available_slots["primary_property"] = concept.properties[0]
                if len(concept.properties) > 1:
                    available_slots["elaboration"] = concept.properties[1]
                if len(concept.properties) > 2:
                    available_slots["properties"] = ", ".join(concept.properties[1:3])
        
        elif category == "continuation":
            if self.conversation_history:
                recent = self.conversation_history.get_recent_turns(n=1)
                if recent:
                    prev_topic = self._extract_topic_from_turn(recent[-1])
                    if prev_topic:
                        available_slots["previous_topic"] = prev_topic
                        if concept:
                            available_slots["new_info"] = f"{concept.term} {concept.properties[0] if concept.properties else ''}"
        
        elif category == "elaboration" and concept:
            available_slots["concept"] = concept.term
            if concept.properties:
                available_slots["examples"] = ", ".join(concept.properties[:2])
                available_slots["properties"] = ", ".join(concept.properties)
        
        # Step 5: Match template that we can fill
        if not available_slots:
            return None
        
        template = self.pragmatic_templates.match_best_template(category, available_slots)
        if not template:
            return None
        
        # Step 6: Fill template
        response_text = self.pragmatic_templates.fill_template(template, available_slots)
        
        # Clean up grammar issues
        response_text = self._clean_composed_response(response_text)
        
        # Step 7: Return composed response
        return ComposedResponse(
            text=response_text,
            fragment_ids=[template.template_id],
            composition_weights=[1.0],
            coherence_score=0.85,
            confidence=0.85,
            is_fallback=False,
            is_low_confidence=False
        )
    
    def _detect_conversation_category(self, user_input: str) -> str:
        """
        Detect conversational category from user input.
        
        Simple heuristic version - in production would use BioNN intent classifier.
        
        Args:
            user_input: User message
            
        Returns:
            Category: "greeting", "definition", "acknowledgment", "opinion", 
                      "confirmation", "teaching", etc.
        """
        user_lower = user_input.lower().strip()
        
        # Greetings
        if any(word in user_lower for word in ["hello", "hi", "hey", "greetings"]):
            return "greeting"
        
        # Opinion/preference questions
        if any(phrase in user_lower for phrase in [
            "do you like",
            "do you prefer",
            "what do you think",
            "what's your opinion",
            "what's your favorite",
            "what's your favourite",
            "how do you feel about",
            "are you interested in"
        ]):
            return "opinion"
        
        # Knowledge queries - "Do you know about X?" (asking about AI's knowledge)
        # These should be treated as definition queries, NOT confirmations
        if any(phrase in user_lower for phrase in [
            "do you know about",
            "do you know what",
            "do you know of",
            "have you heard of",
            "have you heard about",
            "are you familiar with"
        ]):
            return "definition"
        
        # Capability/action requests TO THE AI - "Can you X?", "Could you X?"
        # These are requests for the AI to do something, not yes/no factual questions
        import re
        if re.match(r"^(can|could|would|will)\s+you\s+", user_lower):
            return "capability_request"
        
        # Opinion questions TO THE AI - "Do you like X?", "Do you think X?"
        if re.match(r"^do\s+you\s+(like|love|hate|enjoy|prefer|think|believe|feel)\b", user_lower):
            return "opinion"
        
        # Yes/No confirmation questions - "Is X a Y?", "Are X Y?", "Can X do Y?"
        # Note: These are about FACTS, not requests to the AI
        confirmation_patterns = [
            r"^is\s+(a|an|the)?\s*\w+\s+(a|an|the|type|kind|form|part|sort)\b",  # Is [a] X a/type of Y?
            r"^are\s+(a|an|the)?\s*\w+\s+(a|an|the|type|kind|form|part|sort)\b",  # Are [a] X a/type of Y?
            r"^is\s+(a|an|the)?\s*\w+\s+\w+.*\?$",  # Is X Y? (simple yes/no with ?)
            r"^are\s+(a|an|the)?\s*\w+\s+\w+",  # Are X Y?
            r"^can\s+(?!you\s)\w+\s+\w+",  # Can X do Y? (but NOT "can you")
            r"^does\s+(?!it\s+refer|this\s+refer)\w+\s+\w+",  # Does X have Y?
            r"^do\s+(?!you\s)\w+\s+\w+",  # Do X Y? (but NOT "do you")
        ]
        for pattern in confirmation_patterns:
            if re.match(pattern, user_lower):
                return "confirmation"
        
        # Teaching/Statements - "X is a Y", "X are Y" (declarative statements)
        # These are NOT questions - user is teaching the system
        if not user_input.strip().endswith('?'):
            teaching_patterns = [
                r"^(a|an|the)?\s*\w+\s+is\s+(a|an|the)?\s*",  # "A Wyvern is a Dragon"
                r"^(a|an|the)?\s*\w+\s+are\s+(a|an|the)?\s*",  # "Wyverns are dragons"
                r"^\w+\s+means?\s+",  # "X means Y"
                r"^\w+\s+refers?\s+to\s+",  # "X refers to Y"
            ]
            for pattern in teaching_patterns:
                if re.match(pattern, user_lower):
                    return "teaching"
        
        # Definitions - include "tell me about" for initial queries about a topic
        if any(phrase in user_lower for phrase in ["what is", "what are", "define", "definition of", "tell me about", "describe"]):
            return "definition"
        
        # Acknowledgments (short affirmations)
        if user_lower in ["ok", "okay", "i see", "got it", "thanks", "thank you", "interesting", "cool", "nice"]:
            return "acknowledgment"
        
        # Elaborations - requests for MORE detail on something already discussed
        if any(phrase in user_lower for phrase in ["tell me more", "elaborate", "explain more", "more about", "more details"]):
            return "elaboration"
        
        # Continuations (building on previous)
        if any(phrase in user_lower for phrase in ["and", "also", "what about", "how about"]):
            return "continuation"
        
        # Default to elaboration
        return "elaboration"
    
    def _extract_concept_from_query(self, query: str) -> Optional[str]:
        """
        Extract concept term from query.
        
        Simple version - strips question patterns.
        
        Args:
            query: User query
            
        Returns:
            Concept term or None
        """
        query_lower = query.lower()
        
        # Remove question patterns (including knowledge queries)
        for pattern in [
            "do you know about", "do you know what", "do you know of",
            "have you heard of", "have you heard about", "are you familiar with",
            "what is", "what are", "define", "definition of", "tell me about", "explain"
        ]:
            query_lower = query_lower.replace(pattern, "").strip()
        
        # Remove question mark
        query_lower = query_lower.replace("?", "").strip()
        
        # Remove question words but KEEP the remaining phrase intact
        # Don't remove articles at the start of multi-word concepts
        words = query_lower.split()
        
        # Only remove leading question words, not articles that are part of the concept name
        while words and words[0] in ["what", "how", "why", "when", "where", "who"]:
            words.pop(0)
        
        # Remove trailing articles but keep leading ones for compound proper nouns
        # e.g., "The Elder Scrolls" should stay intact
        concept = " ".join(words).strip()
        
        # Remove leading article only if not part of a title (heuristic: capitalized words follow)
        if concept.startswith(("the ", "a ", "an ")):
            rest = concept.split(" ", 1)[1] if " " in concept else ""
            # If the rest doesn't start with uppercase, it's probably not a title
            if rest and not rest[0].isupper():
                concept = rest
        
        return concept if concept else None
    
    def _extract_property_from_definition(self, definition: str, concept: str) -> Optional[str]:
        """
        Extract the property part from a definition sentence.
        
        Handles definitions like:
        - "The wyvern, sometimes spelled wivern, is a type of mythical dragon"
        - "Python is a programming language"
        - "Parrots (Psittaciformes), also known as psittacines, are birds with..."
        
        Extracts the part after "is/are" to get just the property.
        
        Args:
            definition: Full definition text
            concept: The concept being defined (to match variants)
            
        Returns:
            Property part (e.g., "a type of mythical dragon") or None
        """
        import re
        
        # Try to find "is/are" pattern and extract what comes after
        # Match: "[concept], [optional extra], is/are [property]"
        patterns = [
            r'\b(?:is|are)\s+(.+?)(?:\.|$)',  # General "is/are X" pattern
        ]
        
        for pattern in patterns:
            match = re.search(pattern, definition, re.IGNORECASE)
            if match:
                property_part = match.group(1).strip()
                # Clean up trailing punctuation
                property_part = property_part.rstrip('.,;:')
                if property_part:
                    return property_part
        
        return None
    
    def _clean_composed_response(self, response: str) -> str:
        """
        Clean up common grammar issues in composed responses.
        
        Fixes:
        - Double "is is" or "are are"
        - Singular/plural agreement for common patterns
        
        Args:
            response: Raw composed response
            
        Returns:
            Cleaned response
        """
        import re
        
        # Fix double verb patterns
        response = re.sub(r'\bis\s+is\b', 'is', response, flags=re.IGNORECASE)
        response = re.sub(r'\bare\s+are\b', 'are', response, flags=re.IGNORECASE)
        response = re.sub(r'\bis\s+are\b', 'are', response, flags=re.IGNORECASE)
        response = re.sub(r'\bare\s+is\b', 'is', response, flags=re.IGNORECASE)
        
        # Fix "X is a/an [plural noun]" → "X are [plural noun]"
        # Common plural endings: -s (birds), -es (watches), -ies (countries)
        # Only apply if the subject itself appears to be plural
        
        # Fix subject-verb agreement for plural subjects ending in 's'
        # Pattern: "[Plural] is a" → "[Plural] are"
        # But only if the word before "is" looks plural (ends in s, not a name)
        def fix_plural_agreement(match):
            subject = match.group(1)
            verb = match.group(2)
            rest = match.group(3)
            
            # Check if subject is likely plural (ends in 's' but not common singular words)
            singular_s_words = {'this', 'has', 'is', 'was', 'does', 'its', 'his', 'hers', 
                              'business', 'class', 'glass', 'grass', 'pass', 'mass',
                              'success', 'process', 'address', 'congress', 'progress'}
            
            subject_lower = subject.lower()
            if (subject_lower.endswith('s') and 
                subject_lower not in singular_s_words and
                not subject_lower.endswith('ss') and
                len(subject_lower) > 2):
                # Likely plural - use "are"
                return f"{subject} are {rest}"
            else:
                return match.group(0)  # Keep original
        
        # Apply plural agreement fix
        response = re.sub(
            r'\b(\w+)\s+(is)\s+(a\s+)',
            fix_plural_agreement,
            response,
            flags=re.IGNORECASE
        )
        
        return response
    
    def _extract_concept_from_opinion(self, query: str) -> Optional[str]:
        """
        Extract topic/concept from opinion question.
        
        Args:
            query: Opinion question like "Do you like birds?"
            
        Returns:
            Topic term like "birds"
        """
        query_lower = query.lower()
        
        # Remove opinion question patterns
        patterns = [
            "do you like",
            "do you prefer",
            "what do you think about",
            "what do you think of",
            "what's your opinion on",
            "what's your opinion of",
            "what's your favorite",
            "what's your favourite",
            "how do you feel about",
            "are you interested in"
        ]
        
        for pattern in patterns:
            if pattern in query_lower:
                query_lower = query_lower.replace(pattern, "").strip()
                break
        
        # Remove question mark
        query_lower = query_lower.replace("?", "").strip()
        
        # Remove articles
        words = query_lower.split()
        concept_words = [w for w in words if w not in ["the", "a", "an", "or"]]
        
        return " ".join(concept_words) if concept_words else None
    
    def _extract_confirmation_parts(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract subject and relationship from a confirmation question.
        
        Args:
            query: Question like "Is a Wyvern a type of Dragon?"
            
        Returns:
            Tuple of (subject, relationship) like ("a wyvern", "a type of dragon")
        """
        import re
        query_lower = query.lower().strip().rstrip('?')
        
        # Pattern: "Is X a/an Y?"
        match = re.match(r'^is\s+(a|an|the)?\s*(\w+)\s+(a|an|the)?\s*(.+)$', query_lower)
        if match:
            article1, subject, article2, relationship = match.groups()
            subject_str = f"{article1 or ''} {subject}".strip()
            relationship_str = f"{article2 or ''} {relationship}".strip()
            return subject_str, relationship_str
        
        # Pattern: "Are X Y?"
        match = re.match(r'^are\s+(\w+)\s+(.+)$', query_lower)
        if match:
            subject, relationship = match.groups()
            return subject, relationship
        
        # Pattern: "Can X Y?" / "Does X Y?"
        match = re.match(r'^(can|does|do)\s+(\w+)\s+(.+)$', query_lower)
        if match:
            verb, subject, rest = match.groups()
            return subject, f"{verb} {rest}"
        
        return None, None
    
    def _confirmation_to_statement(self, query: str) -> Optional[str]:
        """
        Convert a confirmation question to a statement for pattern matching.
        
        This helps match "is a parrot a bird?" to stored facts like "A parrot is a bird".
        
        Args:
            query: Confirmation question like "is a parrot a bird?"
            
        Returns:
            Statement form like "a parrot is a bird" or None
        """
        import re
        query_lower = query.lower().strip().rstrip('?')
        
        # Pattern: "is X a/an Y?" -> "X is a/an Y"
        match = re.match(r'^is\s+(a|an|the)?\s*(.+?)\s+(a|an)\s+(.+)$', query_lower)
        if match:
            article1, subject, article2, relationship = match.groups()
            subject_str = f"{article1 + ' ' if article1 else ''}{subject}"
            return f"{subject_str} is {article2} {relationship}"
        
        # Pattern: "are X Y?" -> "X are Y"
        match = re.match(r'^are\s+(a|an|the)?\s*(.+?)\s+(.+)$', query_lower)
        if match:
            article, subject, predicate = match.groups()
            subject_str = f"{article + ' ' if article else ''}{subject}"
            return f"{subject_str} are {predicate}"
        
        # Pattern: "is X adjective?" -> "X is adjective"
        match = re.match(r'^is\s+(a|an|the)?\s*(.+?)\s+(\w+)$', query_lower)
        if match:
            article, subject, adjective = match.groups()
            subject_str = f"{article + ' ' if article else ''}{subject}"
            return f"{subject_str} is {adjective}"
        
        # Pattern: "does X Y?" -> "X does Y" (for verbs)
        match = re.match(r'^does\s+(a|an|the)?\s*(.+?)\s+(.+)$', query_lower)
        if match:
            article, subject, verb_phrase = match.groups()
            subject_str = f"{article + ' ' if article else ''}{subject}"
            return f"{subject_str} does {verb_phrase}"
        
        # Pattern: "can X Y?" -> "X can Y"
        match = re.match(r'^can\s+(a|an|the)?\s*(.+?)\s+(.+)$', query_lower)
        if match:
            article, subject, verb_phrase = match.groups()
            subject_str = f"{article + ' ' if article else ''}{subject}"
            return f"{subject_str} can {verb_phrase}"
        
        return None
    
    def _extract_teaching_parts(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract subject and relationship from a teaching statement.
        
        Args:
            query: Statement like "A Wyvern is a type of Dragon"
            
        Returns:
            Tuple of (subject, relationship) like ("a wyvern", "a type of dragon")
        """
        import re
        query_lower = query.lower().strip()
        
        # Pattern: "X is Y"
        match = re.match(r'^(a|an|the)?\s*(\w+)\s+is\s+(a|an|the)?\s*(.+)$', query_lower)
        if match:
            article1, subject, article2, relationship = match.groups()
            subject_str = f"{article1 or ''} {subject}".strip()
            relationship_str = f"{article2 or ''} {relationship}".strip()
            return subject_str, relationship_str
        
        # Pattern: "X are Y"
        match = re.match(r'^(a|an|the)?\s*(\w+)\s+are\s+(a|an|the)?\s*(.+)$', query_lower)
        if match:
            article1, subject, article2, relationship = match.groups()
            subject_str = f"{article1 or ''} {subject}".strip()
            relationship_str = f"{article2 or ''} {relationship}".strip()
            return subject_str, relationship_str
        
        # Pattern: "X means Y"
        match = re.match(r'^(\w+)\s+means?\s+(.+)$', query_lower)
        if match:
            subject, relationship = match.groups()
            return subject, relationship
        
        return None, None

    def _extract_topic_from_turn(self, turn) -> Optional[str]:
        """Extract main topic from a conversation turn."""
        # Simple: just use first few words of user input
        user_input = turn.user_input if hasattr(turn, 'user_input') else turn.get('user_input', '')
        words = user_input.split()[:3]  # First 3 words
        return " ".join(words) if words else None
    
    def get_metrics(self) -> Dict:
        """Get metrics comparing pattern vs concept approaches."""
        total = self.metrics['pattern_count'] + self.metrics['concept_count']
        
        if total == 0:
            return self.metrics
        
        return {
            **self.metrics,
            'pattern_ratio': self.metrics['pattern_count'] / total,
            'concept_ratio': self.metrics['concept_count'] / total,
            'pattern_success_rate': (
                self.metrics['pattern_success'] / self.metrics['pattern_count']
                if self.metrics['pattern_count'] > 0 else 0
            ),
            'concept_success_rate': (
                self.metrics['concept_success'] / self.metrics['concept_count']
                if self.metrics['concept_count'] > 0 else 0
            )
        }
            
    def _fallback_response(self, user_input: str = "") -> ComposedResponse:
        """
        Smart fallback: Fill knowledge gaps before falling back.
        
        Strategy:
        1. Identify unknown words/concepts in query
        2. Look them up in external sources
        3. Re-attempt pattern matching with enhanced understanding
        4. If successful, teach the pattern automatically
        5. Otherwise, try direct external knowledge lookup
        6. Finally, graceful fallback
        
        Args:
            user_input: User's query (for knowledge lookup and gap analysis)
        """
        # STEP 1: Try to fill knowledge gaps and re-attempt matching
        aug_for_fallbacks = self._ensure_knowledge_augmenter()
        if aug_for_fallbacks and user_input:
            filled_response = self._fill_gaps_and_retry(user_input)
            if filled_response:
                return filled_response
        
        # If gap-filling didn't help, try direct external knowledge lookup
        if aug_for_fallbacks and user_input:
            # Get conversation context for disambiguation
            conv_context = self._get_conversation_context(max_turns=3)
            external_result = aug_for_fallbacks.lookup(user_input, conversation_history=conv_context, min_confidence=0.6)
            
            if external_result:
                response_text, confidence, source = external_result
                
                # Ingest the snippet into vocab/concepts/patterns before replying
                self._ingest_external_knowledge(user_input, response_text, confidence, source)

                # Clean up grammar issues (plural agreement, etc.) and keep concise
                response_text = self._clean_composed_response(self._first_sentence(response_text))
                
                print(f"  💡 Filled knowledge gap from {source} (confidence: {confidence:.2f})")
                
                # Return external knowledge as response
                # The teaching mechanism will learn this pattern automatically
                return ComposedResponse(
                    text=response_text,
                    fragment_ids=[f"external_{source}"],
                    composition_weights=[confidence],
                    coherence_score=confidence,
                    primary_pattern=None,  # No pattern yet - will be learned
                    confidence=confidence,
                    is_fallback=True,  # Triggered by fallback
                    is_low_confidence=False  # But knowledge was found
                )
        
        # Try a lightweight reasoning pass before giving up
        if self.reasoning_stage and user_input:
            try:
                conv_context = self._get_conversation_context(max_turns=3)
                reasoning_result = self.reasoning_stage.deliberate(
                    query=user_input,
                    context=conv_context
                )

                if reasoning_result and reasoning_result.inferences:
                    best_inference = max(
                        reasoning_result.inferences,
                        key=lambda inf: inf.confidence
                    )
                    inferred_text = self._clean_composed_response(
                        f"From reasoning, {best_inference.conclusion}. If that's off, teach me the right answer and upvote with '/+'."
                    )

                    return ComposedResponse(
                        text=inferred_text,
                        fragment_ids=["reasoning_inference"],
                        composition_weights=[best_inference.confidence],
                        coherence_score=best_inference.confidence,
                        primary_pattern=None,
                        confidence=best_inference.confidence,
                        is_fallback=True,
                        is_low_confidence=False
                    )
            except Exception as exc:
                print(f"  ⚠️  Reasoning stage failed in fallback: {exc}")

        # Standard fallback if no external knowledge or reasoning help found
        return ComposedResponse(
            text="I don't have information about that yet. If you know the answer, you can teach me by typing it as your next message, then upvoting with '/+'!",
            fragment_ids=["fallback"],
            composition_weights=[1.0],
            coherence_score=0.0,
            primary_pattern=None,
            confidence=0.0,
            is_fallback=True,
            is_low_confidence=True
        )
    
    def _fill_gaps_and_retry(self, user_input: str) -> Optional[ComposedResponse]:
        """
        Attempt to fill knowledge gaps and retry pattern matching.
        
        ENHANCED LEARNING INTEGRATION (Phase 1 + Phase 2):
        Instead of just memorizing query→response patterns, this now performs
        TRANSPARENT ONLINE LEARNING with REASONING to build true understanding:
        
        Process:
        1. Extract key terms from query (nouns, technical words)
        2. Look up unknown terms in external sources
        3. **LEARN VOCABULARY** - Track terms and their definitions
        4. **LEARN CONCEPTS** - Extract semantic concepts from definitions
        5. **LEARN SYNTAX** - Extract linguistic patterns from definitions
        6. **BUILD CONNECTIONS** (Phase 2) - Use reasoning stage to find relationships
        7. Build enhanced context with learned knowledge
        8. Retry pattern matching with enhanced context
        9. If match found, teach it as new pattern
        
        Phase 2 Integration:
        After learning vocabulary/concepts/syntax, the reasoning stage:
        - Activates newly learned concepts in working memory
        - Runs deliberation to find connections with existing knowledge
        - Generates inferences about concept relationships
        - Builds semantic network of learned information
        
        This enables Lilith to not only learn individual facts, but understand
        how they relate to existing knowledge - creating a connected knowledge graph
        rather than isolated facts.
        
        Args:
            user_input: Original user query
            
        Returns:
            ComposedResponse if gaps filled and match found, None otherwise
        """
        # Extract potential unknown terms
        unknown_terms = self._extract_unknown_terms(user_input)
        
        if not unknown_terms:
            return None

        aug_for_terms = self._ensure_knowledge_augmenter()
        if not aug_for_terms:
            return None
        
        # Look up each unknown term AND learn from definitions
        term_definitions = {}
        learned_count = 0
        
        for term in unknown_terms:
            # Try to get definition/explanation
            # Get conversation context for disambiguation
            conv_context = self._get_conversation_context(max_turns=3)
            result = aug_for_terms.lookup(f"What is {term}?", conversation_history=conv_context, min_confidence=0.6)
            if result:
                definition, confidence, source = result
                term_definitions[term] = {
                    'definition': definition,
                    'confidence': confidence,
                    'source': source
                }
                print(f"  🔍 Learned about '{term}' from {source}")
                
                # ═══════════════════════════════════════════════════════
                # PHASE 1: FULL LEARNING INTEGRATION
                # ═══════════════════════════════════════════════════════
                
                # 1. VOCABULARY LEARNING
                # Track the term and its definition for future queries
                if hasattr(self.fragments, 'vocabulary') and self.fragments.vocabulary:
                    try:
                        # Track the full definition text (includes the term and related words)
                        # VocabularyTracker.track_text() extracts terms automatically
                        tracked = self.fragments.vocabulary.track_text(
                            text=f"{term}: {definition}",
                            source=source
                        )
                        
                        learned_count += 1
                        print(f"     📖 Vocabulary: Tracked '{term}' and {len(tracked)} related terms from definition")
                    except Exception as e:
                        print(f"     ⚠️  Vocabulary tracking failed: {e}")
                
                # 2. CONCEPT LEARNING
                # Extract semantic concepts from the definition
                if hasattr(self.fragments, 'concept_store') and self.fragments.concept_store:
                    try:
                        # Extract key noun phrases from definition as concepts
                        # Simple extraction: take capitalized terms and significant nouns
                        import re
                        
                        # Extract sentences
                        sentences = definition.split('.')
                        concepts_added = 0
                        
                        for sentence in sentences[:2]:  # First 2 sentences
                            sentence = sentence.strip()
                            if not sentence:
                                continue
                            
                            # Extract the main term and add as concept
                            # Use the term as a concept with the sentence as a property
                            self.fragments.concept_store.add_concept(
                                term=term.lower(),
                                properties=[sentence],
                                source=source,
                                confidence=confidence
                            )
                            concepts_added += 1
                            break  # Only add the term once
                        
                        if concepts_added > 0:
                            print(f"     🧠 Concepts: Added '{term}' to concept store")
                            learned_count += 1
                    except Exception as e:
                        print(f"     ⚠️  Concept learning failed: {e}")
                
                # 3. SYNTAX PATTERN LEARNING
                # Extract linguistic patterns from the definition
                if hasattr(self.fragments, 'pattern_extractor') and self.fragments.pattern_extractor:
                    try:
                        # Extract patterns from well-formed definition text
                        # This helps with generating similar explanations in the future
                        # PatternExtractor.extract_patterns() returns PatternMatch objects
                        patterns = self.fragments.pattern_extractor.extract_patterns(
                            text=definition,
                            source=source
                        )
                        
                        if patterns:
                            print(f"     📝 Syntax: Extracted {len(patterns)} linguistic patterns")
                            learned_count += 1
                    except Exception as e:
                        print(f"     ⚠️  Pattern extraction failed: {e}")
                
                # 4. BioNN SEMANTIC LEARNING
                # Train the BioNN on semantic relationships from the definition
                # This allows the neural embeddings to learn concept associations
                if self.contrastive_learner:
                    try:
                        import re
                        pairs_added = 0
                        
                        # Extract key relationships from definition
                        # Pattern: "X is a type of Y" → (X, Y, positive)
                        # Pattern: "X is a Y" → (X, Y, positive)
                        type_match = re.search(
                            r'^.{0,30}?\bis\s+(?:a\s+)?(?:type|kind|form|sort)\s+of\s+(\w+)',
                            definition.lower()
                        )
                        if type_match:
                            related_term = type_match.group(1)
                            self.contrastive_learner.add_pair(
                                anchor=term.lower(),
                                other=related_term,
                                relationship="positive",
                                weight=confidence,
                                source=f"wikipedia_{source}"
                            )
                            pairs_added += 1
                        
                        # Extract other capitalized concepts from definition as related
                        related_concepts = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', definition)
                        for related in related_concepts[:3]:  # Top 3 related concepts
                            related_lower = related.lower()
                            if related_lower != term.lower() and len(related_lower) > 2:
                                self.contrastive_learner.add_pair(
                                    anchor=term.lower(),
                                    other=related_lower,
                                    relationship="positive",
                                    weight=confidence * 0.7,  # Lower weight for inferred relations
                                    source=f"wikipedia_cooccur_{source}"
                                )
                                pairs_added += 1
                        
                        if pairs_added > 0:
                            print(f"     🧠 BioNN: Added {pairs_added} semantic pairs for training")
                            
                            # Periodic incremental training (every 5 new pairs)
                            if len(self.contrastive_learner.pairs) % 5 == 0:
                                self.contrastive_learner.incremental_update(
                                    num_steps=3,  # Quick training update
                                    learning_rate=0.001
                                )
                                print(f"     🎓 BioNN: Incremental training step completed")
                            
                            learned_count += 1
                    except Exception as e:
                        print(f"     ⚠️  BioNN semantic learning failed: {e}")
                
                # ═══════════════════════════════════════════════════════
                # PHASE 2: REASONING STAGE INTEGRATION
                # ═══════════════════════════════════════════════════════
                
                # After learning vocabulary/concepts/syntax, use reasoning stage
                # to build CONNECTIONS between the newly learned concepts.
                # This operates at the SYMBOLIC LEVEL, not language level.
                
                if self.reasoning_stage and hasattr(self.fragments, 'concept_store') and self.fragments.concept_store:
                    try:
                        # Activate the newly learned concept in reasoning stage
                        # This allows it to interact with existing concepts
                        concept_embedding = self.reasoning_stage.encoder.encode(term.lower().split())
                        
                        activated_concept = self.reasoning_stage.activate_concept(
                            term=term.lower(),
                            embedding=concept_embedding,
                            activation=confidence,  # Use external source confidence
                            source="learned_external",
                            properties=[definition.split('.')[0]]  # First sentence as property
                        )
                        
                        # Run deliberation to find connections with existing knowledge
                        # This helps build semantic network of learned concepts
                        deliberation = self.reasoning_stage.deliberate(
                            query=f"{term} in context: {user_input}",
                            context=definition,
                            max_steps=2  # Quick deliberation for learning
                        )
                        
                        # Log any inferences discovered
                        if deliberation.inferences:
                            print(f"     🔗 Reasoning: Found {len(deliberation.inferences)} connections for '{term}'")
                            for inference in deliberation.inferences[:2]:  # Show top 2
                                print(f"        → {inference.inference_type}: {inference.conclusion[:60]}...")
                            learned_count += 1
                        
                    except Exception as e:
                        print(f"     ⚠️  Reasoning stage integration failed: {e}")
        
        if not term_definitions:
            return None
        
        if learned_count > 0:
            print(f"  ✨ Successfully learned {learned_count} knowledge components on-the-fly!")
        
        # Build enhanced context
        enhanced_context = user_input
        for term, info in term_definitions.items():
            # Add definition as context (simplified)
            definition = info['definition'].split('.')[0]  # First sentence
            enhanced_context += f" ({term}: {definition})"
        
        # Retry pattern matching with enhanced context
        print(f"  🔄 Retrying with enhanced context: {enhanced_context[:100]}...")
        
        # ENHANCED: Try pragmatic composition first if available
        if self.pragmatic_templates and self.concept_store and term_definitions:
            print(f"  💡 Attempting compositional reasoning with {len(term_definitions)} learned concepts...")
            
            # Try to compose response using learned concepts
            compositional_response = self._compose_with_learned_concepts(
                user_input, 
                list(term_definitions.keys()),  # List of concept terms
                term_definitions  # Full concept data
            )
            
            if compositional_response and compositional_response.confidence >= 0.70:
                print(f"  ✨ Compositional reasoning succeeded! (confidence: {compositional_response.confidence:.2f})")
                return compositional_response
            else:
                print(f"  ⚠️  Compositional reasoning did not produce high-confidence response, falling back to patterns")
        
        # Get patterns with enhanced context
        patterns = self.fragments.retrieve_patterns(enhanced_context, topk=5, min_score=0.3)
        
        if not patterns:
            return None
        
        best_pattern, best_score = patterns[0]
        
        # Check if enhanced matching improved the score
        # Use a reasonable threshold (0.65 for gap-filled patterns)
        if best_score >= 0.65:
            print(f"  ✨ Gap-filling improved match! Score: {best_score:.3f}")
            
            # Use the matched pattern's response text directly
            response_text = best_pattern.response_text
            
            # Teach a new pattern combining original query with matched response
            # This "fills the gap" in the knowledge base
            if hasattr(self.fragments, 'add_pattern'):
                try:
                    new_pattern_id = self.fragments.add_pattern(
                        trigger_context=user_input,
                        response_text=response_text,
                        intent="gap_filled",
                        success_score=best_score * 0.9  # Slightly lower than direct match
                    )
                    print(f"  📚 Taught gap-filled pattern: {new_pattern_id}")
                except Exception as e:
                    print(f"  ⚠️  Could not teach gap-filled pattern: {e}")
            
            return ComposedResponse(
                text=response_text,
                fragment_ids=[best_pattern.fragment_id],
                composition_weights=[best_score],
                coherence_score=best_score,
                primary_pattern=best_pattern,
                confidence=best_score,
                is_fallback=False,  # Successfully filled gaps!
                is_low_confidence=False
            )
        
        return None

    def _first_sentence(self, text: str) -> str:
        """Return the first sentence (or a short snippet) to avoid verbatim dumps."""

        if not text:
            return ""
        sentence = text.split('.')[0].strip()
        if sentence:
            return sentence
        return text[:240].strip()

    def _ingest_external_knowledge(self, term_hint: str, definition: str, confidence: float, source: str) -> None:
        """Feed external snippets into vocabulary/concepts/patterns without verbatim dumping."""

        if not definition:
            return

        term = self._extract_concept_from_query(term_hint) or term_hint.strip()
        if not term:
            return

        snippet = self._first_sentence(definition)

        # Vocabulary tracking
        if hasattr(self.fragments, 'vocabulary') and self.fragments.vocabulary:
            try:
                self.fragments.vocabulary.track_text(text=f"{term}: {snippet}", source=source)
                print(f"     📖 Vocabulary: tracked '{term}' from {source}")
            except Exception as exc:
                print(f"     ⚠️  Vocabulary tracking failed: {exc}")

        # Concept store
        if hasattr(self.fragments, 'concept_store') and self.fragments.concept_store:
            try:
                self.fragments.concept_store.add_concept(
                    term=term.lower(),
                    properties=[snippet],
                    source=source,
                    confidence=confidence
                )
                print(f"     🧠 Concepts: added '{term}' from {source}")
            except Exception as exc:
                print(f"     ⚠️  Concept learning failed: {exc}")

        # Syntax patterns
        if hasattr(self.fragments, 'pattern_extractor') and self.fragments.pattern_extractor:
            try:
                patterns = self.fragments.pattern_extractor.extract_patterns(text=snippet, source=source)
                if patterns:
                    print(f"     📝 Syntax: extracted {len(patterns)} patterns from {source}")
            except Exception as exc:
                print(f"     ⚠️  Pattern extraction failed: {exc}")

        # Contrastive/BioNN associations
        if self.contrastive_learner:
            try:
                import re
                pairs_added = 0
                type_match = re.search(r"is\s+(?:a\s+)?(?:type|kind|form|sort)\s+of\s+(\w+)", snippet.lower())
                if type_match:
                    related = type_match.group(1)
                    self.contrastive_learner.add_pair(
                        anchor=term.lower(),
                        other=related,
                        relationship="positive",
                        weight=confidence,
                        source=f"external_{source}"
                    )
                    pairs_added += 1

                proper_nouns = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", snippet)
                for related in proper_nouns[:2]:
                    related_lower = related.lower()
                    if related_lower != term.lower():
                        self.contrastive_learner.add_pair(
                            anchor=term.lower(),
                            other=related_lower,
                            relationship="positive",
                            weight=confidence * 0.7,
                            source=f"external_cooccur_{source}"
                        )
                        pairs_added += 1

                if pairs_added:
                    print(f"     🧠 BioNN: added {pairs_added} semantic pairs from {source}")
            except Exception as exc:
                print(f"     ⚠️  BioNN semantic learning failed: {exc}")
    
    def _compose_with_learned_concepts(
        self,
        query: str,
        learned_concepts: List[str],
        concept_data: Dict[str, Dict]
    ) -> Optional[ComposedResponse]:
        """
        Compose response using freshly learned concepts + pragmatic templates.
        
        This is the compositional reasoning approach: Instead of just pattern matching,
        we compose a novel response from learned concepts.
        
        Args:
            query: Original user query
            learned_concepts: List of concept IDs learned from knowledge sources
            concept_data: Dict of concept_id -> {definition, properties, ...}
            
        Returns:
            Composed response or None
        """
        if not self.pragmatic_templates or not learned_concepts:
            return None
        
        # Detect category from query
        category = self._detect_conversation_category(query)
        
        # Build slots from learned concepts
        available_slots = {}
        
        # Use the first learned concept as primary
        primary_concept_id = learned_concepts[0]
        primary_data = concept_data.get(primary_concept_id, {})
        
        if category == "definition" and primary_data:
            # Extract concept term from the query or data
            concept_term = self._extract_concept_from_query(query)
            if not concept_term and 'term' in primary_data:
                concept_term = primary_data['term']
            
            if concept_term:
                available_slots["concept"] = concept_term.capitalize()
                
                # Use learned definition
                if 'definition' in primary_data:
                    definition_text = primary_data['definition']
                    # Extract the property part from "X is Y" style definitions
                    # Handle: "The wyvern, sometimes spelled wivern, is a type of mythical dragon"
                    # Should extract: "a type of mythical dragon"
                    primary_property = self._extract_property_from_definition(definition_text, concept_term)
                    
                    # Split remaining into elaboration
                    sentences = definition_text.split('.')
                    if primary_property:
                        available_slots["primary_property"] = primary_property
                    elif sentences:
                        available_slots["primary_property"] = sentences[0].strip()
                    
                    if len(sentences) > 1:
                        available_slots["elaboration"] = sentences[1].strip()
                
                # Add properties if available
                if 'properties' in primary_data:
                    properties = primary_data['properties']
                    if isinstance(properties, list):
                        available_slots["properties"] = ", ".join(properties[:3])
                    elif isinstance(properties, str):
                        available_slots["properties"] = properties
        
        elif category == "elaboration" and primary_data:
            concept_term = self._extract_concept_from_query(query)
            if concept_term:
                available_slots["concept"] = concept_term
                
                # Use learned information as examples - use full definition, not truncated
                if 'definition' in primary_data:
                    available_slots["examples"] = primary_data['definition']
                
                if 'properties' in primary_data:
                    props = primary_data['properties']
                    if isinstance(props, list):
                        available_slots["properties"] = ", ".join(props[:3])
        
        elif category == "confirmation" and primary_data:
            # Yes/No question like "Is a Wyvern a type of Dragon?"
            # Extract subject and relationship from query
            subject, relationship = self._extract_confirmation_parts(query)
            if subject:
                available_slots["subject"] = subject
                
                # Use learned definition to form the relationship/elaboration
                if 'definition' in primary_data:
                    definition_text = primary_data['definition']
                    sentences = definition_text.split('.')
                    
                    # Check if the definition supports the relationship
                    if relationship:
                        available_slots["relationship"] = relationship
                    elif sentences:
                        available_slots["relationship"] = sentences[0].strip()[:80]
                    
                    if len(sentences) > 1:
                        available_slots["elaboration"] = sentences[1].strip()
        
        elif category == "teaching":
            # User is teaching us something like "A Wyvern is a type of Dragon"
            subject, relationship = self._extract_teaching_parts(query)
            if subject and relationship:
                available_slots["subject"] = subject
                available_slots["relationship"] = relationship
                
                # If we have related concepts, add that too
                if len(learned_concepts) > 1:
                    available_slots["related_concept"] = learned_concepts[1]
        
        elif category == "capability_request" and primary_data:
            # User is asking AI to do something - "Can you name some video games?"
            # Response should list/demonstrate the capability
            concept_term = self._extract_concept_from_query(query)
            if concept_term:
                available_slots["concept"] = concept_term
                
                if 'definition' in primary_data:
                    available_slots["examples"] = primary_data['definition']
                
            # Fall back to elaboration category for template
            category = "elaboration"
        
        elif category == "opinion" and primary_data:
            # Opinion question about a topic - "Do you like X?"
            concept_term = self._extract_concept_from_query(query)
            if concept_term:
                available_slots["subject"] = concept_term
                
                if 'definition' in primary_data:
                    # Use definition to inform the opinion
                    available_slots["elaboration"] = primary_data['definition'].split('.')[0]
        
        # If we have slots, try to match and fill template
        if available_slots:
            template = self.pragmatic_templates.match_best_template(category, available_slots)
            if template:
                response_text = self.pragmatic_templates.fill_template(template, available_slots)
                
                # Clean up grammar issues
                response_text = self._clean_composed_response(response_text)
                
                return ComposedResponse(
                    text=response_text,
                    fragment_ids=[template.template_id] + learned_concepts,
                    composition_weights=[1.0] + [0.5] * len(learned_concepts),
                    coherence_score=0.85,
                    confidence=0.85,
                    is_fallback=False,
                    is_low_confidence=False
                )
        
        return None
    
    def _extract_unknown_terms(self, query: str) -> List[str]:
        """
        Extract potentially unknown terms from a query.
        
        Heuristics:
        - Capitalized words (proper nouns)
        - Consecutive capitalized words (titles like "The Elder Scrolls")
        - Compound terms (Capitalized + number, e.g., "Atari 2600")
        - Technical-sounding words (>8 chars, uncommon patterns)
        - Words in quotes
        - Compound technical terms
        
        Args:
            query: User query
            
        Returns:
            List of potentially unknown terms
        """
        import re
        
        terms = []
        
        # Extract quoted terms
        quoted = re.findall(r'"([^"]+)"', query)
        terms.extend(quoted)
        
        quoted = re.findall(r"'([^']+)'", query)
        terms.extend(quoted)
        
        # Extract CONSECUTIVE capitalized words as a single phrase (proper noun phrases / titles)
        # This handles: "The Elder Scrolls", "World of Warcraft", "Game of Thrones"
        title_pattern = r'\b((?:[A-Z][a-zA-Z]*\s+){1,5}[A-Z][a-zA-Z]*)\b'
        title_matches = re.findall(title_pattern, query)
        for title in title_matches:
            title_clean = title.strip().rstrip('?.,!')
            # Only add if it's more than one word (single words handled below)
            if title_clean and ' ' in title_clean and title_clean not in terms:
                terms.append(title_clean)
        
        # Extract compound terms: Word with capital letter followed by numbers
        # Handles: "Atari 2600", "iPhone 15", "PlayStation 5", "Windows 11"
        compound_patterns = [
            r'\b([A-Z][a-zA-Z]*\s+\d+)\b',         # Standard: Atari 2600, PlayStation 5
            r'\b([a-z]+[A-Z][a-zA-Z]*\s+\d+)\b',   # CamelCase: iPhone 15
        ]
        for pattern in compound_patterns:
            matches = re.findall(pattern, query)
            for compound in matches:
                compound_clean = compound.strip().rstrip('?.,!')
                if compound_clean and compound_clean not in terms:
                    terms.append(compound_clean)
        
        # Build set of words already part of multi-word terms (to avoid duplicates)
        compound_words = set()
        for compound in terms:
            compound_words.update(compound.split())
        
        # Extract single capitalized words (but not first word if it's a question)
        # Skip words that are already part of a compound term
        words = query.split()
        
        for i, word in enumerate(words):
            word_clean = re.sub(r'[^\w]', '', word)
            
            # Skip first word if it's a question word or common sentence starter
            if i == 0 and word_clean.lower() in ['what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'do', 'does', 'can', 'could', 'tell', 'explain', 'describe', 'show', 'give', 'have', 'has', 'would', 'should', 'may', 'might']:
                continue
            
            # Skip if already part of a compound term
            if word_clean in compound_words:
                continue
            
            # Check if capitalized (potential proper noun or technical term)
            if word_clean and word_clean[0].isupper() and len(word_clean) > 2:
                terms.append(word_clean)
        
        # Extract technical-looking words (long, uncommon patterns)
        compound_words_lower = {w.lower() for w in compound_words}  # Lowercase set for comparison
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word).lower()
            
            # Skip if part of compound term
            if word_clean in compound_words_lower:
                continue
            
            # Long words with specific patterns (technical terms)
            if len(word_clean) > 8 and any(pattern in word_clean for pattern in ['tion', 'ment', 'ology', 'ism', 'ics', 'ness']):
                if word_clean not in terms:
                    terms.append(word_clean)
        
        # Remove common words and PRONOUNS (pronouns should be resolved from context, not Wikipedia)
        common_words = {
            'something', 'anything', 'everything', 'nothing',
            'someone', 'anyone', 'everyone',
            'question', 'answer', 'information', 'knowledge',
            # Pronouns - these refer to context, not standalone concepts
            'it', 'its', 'they', 'them', 'their', 'theirs',
            'he', 'him', 'his', 'she', 'her', 'hers',
            'this', 'that', 'these', 'those',
            'who', 'whom', 'which', 'what', 'whose'
        }
        
        terms = [t for t in terms if t.lower() not in common_words]
        
        # Prioritize multi-word terms (titles) over single words
        # Sort by word count (descending), then alphabetically
        terms.sort(key=lambda t: (-len(t.split()), t.lower()))
        
        # Limit to top 3 most likely unknown terms
        return terms[:3]
    
    def _get_conversation_context(self, max_turns: int = 3) -> str:
        """
        Extract recent conversation context for disambiguation.
        
        Args:
            max_turns: Maximum number of recent turns to include
            
        Returns:
            String summarizing recent conversation topics
        """
        if not self.conversation_history:
            return ""
        
        recent_turns = self.conversation_history.get_recent_turns(n=max_turns)
        if not recent_turns:
            return ""
        
        # Build context string from recent user messages
        context_parts = []
        for turn in recent_turns:
            # ConversationTurn is a dataclass, use attribute access
            user_msg = turn.user_input.strip() if hasattr(turn, 'user_input') else ""
            if user_msg:
                context_parts.append(user_msg)
        
        return " ".join(context_parts)
    
    def _fallback_response_low_confidence(
        self, 
        user_input: str,
        best_pattern: ResponsePattern,
        best_score: float
    ) -> ComposedResponse:
        """
        Smart fallback for low confidence matches.
        
        Strategy:
        1. Try to fill knowledge gaps and improve match
        2. If gap-filling helps, use enhanced match
        3. Otherwise, try direct external knowledge lookup
        4. Fall back to graceful acknowledgment if nothing helps
        
        Args:
            user_input: User's query
            best_pattern: Best pattern found (but with low relevance)
            best_score: Relevance score (below threshold)
        
        Returns:
            Enhanced response if gaps filled, external knowledge if found, or graceful fallback
        """
        # STEP 1: Try to fill gaps and improve the match
        aug_for_low_conf = self._ensure_knowledge_augmenter()
        if aug_for_low_conf:
            filled_response = self._fill_gaps_and_retry(user_input)
            if filled_response:
                print(f"  ✨ Low confidence improved by gap-filling!")
                return filled_response
        
        # Try direct external knowledge lookup as fallback
        if aug_for_low_conf:
            # Get conversation context for disambiguation
            conv_context = self._get_conversation_context(max_turns=3)
            external_result = aug_for_low_conf.lookup(user_input, conversation_history=conv_context, min_confidence=0.6)
            
            if external_result:
                response_text, confidence, source = external_result
                
                # Ingest the snippet into vocab/concepts/patterns before replying
                self._ingest_external_knowledge(user_input, response_text, confidence, source)

                # Clean up grammar issues (plural agreement, etc.) and keep concise
                response_text = self._clean_composed_response(self._first_sentence(response_text))
                
                print(f"  💡 Low confidence resolved by {source} (confidence: {confidence:.2f})")
                
                # Return external knowledge as response
                return ComposedResponse(
                    text=response_text,
                    fragment_ids=[f"external_{source}"],
                    composition_weights=[confidence],
                    coherence_score=confidence,
                    primary_pattern=None,  # No pattern yet - will be learned
                    confidence=confidence,
                    is_fallback=True,  # Triggered by fallback
                    is_low_confidence=False  # But knowledge was found
                )
        
        # Try reasoning stage before graceful fallback
        if self.reasoning_stage and user_input:
            try:
                conv_context = self._get_conversation_context(max_turns=3)
                reasoning_result = self.reasoning_stage.deliberate(
                    query=user_input,
                    context=conv_context
                )

                if reasoning_result and reasoning_result.inferences:
                    best_inference = max(
                        reasoning_result.inferences,
                        key=lambda inf: inf.confidence
                    )
                    inferred_text = self._clean_composed_response(
                        f"Reasoning suggests {best_inference.conclusion}. If that's not right, you can teach me and upvote with '/+'."
                    )

                    return ComposedResponse(
                        text=inferred_text,
                        fragment_ids=["reasoning_low_conf"],
                        composition_weights=[best_inference.confidence],
                        coherence_score=best_inference.confidence,
                        primary_pattern=None,
                        confidence=best_inference.confidence,
                        is_fallback=True,
                        is_low_confidence=True
                    )
            except Exception as exc:
                print(f"  ⚠️  Reasoning stage failed in low-confidence fallback: {exc}")

        # Standard graceful fallback if no external knowledge or reasoning help found
        # Analyze query to provide contextual fallback
        user_lower = user_input.lower()
        
        # Check for question markers
        is_question = any(q in user_lower for q in ['what', 'how', 'why', 'when', 'where', 'who', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does', '?'])
        
        # Check for technical terms
        is_technical = any(tech in user_lower for tech in ['neural', 'network', 'algorithm', 'learning', 'model', 'training', 'data', 'transformer', 'cnn', 'rnn', 'gan', 'attention'])
        
        # Generate contextual fallback
        if is_question and is_technical:
            fallback_text = "I don't have specific information about that topic yet. If you know the answer, teach me by typing it and then use '/+' to upvote!"
        elif is_question:
            fallback_text = "I'm not sure how to answer that. You can teach me by providing the answer and upvoting with '/+'."
        elif is_technical:
            fallback_text = "That's an interesting topic, but I haven't learned about it yet. You can teach me by explaining it and using '/+' to confirm!"
        else:
            fallback_text = "I'm not quite sure how to respond. You can teach me the right response by typing it and upvoting with '/+'."
        
        print(f"     ⚠️  Low confidence ({best_score:.3f}) - using graceful fallback")
        
        return ComposedResponse(
            text=fallback_text,
            fragment_ids=["low_confidence_fallback"],
            composition_weights=[1.0],
            coherence_score=0.3,
            primary_pattern=None,
            confidence=best_score,
            is_fallback=True,
            is_low_confidence=True
        )


