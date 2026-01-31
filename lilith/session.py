"""
Unified session management for Lilith conversational AI.

This module provides a shared session abstraction that can be used by
any text-based interface (CLI, Discord, web, etc.). It consolidates
all common logic for message processing, learning, feedback, and state
management.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import re
from collections import deque

import torch

from lilith.personality import (
    PersonalityProfile,
    MoodState,
    apply_style,
    maybe_add_followup,
    update_mood_state,
    mood_confidence_scale,
    mood_plasticity_scale,
)
from lilith.user_preferences import UserPreferenceLearner


@dataclass
class SessionConfig:
    """Configuration for a Lilith session."""
    
    # Core settings
    data_path: str = "data"
    enable_knowledge_augmentation: bool = True
    enable_modal_routing: bool = True
    use_grammar: bool = True
    
    # Compositional response settings (Layer 4 restructured)
    enable_compositional: bool = True  # Enable concept store for compositional responses
    enable_relational_concepts: bool = False  # Enable SQL relational concept lookup
    concept_property_max_len: int = 200
    concept_property_require_term: bool = True
    enable_pragmatic_templates: bool = True  # Enable pragmatic template-based composition
    composition_mode: str = "pragmatic"  # "pattern", "concept", "parallel", or "pragmatic"
    
    # Learning settings
    learning_enabled: bool = True
    enable_auto_learning: bool = True
    auto_train_threshold: int = 10
    auto_train_steps: int = 3
    
    # Feedback settings
    enable_feedback_detection: bool = True
    feedback_min_confidence: float = 0.4
    feedback_apply_threshold: float = 0.5
    
    # Plasticity settings
    plasticity_enabled: bool = True
    syntax_plasticity_interval: int = 5
    pmflow_plasticity_interval: int = 10
    contrastive_interval: int = 5
    
    # Declarative learning
    enable_declarative_learning: bool = True

    # Personality / mood (optional, neutral by default)
    enable_personality: bool = False
    enable_mood: bool = False

    # Preference learning (interests/avoid topics)
    enable_preferences: bool = False
    
    # World model (spatial/temporal/causal grounding)
    enable_world_model: bool = True

    # Reasoning stage (deliberative thinking). Can be disabled to prefer direct pattern/template composition.
    enable_reasoning: bool = True

    # Debugging / observability
    enable_composition_trace: bool = False

    # Capability integration: optionally write deterministic capability outputs
    # back into the pattern store so they become retrievable/learnable.
    enable_capability_writeback: bool = False
    capability_writeback_min_confidence: float = 0.7
    capability_writeback_intent: str = "language_capability"

    # Optional DB-backed memory leaf event log (modality-agnostic substrate)
    enable_memory_leaf_event_log: bool = False
    memory_leaf_db_name: str = "memory_leaf.db"
    memory_leaf_store_user_turns: bool = True
    memory_leaf_store_assistant_turns: bool = True

    # MCP tool stream (remote MCP endpoints as an additional I/O stream)
    enable_mcp_tool_stream: bool = False
    mcp_endpoints: Optional[list] = None  # list of {"name": str, "url": str}
    mcp_max_tools_per_turn: int = 2
    mcp_timeout_seconds: float = 2.5
    mcp_cache_ttl_seconds: float = 60.0

    # MCP stage policy (gating / routing)
    # - off: never sense/call
    # - gated: call only when request likely benefits from tools
    # - always_sense: rank tools but do not call
    # - always_call: call whenever a viable tool matches
    mcp_stage_mode: str = "gated"
    mcp_min_tool_score: float = 0.34
    mcp_direct_answer_enabled: bool = True
    mcp_direct_answer_max_chars: int = 600

    # Ranker / decay controls
    enable_ranking: bool = True
    decay_patterns: bool = True
    decay_concepts: bool = True
    decay_pattern_max_age_days: float = 120.0
    decay_pattern_min_success: float = 0.35
    decay_concept_max_age_days: float = 180.0
    decay_concept_min_confidence: float = 0.35


@dataclass
class SessionResponse:
    """Response from processing a message."""
    
    text: str
    pattern_id: Optional[str] = None
    confidence: float = 0.0
    is_fallback: bool = False
    is_low_confidence: bool = False
    source: str = "internal"  # internal, external_wikipedia, etc.
    learned_fact: Optional[str] = None  # If declarative learning occurred
    personality: Optional[PersonalityProfile] = None  # Optional personality metadata
    mood: Optional[MoodState] = None  # Optional mood metadata for UI/UX
    trace: Optional[list] = None  # Optional ResponseComposer decision trace


class LilithSession:
    """
    Unified session manager for Lilith conversational AI.
    
    Handles all common logic between different interfaces:
    - Message processing and response generation
    - Feedback tracking and application
    - Declarative statement learning
    - Auto semantic learning
    - Neuroplasticity updates
    - Pattern storage and retrieval
    
    Usage:
        session = LilithSession(user_id="user123", config=SessionConfig())
        response = session.process_message("What is Python?")
        print(response.text)
        session.upvote(response.pattern_id)
    """
    
    def __init__(self, 
                 user_id: str,
                 context_id: Optional[str] = None,
                 config: Optional[SessionConfig] = None,
                 store=None,  # Optional pre-configured store
                 display_name: str = "User"):
        """
        Initialize a Lilith session.
        
        Args:
            user_id: Unique user identifier
            context_id: Optional context (e.g., guild_id for multi-tenant)
            config: Session configuration
            store: Optional pre-configured fragment store
            display_name: User's display name
        """
        from lilith.embedding import PMFlowEmbeddingEncoder
        from lilith.response_composer import ResponseComposer
        from lilith.conversation_state import ConversationState
        from lilith.conversation_history import ConversationHistory
        
        self.user_id = user_id
        self.context_id = context_id or "default"
        self.cache_key = f"{user_id}:{self.context_id}"
        self.display_name = display_name
        self.config = config or SessionConfig()

        # Optional MCP tool stream (remote endpoints)
        self.mcp_tool_stream = None
        self.mcp_stage = None
        self._last_mcp_artifact = None
        if self.config.enable_mcp_tool_stream and self.config.mcp_endpoints:
            try:
                from lilith.mcp_tool_stream import MCPRemoteEndpoint, MCPToolStream

                endpoints = []
                for item in list(self.config.mcp_endpoints or []):
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or "").strip()
                    url = str(item.get("url") or "").strip()
                    if name and url:
                        endpoints.append(MCPRemoteEndpoint(name=name, url=url))

                if endpoints:
                    self.mcp_tool_stream = MCPToolStream(
                        endpoints,
                        timeout_seconds=float(self.config.mcp_timeout_seconds),
                        cache_ttl_seconds=float(self.config.mcp_cache_ttl_seconds),
                    )

                    try:
                        from lilith.mcp_stage import MCPStage

                        self.mcp_stage = MCPStage(
                            self.mcp_tool_stream,
                            mode=str(self.config.mcp_stage_mode),
                            topk=int(self.config.mcp_max_tools_per_turn),
                            min_score=float(self.config.mcp_min_tool_score),
                            direct_answer_enabled=bool(self.config.mcp_direct_answer_enabled),
                            direct_answer_max_chars=int(self.config.mcp_direct_answer_max_chars),
                        )
                    except Exception:
                        self.mcp_stage = None
            except Exception:
                self.mcp_tool_stream = None
                self.mcp_stage = None

        # Preference learner (optional)
        self.preference_learner = None
        self.user_preferences = None
        if self.config.enable_preferences:
            self.preference_learner = UserPreferenceLearner(base_path=self.config.data_path)
            self.user_preferences = self.preference_learner.store.load(self.user_id)
            if self.user_preferences.display_name:
                self.display_name = self.user_preferences.display_name
        
        # Initialize encoder
        self.encoder = PMFlowEmbeddingEncoder()

        # Per-user persistence root (ignored by git)
        user_root = Path(self.config.data_path) / "users" / self.user_id
        user_root.mkdir(parents=True, exist_ok=True)

        # Optional modality-agnostic memory leaf (DB-backed, embedding-addressable)
        self.memory_leaf_adapter = None
        self._memory_leaf_store = None
        self._memory_leaf_scenario = None
        if self.config.enable_memory_leaf_event_log:
            try:
                from lilith.memory import MemoryEvent, MemoryLeaf, MemoryLeafAdapter
                from lilith.storage.sqlite_memory_store import SQLiteMemoryStore

                memory_db_path = user_root / self.config.memory_leaf_db_name

                self._memory_leaf_store = SQLiteMemoryStore(memory_db_path)
                self._memory_leaf_scenario = self.context_id
                leaf = MemoryLeaf(
                    store=self._memory_leaf_store,
                    encoder=self.encoder,
                    scenario=self._memory_leaf_scenario,
                )
                self.memory_leaf_adapter = MemoryLeafAdapter(leaf)
                self._MemoryEvent = MemoryEvent
            except Exception:
                # Memory leaf is optional and should never block sessions.
                self.memory_leaf_adapter = None
                self._memory_leaf_store = None
                self._memory_leaf_scenario = None
        
        # Use provided store or create default
        self.store = store
        if self.store is None:
            from lilith.multi_tenant_store import MultiTenantFragmentStore
            from lilith.user_auth import UserIdentity, AuthMode
            
            identity = UserIdentity(
                user_id=user_id,
                auth_mode=AuthMode.TRUSTED,
                display_name=display_name
            )
            
            self.store = MultiTenantFragmentStore(
                encoder=self.encoder,
                user_identity=identity,
                base_data_path=self.config.data_path,
                enable_relational_concepts=self.config.enable_relational_concepts,
                concept_property_max_len=self.config.concept_property_max_len,
                concept_property_require_term=self.config.concept_property_require_term,
            )
        
        # Create conversation state (working memory - active topics with decay)
        self.state = ConversationState(self.encoder)
        
        # Create conversation history (short-term memory - recent turns, sliding window)
        self.conversation_history = ConversationHistory(max_turns=10)

        # Personality and mood (neutral/no-op unless enabled)
        self.personality_profile = PersonalityProfile.neutral() if self.config.enable_personality else None
        if self.personality_profile:
            # Make persona slightly warm and proactive by default
            self.personality_profile.warmth = 0.65
            self.personality_profile.humor = 0.1
            self.personality_profile.proactivity = 0.45
        self.mood_state = MoodState.neutral() if self.config.enable_mood else None

        # Seed personality with stored preferences when available
        if self.config.enable_preferences and self.personality_profile and self.user_preferences:
            self.personality_profile.interests = list(self.user_preferences.interests)
            self.personality_profile.aversions = list(self.user_preferences.aversions)
        
        # Initialize pragmatic templates (Layer 4: linguistic patterns)
        pragmatic_templates = None
        if self.config.enable_pragmatic_templates:
            try:
                from lilith.pragmatic_templates import PragmaticTemplateStore
                templates_path = user_root / "pragmatic_templates.json"
                
                if templates_path.exists():
                    # Load existing templates
                    pragmatic_templates = PragmaticTemplateStore.load(str(templates_path))
                else:
                    # Create with default templates
                    pragmatic_templates = PragmaticTemplateStore()
                    # Save for next time
                    pragmatic_templates.save(str(templates_path))
            except ImportError:
                pass
        
        # Initialize concept store (Layer 4: semantic knowledge)
        # Prefer the concept store from the multi-tenant store (which handles user paths correctly)
        concept_store = None
        if self.config.enable_compositional:
            # First, try to use concept store from the fragment store (multi-tenant aware)
            if hasattr(self.store, 'concept_store') and self.store.concept_store is not None:
                concept_store = self.store.concept_store
            else:
                # Fallback: create standalone concept store (for non-multi-tenant use)
                try:
                    from lilith.production_concept_store import ProductionConceptStore
                    concept_db_path = Path(self.config.data_path) / "concept_store.db"
                    concept_store = ProductionConceptStore(
                        semantic_encoder=self.encoder,
                        db_path=str(concept_db_path),
                        enable_relational=self.config.enable_relational_concepts,
                        property_max_len=self.config.concept_property_max_len,
                        property_require_term=self.config.concept_property_require_term,
                    )
                except ImportError:
                    pass
        
        # Create composer with conversation history, pragmatic templates, and concept store
        self.composer = ResponseComposer(
            self.store,
            self.state,
            self.conversation_history,
            semantic_encoder=self.encoder,
            enable_knowledge_augmentation=self.config.enable_knowledge_augmentation,
            enable_modal_routing=self.config.enable_modal_routing,
            use_grammar=self.config.use_grammar,
            concept_store=concept_store,
            pragmatic_templates=pragmatic_templates,
            enable_pragmatic_templates=self.config.enable_pragmatic_templates,
            composition_mode=self.config.composition_mode,
            enable_relational_concepts=self.config.enable_relational_concepts,
            enable_world_model=self.config.enable_world_model,
            enable_reasoning=self.config.enable_reasoning,
            enable_trace=self.config.enable_composition_trace,
            data_path=self.config.data_path,
            syntax_storage_path=(user_root / "syntax_patterns.json"),
        )
        
        # Wire personality bias into composer for limbic-style BNN modulation (after composer exists)
        if self.config.enable_personality:
            self.composer._personality_bias_fn = self._compute_personality_bias
        
        # Load contrastive weights if available
        contrastive_path = Path(self.config.data_path) / "contrastive_learner"
        if contrastive_path.with_suffix('.json').exists():
            self.composer.load_contrastive_weights(str(contrastive_path))
        
        # Initialize auto semantic learner
        self.auto_learner = None
        if self.config.enable_auto_learning:
            try:
                from lilith.auto_semantic_learner import AutoSemanticLearner
                if self.composer.contrastive_learner:
                    self.auto_learner = AutoSemanticLearner(
                        contrastive_learner=self.composer.contrastive_learner,
                        auto_train_threshold=self.config.auto_train_threshold,
                        auto_train_steps=self.config.auto_train_steps
                    )
                    # Load previous state
                    state_path = Path(self.config.data_path) / "auto_learner_state.json"
                    if state_path.exists():
                        self.auto_learner.load_state(state_path)
            except ImportError:
                pass
        
        # Initialize topic extractor for BioNN-based query cleaning
        self.topic_extractor = None
        try:
            from lilith.topic_extractor import TopicExtractor
            topics_path = user_root / "topics.json"
            self.topic_extractor = TopicExtractor(
                encoder=self.encoder,
                storage_path=topics_path
            )
            # Wire up to knowledge augmenter
            if self.composer.knowledge_augmenter:
                self.composer.knowledge_augmenter.set_topic_extractor(self.topic_extractor)
        except ImportError:
            pass
        
        # Initialize feedback tracker
        self.feedback_tracker = None
        if self.config.enable_feedback_detection:
            try:
                from lilith.feedback_detector import FeedbackDetector, FeedbackTracker
                self.feedback_tracker = FeedbackTracker(
                    detector=FeedbackDetector(
                        min_confidence=self.config.feedback_min_confidence,
                        apply_threshold=self.config.feedback_apply_threshold
                    )
                )
            except ImportError:
                pass
        
        # Tracking state
        self.interaction_count = 0
        self.last_pattern_id = None
        self.last_user_input = None
        self.last_response_text = None

        # Eligibility trace: retain a small buffer of recent decision contexts so
        # delayed feedback can reinforce the correct prior associations.
        self._eligibility_buffer = deque(maxlen=25)
    
    def process_message(self, content: str, passive_mode: bool = False) -> SessionResponse:
        """
        Process a user message and generate a response.
        
        Args:
            content: User message text
            passive_mode: If True, learn but don't generate response
            
        Returns:
            SessionResponse with text and metadata
        """
        learned_fact = None

        # Optional: log incoming observation to the DB-backed memory leaf.
        self._emit_memory_turn(role="user", text=content)
        
        # Detect and learn from declarative statements
        if self.config.learning_enabled and self.config.enable_declarative_learning:
            learned_fact = self._detect_and_learn_declarative(content)
        
        # Update conversation state for topic tracking and pronoun resolution
        enriched_context = self._update_conversation_context(content)

        # Optional: MCP tool stage (structured artifact, not just string injection).
        mcp_artifact_dict = None
        self._last_mcp_artifact = None
        if self.mcp_stage is not None:
            try:
                art = self.mcp_stage.process(content)
                self._last_mcp_artifact = art
                mcp_artifact_dict = art.to_dict()

                # Emit tool observations into the memory leaf (if enabled).
                # Keep the text small and provenance-carrying.
                if getattr(art, "summary", ""):
                    self._emit_memory_turn(
                        role="assistant",
                        text=art.summary,
                        extra={
                            "source": "mcp_stage",
                            "decision": getattr(art, "decision", None),
                            "reason": getattr(art, "reason", None),
                            "confidence": float(getattr(art, "confidence", 0.0) or 0.0),
                            "tool_ok_count": len([r for r in (getattr(art, "results", []) or []) if getattr(r, "ok", False)]),
                        },
                    )
            except Exception:
                mcp_artifact_dict = None

        # Learn preferences (name, interests, aversions) from the incoming text
        learned_preferences = self._process_preferences(content)
        
        # In passive mode, just learn and return
        if passive_mode:
            # Still update auto-learner with observed message
            if self.auto_learner and self.config.learning_enabled:
                self.auto_learner.process_conversation(content, "")
            
            # Apply plasticity periodically
            if self.config.plasticity_enabled and self.config.learning_enabled:
                self.interaction_count += 1
                if self.interaction_count % 10 == 0:
                    self._apply_plasticity()

            if self.config.enable_mood:
                self.mood_state = update_mood_state(self.mood_state, content)
            
            return SessionResponse(
                text="",
                learned_fact=learned_fact,
                personality=self.personality_profile if self.config.enable_personality else None,
                mood=self.mood_state if self.config.enable_mood else None,
                # Preferences learning in passive mode is implicit
            )
        
        # Check for feedback from previous message
        feedback_applied = False
        if self.feedback_tracker and self.config.learning_enabled:
            if self.feedback_tracker.history:
                feedback_result = self.feedback_tracker.check_feedback(content)
                if feedback_result:
                    result, pattern_id = feedback_result
                    if result.should_apply and pattern_id:
                        if result.is_positive:
                            self.upvote(pattern_id, strength=result.strength)
                        elif result.is_negative:
                            self.downvote(pattern_id, strength=result.strength)
                        feedback_applied = True
        
        # Check if this is ONLY feedback (emoji or short feedback phrase)
        # If so, don't generate a response - just acknowledge the feedback
        if feedback_applied and self._is_pure_feedback(content):
            if self.config.enable_mood:
                self.mood_state = update_mood_state(self.mood_state, content)
            # Return empty response - feedback was applied, no need to respond
            return SessionResponse(
                text="",
                learned_fact=learned_fact,
                personality=self.personality_profile if self.config.enable_personality else None,
                mood=self.mood_state if self.config.enable_mood else None,
            )
        
        # Generate response using enriched context (includes topic history for pronoun resolution)
        response = self.composer.compose_response(context=enriched_context, user_input=content, tool_artifact=mcp_artifact_dict)

        # Record eligibility context for delayed credit assignment.
        self._record_eligibility(content, response)

        # Optional: write deterministic capability outputs back into the pattern store.
        if self.config.learning_enabled:
            self._maybe_writeback_capability(content, response)

        # Optional: log assistant output to the DB-backed memory leaf.
        self._emit_memory_turn(
            role="assistant",
            text=getattr(response, "text", ""),
            extra={
                "pattern_id": (response.fragment_ids[0] if getattr(response, "fragment_ids", None) else None),
                "confidence": float(getattr(response, "confidence", 0.0) or 0.0),
                "is_fallback": bool(getattr(response, "is_fallback", False)),
            },
        )

        # Gentle bias: adjust confidence by learned interests/aversions on primary intent
        if getattr(response, "primary_pattern", None):
            intent = getattr(response.primary_pattern, "intent", None)
            if intent:
                boost = 1.0
                interests, aversions = self._preference_terms()
                if intent in interests:
                    boost *= 1.05
                if intent in aversions:
                    boost *= 0.85
                response.confidence = max(0.0, min(1.0, response.confidence * boost))

        # Apply optional personality style (minimal - most influence is at BNN level)
        if self.config.enable_personality and self.personality_profile:
            # Only apply subtle post-processing (main influence is embedding bias)
            response.text = apply_style(response.text, self.personality_profile)
            if self.personality_profile.proactivity > 0.7:  # Only for high proactivity
                response.text = maybe_add_followup(response.text, self.personality_profile, getattr(response, 'confidence', 0.0))
        
        # Record turn in conversation history for continuity tracking
        if self.conversation_history:
            # Get current working memory state for this turn
            state_snapshot = self.state.snapshot()
            working_memory_state = {
                'activation_energy': state_snapshot.activation_energy,
                'novelty': state_snapshot.novelty,
                'topic_count': len(state_snapshot.topics),
                'dominant_topic': state_snapshot.dominant.summary if state_snapshot.dominant else None
            }
            
            self.conversation_history.add_turn(
                user_input=content,
                bot_response=response.text,
                user_embedding=None,  # Could add embeddings if needed
                response_embedding=None,
                working_memory_state=working_memory_state
            )
            
            # Update success score based on response confidence
            success_score = response.confidence if hasattr(response, 'confidence') else 0.5
            if getattr(response, 'is_fallback', False):
                success_score = 0.3  # Fallback responses are lower success
            
            self.conversation_history.update_last_success(success_score)
        
        # Track for feedback
        if self.feedback_tracker and self.config.learning_enabled:
            pattern_id = response.fragment_ids[0] if response.fragment_ids else None
            self.feedback_tracker.record_interaction(content, response.text, pattern_id)
        
        # Auto-learn semantic relationships
        if self.auto_learner and self.config.learning_enabled:
            self.auto_learner.process_conversation(content, response.text)
        
        # Track for potential upvote/downvote
        self.last_pattern_id = response.fragment_ids[0] if response.fragment_ids else None
        self.last_user_input = content
        self.last_response_text = response.text
        
        # Apply neuroplasticity
        if self.config.plasticity_enabled and self.config.learning_enabled:
            self.interaction_count += 1
            self._apply_plasticity()

        if self.config.enable_mood:
            # Prefer BNN-derived sentiment when it produces a strong signal.
            # Otherwise fall back to lightweight text heuristics so mood updates
            # remain intuitive even with deterministic/weak embeddings.
            sentiment_score = None
            try:
                from lilith.personality import compute_sentiment_from_embedding
                content_emb = self.encoder.encode(content)
                sentiment_score = float(compute_sentiment_from_embedding(content_emb, self.encoder))
            except Exception:
                sentiment_score = None

            if sentiment_score is None or (-0.5 < sentiment_score < 0.5):
                self.mood_state = update_mood_state(self.mood_state, content)
            else:
                self.mood_state = update_mood_state(self.mood_state, sentiment_score)
            
            # Limbic-style modulation: adjust confidence with mood
            response.confidence = max(0.0, min(1.0, response.confidence * mood_confidence_scale(self.mood_state)))
        
        return SessionResponse(
            text=response.text,
            pattern_id=self.last_pattern_id,
            confidence=response.confidence if hasattr(response, 'confidence') else 0.0,
            is_fallback=getattr(response, 'is_fallback', False),
            is_low_confidence=getattr(response, 'is_low_confidence', False),
            source=self._determine_source(self.last_pattern_id),
            learned_fact=learned_fact,
            personality=self.personality_profile if self.config.enable_personality else None,
            mood=self.mood_state if self.config.enable_mood else None,
            trace=getattr(response, 'trace', None),
        )

    def _record_eligibility(self, user_input: str, response: Any) -> None:
        """Record a compact trace of what was used to answer.

        This is used to map later feedback back to the correct decision context.
        """

        try:
            # Data-driven label of what won routing for this turn.
            # This is intentionally an opaque string ("pattern", "pragmatic", "math", etc.)
            # so adding new routes does not require updating session logic.
            channel = getattr(self.composer, "last_approach", None)
            fragment_ids = list(getattr(response, "fragment_ids", []) or [])
            weights = list(getattr(response, "composition_weights", []) or [])
            self._eligibility_buffer.append(
                {
                    "channel": channel,
                    "user_input": user_input,
                    "response_text": getattr(response, "text", ""),
                    "fragment_ids": fragment_ids,
                    "weights": weights,
                    "mcp": (self._last_mcp_artifact.to_dict() if getattr(self, "_last_mcp_artifact", None) is not None else None),
                }
            )
        except Exception:
            return

    def _find_eligibility_record(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Find the most recent eligibility record containing the given fragment id."""

        if not pattern_id:
            return None

        for record in reversed(self._eligibility_buffer):
            try:
                if pattern_id in (record.get("fragment_ids") or []):
                    return record
            except Exception:
                continue

        # Fallback: if we can't locate by fragment id (e.g., some callers pass
        # a computed/non-stored id), use the most recent eligibility record.
        # This keeps delayed feedback usable without encoding modality-specific rules.
        try:
            return self._eligibility_buffer[-1] if self._eligibility_buffer else None
        except Exception:
            return None
        return None

    def _is_language_capability_fragment(self, fragment_id: Optional[str]) -> bool:
        if not fragment_id:
            return False
        return fragment_id in {"rewrite", "generated_sentence_about", "generated_sentence_words"} or fragment_id.startswith(
            "capability_"
        )

    def _maybe_writeback_capability(self, user_input: str, response: Any) -> None:
        """Write back deterministic language capability responses into the pattern store.

        This bridges the bootstrap capability path into the same memory substrate used
        by retrieval/learning, keeping behavior aligned with the BioNN+DB philosophy.
        """

        if not self.config.enable_capability_writeback:
            return
        if not self.config.learning_enabled:
            return

        fragment_ids = list(getattr(response, "fragment_ids", []) or [])
        if not fragment_ids or not self._is_language_capability_fragment(fragment_ids[0]):
            return

        confidence = float(getattr(response, "confidence", 0.0) or 0.0)
        if confidence < float(self.config.capability_writeback_min_confidence):
            return

        if not hasattr(self.store, "add_pattern"):
            return

        # Seed with confidence, then limbic-gate the initial prior.
        seeded = max(0.0, min(1.0, confidence))
        seeded = self._limbic_reinforcement_scale(
            text=user_input,
            base_strength=seeded,
            channel="writeback",
        )

        try:
            self.store.add_pattern(
                trigger_context=user_input,
                response_text=getattr(response, "text", ""),
                intent=self.config.capability_writeback_intent,
                success_score=seeded,
            )
        except Exception:
            return

    def _emit_memory_turn(self, *, role: str, text: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Best-effort event emission into the modality-agnostic memory leaf."""

        adapter = getattr(self, "memory_leaf_adapter", None)
        if adapter is None:
            return

        if role == "user" and not self.config.memory_leaf_store_user_turns:
            return
        if role == "assistant" and not self.config.memory_leaf_store_assistant_turns:
            return

        try:
            payload: Dict[str, Any] = {
                "role": role,
                "user_id": self.user_id,
                "context_id": self.context_id,
                "cache_key": self.cache_key,
            }
            if extra:
                payload.update(extra)

            event = self._MemoryEvent(modality="text", text=text, payload=payload)
            adapter.observe(event)
        except Exception:
            # Never let memory logging affect chat/session behavior.
            return

    def _process_preferences(self, content: str) -> Dict[str, Any]:
        """Extract and persist preferences from user text, refreshing in-memory state."""

        if not self.config.enable_preferences or not self.preference_learner:
            return {}

        learned = self.preference_learner.process_input(self.user_id, content)
        self.user_preferences = self.preference_learner.store.load(self.user_id)

        # Keep personality profile in sync when enabled
        if self.personality_profile:
            self.personality_profile.interests = list(self.user_preferences.interests)
            self.personality_profile.aversions = list(self.user_preferences.aversions)

        # Refresh display name if newly learned
        if 'name' in learned and self.user_preferences.display_name:
            self.display_name = self.user_preferences.display_name

        return learned

    def _preference_terms(self) -> Tuple[list, list]:
        """Return (interests, aversions) from stored preferences/personality."""

        interests = []
        aversions = []

        if self.config.enable_preferences and self.user_preferences:
            interests.extend(self.user_preferences.interests)
            aversions.extend(self.user_preferences.aversions)

        if self.personality_profile:
            interests.extend(x for x in self.personality_profile.interests if x not in interests)
            aversions.extend(x for x in self.personality_profile.aversions if x not in aversions)

        return interests, aversions
    
    def _compute_personality_bias(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Apply limbic-style bias to query embedding based on interests/aversions.
        
        This modulates BNN retrieval by pulling embeddings toward interests
        and pushing away from aversions - analogous to emotional attention.
        
        Args:
            query_embedding: Original query embedding from encoder
            
        Returns:
            Biased embedding for retrieval
        """
        import torch.nn.functional as F
        
        if not self.config.enable_personality or not self.personality_profile:
            return query_embedding
        
        interests, aversions = self._preference_terms()
        if not interests and not aversions:
            return query_embedding
        
        biased = query_embedding.clone()
        
        # Compute interest embeddings and bias toward them
        for interest in interests[:5]:  # Limit to top 5 to avoid over-biasing
            try:
                interest_emb = self.encoder.encode(interest)
                similarity = F.cosine_similarity(query_embedding, interest_emb, dim=-1)
                
                if similarity > 0.25:  # Related to interest
                    # Pull query toward interest (limbic attention boost)
                    bias_strength = 0.20 * self.personality_profile.proactivity
                    biased = biased + bias_strength * interest_emb
            except Exception:
                continue
        
        # Compute aversion embeddings and bias away from them
        for aversion in aversions[:5]:
            try:
                aversion_emb = self.encoder.encode(aversion)
                similarity = F.cosine_similarity(query_embedding, aversion_emb, dim=-1)
                
                if similarity > 0.25:  # Related to aversion
                    # Push query away from aversion (limbic avoidance)
                    bias_strength = 0.15 * self.personality_profile.proactivity
                    biased = biased - bias_strength * aversion_emb
            except Exception:
                continue
        
        # Re-normalize to keep embedding in valid space
        biased = F.normalize(biased, p=2, dim=-1)
        
        return biased

    def _persona_engagement(self, response: Any, user_input: str) -> Optional[str]:
        """Add a short persona-driven opinion or invitation to chat.

        Activated only when personality is enabled and proactivity is non-zero.
        Uses learned interests/aversions to stay grounded while avoiding
        hallucinated facts when confidence is low.
        """

        profile = self.personality_profile
        if not profile or profile.proactivity <= 0.0:
            return None

        interests, aversions = self._preference_terms()
        user_lower = user_input.lower()

        # Look for overlap between the user's message and known interests/aversions
        matched_interest = next((t for t in interests if t.lower() in user_lower), None)
        matched_aversion = next((t for t in aversions if t.lower() in user_lower), None)

        is_low_conf = getattr(response, 'is_low_confidence', False)
        is_fallback = getattr(response, 'is_fallback', False)

        # For fallbacks/low confidence, ONLY add engagement if there's a matched interest/aversion
        # Don't randomly mention unrelated interests
        if is_fallback or is_low_conf:
            if matched_aversion:
                return f"I usually keep some distance from {matched_aversion}, but I'm listening -- what matters to you about it?"
            if matched_interest:
                return f"I haven't stored much on {matched_interest} yet, but I'm curious. What part should I learn first?"
            # Don't add generic engagement for fallbacks - let the fallback message speak for itself
            return None

        # For successful responses, add engagement only if there's a match
        if matched_aversion:
            return f"{matched_aversion} isn't my favorite area, but I'm listening. What draws you to it?"

        if matched_interest:
            return f"I'm into {matched_interest}. What's your take?"

        # Don't add generic engagement for unrelated topics
        return None
    
    def _update_conversation_context(self, content: str) -> str:
        """
        Update conversation state and build enriched context for pronoun resolution.
        
        Args:
            content: Current user message
            
        Returns:
            Enriched context string that includes recent topics, with pronouns resolved
        """
        # If conversation state is not active, just return content
        if not self.state.is_active():
            return content
        
        # Parse the user input to get a PipelineArtifact
        from lilith.pipeline import SymbolicPipeline
        from lilith.base import Utterance
        import re
        
        try:
            # Create a minimal pipeline for parsing
            pipeline = SymbolicPipeline(encoder=self.encoder)
            utterance = Utterance(text=content)
            artifact = pipeline.process(utterance)
            
            # Update conversation state with the new message
            snapshot = self.state.update(artifact)
            
            # Check for pronouns and resolve them
            pronouns = {
                'they', 'them', 'their', 'theirs',
                'it', 'its',
                'this', 'that', 'these', 'those',
                'he', 'him', 'his',
                'she', 'her', 'hers'
            }
            
            content_lower = content.lower()
            has_pronoun = any(f" {p} " in f" {content_lower} " or content_lower.startswith(f"{p} ") 
                            for p in pronouns)
            
            # Build enriched context from active topics
            if snapshot.topics and has_pronoun:
                # Get the strongest/most recent topics to find likely referent
                topic_summaries = [topic.summary for topic in snapshot.topics[:3]]  # Top 3 topics
                
                if topic_summaries:
                    # Try to find a noun phrase (not pronouns, not action words)
                    referent = None
                    for summary in topic_summaries:
                        # Clean up summary: remove common words that aren't the main topic
                        cleaned = summary.lower()
                        # Remove pronouns and common verbs from the beginning
                        for prefix in ['me ', 'you ', 'they ', 'do ', 'does ', 'did ', 'is ', 'are ', 'was ', 'were ']:
                            if cleaned.startswith(prefix):
                                cleaned = cleaned[len(prefix):].strip()
                        
                        # Skip if it's too short or still contains only pronouns/verbs
                        if len(cleaned) < 3:
                            continue
                        words = cleaned.split()
                        if all(w in pronouns | {'do', 'does', 'did', 'is', 'are', 'was', 'were', 'me', 'you'} 
                               for w in words):
                            continue
                        
                        referent = cleaned
                        break
                    
                    if referent:
                        # Validate referent quality before using it
                        # Skip if referent looks broken (very short, all common words, etc)
                        referent_words = referent.split()
                        
                        # Don't use if it's too short or looks like garbage
                        if len(referent_words) < 1 or len(referent) < 4:
                            referent = None
                        # Don't use if ALL words are in common set (likely broken)
                        elif referent_words and all(w in pronouns | {'do', 'does', 'did', 'is', 'are', 'was', 'were', 'me', 'you', 'i', 'have', 'has', 'had'} for w in referent_words):
                            referent = None
                        # Don't use if it contains too many pronouns (sign of bad extraction)
                        # Changed from > to >= to reject 50% (was allowing exactly 50%)
                        elif sum(1 for w in referent_words if w in pronouns) >= len(referent_words) // 2:
                            referent = None
                    
                    if referent:
                        # Replace pronouns with referent
                        resolved = content
                        for pronoun in pronouns:
                            # Only replace standalone pronouns, not parts of words
                            pattern = r'\b' + pronoun + r'\b'
                            replacement = referent
                            resolved = re.sub(pattern, replacement, resolved, flags=re.IGNORECASE | re.MULTILINE)
                        
                        if resolved.lower() != content.lower():
                            print(f"  🔗 Resolved pronoun: '{content}' → '{resolved}'")
                            return resolved
            
            # If no pronoun resolution, avoid injecting previous facts to reduce echoes
            # Keep raw content to prevent repeating past answers verbatim
            return content
                    
        except Exception as e:
            # If parsing fails, fall back to raw content
            print(f"  ⚠️ Context update failed: {e}")
        
        return content
    
    def _is_pure_feedback(self, content: str) -> bool:
        """
        Check if the message is purely feedback (emoji or short feedback phrase).
        
        Args:
            content: User message
            
        Returns:
            True if this is only feedback with no substantive question
        """
        text = content.strip()
        
        # Check for emoji-only feedback
        feedback_emojis = {'👍', '👎', '❤️', '✅', '🎉', '💯', '🙏', '❌', '😕', '🤔', '🙄', '💩', '🚫'}
        
        # Remove whitespace and check if it's just emojis
        text_no_space = text.replace(' ', '')
        if all(c in feedback_emojis or c in '!?.' for c in text_no_space):
            return True
        
        # Check for short standalone feedback phrases (3 words or less)
        words = text.lower().split()
        if len(words) <= 3:
            feedback_phrases = {
                'thanks', 'thank you', 'thx', 'ty',
                'perfect', 'exactly', 'great', 'excellent',
                'awesome', 'wrong', 'incorrect', 'no',
                'yes', 'right', 'correct', 'got it',
                'makes sense', 'i see', 'ok', 'okay',
                'cool', 'nice', 'good', 'nope'
            }
            text_clean = text.lower().rstrip('!?.').strip()
            if text_clean in feedback_phrases:
                return True
        
        return False
    
    def upvote(self, pattern_id: Optional[str] = None, strength: float = 0.2) -> bool:
        """
        Upvote a pattern to reinforce it.
        
        Args:
            pattern_id: Pattern to upvote (None = last pattern)
            strength: Strength of upvote (0.0-1.0)
            
        Returns:
            True if upvote was applied
        """
        target_id = pattern_id or self.last_pattern_id
        if not target_id:
            return False

        # If feedback arrives later, use the correct historical user_input/response_text
        # (eligibility trace) rather than the most recent turn.
        eligibility = self._find_eligibility_record(target_id)
        reinforcement_text = (eligibility.get("user_input") if eligibility else None) or (self.last_user_input or "")
        reinforcement_response_text = (eligibility.get("response_text") if eligibility else None) or (self.last_response_text or "")

        # Limbic-style modulation: scale reinforcement strength based on mood/personality/preferences.
        effective_strength = self._limbic_reinforcement_scale(
            text=reinforcement_text,
            base_strength=strength,
            channel="upvote",
        )
        
        # Check if this is external knowledge that should be learned.
        # Prefer eligibility-trace-mapped (query, response_text) so delayed feedback
        # reinforces the correct historical turn.
        if target_id.startswith('external_') and reinforcement_text and reinforcement_response_text:
            # Learn from Wikipedia/external source
            if hasattr(self.store, 'learn_from_wikipedia'):
                # Use limbic modulation to bias how strongly we seed new knowledge.
                # (This is not a learning-rate update; it's the initial success prior.)
                seeded_success = max(0.0, min(1.0, 0.8 * (0.9 + 0.2 * min(effective_strength / max(strength, 1e-6), 2.0))))
                new_pattern_id = self.store.learn_from_wikipedia(
                    query=reinforcement_text,
                    response_text=reinforcement_response_text,
                    success_score=seeded_success,
                    intent="learned_knowledge"
                )
                print(f"📚 Learned from external knowledge: {new_pattern_id}")
                return True
        
        # Check if this is world model answer that should be reinforced
        if target_id.startswith('world_model_') and self.last_user_input:
            if hasattr(self.composer, 'world_model') and self.composer.world_model:
                # Re-process the query to find the relevant situation
                try:
                    results = self.composer.world_model.retrieve_similar_situations(
                        reinforcement_text, 
                        topk=1
                    )
                    if results:
                        # Reinforce the top matching situation
                        pattern = results[0].pattern
                        pattern.success_score = min(1.0, pattern.success_score + effective_strength)
                        pattern.usage_count += 1
                        self.composer.world_model._update_pattern(pattern)
                        print(f"🌍 Reinforced world model knowledge")
                        return True
                except Exception as e:
                    print(f"  ⚠️  World model reinforcement failed: {e}")
        
        # Regular upvote
        self._apply_weighted_feedback(
            target_id=target_id,
            positive=True,
            total_strength=effective_strength,
        )
        return True
    
    def downvote(self, pattern_id: Optional[str] = None, strength: float = 0.2) -> bool:
        """
        Downvote a pattern to weaken it.
        
        Args:
            pattern_id: Pattern to downvote (None = last pattern)
            strength: Strength of downvote (0.0-1.0)
            
        Returns:
            True if downvote was applied
        """
        target_id = pattern_id or self.last_pattern_id
        if not target_id:
            return False

        eligibility = self._find_eligibility_record(target_id)
        reinforcement_text = (eligibility.get("user_input") if eligibility else None) or (self.last_user_input or "")

        effective_strength = self._limbic_reinforcement_scale(
            text=reinforcement_text,
            base_strength=strength,
            channel="downvote",
        )
        self._apply_weighted_feedback(
            target_id=target_id,
            positive=False,
            total_strength=effective_strength,
        )
        return True

    def _apply_weighted_feedback(self, *, target_id: str, positive: bool, total_strength: float) -> None:
        """Apply feedback across all fragments that contributed to a response.

        This is an eligibility-trace style credit assignment: when a response was composed
        from multiple fragments, distribute reinforcement proportional to their contribution.
        """

        if total_strength <= 0.0:
            return

        record = self._find_eligibility_record(target_id)
        fragment_ids = list((record or {}).get("fragment_ids") or [])
        weights = list((record or {}).get("weights") or [])

        if not fragment_ids:
            fragment_ids = [target_id]

        if len(weights) != len(fragment_ids):
            weights = [1.0 for _ in fragment_ids]

        cleaned_weights = [max(0.0, float(w)) for w in weights]
        weight_sum = sum(cleaned_weights)
        if weight_sum <= 0.0:
            cleaned_weights = [1.0 for _ in fragment_ids]
            weight_sum = float(len(fragment_ids))

        # Distribute total_strength across fragments.
        for fid, w in zip(fragment_ids, cleaned_weights):
            portion = total_strength * (w / weight_sum)
            if portion <= 0.0:
                continue
            portion = max(0.0, min(1.0, portion))

            # Skip non-pattern fragment IDs when possible (avoids noisy warnings).
            store = None
            if hasattr(self.store, "_get_pattern_store"):
                try:
                    store = self.store._get_pattern_store(fid)  # type: ignore[attr-defined]
                except Exception:
                    store = None

            try:
                if store is not None:
                    if positive:
                        store.upvote(fid, portion)
                    else:
                        store.downvote(fid, portion)
                else:
                    if positive:
                        self.store.upvote(fid, strength=portion)
                    else:
                        self.store.downvote(fid, strength=portion)
            except Exception:
                continue

    def _limbic_reinforcement_scale(self, *, text: str, base_strength: float, channel: str) -> float:
        """Compute a limbic-style scaling factor for reinforcement.

        Goal: let mood/personality/preferences *gate learning intensity*.
        - Mood influences overall plasticity (already used for syntax plasticity);
          this extends it to reinforcement learning updates.
        - Preferences can dampen learning on aversive topics and mildly boost interests.
        - Personality proactivity provides a small global gain ("engage/learn" vs "conserve").

        Returns the scaled strength clamped to [0.0, 1.0].
        """

        strength = float(base_strength)
        if strength <= 0.0:
            return 0.0

        # Start with mood-based plasticity gain.
        scale = mood_plasticity_scale(self.mood_state) if self.config.enable_mood else 1.0

        # Small global personality gain.
        if self.config.enable_personality and self.personality_profile is not None:
            scale *= 0.9 + 0.2 * float(max(0.0, min(self.personality_profile.proactivity, 1.0)))

        # Preferences gate learning on-topic.
        if self.config.enable_preferences and (text or "").strip():
            t = text.lower()
            interests, aversions = self._preference_terms()
            if any(a.lower() in t for a in aversions):
                scale *= 0.75
            if any(i.lower() in t for i in interests):
                scale *= 1.10

        # Slight asymmetry: when "concerned", make negative feedback a bit stronger
        # (helps avoid repeating mistakes) while still damping positive plasticity.
        if self.config.enable_mood and self.mood_state is not None and self.mood_state.label == "concerned":
            if channel == "downvote":
                scale *= 1.10
            elif channel == "upvote":
                scale *= 0.90

        # Clamp and apply.
        scale = max(0.5, min(scale, 1.5))
        out = strength * scale
        return max(0.0, min(1.0, out))
    
    def teach(self, question: str, answer: str, intent: str = "user_teaching") -> str:
        """
        Teach the system a new question/answer pair.
        
        Args:
            question: Question text
            answer: Answer text
            intent: Intent category (default: "user_teaching")
            
        Returns:
            Pattern ID of the new pattern
        """
        pattern_id = self.store.add_pattern(
            question, 
            answer, 
            success_score=0.8,
            intent=intent
        )
        return pattern_id

    def learn_syntax_correction(self, incorrect: Optional[str], correct: str, *, use_last_response: bool = False) -> bool:
        """Teach the syntax stage a text-level correction.

        This is intentionally explicit (caller must provide a correction), to avoid
        over-learning from ambiguous feedback.
        """

        correct = (correct or "").strip()
        if not correct:
            return False

        if use_last_response and not (incorrect or "").strip():
            incorrect = self.last_response_text

        incorrect = (incorrect or "").strip()
        if not incorrect:
            return False

        stage = getattr(self.composer, "syntax_stage", None)
        if stage is None or not hasattr(stage, "learn_correction"):
            return False

        try:
            stage.learn_correction(incorrect, correct)
            return True
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'interaction_count': self.interaction_count,
            'pattern_counts': self.store.get_pattern_count() if hasattr(self.store, 'get_pattern_count') else {},
            'last_pattern_id': self.last_pattern_id,
        }
        
        # Add vocabulary stats if available
        if hasattr(self.store, 'get_vocabulary_stats'):
            stats['vocabulary'] = self.store.get_vocabulary_stats()
        
        # Add pattern stats if available
        if hasattr(self.store, 'get_pattern_stats'):
            stats['patterns'] = self.store.get_pattern_stats()
        
        # Add auto-learner stats
        if self.auto_learner:
            stats['auto_learning'] = self.auto_learner.get_stats()
        
        return stats
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get feedback detection statistics.
        
        Returns:
            Dictionary of feedback stats
        """
        if not self.feedback_tracker:
            return {}
        
        return self.feedback_tracker.get_stats()
    
    def save_state(self):
        """Save session state to disk."""
        if self.auto_learner:
            state_path = Path(self.config.data_path) / "auto_learner_state.json"
            self.auto_learner.save_state(state_path)
    
    def cleanup(self):
        """Cleanup session resources."""
        # Force train any pending auto-learning
        if self.auto_learner:
            self.auto_learner.force_train()
            self.save_state()
    
    def _detect_and_learn_declarative(self, content: str) -> Optional[str]:
        """
        Detect declarative statements and store them for learning.
        
        Only learns objective facts, not personal statements or opinions.
        
        Returns:
            String describing what was learned, or None
        """
        # Normalize content
        text = content.strip().rstrip('.!?')
        
        # Skip personal/subjective statements - these are not facts to learn
        personal_starts = (
            'i ', 'i\'', 'my ', 'me ', 'we ', 'our ',  # First person
            'you ', 'your ',  # Second person
            'he ', 'she ', 'they ', 'his ', 'her ', 'their ',  # Third person pronouns (often about people, not facts)
            'it\'s ', 'that\'s ',  # Contractions that often indicate opinion
        )
        if text.lower().startswith(personal_starts):
            return None
        
        # Skip statements that contain personal indicators anywhere
        personal_indicators = (
            ' i ', ' i\'', ' my ', ' me ', ' myself ',
            ' you ', ' your ', ' yourself ',
            ' think ', ' feel ', ' believe ', ' guess ', ' suppose ',
            ' maybe ', ' probably ', ' might ', ' could be ',
            ' used to ', ' don\'t ', ' doesn\'t ', ' won\'t ', ' can\'t ',
        )
        text_lower = f" {text.lower()} "
        for indicator in personal_indicators:
            if indicator in text_lower:
                return None
        
        # Skip very short statements (but allow factual statements like "X is Y")
        if len(text) < 10 or len(text.split()) < 3:
            return None
        
        # Patterns for declarative statements
        patterns = [
            (r'^(.+?)\s+(?:is|are|was|were)\s+(.+)$', 'is'),
            (r'^(.+?)\s+(?:does|do|did)\s+(.+)$', 'does'),
            (r'^(.+?)\s+(?:has|have|had)\s+(.+)$', 'has'),
            (r'^(.+?)\s+(?:drink|drinks|eat|eats|sleep|sleeps|live|lives|hunt|hunts)\s+(.+)$', 'verb'),
        ]
        
        for pattern, relation_type in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                subject = match.group(1).strip()
                predicate = match.group(2).strip()
                
                # Don't learn questions or very short statements
                if len(subject) < 2 or len(predicate) < 2:
                    continue
                
                # Skip if subject IS a question word (not just starts with one)
                # This prevents learning "what is X" as a fact, but allows "dogs", "dolphins", etc.
                question_words = {'what', 'who', 'where', 'when', 'why', 'how', 'do', 'does', 'did', 'is', 'are', 'was', 'were', 'can', 'could', 'would', 'should'}
                subject_first_word = subject.lower().split()[0] if subject.split() else ''
                if subject_first_word in question_words:
                    continue
                
                # Skip if subject looks personal or conversational
                subject_lower = subject.lower()
                if subject_lower in ('it', 'this', 'that', 'there', 'here', 'things', 'stuff'):
                    continue
                # Skip subjects that are clearly about the speaker or listener
                if any(word in subject_lower for word in ('i ', 'my ', 'you ', 'your ', 'we ', 'our ')):
                    continue
                
                # SIMPLIFIED: Store the statement as-is for BioNN semantic matching
                # The BioNN embedding already recognizes that:
                #   "are games edible" ↔ "games are not edible" (0.88 similarity)
                #   "is a parrot a bird" ↔ "a parrot is a bird" (1.00 similarity)
                # So we don't need complex question-form generation - just store the fact
                # and let the neural network handle the semantic matching!
                
                answer = text  # The full statement is both trigger and answer
                
                try:
                    # Limbic-style modulation: mood can slightly gate how strongly we seed a new fact.
                    seeded_success = 0.75
                    if self.config.enable_mood:
                        seeded_success = max(0.0, min(1.0, seeded_success * mood_plasticity_scale(self.mood_state)))
                    # Store the statement itself as the pattern trigger
                    # BioNN semantic matching will find it when similar questions are asked
                    self.store.add_pattern(
                        text,  # Use statement as trigger (best semantic match)
                        answer, 
                        success_score=seeded_success,
                        intent='declarative_learning'
                    )
                    
                    # Learn the topic for BioNN-based query extraction
                    # This allows future "do you know about {subject}?" queries
                    # to be resolved via semantic similarity instead of regex
                    if self.topic_extractor:
                        self.topic_extractor.learn_topic(subject, text)
                    
                    return f"{subject} -> {predicate}"
                except Exception as e:
                    print(f"  ⚠️  Failed to store declarative: {e}")
                    return None
        
        return None
    
    def _apply_plasticity(self):
        """Apply neuroplasticity updates based on interaction count."""
        if not self.composer.syntax_stage:
            return

        plasticity_scale = mood_plasticity_scale(self.mood_state) if self.config.enable_mood else 1.0
        
        # Syntax plasticity
        if self.interaction_count % self.config.syntax_plasticity_interval == 0:
            try:
                if hasattr(self.composer.syntax_stage, 'patterns') and self.composer.syntax_stage.patterns:
                    pattern_sample = list(self.composer.syntax_stage.patterns.values())[-5:]
                    for pattern in pattern_sample:
                        if hasattr(self.composer.syntax_stage, '_apply_plasticity'):
                            self.composer.syntax_stage._apply_plasticity(
                                pattern=pattern,
                                feedback=0.8 * plasticity_scale,
                                contrastive_pairs=None
                            )
            except Exception as e:
                print(f"  ⚠️  Syntax plasticity error: {e}")
        
        # World model plasticity (learns from stored situations)
        if self.interaction_count % self.config.syntax_plasticity_interval == 0:
            try:
                if hasattr(self.composer, 'world_model') and self.composer.world_model:
                    if hasattr(self.composer.world_model, 'apply_plasticity'):
                        # Calculate success rate from recent interactions
                        success_rate = self._calculate_world_model_success_rate()
                        self.composer.world_model.apply_plasticity(success_rate=success_rate)
                        print(f"  🌍 World model plasticity applied (success_rate={success_rate:.2f})")
            except Exception as e:
                print(f"  ⚠️  World model plasticity error: {e}")
        
        # Contrastive learning
        if self.interaction_count % self.config.contrastive_interval == 0:
            try:
                if hasattr(self.composer.syntax_stage, 'apply_contrastive_learning'):
                    try:
                        self.composer.syntax_stage.apply_contrastive_learning(scale=plasticity_scale)
                    except TypeError:
                        self.composer.syntax_stage.apply_contrastive_learning()
            except Exception as e:
                print(f"  ⚠️  Contrastive learning error: {e}")
    
    def _calculate_world_model_success_rate(self) -> float:
        """Calculate success rate for world model learning."""
        # Use general success rate as proxy
        # In future, could track world-model-specific feedback
        if hasattr(self, 'feedback_tracker') and self.feedback_tracker:
            if hasattr(self.feedback_tracker, 'get_success_rate'):
                return self.feedback_tracker.get_success_rate()
        # Default to moderate success for gradual learning
        return 0.6
    
    def _determine_source(self, pattern_id: Optional[str]) -> str:
        """Determine the source of a pattern."""
        if not pattern_id:
            return "unknown"
        if pattern_id.startswith('external_'):
            return "external_wikipedia"
        if pattern_id.startswith('low_confidence'):
            return "fallback"
        return "internal"
