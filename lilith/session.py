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


@dataclass
class SessionConfig:
    """Configuration for a Lilith session."""
    
    # Core settings
    data_path: str = "data"
    enable_knowledge_augmentation: bool = True
    enable_modal_routing: bool = True
    use_grammar: bool = True
    
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
        
        # Initialize encoder
        self.encoder = PMFlowEmbeddingEncoder()
        
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
                base_data_path=self.config.data_path
            )
        
        # Create conversation state (working memory - active topics with decay)
        self.state = ConversationState(self.encoder)
        
        # Create conversation history (short-term memory - recent turns, sliding window)
        self.conversation_history = ConversationHistory(max_turns=10)
        
        # Create composer with conversation history
        self.composer = ResponseComposer(
            self.store,
            self.state,
            self.conversation_history,
            semantic_encoder=self.encoder,
            enable_knowledge_augmentation=self.config.enable_knowledge_augmentation,
            enable_modal_routing=self.config.enable_modal_routing,
            use_grammar=self.config.use_grammar
        )
        
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
        
        # Detect and learn from declarative statements
        if self.config.learning_enabled and self.config.enable_declarative_learning:
            learned_fact = self._detect_and_learn_declarative(content)
        
        # Update conversation state for topic tracking and pronoun resolution
        enriched_context = self._update_conversation_context(content)
        
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
            
            return SessionResponse(
                text="",
                learned_fact=learned_fact
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
            # Return empty response - feedback was applied, no need to respond
            return SessionResponse(
                text="",
                learned_fact=learned_fact
            )
        
        # Generate response using enriched context (includes topic history for pronoun resolution)
        response = self.composer.compose_response(context=enriched_context, user_input=content)
        
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
        
        return SessionResponse(
            text=response.text,
            pattern_id=self.last_pattern_id,
            confidence=response.confidence if hasattr(response, 'confidence') else 0.0,
            is_fallback=getattr(response, 'is_fallback', False),
            is_low_confidence=getattr(response, 'is_low_confidence', False),
            source=self._determine_source(self.last_pattern_id),
            learned_fact=learned_fact
        )
    
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
            
            # If no pronoun resolution, still build enriched context for retrieval
            if snapshot.topics:
                topic_summaries = [topic.summary for topic in snapshot.topics[:3]]
                if topic_summaries:
                    topics_str = ", ".join(topic_summaries)
                    enriched = f"Recent topics: {topics_str} | Current: {content}"
                    return enriched
                    
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
        
        # Check if this is external knowledge that should be learned
        if target_id.startswith('external_') and self.last_user_input and self.last_response_text:
            # Learn from Wikipedia/external source
            if hasattr(self.store, 'learn_from_wikipedia'):
                new_pattern_id = self.store.learn_from_wikipedia(
                    query=self.last_user_input,
                    response_text=self.last_response_text,
                    success_score=0.8,
                    intent="learned_knowledge"
                )
                print(f"📚 Learned from external knowledge: {new_pattern_id}")
                return True
        
        # Regular upvote
        self.store.upvote(target_id, strength=strength)
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
        
        self.store.downvote(target_id, strength=strength)
        return True
    
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
        
        Returns:
            String describing what was learned, or None
        """
        # Normalize content
        text = content.strip().rstrip('.!?')
        
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
                if subject.lower().startswith(('what', 'who', 'where', 'when', 'why', 'how', 'do', 'does', 'is', 'are')):
                    continue
                
                # Create Q&A pattern from declarative statement
                if relation_type == 'is':
                    question = f"What is {subject}?"
                    answer = f"{subject} is {predicate}"
                elif relation_type == 'does':
                    question = f"What does {subject} do?"
                    answer = f"{subject} does {predicate}"
                elif relation_type == 'has':
                    question = f"What does {subject} have?"
                    answer = f"{subject} has {predicate}"
                else:  # verb
                    question = f"What about {subject}?"
                    answer = text
                
                try:
                    self.store.add_pattern(
                        question, 
                        answer, 
                        success_score=0.7,
                        intent='declarative_learning'
                    )
                    return f"{subject} -> {predicate}"
                except Exception as e:
                    print(f"  ⚠️  Failed to store declarative: {e}")
                    return None
        
        return None
    
    def _apply_plasticity(self):
        """Apply neuroplasticity updates based on interaction count."""
        if not self.composer.syntax_stage:
            return
        
        # Syntax plasticity
        if self.interaction_count % self.config.syntax_plasticity_interval == 0:
            try:
                if hasattr(self.composer.syntax_stage, 'patterns') and self.composer.syntax_stage.patterns:
                    pattern_sample = list(self.composer.syntax_stage.patterns.values())[-5:]
                    for pattern in pattern_sample:
                        if hasattr(self.composer.syntax_stage, '_apply_plasticity'):
                            self.composer.syntax_stage._apply_plasticity(
                                pattern=pattern,
                                feedback=0.8,
                                contrastive_pairs=None
                            )
            except Exception as e:
                print(f"  ⚠️  Syntax plasticity error: {e}")
        
        # Contrastive learning
        if self.interaction_count % self.config.contrastive_interval == 0:
            try:
                if hasattr(self.composer.syntax_stage, 'apply_contrastive_learning'):
                    self.composer.syntax_stage.apply_contrastive_learning()
            except Exception as e:
                print(f"  ⚠️  Contrastive learning error: {e}")
    
    def _determine_source(self, pattern_id: Optional[str]) -> str:
        """Determine the source of a pattern."""
        if not pattern_id:
            return "unknown"
        if pattern_id.startswith('external_'):
            return "external_wikipedia"
        if pattern_id.startswith('low_confidence'):
            return "fallback"
        return "internal"
