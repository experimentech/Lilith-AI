"""
Discourse Management Stage for Lilith v2.

The "Conductor" - tracks dialogue state, user intent, topic continuity,
and conversational obligations.

This stage answers: "Where are we in the conversation, and what's expected?"
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class DialogueAct:
    """A single communicative action from the user."""
    act_type: str  # "question", "statement", "greeting", "feedback", "correction", "farewell"
    topic: Optional[str] = None
    referents: List[str] = field(default_factory=list)  # Entities mentioned
    timestamp: float = field(default_factory=time.time)
    raw_text: str = ""


@dataclass
class DialogueState:
    """Complete state of the current dialogue."""
    phase: str = "opening"  # "opening", "topic_exploration", "teaching", "closing"
    current_topic: Optional[str] = None
    topic_stack: List[str] = field(default_factory=list)  # For nested topics
    last_n_acts: List[DialogueAct] = field(default_factory=list)
    pending_obligations: List[str] = field(default_factory=list)  # "answer_question", "acknowledge"
    turn_count: int = 0
    last_user_intent: str = "unknown"
    last_bot_response: str = ""
    referent_cache: Dict[str, str] = field(default_factory=dict)  # pronoun -> entity


class DiscourseManager:
    """
    Manages dialogue state and conversational flow.
    
    Responsibilities:
    1. Track dialogue phase (opening, exploration, closing)
    2. Identify user's communicative intent
    3. Manage topic continuity across turns
    4. Resolve anaphora (pronouns to entities)
    5. Track conversational obligations
    """
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.state = DialogueState()
        self._act_history: deque = deque(maxlen=max_history)
        
        # Intent patterns (bootstrapped, should be learned)
        self._intent_patterns = self._bootstrap_intent_patterns()
        
        # Referent tracking
        self._entity_buffer: List[Tuple[str, str]] = []  # (entity, type)
    
    def _bootstrap_intent_patterns(self) -> Dict[str, List[str]]:
        """Initial patterns for intent detection. Should be learned over time."""
        return {
            "question": [
                r"^(what|who|where|when|why|how|which|is|are|can|could|do|does|did|will|would)\b",
                r"\?$",
                r"^(tell me|explain|describe|define)\b",
            ],
            "greeting": [
                r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b",
                r"^(howdy|hiya|yo)\b",
                r"^how are you",  # Social greeting, not genuine question
                r"^how('s| is) it going",
                r"^what's up\b",
                r"how are you\??\s*$",  # Embedded greeting at end: "I'm good. how are you?"
                r"\bhow about you\??\s*$",  # "I'm fine. How about you?"
                r"\band you\??\s*$",  # "Good, and you?"
            ],
            "farewell": [
                r"^(bye|goodbye|see you|later|take care|farewell)\b",
                r"^(gotta go|have to go|leaving)\b",
            ],
            "feedback_positive": [
                r"^(thanks|thank you|great|good|nice|perfect|awesome|excellent)\b",
                r"^(that helps|that's helpful|makes sense)\b",
            ],
            "feedback_negative": [
                r"^(no|wrong|incorrect|that's not|actually)\b",
                r"^(i don't think|that doesn't seem)\b",
            ],
            "teaching": [
                r"\bis\s+(a|an|the)\s+\w+",  # "X is a Y"
                r"\bare\s+\w+",  # "X are Y"
                r"\bmeans?\b",  # "X means Y"
            ],
            "imperative": [
                r"^(please\s+)?(do|make|create|write|read|open|close|start|stop|run|send|get|set|put|move|copy|delete|download|upload|install|sign\s*up|log\s*in|register)\b",
                r"^(please\s+)?(go to|navigate to|browse to|click|type|enter|submit|fill|check|select)\b",
                r"^(please\s+)?(can you|could you|would you|will you)\s+(do|make|create|write|read|open|send|get)\b",
                r"^(i need you to|i want you to|i'd like you to)\b",
                r"^(help me|assist me)\s+(to\s+)?\w+",
            ],
            "elaboration_request": [
                r"^(tell me more|more about|elaborate|expand|go on)\b",
                r"^(what else|anything else|more details)\b",
            ],
            "acknowledgment": [
                r"^(ok|okay|i see|got it|understood|right|alright)\b",
                r"^(uh huh|mm|hmm|interesting)\b",
            ],
        }
    
    def update(
        self,
        user_input: str,
        extracted_entities: Optional[List[str]] = None,
        extracted_topic: Optional[str] = None,
        bot_response: Optional[str] = None,
    ) -> DialogueState:
        """
        Update dialogue state based on new user input.
        
        Args:
            user_input: Raw user input text
            extracted_entities: Entities extracted by semantic analysis
            extracted_topic: Topic extracted by topic extractor
            bot_response: The response we're about to give (for tracking)
            
        Returns:
            Updated DialogueState
        """
        self.state.turn_count += 1
        
        # 1. Identify user's communicative intent
        intent = self._classify_intent(user_input)
        self.state.last_user_intent = intent
        
        # 2. Create dialogue act
        act = DialogueAct(
            act_type=intent,
            topic=extracted_topic,
            referents=extracted_entities or [],
            raw_text=user_input,
        )
        self._act_history.append(act)
        self.state.last_n_acts = list(self._act_history)
        
        # 3. Update entity buffer for anaphora resolution
        if extracted_entities:
            for entity in extracted_entities:
                self._update_entity_buffer(entity, user_input)
        
        # 4. Update topic tracking
        if extracted_topic:
            self._update_topic(extracted_topic)
        
        # 5. Update dialogue phase
        self._update_phase(intent)
        
        # 6. Determine obligations
        self._update_obligations(intent, user_input)
        
        # 7. Track last bot response
        if bot_response:
            self.state.last_bot_response = bot_response
        
        return self.state
    
    def _classify_intent(self, text: str) -> str:
        """Classify user's communicative intent from text."""
        text_lower = text.lower().strip()
        
        # Check patterns in priority order
        priority_order = [
            "greeting", "farewell", "feedback_positive", "feedback_negative",
            "elaboration_request", "acknowledgment", "imperative", "question", "teaching"
        ]
        
        for intent in priority_order:
            patterns = self._intent_patterns.get(intent, [])
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return intent
        
        # Default: statement
        return "statement"
    
    def _update_entity_buffer(self, entity: str, context: str) -> None:
        """Track entity for anaphora resolution."""
        # Determine entity type heuristically
        entity_lower = entity.lower()
        
        # Simple type detection (should use NER in production)
        if entity[0].isupper() and " " not in entity:
            entity_type = "proper_noun"
        elif any(w in context.lower() for w in ["he", "him", "his"]):
            entity_type = "person_male"
        elif any(w in context.lower() for w in ["she", "her", "hers"]):
            entity_type = "person_female"
        else:
            entity_type = "thing"
        
        # Add to buffer (recent first)
        self._entity_buffer.insert(0, (entity, entity_type))
        
        # Keep buffer manageable
        if len(self._entity_buffer) > 20:
            self._entity_buffer = self._entity_buffer[:20]
        
        # Update referent cache for pronouns
        if entity_type == "person_male":
            self.state.referent_cache["he"] = entity
            self.state.referent_cache["him"] = entity
            self.state.referent_cache["his"] = entity
        elif entity_type == "person_female":
            self.state.referent_cache["she"] = entity
            self.state.referent_cache["her"] = entity
            self.state.referent_cache["hers"] = entity
        
        # "it" refers to most recent non-person entity
        if entity_type in ("thing", "proper_noun"):
            self.state.referent_cache["it"] = entity
            self.state.referent_cache["that"] = entity
            self.state.referent_cache["this"] = entity
    
    def _update_topic(self, topic: str) -> None:
        """Update topic tracking with continuity."""
        if self.state.current_topic:
            # Push old topic to stack if different
            if self.state.current_topic.lower() != topic.lower():
                if self.state.current_topic not in self.state.topic_stack:
                    self.state.topic_stack.append(self.state.current_topic)
                # Keep stack bounded
                if len(self.state.topic_stack) > 5:
                    self.state.topic_stack = self.state.topic_stack[-5:]
        
        self.state.current_topic = topic
    
    def _update_phase(self, intent: str) -> None:
        """Update dialogue phase based on intent and history."""
        if intent == "greeting" and self.state.turn_count <= 2:
            self.state.phase = "opening"
        elif intent == "farewell":
            self.state.phase = "closing"
        elif intent == "teaching":
            self.state.phase = "teaching"
        elif intent in ("question", "statement", "elaboration_request"):
            self.state.phase = "topic_exploration"
        # Keep current phase for acknowledgments
    
    def _update_obligations(self, intent: str, text: str) -> None:
        """Determine what the response must address."""
        self.state.pending_obligations.clear()
        
        if intent == "question":
            self.state.pending_obligations.append("answer_question")
        elif intent == "greeting":
            self.state.pending_obligations.append("reciprocate_greeting")
        elif intent == "farewell":
            self.state.pending_obligations.append("reciprocate_farewell")
        elif intent == "teaching":
            self.state.pending_obligations.append("acknowledge_teaching")
        elif intent == "feedback_negative":
            self.state.pending_obligations.append("acknowledge_correction")
        elif intent == "feedback_positive":
            self.state.pending_obligations.append("acknowledge_thanks")
        elif intent == "elaboration_request":
            self.state.pending_obligations.append("elaborate")
        # Statements and acknowledgments have no hard obligations
    
    def resolve_anaphora(self, text: str) -> str:
        """
        Replace pronouns with their referents from context.
        
        Args:
            text: Text with potential pronouns
            
        Returns:
            Text with pronouns resolved where possible
        """
        if not self.state.referent_cache:
            return text
        
        result = text
        
        # Pattern: "Tell me about it" -> "Tell me about [dolphin]"
        # Only replace standalone pronouns, not parts of words
        pronouns = ["it", "that", "this", "he", "she", "him", "her", "they", "them"]
        
        for pronoun in pronouns:
            if pronoun in self.state.referent_cache:
                referent = self.state.referent_cache[pronoun]
                # Match whole word only
                pattern = rf"\b{pronoun}\b"
                # Only replace if it's clearly anaphoric (not "it is", "that is")
                # This is a simplification - real anaphora resolution is complex
                if re.search(pattern, result.lower()):
                    # Check if it's in a question context about the referent
                    question_patterns = [
                        rf"(what|tell me|more) about {pronoun}\b",
                        rf"(is|are) {pronoun}\b",
                        rf"{pronoun} (is|are|has|have)\b",
                    ]
                    for qp in question_patterns:
                        if re.search(qp, result.lower()):
                            result = re.sub(pattern, referent, result, flags=re.IGNORECASE)
                            break
        
        return result
    
    def get_topic_context(self) -> Dict[str, Any]:
        """Get current topic context for response generation."""
        return {
            "current_topic": self.state.current_topic,
            "topic_stack": self.state.topic_stack.copy(),
            "recent_entities": [e for e, _ in self._entity_buffer[:5]],
            "referent_cache": self.state.referent_cache.copy(),
        }
    
    def get_obligations(self) -> List[str]:
        """Get current conversational obligations."""
        return self.state.pending_obligations.copy()
    
    def get_last_intent(self) -> str:
        """Get the user's last communicative intent."""
        return self.state.last_user_intent
    
    def get_phase(self) -> str:
        """Get current dialogue phase."""
        return self.state.phase
    
    def is_topic_continuation(self, topic: str) -> bool:
        """Check if topic is a continuation of current discussion."""
        if not self.state.current_topic:
            return False
        return (
            topic.lower() == self.state.current_topic.lower() or
            topic.lower() in [t.lower() for t in self.state.topic_stack]
        )
    
    def get_previous_topic(self) -> Optional[str]:
        """Get the previous topic (for topic bridging)."""
        if self.state.topic_stack:
            return self.state.topic_stack[-1]
        return None
    
    def mark_obligation_fulfilled(self, obligation: str) -> None:
        """Mark an obligation as fulfilled."""
        if obligation in self.state.pending_obligations:
            self.state.pending_obligations.remove(obligation)
    
    def reset(self) -> None:
        """Reset state for a new conversation."""
        self.state = DialogueState()
        self._act_history.clear()
        self._entity_buffer.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics about discourse state."""
        return {
            "turn_count": self.state.turn_count,
            "phase": self.state.phase,
            "current_topic": self.state.current_topic,
            "topic_stack_depth": len(self.state.topic_stack),
            "pending_obligations": len(self.state.pending_obligations),
            "tracked_entities": len(self._entity_buffer),
            "referent_cache_size": len(self.state.referent_cache),
        }
