"""
Pragmatic Template System for Lilith v2 (Ported and Enhanced from v1).

This manages 'Dialogue Acts' and high-level conversational patterns.
Includes support for:
1. Dynamic Template Selection (Categories/Intents)
2. Teaching Detection (Pattern Learning)
3. Engagement Evaluation (Conversational Success)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
import random
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class PragmaticTemplate:
    template_id: str
    category: str    # "greeting", "acknowledgment", "definition"
    template: str    # "{greeting}! {offer_help}"
    slots: List[str] # ["greeting", "offer_help"]
    priority: int = 5
    intent: str = "general" # Add intent for finer granulartiy (omitted in v2 bootstrap, adding now)

@dataclass
class InteractionSnapshot:
    """Stores the state of an interaction for learning."""
    user_input: str
    bot_response: str
    topic_signature: str
    timestamp: float

class PragmaticSystem:
    def __init__(self):
        self.templates: Dict[str, PragmaticTemplate] = {}
        self._bootstrap()
        
        # Learning state
        self.history: List[InteractionSnapshot] = []

    def _bootstrap(self):
        """Load default conversational templates (Expanded set)."""
        # 1. Greetings
        self.register(PragmaticTemplate(
            "greet_simple", "greeting", 
            "Hello! I am ready to help.", [], priority=5
        ))
        
        # 2. Definitions
        self.register(PragmaticTemplate(
            "def_simple", "definition",
            "{concept} is {property}.",
            ["concept", "property"], priority=8
        ))
        self.register(PragmaticTemplate(
            "def_detailed", "definition",
            "{concept} is {property}. It is known for {detail}.",
            ["concept", "property", "detail"], priority=9
        ))

        # 3. Teaching Acknowledgment (Crucial for Autodidact loop)
        self.register(PragmaticTemplate(
            "teaching_ack", "teaching",
            "I see! So {subject} is {object}. I've learned that now.",
            ["subject", "object"], priority=9, intent="acknowledge_learning"
        ))
        
        # 4. Clarification (From v1)
        self.register(PragmaticTemplate(
            "clarify_context", "clarification",
            "I'd like to help, but could you provide more context about {topic}?",
            ["topic"], priority=7
        ))
        
        # 5. Confirmation (From v1)
        self.register(PragmaticTemplate(
            "confirm_yes", "confirmation",
            "Yes, that is correct. {subject} is {relationship}.",
            ["subject", "relationship"], priority=8
        ))

        # 6. Uncertain / Fuzzy
        self.register(PragmaticTemplate(
            "uncertain", "uncertainty",
            "I'm not entirely sure, but {concept} might be {property}.",
            ["concept", "property"]
        ))

    def register(self, template: PragmaticTemplate):
        self.templates[template.template_id] = template

    def get_template(self, category: str, available_slots: List[str], intent: str = None) -> Optional[PragmaticTemplate]:
        """
        Select best template, filtering by category, slots, and optional specific intent.
        """
        candidates = [
            t for t in self.templates.values() 
            if t.category == category
            and all(s in available_slots for s in t.slots)
        ]
        
        if intent:
            candidates = [t for t in candidates if t.intent == intent]
            
        if not candidates:
            return None
            
        candidates.sort(key=lambda t: t.priority, reverse=True)
        top_priority = candidates[0].priority
        top_tier = [c for c in candidates if c.priority == top_priority]
        return random.choice(top_tier)

    def fill(self, template: PragmaticTemplate, context: Dict[str, str]) -> str:
        """Hydrate template."""
        try:
            result = template.template
            for slot in template.slots:
                if slot in context:
                    val = str(context[slot])
                    result = result.replace(f"{{{slot}}}", val)
            return result
        except Exception as e:
            logger.error(f"Template fill failed: {e}")
            return template.template

    # ------------------------------------------------------------------
    # Learning Logic (Ported from v1 PragmaticLearner)
    # ------------------------------------------------------------------

    def evaluate_engagement(self, user_input: str) -> float:
        """
        Evaluate engagement score based on user input (From v1).
        
        Signals:
        - "+" Length (>10 words)
        - "+" Curiosity ("why", "how", "tell me more")
        - "-" Confusion ("what", "huh")
        """
        score = 0.5
        text = user_input.lower()
        
        # Negative signals
        if any(w in text for w in ["what?", "huh", "confused"]):
            score -= 0.2
            
        # Positive signals
        if any(w in text for w in ["interesting", "tell me more", "why", "how"]):
            score += 0.3
            
        # Length signal
        words = len(text.split())
        if words > 10: score += 0.2
        elif words < 3: score -= 0.1
        
        return max(0.0, min(1.0, score))

    def detect_teaching_intent(self, previous_response: str, current_input: str) -> bool:
        """
        Heuristic to detect if user is correcting/teaching Lilith (From v1).
        
        Pattern: Bot fails ("I don't know") -> User states fact ("X is Y")
        """
        result = False
        
        # Check if previous response indicated failure/uncertanty
        # We need to be careful with exact wording, but looking for key markers
        fallback_markers = ["listening", "don't know", "not sure", "provide more context"]
        bot_failed = any(m in previous_response.lower() for m in fallback_markers)
        
        # Check if current input looks like a statement of fact
        # A simple heuristic: "X is Y" structure
        fact_markers = [" is ", " are ", " means "]
        user_teaching = any(m in current_input.lower() for m in fact_markers)
        
        if bot_failed and user_teaching:
            logger.info("🎓 Teaching Intent Detected via Pragmatic Heuristic")
            result = True
            
        return result

    def learn_from_interaction(self, user_input: str, bot_response: str, topic: str):
        """Record interaction for future pattern extraction."""
        self.history.append(InteractionSnapshot(
            user_input=user_input,
            bot_response=bot_response,
            topic_signature=topic,
            timestamp=time.time()
        ))
        # Keep history short (Working Memory)
        if len(self.history) > 10:
            self.history.pop(0)

