"""
Communication Planning Stage for Lilith v2.

The "Intent Generator" - decides WHAT to communicate and WHY.
This is the missing "motivation" layer that bridges knowing and speaking.

Consumes:
- DialogueState (what's expected)
- Working memory (what we know)
- Affective state (our drives)
- Inferences (what connections we made)

Produces:
- CommunicationPlan (what to say and how)
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CommunicationPlan:
    """
    A plan for what to communicate.
    
    This bridges the gap between having knowledge and expressing it purposefully.
    """
    # Primary communication goal
    primary_goal: str  # "inform", "ask", "acknowledge", "elaborate", "clarify", "greet", "farewell"
    
    # Current topic
    topic: Optional[str] = None  # The main subject of communication
    
    # Content to express
    content_concepts: List[str] = field(default_factory=list)  # Concept IDs/terms
    content_relations: List[Tuple[str, str, str]] = field(default_factory=list)  # (subj, pred, obj)
    content_facts: List[str] = field(default_factory=list)  # Raw facts to convey
    
    # Communicative stance
    stance: str = "neutral"  # "certain", "curious", "empathetic", "uncertain", "enthusiastic"
    
    # Follow-up behavior (secondary goal)
    follow_up_goal: Optional[str] = None  # "ask_experience", "ask_elaboration", "ask_opinion"
    follow_up_topic: Optional[str] = None
    
    # Discourse markers to use
    discourse_markers: List[str] = field(default_factory=list)  # "Actually...", "Speaking of..."
    
    # Initiative
    take_initiative: bool = False  # Should Lilith drive the conversation?
    
    # Confidence in plan
    confidence: float = 0.5
    
    # Explanation trace
    reasoning: List[str] = field(default_factory=list)


class CommunicationPlanner:
    """
    Plans what to communicate based on dialogue state and internal drives.
    
    This is the "motivation" layer that answers:
    1. What should I say? (goal selection)
    2. What content should I include? (content selection)
    3. How should I say it? (stance selection)
    4. Should I ask something? (follow-up generation)
    """
    
    def __init__(
        self,
        curiosity_threshold: float = 0.6,
        confidence_threshold: float = 0.5,
        warmth_threshold: float = 0.5,
        initiative_threshold: float = 0.4,
    ):
        self.curiosity_threshold = curiosity_threshold
        self.confidence_threshold = confidence_threshold
        self.warmth_threshold = warmth_threshold
        self.initiative_threshold = initiative_threshold
        
        # Track engagement for initiative decisions
        self._engagement_history: List[float] = []
        
        # Follow-up question templates (should be learned)
        self._followup_templates = self._bootstrap_followup_templates()
        
        # Discourse markers (should be learned from input)
        self._discourse_markers = self._bootstrap_discourse_markers()
    
    def _bootstrap_followup_templates(self) -> Dict[str, List[str]]:
        """Initial follow-up question templates."""
        return {
            "ask_experience": [
                "Have you encountered {topic} before?",
                "What's your experience with {topic}?",
                "Have you seen {topic} yourself?",
            ],
            "ask_elaboration": [
                "What else would you like to know about {topic}?",
                "Is there a specific aspect of {topic} you're curious about?",
                "Would you like to know more about {topic}?",
            ],
            "ask_opinion": [
                "What do you think about {topic}?",
                "How do you feel about {topic}?",
                "What's your take on {topic}?",
            ],
            "ask_related": [
                "Speaking of {topic}, have you heard about {related}?",
                "That reminds me of {related}. Are you familiar with it?",
            ],
        }
    
    def _bootstrap_discourse_markers(self) -> Dict[str, List[str]]:
        """Initial discourse markers by function."""
        return {
            "topic_continuation": ["Speaking of which, ", "On that note, ", "Related to that, "],
            "topic_shift": ["By the way, ", "On another note, ", "Changing topics, "],
            "elaboration": ["In fact, ", "Actually, ", "To add to that, "],
            "contrast": ["However, ", "On the other hand, ", "That said, "],
            "emphasis": ["Interestingly, ", "Notably, ", "Importantly, "],
            "uncertainty": ["I think ", "Perhaps ", "It seems that "],
            "certainty": ["Indeed, ", "Certainly, ", "Definitely, "],
            "acknowledgment": ["I see. ", "That makes sense. ", "Understood. "],
            "enthusiasm": ["That's fascinating! ", "How interesting! ", "Wonderful! "],
        }
    
    def plan(
        self,
        obligations: List[str],
        working_memory: Dict[str, Any],  # concept_id -> ActivatedConcept
        affective_state: Dict[str, float],
        inferences: Optional[List[Any]] = None,
        external_knowledge: Optional[List[Any]] = None,
        current_topic: Optional[str] = None,
        topic_context: Optional[Dict] = None,
        user_intent: str = "unknown",
    ) -> CommunicationPlan:
        """
        Generate a communication plan.
        
        Priority order:
        1. Fulfill obligations (must do)
        2. Share relevant inferences (if interesting)
        3. Add follow-up based on drives (if curious)
        4. Choose stance based on confidence
        """
        plan = CommunicationPlan(primary_goal="neutral", topic=current_topic)
        
        # Extract affective drives
        curiosity = affective_state.get("curiosity_drive", 0.5)
        warmth = affective_state.get("warmth", 0.5)
        confidence = affective_state.get("confidence", 0.5)
        valence = affective_state.get("mood_valence", 0.0)
        
        # Step 1: Determine primary goal from obligations
        primary_goal = self._goal_from_obligations(obligations, user_intent)
        plan.primary_goal = primary_goal
        plan.reasoning.append(f"Goal '{primary_goal}' from obligations: {obligations}")
        
        # Step 2: Select content based on goal
        self._select_content(
            plan, 
            working_memory, 
            inferences, 
            external_knowledge,
            current_topic,
        )
        
        # Step 3: Determine stance from affective state
        plan.stance = self._select_stance(confidence, valence, warmth, plan.primary_goal)
        plan.reasoning.append(f"Stance '{plan.stance}' from confidence={confidence:.2f}, valence={valence:.2f}")
        
        # Step 4: Consider follow-up based on curiosity
        if curiosity > self.curiosity_threshold:
            self._add_followup(plan, current_topic, topic_context)
            plan.reasoning.append(f"Added follow-up due to curiosity={curiosity:.2f}")
        
        # Step 5: Decide on initiative
        plan.take_initiative = self._should_take_initiative(
            plan, curiosity, valence, user_intent
        )
        
        # Step 6: Select discourse markers
        self._add_discourse_markers(plan, topic_context, user_intent)
        
        # Step 7: Calculate confidence in plan
        plan.confidence = self._calculate_plan_confidence(plan, working_memory)
        
        logger.debug(f"Communication plan: goal={plan.primary_goal}, stance={plan.stance}, "
                    f"content={len(plan.content_concepts)}, follow_up={plan.follow_up_goal}")
        
        return plan
    
    def _goal_from_obligations(self, obligations: List[str], user_intent: str) -> str:
        """Map obligations to communication goals."""
        if not obligations:
            # No obligations - default based on user intent
            if user_intent == "statement":
                return "acknowledge"
            elif user_intent == "acknowledgment":
                return "continue"  # Keep conversation going
            return "neutral"
        
        # Priority mapping
        obligation_to_goal = {
            "answer_question": "inform",
            "reciprocate_greeting": "greet",
            "reciprocate_farewell": "farewell",
            "acknowledge_teaching": "acknowledge_learning",
            "acknowledge_correction": "accept_correction",
            "acknowledge_thanks": "acknowledge",
            "elaborate": "elaborate",
        }
        
        # Take first obligation (highest priority)
        for obligation in obligations:
            if obligation in obligation_to_goal:
                return obligation_to_goal[obligation]
        
        return "inform"  # Default
    
    def _select_content(
        self,
        plan: CommunicationPlan,
        working_memory: Dict[str, Any],
        inferences: Optional[List[Any]],
        external_knowledge: Optional[List[Any]],
        current_topic: Optional[str],
    ) -> None:
        """Select content from available sources."""
        
        # Priority 1: Inferences (things we reasoned about)
        if inferences:
            for inf in inferences[:2]:  # Max 2 inferences
                if hasattr(inf, 'source_concept') and hasattr(inf, 'target_concept'):
                    plan.content_concepts.append(inf.source_concept)
                    plan.content_concepts.append(inf.target_concept)
                    if hasattr(inf, 'relation'):
                        plan.content_relations.append((
                            inf.source_concept, inf.relation, inf.target_concept
                        ))
                elif isinstance(inf, dict):
                    if 'subject' in inf and 'object' in inf:
                        plan.content_relations.append((
                            inf['subject'], 
                            inf.get('predicate', 'relates_to'),
                            inf['object']
                        ))
            plan.reasoning.append(f"Selected {len(inferences)} inferences for content")
        
        # Priority 2: External knowledge
        if external_knowledge:
            for frag in external_knowledge[:2]:
                if hasattr(frag, 'content'):
                    plan.content_facts.append(frag.content)
                elif isinstance(frag, str):
                    plan.content_facts.append(frag)
            plan.reasoning.append(f"Added {len(external_knowledge)} external facts")
        
        # Priority 3: Working memory (activated concepts)
        if working_memory and current_topic:
            topic_lower = current_topic.lower()
            for cid, concept in working_memory.items():
                term = getattr(concept, 'term', str(concept))
                if topic_lower in term.lower():
                    plan.content_concepts.append(term)
                    # Get properties if available
                    if hasattr(concept, 'properties') and concept.properties:
                        for prop in concept.properties[:2]:
                            plan.content_relations.append((term, "has_property", prop))
            plan.reasoning.append(f"Selected concepts from working memory")
    
    def _select_stance(
        self, 
        confidence: float, 
        valence: float, 
        warmth: float,
        goal: str,
    ) -> str:
        """Select communicative stance based on affective state."""
        
        # Goal-specific stances
        if goal in ("greet", "farewell"):
            return "warm" if warmth > 0.5 else "neutral"
        
        if goal == "acknowledge_learning":
            return "enthusiastic" if valence > 0.2 else "appreciative"
        
        if goal == "accept_correction":
            return "humble"
        
        # General stance selection
        if confidence > 0.7:
            return "certain"
        elif confidence < 0.4:
            return "uncertain"
        elif valence > 0.3:
            return "enthusiastic"
        elif warmth > self.warmth_threshold:
            return "empathetic"
        
        return "neutral"
    
    def _add_followup(
        self,
        plan: CommunicationPlan,
        current_topic: Optional[str],
        topic_context: Optional[Dict],
    ) -> None:
        """Add a follow-up question based on curiosity."""
        if not current_topic:
            return
        
        # Decide type of follow-up
        if plan.primary_goal == "inform":
            # We just informed - ask about their experience
            plan.follow_up_goal = "ask_experience"
        elif plan.primary_goal == "acknowledge_learning":
            # They taught us - ask for more
            plan.follow_up_goal = "ask_elaboration"
        else:
            # Default to asking opinion
            plan.follow_up_goal = "ask_opinion"
        
        plan.follow_up_topic = current_topic
    
    def _should_take_initiative(
        self,
        plan: CommunicationPlan,
        curiosity: float,
        valence: float,
        user_intent: str,
    ) -> bool:
        """Decide whether to take conversational initiative."""
        # Always respond to questions without taking over
        if user_intent == "question":
            return False
        
        # Take initiative if curious and positive mood
        if curiosity > 0.7 and valence > 0:
            return True
        
        # Take initiative if they're just acknowledging
        if user_intent == "acknowledgment":
            return True
        
        # Take initiative if we have follow-up
        if plan.follow_up_goal:
            return True
        
        return False
    
    def _add_discourse_markers(
        self,
        plan: CommunicationPlan,
        topic_context: Optional[Dict],
        user_intent: str,
    ) -> None:
        """Select appropriate discourse markers."""
        markers = []
        
        # Acknowledgment markers for teaching
        if plan.primary_goal == "acknowledge_learning":
            markers.extend(random.sample(
                self._discourse_markers.get("acknowledgment", []), 
                min(1, len(self._discourse_markers.get("acknowledgment", [])))
            ))
        
        # Stance-based markers
        if plan.stance == "certain":
            markers.extend(random.sample(
                self._discourse_markers.get("certainty", []),
                min(1, len(self._discourse_markers.get("certainty", [])))
            ))
        elif plan.stance == "uncertain":
            markers.extend(random.sample(
                self._discourse_markers.get("uncertainty", []),
                min(1, len(self._discourse_markers.get("uncertainty", [])))
            ))
        elif plan.stance == "enthusiastic":
            markers.extend(random.sample(
                self._discourse_markers.get("enthusiasm", []),
                min(1, len(self._discourse_markers.get("enthusiasm", [])))
            ))
        
        # Topic continuation markers
        if topic_context and topic_context.get("topic_stack"):
            if random.random() > 0.7:  # Don't always use
                markers.extend(random.sample(
                    self._discourse_markers.get("topic_continuation", []),
                    min(1, len(self._discourse_markers.get("topic_continuation", [])))
                ))
        
        plan.discourse_markers = markers
    
    def _calculate_plan_confidence(
        self,
        plan: CommunicationPlan,
        working_memory: Dict[str, Any],
    ) -> float:
        """Calculate confidence in the communication plan."""
        confidence = 0.5  # Base
        
        # More content = more confidence
        content_count = (
            len(plan.content_concepts) + 
            len(plan.content_relations) + 
            len(plan.content_facts)
        )
        confidence += min(0.3, content_count * 0.1)
        
        # Clear goal = more confidence
        if plan.primary_goal not in ("neutral", "unknown"):
            confidence += 0.1
        
        # Follow-up planned = more confidence
        if plan.follow_up_goal:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def update_engagement(self, score: float) -> None:
        """Track engagement for initiative decisions."""
        self._engagement_history.append(score)
        if len(self._engagement_history) > 10:
            self._engagement_history.pop(0)
    
    def get_average_engagement(self) -> float:
        """Get average engagement score."""
        if not self._engagement_history:
            return 0.5
        return sum(self._engagement_history) / len(self._engagement_history)
    
    def generate_followup_question(
        self,
        goal: str,
        topic: str,
        related: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a follow-up question from templates."""
        templates = self._followup_templates.get(goal, [])
        if not templates:
            return None
        
        template = random.choice(templates)
        
        try:
            if "{related}" in template and related:
                return template.format(topic=topic, related=related)
            return template.format(topic=topic)
        except KeyError:
            return None
    
    def learn_discourse_marker(self, marker: str, function: str) -> None:
        """Learn a new discourse marker from input."""
        if function not in self._discourse_markers:
            self._discourse_markers[function] = []
        if marker not in self._discourse_markers[function]:
            self._discourse_markers[function].append(marker)
            logger.debug(f"Learned discourse marker '{marker}' for function '{function}'")
    
    def learn_followup_template(self, template: str, goal: str) -> None:
        """Learn a new follow-up template from observation."""
        if goal not in self._followup_templates:
            self._followup_templates[goal] = []
        if template not in self._followup_templates[goal]:
            self._followup_templates[goal].append(template)
            logger.debug(f"Learned follow-up template for goal '{goal}': {template}")
