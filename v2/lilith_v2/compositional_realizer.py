"""
Compositional Realization Stage for Lilith v2.

The "Composer" - builds language from concepts and relations, not templates.
This is where learning manifests as behavior - syntactic frames learned from
input are used to generate output.

Key principle: Compose like building with Lego
1. Select concepts to express
2. Retrieve relations between them  
3. Choose syntactic frame for relation type (LEARNED from input)
4. Apply stance modifiers and discourse markers
5. Ensure grammaticality
"""

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Import v1's rich pragmatic template system
try:
    from lilith.pragmatic_templates import PragmaticTemplateStore
    PRAGMATIC_TEMPLATES_AVAILABLE = True
except ImportError:
    PRAGMATIC_TEMPLATES_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SyntacticFrame:
    """A learned syntactic pattern for expressing a relation type."""
    frame_id: str
    predicate: str  # "is_a", "has_property", "part_of", etc.
    template: str   # "{subject} is a type of {object}"
    pos_sequence: List[str] = field(default_factory=list)  # Optional POS tags
    usage_count: int = 0
    success_rate: float = 0.5


@dataclass
class RealizationResult:
    """The output of compositional realization."""
    text: str
    frames_used: List[str]  # Frame IDs used
    confidence: float
    components: Dict[str, str] = field(default_factory=dict)  # For debugging


class CompositionalRealizer:
    """
    Builds language compositionally from communication plans.
    
    Unlike template-based generation, this:
    1. Uses LEARNED syntactic frames (not hard-coded)
    2. Composes from relations and concepts
    3. Applies stance and discourse markers dynamically
    4. Queries pattern store for learned responses (learning-driven)
    """
    
    def __init__(self, graph_store: Any = None, pattern_store: Any = None):
        self.graph = graph_store
        self.pattern_store = pattern_store  # For learned response patterns
        
        # Syntactic frames: predicate -> List[SyntacticFrame]
        # These should be learned from input and persisted to graph
        self._frames: Dict[str, List[SyntacticFrame]] = {}
        
        # Load v1's pragmatic templates if available
        self._pragmatic_store = None
        if PRAGMATIC_TEMPLATES_AVAILABLE:
            try:
                self._pragmatic_store = PragmaticTemplateStore()
                logger.debug("Loaded PragmaticTemplateStore from v1")
            except Exception as e:
                logger.debug(f"Could not load PragmaticTemplateStore: {e}")
        
        # Stance modifiers: affect how content is expressed
        self._stance_modifiers: Dict[str, List[str]] = self._bootstrap_stance_modifiers()
        
        # Connectors for multi-clause sentences
        self._connectors: Dict[str, List[str]] = self._bootstrap_connectors()
        
        # Acknowledgment phrases
        self._acknowledgments: Dict[str, List[str]] = self._bootstrap_acknowledgments()
        
        # Follow-up question frames
        self._followup_frames: Dict[str, List[str]] = self._bootstrap_followup_frames()
        
        # Load any persisted frames from graph
        self._hydrate_frames_from_graph()
        
        # Bootstrap minimal frames if empty
        if not self._frames:
            self._bootstrap_frames()
    
    def _bootstrap_stance_modifiers(self) -> Dict[str, List[str]]:
        """Initial stance modifiers - should be learned."""
        return {
            "certain": ["", "Indeed, ", "Yes, ", "Absolutely, "],
            "uncertain": ["I think ", "Perhaps ", "It seems ", "I believe "],
            "enthusiastic": ["How fascinating! ", "Interestingly, ", "Remarkably, "],
            "empathetic": ["I understand. ", "That makes sense. ", "I see. "],
            "neutral": [""],
            "warm": ["", ""],
            "humble": ["I see, ", "You're right, ", "Thank you for clarifying. "],
            "appreciative": ["Thank you! ", "I appreciate that. ", "Great, "],
        }
    
    def _bootstrap_connectors(self) -> Dict[str, List[str]]:
        """Connectors for joining clauses."""
        return {
            "addition": [" and ", ". Also, ", ". Additionally, ", ". Furthermore, "],
            "contrast": [" but ", ". However, ", ". On the other hand, "],
            "elaboration": [" which ", ". In fact, ", ". Specifically, "],
            "result": [" so ", ". Therefore, ", ". As a result, "],
            "sequence": [". Then, ", ". Next, ", ". After that, "],
        }
    
    def _bootstrap_acknowledgments(self) -> Dict[str, List[str]]:
        """Acknowledgment phrases by type."""
        return {
            "learning": [
                "I understand now. {content}",
                "I see, so {content}",
                "Got it. {content}",
                "I've learned that {content}",
            ],
            "greeting": [
                "Hello! {content}",
                "Hi there! {content}",
                "Hey! {content}",
            ],
            "farewell": [
                "Goodbye! {content}",
                "Take care! {content}",
                "See you! {content}",
            ],
            "thanks": [
                "You're welcome! {content}",
                "Happy to help! {content}",
                "Glad I could assist. {content}",
            ],
            "correction": [
                "I see, I was mistaken. {content}",
                "Thank you for the correction. {content}",
                "I stand corrected. {content}",
            ],
            "statement": [
                "That's interesting about {topic}!",
                "How fascinating! {topic}!",
                "I see, {topic}.",
                "That's exciting!",
                "Really? Tell me more!",
                "How wonderful!",
            ],
        }
    
    def _bootstrap_followup_frames(self) -> Dict[str, List[str]]:
        """Follow-up question frames."""
        return {
            "ask_experience": [
                "Have you encountered {topic} before?",
                "What's your experience with {topic}?",
                "Have you experienced {topic} yourself?",
            ],
            "ask_elaboration": [
                "Would you like to know more about {topic}?",
                "Is there anything specific about {topic} you're curious about?",
                "What aspect of {topic} interests you?",
            ],
            "ask_opinion": [
                "What do you think about {topic}?",
                "How do you feel about {topic}?",
                "What's your view on {topic}?",
            ],
        }
    
    def _bootstrap_frames(self) -> None:
        """Minimal syntactic frames - real frames should come from learning."""
        defaults = [
            # is_a frames
            SyntacticFrame("default_is_a_1", "is_a", "{subject} is a {object}"),
            SyntacticFrame("default_is_a_2", "is_a", "{subject} is a type of {object}"),
            SyntacticFrame("default_is_a_3", "is_a", "A {subject} is {object}"),
            
            # has_property frames  
            SyntacticFrame("default_has_prop_1", "has_property", "{subject} is {object}"),
            SyntacticFrame("default_has_prop_2", "has_property", "{subject} has {object}"),
            SyntacticFrame("default_has_prop_3", "has_property", "{subject} are known for {object}"),
            
            # part_of frames
            SyntacticFrame("default_part_of_1", "part_of", "{subject} is part of {object}"),
            SyntacticFrame("default_part_of_2", "part_of", "{subject} belongs to {object}"),
            
            # has_a frames
            SyntacticFrame("default_has_a_1", "has_a", "{subject} has {object}"),
            SyntacticFrame("default_has_a_2", "has_a", "{subject} contains {object}"),
            
            # related_to (generic)
            SyntacticFrame("default_related_1", "related_to", "{subject} is related to {object}"),
            SyntacticFrame("default_related_2", "related_to", "{subject} connects to {object}"),
        ]
        
        for frame in defaults:
            self._add_frame(frame)
    
    def _add_frame(self, frame: SyntacticFrame) -> None:
        """Add a frame to the collection."""
        if frame.predicate not in self._frames:
            self._frames[frame.predicate] = []
        
        # Check for duplicate templates
        existing_templates = [f.template for f in self._frames[frame.predicate]]
        if frame.template not in existing_templates:
            self._frames[frame.predicate].append(frame)
    
    def _hydrate_frames_from_graph(self) -> None:
        """Load learned frames from graph store."""
        if not self.graph:
            return
        
        try:
            # Query for syntax_pattern nodes
            patterns = self.graph.get_nodes_by_type("syntax_pattern")
            for node in patterns:
                data = node.get("data", {})
                predicate = data.get("predicate", "related_to")
                template = node.get("term", "")
                if template and "{subject}" in template and "{object}" in template:
                    frame = SyntacticFrame(
                        frame_id=node.get("node_id", f"learned_{len(self._frames)}"),
                        predicate=predicate,
                        template=template,
                        pos_sequence=data.get("pos_sequence", []),
                        usage_count=data.get("usage_count", 0),
                    )
                    self._add_frame(frame)
            logger.debug(f"Loaded {len(patterns)} syntactic frames from graph")
        except Exception as e:
            logger.debug(f"Could not load frames from graph: {e}")
    
    def _query_learned_pattern(
        self,
        intent: str,
        topic: str,
        context: Optional[Dict] = None
    ) -> Optional[RealizationResult]:
        """
        Query the pattern store for a learned response pattern.
        
        This is the learning-driven path - if we've seen successful responses
        for similar contexts, we reuse them.
        
        Args:
            intent: The communication intent (greet, acknowledge, inform, etc.)
            topic: Current topic of conversation
            context: Additional context
            
        Returns:
            RealizationResult if a learned pattern is found, None otherwise
        """
        if not self.pattern_store:
            return None
        
        try:
            # Build query from intent and topic
            query = f"{intent} {topic}" if topic else intent
            
            # Query pattern store for matches
            # Pattern store should have search/retrieve method
            if hasattr(self.pattern_store, 'search'):
                patterns = self.pattern_store.search(query, top_k=3)
            elif hasattr(self.pattern_store, 'get_by_intent'):
                patterns = self.pattern_store.get_by_intent(intent)
            elif hasattr(self.pattern_store, 'list'):
                # Fall back to listing all and filtering
                all_patterns = self.pattern_store.list()
                patterns = [p for p in all_patterns if p.get('intent') == intent][:3]
            else:
                return None
            
            if not patterns:
                return None
            
            # Select best pattern (highest success score)
            best = max(patterns, key=lambda p: p.get('success_score', 0.5))
            
            # Only use if success score is above threshold
            if best.get('success_score', 0.5) < 0.6:
                return None
            
            response_text = best.get('response_text', '')
            if not response_text:
                return None
            
            logger.debug(f"Using learned pattern: {best.get('fragment_id', 'unknown')}")
            
            return RealizationResult(
                text=response_text,
                frames_used=[f"learned:{best.get('fragment_id', 'pattern')}"],
                confidence=best.get('success_score', 0.7),
                components={'source': 'pattern_store', 'intent': intent}
            )
            
        except Exception as e:
            logger.debug(f"Pattern store query failed: {e}")
            return None
    
    def realize(
        self,
        plan: Any,  # CommunicationPlan
        topic_context: Optional[Dict] = None,
    ) -> RealizationResult:
        """
        Realize a communication plan as natural language.
        
        Process (learning-first):
        1. Try learned patterns from pattern store (accumulated successes)
        2. Fall back to pragmatic templates (static but rich)
        3. Fall back to hardcoded frames (bootstrap only)
        
        This prioritizes LEARNED behavior over hardcoded behavior.
        """
        components = {}
        frames_used = []
        
        # Get topic from plan
        topic = getattr(plan, 'topic', None) or ""
        if not topic and topic_context:
            topic = topic_context.get('current_topic', '')
        
        # Step 0: Try learned patterns first (learning-driven)
        # This is the key insight from V1's PragmaticLearner - use what worked before
        learned_result = self._query_learned_pattern(
            intent=plan.primary_goal,
            topic=topic,
            context={'plan': plan, 'topic_context': topic_context}
        )
        if learned_result and learned_result.confidence > 0.65:
            logger.debug(f"Using learned pattern for '{plan.primary_goal}'")
            return learned_result
        
        # Step 1: Handle special goals with fallback to templates
        if plan.primary_goal == "greet":
            return self._realize_greeting(plan)
        elif plan.primary_goal == "farewell":
            return self._realize_farewell(plan)
        elif plan.primary_goal == "acknowledge_learning":
            return self._realize_learning_ack(plan)
        elif plan.primary_goal == "accept_correction":
            return self._realize_correction_ack(plan)
        elif plan.primary_goal == "acknowledge":
            return self._realize_acknowledgment(plan)
        
        # Step 2: Build content from relations
        content_parts = []
        
        # Realize each relation using learned frames
        for subj, pred, obj in plan.content_relations:
            realized = self._realize_relation(subj, pred, obj)
            if realized:
                content_parts.append(realized["text"])
                frames_used.append(realized["frame_id"])
        
        # Add facts
        for fact in plan.content_facts[:2]:  # Limit facts
            content_parts.append(self._clean_fact(fact))
        
        # If no content yet, try to build from concepts alone
        if not content_parts and plan.content_concepts:
            concept = plan.content_concepts[0]
            content_parts.append(f"Regarding {concept}")
        
        # Join content with appropriate connectors
        if len(content_parts) > 1:
            content = self._join_clauses(content_parts, "addition")
        elif content_parts:
            content = content_parts[0]
        else:
            content = ""
        
        components["core_content"] = content
        
        # Step 3: Apply stance modifier
        stance_prefix = self._get_stance_modifier(plan.stance)
        components["stance"] = stance_prefix
        
        # Step 4: Prepend discourse markers
        discourse_prefix = ""
        if plan.discourse_markers:
            discourse_prefix = plan.discourse_markers[0]
        components["discourse"] = discourse_prefix
        
        # Step 5: Add follow-up question (only if we have content to follow up on)
        followup = ""
        if plan.follow_up_goal and plan.follow_up_topic and content:
            # Validate topic is a reasonable length (not a full sentence)
            if len(plan.follow_up_topic) < 30 and ' ' not in plan.follow_up_topic or len(plan.follow_up_topic.split()) <= 3:
                followup = self._generate_followup(
                    plan.follow_up_goal, 
                    plan.follow_up_topic
                )
        components["followup"] = followup
        
        # Step 6: Compose final text
        text = self._compose_final(
            discourse_prefix,
            stance_prefix,
            content,
            followup,
        )
        
        # Step 7: Clean up grammar
        text = self._cleanup_grammar(text)
        
        return RealizationResult(
            text=text,
            frames_used=frames_used,
            confidence=plan.confidence,
            components=components,
        )
    
    def _realize_relation(
        self, 
        subject: str, 
        predicate: str, 
        obj: str,
    ) -> Optional[Dict[str, str]]:
        """Realize a single relation using learned frames."""
        # Normalize predicate
        pred_normalized = predicate.lower().replace(" ", "_")
        
        # Find frames for this predicate
        frames = self._frames.get(pred_normalized, [])
        
        # Fall back to generic if not found
        if not frames:
            frames = self._frames.get("related_to", [])
        
        if not frames:
            # Ultimate fallback
            return {
                "text": f"{subject} is related to {obj}",
                "frame_id": "fallback",
            }
        
        # Select frame (prefer more used ones, but add randomness)
        frame = self._select_frame(frames)
        
        # Fill template
        try:
            text = frame.template.format(subject=subject, object=obj)
            # Capitalize first letter
            text = text[0].upper() + text[1:] if text else text
            
            # Update usage count
            frame.usage_count += 1
            
            return {
                "text": text,
                "frame_id": frame.frame_id,
            }
        except KeyError:
            return None
    
    def _select_frame(self, frames: List[SyntacticFrame]) -> SyntacticFrame:
        """Select a frame with preference for higher usage but some randomness."""
        if len(frames) == 1:
            return frames[0]
        
        # Weight by usage + random factor
        weights = []
        for frame in frames:
            weight = frame.usage_count + 1 + random.random() * 5
            weights.append(weight)
        
        total = sum(weights)
        r = random.random() * total
        
        cumulative = 0
        for frame, weight in zip(frames, weights):
            cumulative += weight
            if r <= cumulative:
                return frame
        
        return frames[-1]
    
    def _get_stance_modifier(self, stance: str) -> str:
        """Get a stance modifier phrase."""
        modifiers = self._stance_modifiers.get(stance, [""])
        return random.choice(modifiers)
    
    def _join_clauses(self, clauses: List[str], connection_type: str) -> str:
        """Join multiple clauses with appropriate connectors."""
        if len(clauses) == 1:
            return clauses[0]
        
        connectors = self._connectors.get(connection_type, [". "])
        connector = random.choice(connectors)
        
        # Join with connector (handle punctuation)
        result = clauses[0]
        for clause in clauses[1:]:
            # Remove redundant periods
            if result.endswith(".") and connector.startswith("."):
                result = result[:-1]
            result += connector + clause
        
        return result
    
    def _generate_followup(self, goal: str, topic: str) -> str:
        """Generate a follow-up question."""
        frames = self._followup_frames.get(goal, [])
        if not frames:
            return ""
        
        template = random.choice(frames)
        try:
            return template.format(topic=topic)
        except KeyError:
            return ""
    
    def _compose_final(
        self,
        discourse: str,
        stance: str,
        content: str,
        followup: str,
    ) -> str:
        """Compose the final response."""
        parts = []
        
        # Discourse + stance + content
        if content:
            main = f"{discourse}{stance}{content}"
            # Ensure ends with period
            if main and not main.endswith((".", "!", "?")):
                main += "."
            parts.append(main)
        
        # Follow-up
        if followup:
            parts.append(followup)
        
        return " ".join(parts).strip()
    
    def _cleanup_grammar(self, text: str) -> str:
        """Clean up grammatical issues."""
        if not text:
            return text
        
        # Fix double spaces
        text = re.sub(r"\s+", " ", text)
        
        # Fix double periods
        text = re.sub(r"\.+", ".", text)
        
        # Fix space before punctuation
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        
        # Ensure capitalization after sentence-ending punctuation
        def cap_after_punct(match):
            return match.group(1) + " " + match.group(2).upper()
        text = re.sub(r"([.!?])\s+([a-z])", cap_after_punct, text)
        
        # Capitalize first character
        if text:
            text = text[0].upper() + text[1:]
        
        return text.strip()
    
    def _clean_fact(self, fact: str) -> str:
        """Clean a fact for inclusion."""
        # Truncate if too long (first sentence)
        sentences = fact.split(".")
        if sentences:
            clean = sentences[0].strip()
            if len(clean) > 150:
                clean = clean[:147] + "..."
            return clean
        return fact[:150]
    
    def _realize_greeting(self, plan: Any) -> RealizationResult:
        """Realize a greeting."""
        templates = self._acknowledgments.get("greeting", ["Hello!"])
        template = random.choice(templates)
        
        content = ""
        if plan.content_facts:
            content = plan.content_facts[0]
        
        text = template.format(content=content).strip()
        if text.endswith("{content}"):
            text = text.replace("{content}", "").strip()
        
        return RealizationResult(
            text=text or "Hello!",
            frames_used=["greeting"],
            confidence=0.9,
        )
    
    def _realize_farewell(self, plan: Any) -> RealizationResult:
        """Realize a farewell."""
        templates = self._acknowledgments.get("farewell", ["Goodbye!"])
        template = random.choice(templates)
        text = template.format(content="").strip()
        
        return RealizationResult(
            text=text or "Goodbye!",
            frames_used=["farewell"],
            confidence=0.9,
        )
    
    def _realize_learning_ack(self, plan: Any) -> RealizationResult:
        """
        Realize acknowledgment of learning.
        
        Philosophy: Simple response - learning will discover what works.
        """
        simple_learning_acks = [
            "I've noted that.",
            "Thank you for sharing.",
            "I understand.",
            "Got it, thanks.",
            "Interesting, I'll remember that.",
        ]
        
        return RealizationResult(
            text=random.choice(simple_learning_acks),
            frames_used=["simple_learning_ack"],
            confidence=0.6,  # Lower confidence encourages learning
        )
    
    def _realize_correction_ack(self, plan: Any) -> RealizationResult:
        """Realize acknowledgment of correction."""
        templates = self._acknowledgments.get("correction", [])
        
        content = ""
        for subj, pred, obj in plan.content_relations:
            content = f"{subj} is {obj}"
            break
        
        if templates:
            template = random.choice(templates)
            text = template.format(content=content)
        else:
            text = f"I see, thank you for the correction. {content}."
        
        return RealizationResult(
            text=text,
            frames_used=["correction_ack"],
            confidence=0.8,
        )
    
    def _realize_acknowledgment(self, plan: Any) -> RealizationResult:
        """
        Realize a general acknowledgment.
        
        Philosophy: Keep this SIMPLE. The learning system will discover
        what acknowledgments work best in different contexts. We just
        provide a reasonable fallback.
        """
        # Simple acknowledgments - let learning discover better ones
        simple_acks = [
            "I see.",
            "Got it.",
            "Understood.",
            "I understand.",
            "Interesting.",
        ]
        
        return RealizationResult(
            text=random.choice(simple_acks),
            frames_used=["simple_ack"],
            confidence=0.6,  # Lower confidence encourages learning
        )
    
    # ===== Learning Methods =====
    
    def learn_frame(
        self,
        predicate: str,
        template: str,
        pos_sequence: Optional[List[str]] = None,
    ) -> None:
        """
        Learn a new syntactic frame from observed input.
        
        This is where learning manifests as behavior -
        frames learned here get used in realize().
        """
        frame_id = f"learned_{predicate}_{len(self._frames.get(predicate, []))}"
        
        frame = SyntacticFrame(
            frame_id=frame_id,
            predicate=predicate,
            template=template,
            pos_sequence=pos_sequence or [],
            usage_count=0,
            success_rate=0.5,
        )
        
        self._add_frame(frame)
        
        # Persist to graph if available
        if self.graph:
            try:
                self.graph.add_node(
                    node_id=frame_id,
                    node_type="syntax_pattern",
                    term=template,
                    confidence=0.5,
                    data={
                        "predicate": predicate,
                        "pos_sequence": pos_sequence or [],
                        "usage_count": 0,
                    }
                )
            except Exception as e:
                logger.debug(f"Could not persist frame: {e}")
        
        logger.info(f"Learned syntactic frame for '{predicate}': {template}")
    
    def learn_stance_modifier(self, stance: str, modifier: str) -> None:
        """Learn a stance modifier from observation."""
        if stance not in self._stance_modifiers:
            self._stance_modifiers[stance] = []
        if modifier not in self._stance_modifiers[stance]:
            self._stance_modifiers[stance].append(modifier)
            logger.debug(f"Learned stance modifier for '{stance}': {modifier}")
    
    def learn_connector(self, connection_type: str, connector: str) -> None:
        """Learn a clause connector from observation."""
        if connection_type not in self._connectors:
            self._connectors[connection_type] = []
        if connector not in self._connectors[connection_type]:
            self._connectors[connection_type].append(connector)
            logger.debug(f"Learned connector for '{connection_type}': {connector}")
    
    def update_frame_success(self, frame_id: str, success: bool) -> None:
        """Update success rate of a frame based on feedback."""
        for pred_frames in self._frames.values():
            for frame in pred_frames:
                if frame.frame_id == frame_id:
                    alpha = 0.1  # Learning rate
                    target = 1.0 if success else 0.0
                    frame.success_rate = (
                        (1 - alpha) * frame.success_rate + alpha * target
                    )
                    return
    
    def get_frame_stats(self) -> Dict[str, Any]:
        """Get statistics about learned frames."""
        total_frames = sum(len(f) for f in self._frames.values())
        predicates = list(self._frames.keys())
        
        learned_count = sum(
            1 for frames in self._frames.values()
            for f in frames if f.frame_id.startswith("learned_")
        )
        
        return {
            "total_frames": total_frames,
            "predicates": predicates,
            "learned_frames": learned_count,
            "stance_modifiers": len(self._stance_modifiers),
            "connectors": len(self._connectors),
        }
