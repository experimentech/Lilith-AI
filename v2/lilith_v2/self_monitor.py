"""
Self-Monitoring Stage for Lilith v2.

The "Editor" - evaluates responses before output and triggers revision if needed.

Checks:
- Coherence: Does this fit the conversation?
- Informativeness: Am I adding value?
- Appropriateness: Right tone/length?
- Completeness: Did I address the need?
- Repetition: Have I said this recently?
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class MonitoringResult:
    """Result of self-monitoring evaluation."""
    score: float  # 0.0 to 1.0
    passed: bool
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class SelfMonitor:
    """
    Evaluates response quality before output.
    
    This creates a "generate → evaluate → revise" loop where
    responses are checked against quality criteria.
    """
    
    def __init__(
        self,
        min_score: float = 0.5,
        recent_responses_size: int = 10,
        min_length: int = 5,
        max_length: int = 300,
    ):
        self.min_score = min_score
        self.min_length = min_length
        self.max_length = max_length
        
        # Track recent responses for repetition detection
        self._recent_responses: deque = deque(maxlen=recent_responses_size)
        self._recent_phrases: Set[str] = set()
        
        # Track topics discussed
        self._discussed_topics: Set[str] = set()
    
    def evaluate(
        self,
        draft: str,
        plan: Any,  # CommunicationPlan
        dialogue_state: Any,  # DialogueState
        user_input: str,
    ) -> MonitoringResult:
        """
        Evaluate a draft response against quality criteria.
        
        Returns:
            MonitoringResult with score, pass/fail, and issues
        """
        issues = []
        suggestions = []
        score = 1.0  # Start perfect, deduct for issues
        
        # Check 1: Length appropriateness
        length_result = self._check_length(draft, plan)
        score -= length_result["penalty"]
        if length_result["issue"]:
            issues.append(length_result["issue"])
            suggestions.append(length_result["suggestion"])
        
        # Check 2: Coherence with context
        coherence_result = self._check_coherence(draft, dialogue_state, user_input)
        score -= coherence_result["penalty"]
        if coherence_result["issue"]:
            issues.append(coherence_result["issue"])
            suggestions.append(coherence_result["suggestion"])
        
        # Check 3: Informativeness
        info_result = self._check_informativeness(draft, plan, user_input)
        score -= info_result["penalty"]
        if info_result["issue"]:
            issues.append(info_result["issue"])
            suggestions.append(info_result["suggestion"])
        
        # Check 4: Repetition
        repeat_result = self._check_repetition(draft)
        score -= repeat_result["penalty"]
        if repeat_result["issue"]:
            issues.append(repeat_result["issue"])
            suggestions.append(repeat_result["suggestion"])
        
        # Check 5: Obligation fulfillment
        oblig_result = self._check_obligations(draft, plan, dialogue_state)
        score -= oblig_result["penalty"]
        if oblig_result["issue"]:
            issues.append(oblig_result["issue"])
            suggestions.append(oblig_result["suggestion"])
        
        # Check 6: Grammar/fluency (basic)
        grammar_result = self._check_grammar(draft)
        score -= grammar_result["penalty"]
        if grammar_result["issue"]:
            issues.append(grammar_result["issue"])
        
        # Clamp score
        score = max(0.0, min(1.0, score))
        passed = score >= self.min_score
        
        if issues:
            logger.debug(f"Self-monitoring found {len(issues)} issues, score={score:.2f}")
        
        return MonitoringResult(
            score=score,
            passed=passed,
            issues=issues,
            suggestions=suggestions,
        )
    
    def _check_length(self, draft: str, plan: Any) -> Dict[str, Any]:
        """Check if response length is appropriate."""
        words = len(draft.split())
        
        # Too short
        if words < 3 and plan.primary_goal not in ("greet", "farewell", "acknowledge"):
            return {
                "penalty": 0.3,
                "issue": "Response too short",
                "suggestion": "Add more detail or follow-up",
            }
        
        # Too long
        if len(draft) > self.max_length:
            return {
                "penalty": 0.2,
                "issue": "Response too long",
                "suggestion": "Condense to key points",
            }
        
        return {"penalty": 0.0, "issue": None, "suggestion": None}
    
    def _check_coherence(
        self, 
        draft: str, 
        dialogue_state: Any,
        user_input: str,
    ) -> Dict[str, Any]:
        """Check if response fits the conversation context."""
        issues = []
        penalty = 0.0
        
        # Check topic relevance
        if dialogue_state and hasattr(dialogue_state, 'current_topic'):
            topic = dialogue_state.current_topic
            if topic and len(topic) > 2:
                # Response should reference the topic or related concepts
                topic_lower = topic.lower()
                draft_lower = draft.lower()
                
                # Very loose check - topic word should appear OR response should be short ack
                if topic_lower not in draft_lower:
                    # Not necessarily wrong if it's a short acknowledgment
                    words = len(draft.split())
                    if words > 10:
                        penalty += 0.1
                        issues.append("Response may not address the current topic")
        
        # Check for greeting coherence
        if dialogue_state and hasattr(dialogue_state, 'phase'):
            phase = dialogue_state.phase
            if phase != "opening" and any(g in draft.lower() for g in ["hello", "hi there", "hey"]):
                penalty += 0.2
                issues.append("Greeting seems out of place mid-conversation")
        
        # Check for farewell coherence
        if dialogue_state and hasattr(dialogue_state, 'phase'):
            if dialogue_state.phase != "closing":
                if any(f in draft.lower() for f in ["goodbye", "bye", "farewell"]):
                    # User didn't say goodbye - we shouldn't either
                    user_lower = user_input.lower()
                    if not any(f in user_lower for f in ["bye", "goodbye", "farewell"]):
                        penalty += 0.3
                        issues.append("Farewell without user indicating departure")
        
        issue_text = "; ".join(issues) if issues else None
        
        return {
            "penalty": penalty,
            "issue": issue_text,
            "suggestion": "Ensure response addresses conversation context" if issues else None,
        }
    
    def _check_informativeness(
        self, 
        draft: str, 
        plan: Any,
        user_input: str,
    ) -> Dict[str, Any]:
        """Check if response adds value."""
        draft_lower = draft.lower()
        
        # Generic/empty responses
        generic_phrases = [
            "i am listening",
            "please provide more context",
            "i don't know",
            "i'm not sure what you mean",
        ]
        
        for phrase in generic_phrases:
            if phrase in draft_lower:
                # This is okay sometimes, but check if we have content to share
                if plan.content_relations or plan.content_facts:
                    return {
                        "penalty": 0.4,
                        "issue": "Generic response despite having content available",
                        "suggestion": "Use available content instead of generic fallback",
                    }
        
        # Check if response is just echoing input
        user_words = set(user_input.lower().split())
        draft_words = set(draft_lower.split())
        
        if len(user_words) > 3:
            overlap = len(user_words & draft_words) / len(user_words)
            if overlap > 0.8:
                return {
                    "penalty": 0.2,
                    "issue": "Response mostly echoes user input",
                    "suggestion": "Add new information or perspective",
                }
        
        return {"penalty": 0.0, "issue": None, "suggestion": None}
    
    def _check_repetition(self, draft: str) -> Dict[str, Any]:
        """Check if we're repeating ourselves."""
        draft_lower = draft.lower().strip()
        
        # Exact repetition
        for recent in self._recent_responses:
            if draft_lower == recent.lower():
                return {
                    "penalty": 0.5,
                    "issue": "Exact repetition of recent response",
                    "suggestion": "Vary response or add new information",
                }
        
        # High similarity (substring)
        for recent in self._recent_responses:
            recent_lower = recent.lower()
            # Check if significant overlap
            if len(draft_lower) > 20 and len(recent_lower) > 20:
                if draft_lower in recent_lower or recent_lower in draft_lower:
                    return {
                        "penalty": 0.3,
                        "issue": "Very similar to recent response",
                        "suggestion": "Provide variation in phrasing",
                    }
        
        # Phrase repetition (key phrases used too often)
        # Extract key phrases (3-grams)
        words = draft_lower.split()
        if len(words) >= 3:
            for i in range(len(words) - 2):
                phrase = " ".join(words[i:i+3])
                if phrase in self._recent_phrases:
                    # Not a hard issue, just note it
                    pass  # Could add soft penalty here
        
        return {"penalty": 0.0, "issue": None, "suggestion": None}
    
    def _check_obligations(
        self, 
        draft: str, 
        plan: Any,
        dialogue_state: Any,
    ) -> Dict[str, Any]:
        """Check if response fulfills conversational obligations."""
        if not dialogue_state:
            return {"penalty": 0.0, "issue": None, "suggestion": None}
        
        obligations = getattr(dialogue_state, 'pending_obligations', [])
        if not obligations:
            return {"penalty": 0.0, "issue": None, "suggestion": None}
        
        draft_lower = draft.lower()
        
        # Check each obligation
        unfulfilled = []
        
        for obligation in obligations:
            if obligation == "answer_question":
                # Should contain some informative content
                if "?" in draft and "i don't" not in draft_lower:
                    continue  # Might be asking for clarification - okay
                if len(draft.split()) < 5:
                    unfulfilled.append("answer_question")
            
            elif obligation == "reciprocate_greeting":
                greetings = ["hello", "hi", "hey", "greetings"]
                if not any(g in draft_lower for g in greetings):
                    unfulfilled.append("reciprocate_greeting")
            
            elif obligation == "acknowledge_teaching":
                ack_phrases = ["understand", "learned", "got it", "i see", "thank"]
                if not any(a in draft_lower for a in ack_phrases):
                    unfulfilled.append("acknowledge_teaching")
        
        if unfulfilled:
            return {
                "penalty": 0.25 * len(unfulfilled),
                "issue": f"Unfulfilled obligations: {', '.join(unfulfilled)}",
                "suggestion": f"Address: {', '.join(unfulfilled)}",
            }
        
        return {"penalty": 0.0, "issue": None, "suggestion": None}
    
    def _check_grammar(self, draft: str) -> Dict[str, Any]:
        """Basic grammar and fluency check."""
        issues = []
        penalty = 0.0
        
        # Double spaces
        if "  " in draft:
            penalty += 0.05
            issues.append("Double spaces")
        
        # Double punctuation (except ...)
        if re.search(r"[.!?]{2}", draft) and "..." not in draft:
            penalty += 0.05
            issues.append("Double punctuation")
        
        # Missing capital after sentence end
        if re.search(r"[.!?]\s+[a-z]", draft):
            penalty += 0.05
            issues.append("Missing capitalization")
        
        # Unclosed parentheses/brackets
        if draft.count("(") != draft.count(")"):
            penalty += 0.1
            issues.append("Unbalanced parentheses")
        if draft.count("[") != draft.count("]"):
            penalty += 0.1
            issues.append("Unbalanced brackets")
        
        issue_text = "; ".join(issues) if issues else None
        
        return {
            "penalty": penalty,
            "issue": issue_text,
            "suggestion": None,
        }
    
    def record_response(self, response: str, topic: Optional[str] = None) -> None:
        """Record a response for future repetition detection."""
        self._recent_responses.append(response)
        
        # Extract and record key phrases
        words = response.lower().split()
        if len(words) >= 3:
            for i in range(len(words) - 2):
                phrase = " ".join(words[i:i+3])
                self._recent_phrases.add(phrase)
        
        # Limit phrase cache
        if len(self._recent_phrases) > 100:
            # Remove oldest (arbitrary selection)
            self._recent_phrases = set(list(self._recent_phrases)[:75])
        
        # Track topics
        if topic:
            self._discussed_topics.add(topic.lower())
    
    def suggest_revision(
        self, 
        draft: str, 
        result: MonitoringResult,
        plan: Any,
    ) -> str:
        """
        Attempt to revise a response to address issues.
        
        This is a simple revision system - more sophisticated
        revision would use the same generation pipeline with adjusted parameters.
        """
        revised = draft
        
        for issue, suggestion in zip(result.issues, result.suggestions):
            if not suggestion:
                continue
            
            # Handle specific issues
            if "too short" in issue.lower():
                # Try to add a follow-up if we have one planned
                if plan.follow_up_goal and plan.follow_up_topic:
                    followup = f"What do you think about {plan.follow_up_topic}?"
                    if followup not in revised:
                        revised = revised.rstrip(".")
                        revised += f". {followup}"
            
            elif "too long" in issue.lower():
                # Truncate to first two sentences
                sentences = re.split(r'[.!?]+', revised)
                if len(sentences) > 2:
                    revised = ". ".join(sentences[:2]) + "."
            
            elif "generic response" in issue.lower():
                # We have content but used generic fallback - can't easily fix here
                # Would need to re-run generation
                pass
            
            elif "repetition" in issue.lower():
                # Try adding a slight variation
                variations = [
                    ("I understand", "I see"),
                    ("I see", "Got it"),
                    ("That's interesting", "How fascinating"),
                ]
                for old, new in variations:
                    if old in revised:
                        revised = revised.replace(old, new, 1)
                        break
        
        # Final cleanup
        revised = re.sub(r"\s+", " ", revised)
        revised = re.sub(r"\.+", ".", revised)
        
        return revised.strip()
    
    def reset(self) -> None:
        """Reset monitoring state for new conversation."""
        self._recent_responses.clear()
        self._recent_phrases.clear()
        self._discussed_topics.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "recent_responses_tracked": len(self._recent_responses),
            "recent_phrases_tracked": len(self._recent_phrases),
            "topics_discussed": len(self._discussed_topics),
        }
