"""
Automatic Feedback Detection for Lilith AI.

Analyzes user follow-up messages to automatically detect success/failure
signals and provide implicit feedback to learned patterns.

Success signals:
- Gratitude: "thanks", "thank you", "great", "perfect"
- Confirmation: "yes", "exactly", "correct", "right"
- Engagement: follow-up questions that build on the answer
- Topic continuation: discussing related concepts

Failure signals:
- Confusion: "what?", "huh?", "I don't understand"
- Corrections: "no", "wrong", "I meant", "actually"
- Repetition: asking the same question again
- Abandonment: sudden topic change (weak signal)
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple
import difflib


class FeedbackSignal(Enum):
    """Detected feedback signal type."""
    STRONG_POSITIVE = "strong_positive"   # Clear success indicators
    WEAK_POSITIVE = "weak_positive"       # Probable success
    NEUTRAL = "neutral"                   # No clear signal
    WEAK_NEGATIVE = "weak_negative"       # Probable failure
    STRONG_NEGATIVE = "strong_negative"   # Clear failure indicators


@dataclass
class FeedbackResult:
    """Result of feedback detection."""
    signal: FeedbackSignal
    confidence: float           # 0.0 to 1.0
    reason: str                 # Human-readable explanation
    strength: float             # Suggested reward/penalty strength (0.0 to 0.5)
    patterns_matched: List[str] # Which patterns triggered this detection
    
    @property
    def is_positive(self) -> bool:
        return self.signal in (FeedbackSignal.STRONG_POSITIVE, FeedbackSignal.WEAK_POSITIVE)
    
    @property
    def is_negative(self) -> bool:
        return self.signal in (FeedbackSignal.STRONG_NEGATIVE, FeedbackSignal.WEAK_NEGATIVE)
    
    @property
    def should_apply(self) -> bool:
        """Whether this feedback should be applied (skip neutral)."""
        return self.signal != FeedbackSignal.NEUTRAL


class FeedbackDetector:
    """
    Detects implicit feedback signals from user follow-up messages.
    
    Uses pattern matching and contextual analysis to determine if the user
    found the previous response helpful or not.
    """
    
    # Strong positive indicators - clear satisfaction
    STRONG_POSITIVE_PATTERNS = [
        r'\b(thanks|thank\s*you|thx|ty)\b',
        r'\b(perfect|exactly|precisely)\b',
        r'\b(great|excellent|awesome|amazing|wonderful)\b',
        r'\b(that\'?s?\s*(right|correct|it|perfect))\b',
        r'\b(yes[,!.\s]*(that\'?s?\s*)?(it|right|correct)?)\b',  # More flexible "yes" matching
        r'\b(makes?\s*sense)\b',
        r'\b(got\s*it|i\s*see|i\s*understand|now\s*i\s*(get|understand))\b',
        r'^(yes|yeah|yep|yup|right)[!.]*$',  # Standalone affirmations
        r'\b(helpful|useful)\b',
        r'👍|🎉|✅|💯|🙏',
    ]
    
    # Weak positive indicators - probable satisfaction
    WEAK_POSITIVE_PATTERNS = [
        r'^(ok|okay|cool|nice|good)\b',
        r'\b(alright|sure)\b',
        r'\b(interesting|neat)\b',
        r'^(ah|oh)[,!.\s]',  # "Ah, I see"
        r'\b(so\s+(you\'re|it\'s|that)\s+saying)\b',  # Paraphrasing = understanding
    ]
    
    # Strong negative indicators - clear dissatisfaction
    STRONG_NEGATIVE_PATTERNS = [
        r'\b(wrong|incorrect|false)\b',
        r'\b(no[,.]?\s*that\'?s?\s*(not|wrong))\b',
        r'\b(i\s*(meant|said|asked))\b',  # Correction
        r'\b(not\s+what\s+i\s+(meant|asked|wanted))\b',
        r'\b(completely|totally)\s+wrong\b',
        r'\b(makes?\s*no\s*sense)\b',
        r'\b(useless|unhelpful|terrible)\b',
        r'👎|❌|🚫|💩',
    ]
    
    # Weak negative indicators - probable confusion
    WEAK_NEGATIVE_PATTERNS = [
        r'^\s*(what|huh|eh)\s*\?+\s*$',  # Standalone confusion
        r'\b(i\s*don\'?t\s*(get|understand))\b',
        r'\b(confused|confusing)\b',
        r'\b(can\s*you\s*(explain|clarify|rephrase))\b',
        r'\b(what\s*do\s*you\s*mean)\b',
        r'\b(sorry[,]?\s*but)\b',  # Polite disagreement
        r'\b(actually[,]?)\s',  # Correction lead-in
        r'🤔|😕|🙄',
    ]
    
    # Question patterns that might indicate confusion
    CLARIFICATION_PATTERNS = [
        r'\b(what\s+is|what\'?s)\s+that\b',
        r'\b(can\s+you\s+be\s+more\s+specific)\b',
        r'\b(i\'?m\s+not\s+sure)\b',
        r'\b(in\s+other\s+words)\b',
    ]
    
    # Follow-up question patterns (generally positive - engagement)
    FOLLOWUP_QUESTION_PATTERNS = [
        r'\b(what\s+about)\b',
        r'\b(how\s+(about|does|do|would|can))\b',
        r'\b(can\s+you\s+(also|tell\s+me\s+more))\b',
        r'\b(and\s+what)\b',
        r'\b(so\s+(then|if|does))\b',
    ]
    
    def __init__(self, 
                 min_confidence: float = 0.4,
                 apply_threshold: float = 0.5,
                 strong_strength: float = 0.25,
                 weak_strength: float = 0.1):
        """
        Initialize feedback detector.
        
        Args:
            min_confidence: Minimum confidence to return non-neutral result
            apply_threshold: Minimum confidence to actually apply feedback
            strong_strength: Reward/penalty strength for strong signals
            weak_strength: Reward/penalty strength for weak signals
        """
        self.min_confidence = min_confidence
        self.apply_threshold = apply_threshold
        self.strong_strength = strong_strength
        self.weak_strength = weak_strength
        
        # Compile patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns."""
        self.compiled_strong_positive = [
            (re.compile(p, re.IGNORECASE), p) for p in self.STRONG_POSITIVE_PATTERNS
        ]
        self.compiled_weak_positive = [
            (re.compile(p, re.IGNORECASE), p) for p in self.WEAK_POSITIVE_PATTERNS
        ]
        self.compiled_strong_negative = [
            (re.compile(p, re.IGNORECASE), p) for p in self.STRONG_NEGATIVE_PATTERNS
        ]
        self.compiled_weak_negative = [
            (re.compile(p, re.IGNORECASE), p) for p in self.WEAK_NEGATIVE_PATTERNS
        ]
        self.compiled_clarification = [
            (re.compile(p, re.IGNORECASE), p) for p in self.CLARIFICATION_PATTERNS
        ]
        self.compiled_followup = [
            (re.compile(p, re.IGNORECASE), p) for p in self.FOLLOWUP_QUESTION_PATTERNS
        ]
    
    def detect(self, 
               current_input: str, 
               previous_input: str,
               previous_response: str) -> FeedbackResult:
        """
        Analyze current input for feedback signals about the previous response.
        
        Args:
            current_input: The user's current message
            previous_input: The user's previous message (that got the response)
            previous_response: Lilith's previous response
            
        Returns:
            FeedbackResult with detected signal and metadata
        """
        text = current_input.strip()
        
        # Skip if this looks like a command
        if text.startswith('/'):
            return FeedbackResult(
                signal=FeedbackSignal.NEUTRAL,
                confidence=0.0,
                reason="Command input (skipped)",
                strength=0.0,
                patterns_matched=[]
            )
        
        # Check for repetition (asking the same thing again = failure)
        repetition_score = self._check_repetition(text, previous_input)
        if repetition_score > 0.8:
            return FeedbackResult(
                signal=FeedbackSignal.STRONG_NEGATIVE,
                confidence=repetition_score,
                reason="Repeated question (likely didn't understand response)",
                strength=self.strong_strength,
                patterns_matched=["repetition_detected"]
            )
        elif repetition_score > 0.5:
            return FeedbackResult(
                signal=FeedbackSignal.WEAK_NEGATIVE,
                confidence=repetition_score,
                reason="Similar question repeated (possible confusion)",
                strength=self.weak_strength,
                patterns_matched=["similar_repetition"]
            )
        
        # Count pattern matches
        strong_pos_matches = self._match_patterns(text, self.compiled_strong_positive)
        weak_pos_matches = self._match_patterns(text, self.compiled_weak_positive)
        strong_neg_matches = self._match_patterns(text, self.compiled_strong_negative)
        weak_neg_matches = self._match_patterns(text, self.compiled_weak_negative)
        clarification_matches = self._match_patterns(text, self.compiled_clarification)
        followup_matches = self._match_patterns(text, self.compiled_followup)
        
        # Calculate scores
        positive_score = (
            len(strong_pos_matches) * 1.0 +
            len(weak_pos_matches) * 0.5 +
            len(followup_matches) * 0.3  # Follow-up questions = engagement
        )
        
        negative_score = (
            len(strong_neg_matches) * 1.0 +
            len(weak_neg_matches) * 0.6 +  # Raised from 0.5 - confusion is important
            len(clarification_matches) * 0.4  # Requests for clarification = weak negative
        )
        
        # Boost confidence for short, definitive responses (e.g., "What?", "Thanks!")
        text_len = len(text.split())
        short_response_boost = 1.0
        if text_len <= 3:
            # Short responses with feedback patterns are usually unambiguous
            short_response_boost = 1.8  # Boost confidence for short clear feedback
        elif text_len <= 6:
            short_response_boost = 1.3
        
        # Determine result
        all_matches = (
            strong_pos_matches + weak_pos_matches + 
            strong_neg_matches + weak_neg_matches +
            clarification_matches + followup_matches
        )
        
        if positive_score == 0 and negative_score == 0:
            # Check for topic continuation vs topic change
            topic_score = self._check_topic_continuation(
                text, previous_input, previous_response
            )
            if topic_score > 0.5:
                return FeedbackResult(
                    signal=FeedbackSignal.WEAK_POSITIVE,
                    confidence=topic_score * 0.5,  # Lower confidence for implicit
                    reason="Topic continuation (implicit success)",
                    strength=self.weak_strength * 0.5,
                    patterns_matched=["topic_continuation"]
                )
            return FeedbackResult(
                signal=FeedbackSignal.NEUTRAL,
                confidence=0.0,
                reason="No clear feedback signal",
                strength=0.0,
                patterns_matched=[]
            )
        
        # Calculate net score and confidence
        net_score = positive_score - negative_score
        total_matches = positive_score + negative_score
        
        # Base confidence from match count, boosted for short definitive responses
        base_confidence = min(1.0, total_matches / 2.0)  # Max at 2+ matches
        confidence = min(1.0, base_confidence * short_response_boost)
        
        if net_score > 0:
            if strong_pos_matches:
                signal = FeedbackSignal.STRONG_POSITIVE
                strength = self.strong_strength
                reason = f"Positive feedback detected: {', '.join(strong_pos_matches[:3])}"
            else:
                signal = FeedbackSignal.WEAK_POSITIVE
                strength = self.weak_strength
                reason = f"Probable positive feedback: {', '.join(weak_pos_matches[:3])}"
        elif net_score < 0:
            if strong_neg_matches:
                signal = FeedbackSignal.STRONG_NEGATIVE
                strength = self.strong_strength
                reason = f"Negative feedback detected: {', '.join(strong_neg_matches[:3])}"
            else:
                signal = FeedbackSignal.WEAK_NEGATIVE
                strength = self.weak_strength
                reason = f"Probable negative feedback: {', '.join(weak_neg_matches[:3])}"
        else:
            # Mixed signals - look at strong indicators
            if strong_pos_matches and not strong_neg_matches:
                signal = FeedbackSignal.WEAK_POSITIVE
                strength = self.weak_strength
                reason = "Mixed signals, but strong positive present"
            elif strong_neg_matches and not strong_pos_matches:
                signal = FeedbackSignal.WEAK_NEGATIVE
                strength = self.weak_strength
                reason = "Mixed signals, but strong negative present"
            else:
                signal = FeedbackSignal.NEUTRAL
                strength = 0.0
                reason = "Mixed signals, unclear feedback"
        
        # Apply confidence threshold
        if confidence < self.min_confidence:
            return FeedbackResult(
                signal=FeedbackSignal.NEUTRAL,
                confidence=confidence,
                reason=f"Low confidence ({confidence:.2f}): {reason}",
                strength=0.0,
                patterns_matched=all_matches
            )
        
        return FeedbackResult(
            signal=signal,
            confidence=confidence,
            reason=reason,
            strength=strength,
            patterns_matched=all_matches
        )
    
    def _match_patterns(self, text: str, patterns: List[Tuple[re.Pattern, str]]) -> List[str]:
        """Match text against compiled patterns, return matched pattern strings."""
        matches = []
        for pattern, pattern_str in patterns:
            if pattern.search(text):
                # Clean up pattern string for display
                clean = pattern_str.replace(r'\b', '').replace(r'\s*', ' ')
                clean = re.sub(r'[\\()[\]|+*?]', '', clean)
                matches.append(clean.strip())
        return matches
    
    def _check_repetition(self, current: str, previous: str) -> float:
        """
        Check if current input is a repetition of previous (indicating failure).
        
        Returns similarity score 0.0 to 1.0
        """
        # Normalize for comparison
        current_norm = self._normalize_for_comparison(current)
        previous_norm = self._normalize_for_comparison(previous)
        
        if not current_norm or not previous_norm:
            return 0.0
        
        # Use sequence matcher for similarity
        similarity = difflib.SequenceMatcher(
            None, current_norm, previous_norm
        ).ratio()
        
        return similarity
    
    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        # Remove common filler words
        fillers = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'please', 'can', 'you'}
        words = [w for w in text.split() if w not in fillers]
        return ' '.join(words)
    
    def _check_topic_continuation(self, 
                                   current: str, 
                                   previous_input: str,
                                   previous_response: str) -> float:
        """
        Check if current input continues the topic (positive signal).
        
        Returns a score 0.0 to 1.0 indicating topic continuation.
        """
        # Extract key terms from previous context
        context_terms = set()
        
        for text in [previous_input, previous_response]:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            # Filter common words
            common = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                     'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                     'have', 'been', 'were', 'they', 'this', 'that', 'with',
                     'what', 'when', 'where', 'which', 'while', 'your', 'from'}
            context_terms.update(w for w in words if w not in common)
        
        # Check how many context terms appear in current input
        current_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', current.lower()))
        
        if not context_terms or not current_words:
            return 0.0
        
        overlap = len(context_terms & current_words)
        max_possible = min(len(context_terms), 5)  # Cap at 5 for scoring
        
        return min(1.0, overlap / max_possible) if max_possible > 0 else 0.0
    
    def get_feedback_emoji(self, result: FeedbackResult) -> str:
        """Get an emoji representation of the feedback signal."""
        emoji_map = {
            FeedbackSignal.STRONG_POSITIVE: "✨",
            FeedbackSignal.WEAK_POSITIVE: "👍",
            FeedbackSignal.NEUTRAL: "➖",
            FeedbackSignal.WEAK_NEGATIVE: "🤔",
            FeedbackSignal.STRONG_NEGATIVE: "❌",
        }
        return emoji_map.get(result.signal, "❓")


class FeedbackTracker:
    """
    Tracks feedback history and applies it to learned patterns.
    
    Maintains a window of recent interactions to apply delayed feedback
    and track overall response quality.
    """
    
    def __init__(self, 
                 detector: Optional[FeedbackDetector] = None,
                 history_size: int = 10):
        """
        Initialize feedback tracker.
        
        Args:
            detector: FeedbackDetector instance (creates default if None)
            history_size: Number of recent interactions to track
        """
        self.detector = detector or FeedbackDetector()
        self.history_size = history_size
        
        # Interaction history: (user_input, response, pattern_id, feedback_applied)
        self.history: List[dict] = []
        
        # Cumulative stats
        self.total_positive = 0
        self.total_negative = 0
        self.total_neutral = 0
    
    def record_interaction(self, 
                           user_input: str, 
                           response_text: str,
                           pattern_id: Optional[str] = None):
        """Record an interaction for potential feedback tracking."""
        self.history.append({
            'user_input': user_input,
            'response': response_text,
            'pattern_id': pattern_id,
            'feedback_applied': False,
            'feedback_result': None
        })
        
        # Trim history
        while len(self.history) > self.history_size:
            self.history.pop(0)
    
    def check_feedback(self, current_input: str) -> Optional[Tuple[FeedbackResult, Optional[str]]]:
        """
        Check if current input contains feedback about the previous response.
        
        Returns:
            Tuple of (FeedbackResult, pattern_id) if feedback detected, else None
        """
        if len(self.history) < 1:
            return None
        
        last = self.history[-1]
        
        # Skip if we already applied feedback to this interaction
        if last['feedback_applied']:
            return None
        
        # Get context for detection
        previous_input = last['user_input']
        previous_response = last['response']
        
        # We also need the second-to-last input if available
        if len(self.history) >= 2:
            earlier_input = self.history[-2]['user_input']
        else:
            earlier_input = previous_input
        
        # Detect feedback
        result = self.detector.detect(
            current_input=current_input,
            previous_input=previous_input,
            previous_response=previous_response
        )
        
        # Mark as processed
        last['feedback_applied'] = True
        last['feedback_result'] = result
        
        # Update stats
        if result.is_positive:
            self.total_positive += 1
        elif result.is_negative:
            self.total_negative += 1
        else:
            self.total_neutral += 1
        
        if result.should_apply and result.confidence >= self.detector.apply_threshold:
            return (result, last['pattern_id'])
        
        return None
    
    def get_stats(self) -> dict:
        """Get feedback statistics."""
        total = self.total_positive + self.total_negative + self.total_neutral
        return {
            'total_interactions': total,
            'positive': self.total_positive,
            'negative': self.total_negative,
            'neutral': self.total_neutral,
            'positive_rate': self.total_positive / total if total > 0 else 0,
            'negative_rate': self.total_negative / total if total > 0 else 0,
        }
    
    def get_recent_feedback(self, n: int = 5) -> List[dict]:
        """Get the n most recent interactions with their feedback."""
        recent = self.history[-n:] if len(self.history) >= n else self.history
        return [
            {
                'input': h['user_input'][:50] + '...' if len(h['user_input']) > 50 else h['user_input'],
                'pattern_id': h['pattern_id'],
                'feedback': h['feedback_result'].signal.value if h['feedback_result'] else 'pending'
            }
            for h in recent
        ]
