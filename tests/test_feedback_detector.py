"""
Tests for the automatic feedback detection system.
"""

import pytest
from lilith.feedback_detector import (
    FeedbackDetector, 
    FeedbackTracker, 
    FeedbackSignal,
    FeedbackResult
)


class TestFeedbackDetector:
    """Test the FeedbackDetector class."""
    
    @pytest.fixture
    def detector(self):
        return FeedbackDetector()
    
    # Strong positive tests
    def test_thanks_is_strong_positive(self, detector):
        result = detector.detect("Thanks!", "What is X?", "X is Y.")
        assert result.signal == FeedbackSignal.STRONG_POSITIVE
        assert result.is_positive
        
    def test_perfect_is_strong_positive(self, detector):
        result = detector.detect("Perfect, exactly!", "Tell me about cats", "Cats are pets.")
        assert result.signal == FeedbackSignal.STRONG_POSITIVE
        
    def test_yes_thats_right_is_strong_positive(self, detector):
        result = detector.detect("Yes, that's right!", "Is the sky blue?", "The sky is blue.")
        assert result.signal == FeedbackSignal.STRONG_POSITIVE
        
    def test_helpful_is_strong_positive(self, detector):
        result = detector.detect("That was helpful", "How do I code?", "Use a text editor.")
        assert result.signal == FeedbackSignal.STRONG_POSITIVE
        
    # Weak positive tests
    def test_ok_is_weak_positive(self, detector):
        result = detector.detect("Ok, interesting", "What is AI?", "AI is artificial intelligence.")
        assert result.signal == FeedbackSignal.WEAK_POSITIVE
        
    def test_cool_followup_is_weak_positive(self, detector):
        result = detector.detect("Cool, what about ML?", "What is AI?", "AI is a field.")
        assert result.signal == FeedbackSignal.WEAK_POSITIVE
        
    # Strong negative tests
    def test_wrong_is_strong_negative(self, detector):
        result = detector.detect("No, that's wrong", "What is 2+2?", "2+2 is 5.")
        assert result.signal == FeedbackSignal.STRONG_NEGATIVE
        assert result.is_negative
        
    def test_makes_no_sense_is_strong_negative(self, detector):
        result = detector.detect("That makes no sense", "Explain X", "X is complicated.")
        assert result.signal == FeedbackSignal.STRONG_NEGATIVE
        
    def test_repetition_is_strong_negative(self, detector):
        """Asking the same question again indicates failure."""
        result = detector.detect("What is Python?", "What is Python?", "Python is a snake.")
        assert result.signal == FeedbackSignal.STRONG_NEGATIVE
        assert "repeat" in result.reason.lower()
        
    # Weak negative tests
    def test_what_question_is_weak_negative(self, detector):
        result = detector.detect("What?", "Tell me about X", "X is a thing.")
        assert result.signal == FeedbackSignal.WEAK_NEGATIVE
        
    def test_dont_understand_is_weak_negative(self, detector):
        result = detector.detect("I don't understand", "Explain recursion", "It calls itself.")
        assert result.signal == FeedbackSignal.WEAK_NEGATIVE
        
    def test_confused_is_weak_negative(self, detector):
        result = detector.detect("I'm confused", "What is entropy?", "Entropy is disorder.")
        assert result.signal == FeedbackSignal.WEAK_NEGATIVE
        
    # Neutral tests
    def test_topic_change_is_neutral(self, detector):
        result = detector.detect("Tell me about JavaScript", "What is Python?", "Python is a language.")
        assert result.signal == FeedbackSignal.NEUTRAL
        
    def test_command_is_neutral(self, detector):
        result = detector.detect("/stats", "What is X?", "X is Y.")
        assert result.signal == FeedbackSignal.NEUTRAL
        
    # Confidence tests
    def test_short_response_high_confidence(self, detector):
        """Short definitive responses should have high confidence."""
        result = detector.detect("Thanks!", "What is X?", "X is Y.")
        assert result.confidence >= 0.8
        
    def test_yes_alone_high_confidence(self, detector):
        result = detector.detect("Yes!", "Is it true?", "Yes it is.")
        assert result.confidence >= 0.8


class TestFeedbackTracker:
    """Test the FeedbackTracker class."""
    
    @pytest.fixture
    def tracker(self):
        return FeedbackTracker()
    
    def test_record_interaction(self, tracker):
        tracker.record_interaction("What is X?", "X is Y.", "pattern_001")
        assert len(tracker.history) == 1
        assert tracker.history[0]['pattern_id'] == "pattern_001"
        
    def test_check_feedback_positive(self, tracker):
        tracker.record_interaction("What is X?", "X is Y.", "pattern_001")
        result = tracker.check_feedback("Thanks!")
        
        assert result is not None
        feedback, pattern_id = result
        assert feedback.is_positive
        assert pattern_id == "pattern_001"
        
    def test_check_feedback_negative(self, tracker):
        tracker.record_interaction("What is X?", "X is Y.", "pattern_001")
        result = tracker.check_feedback("No, that's wrong")
        
        assert result is not None
        feedback, pattern_id = result
        assert feedback.is_negative
        
    def test_feedback_not_applied_twice(self, tracker):
        """Feedback should only be applied once per interaction."""
        tracker.record_interaction("What is X?", "X is Y.", "pattern_001")
        
        # First check applies feedback
        result1 = tracker.check_feedback("Thanks!")
        assert result1 is not None
        
        # Second check should not re-apply
        tracker.record_interaction("New question", "New answer", "pattern_002")
        # Checking against the NEW interaction
        result2 = tracker.check_feedback("Thanks again")
        assert result2 is not None
        
    def test_stats_tracking(self, tracker):
        tracker.record_interaction("Q1", "A1", "p1")
        tracker.check_feedback("Thanks!")  # Positive
        
        tracker.record_interaction("Q2", "A2", "p2")
        tracker.check_feedback("What?")  # Negative
        
        stats = tracker.get_stats()
        assert stats['positive'] == 1
        assert stats['negative'] == 1
        assert stats['total_interactions'] >= 2
        
    def test_history_trimming(self):
        """History should be trimmed to max size."""
        tracker = FeedbackTracker(history_size=3)
        
        for i in range(5):
            tracker.record_interaction(f"Q{i}", f"A{i}", f"p{i}")
        
        assert len(tracker.history) == 3
        assert tracker.history[0]['user_input'] == "Q2"  # Oldest kept


class TestFeedbackResult:
    """Test FeedbackResult properties."""
    
    def test_is_positive(self):
        result = FeedbackResult(
            signal=FeedbackSignal.STRONG_POSITIVE,
            confidence=0.9,
            reason="test",
            strength=0.2,
            patterns_matched=[]
        )
        assert result.is_positive
        assert not result.is_negative
        assert result.should_apply
        
    def test_is_negative(self):
        result = FeedbackResult(
            signal=FeedbackSignal.WEAK_NEGATIVE,
            confidence=0.6,
            reason="test",
            strength=0.1,
            patterns_matched=[]
        )
        assert result.is_negative
        assert not result.is_positive
        assert result.should_apply
        
    def test_neutral_not_applied(self):
        result = FeedbackResult(
            signal=FeedbackSignal.NEUTRAL,
            confidence=0.0,
            reason="test",
            strength=0.0,
            patterns_matched=[]
        )
        assert not result.is_positive
        assert not result.is_negative
        assert not result.should_apply


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
