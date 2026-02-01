import pytest
from unittest.mock import MagicMock
from v2.lilith_v2.pragmatic_system import PragmaticSystem, PragmaticTemplate

def test_teaching_detection_heuristic():
    """Test standard fallback -> teaching pattern."""
    p_sys = PragmaticSystem()
    
    # 1. Bot fails
    prev_response = "I don't know about that. I am listening."
    
    # 2. User teaches
    user_input = "A dog is a four-legged animal."
    
    is_teaching = p_sys.detect_teaching_intent(prev_response, user_input)
    assert is_teaching is True, "Should detect teaching intent after fallback"

def test_teaching_detection_negative():
    """Test normal conversation doesn't trigger teaching."""
    p_sys = PragmaticSystem()
    
    prev_response = "Hello! I am ready to help."
    user_input = "What time is it?"
    
    is_teaching = p_sys.detect_teaching_intent(prev_response, user_input)
    assert is_teaching is False, "Should not detect teaching intent"

def test_intent_filtering():
    """Test fetching template by intent."""
    p_sys = PragmaticSystem()
    
    # Register specific intent template
    p_sys.register(PragmaticTemplate(
        "test_ack", "teaching", "Got it: {subject}={object}", 
        ["subject", "object"], priority=10, intent="acknowledge_learning"
    ))
    
    template = p_sys.get_template("teaching", ["subject", "object"], intent="acknowledge_learning")
    assert template is not None
    assert template.template_id == "test_ack"
    
def test_evaluate_engagement():
    """Test the ported engagement scoring."""
    p_sys = PragmaticSystem()
    
    # Positive: Long + Curiosity
    pos_input = "That is very interesting, tell me how it works explicitly." 
    score = p_sys.evaluate_engagement(pos_input)
    assert score > 0.5
    
    # Negative: Confusion
    neg_input = "Huh? What?"
    score2 = p_sys.evaluate_engagement(neg_input)
    assert score2 < 0.5

def test_history_tracking():
    """Test that interactions are recorded."""
    p_sys = PragmaticSystem()
    
    p_sys.learn_from_interaction("Hi", "Hello", "greeting")
    assert len(p_sys.history) == 1
    assert p_sys.history[0].user_input == "Hi"
