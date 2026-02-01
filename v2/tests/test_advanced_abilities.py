import pytest
from unittest.mock import MagicMock, patch
import torch
from v2.lilith_v2.cognitive_stage import CognitiveStage
from v2.lilith_v2.knowledge_service import KnowledgeService, KnowledgeFragment
from v2.lilith_v2.affective_system import AffectiveSystem

class MockEncoder:
    def encode(self, text):
        return torch.tensor([0.1, 0.2, 0.3])

@pytest.fixture
def advanced_stage():
    pmflow = MagicMock()
    graph = MagicMock()
    encoder = MockEncoder()
    
    # Mock knowledge service to avoid network calls
    knowledge_service = MagicMock(spec=KnowledgeService)
    knowledge_service.enabled = True
    knowledge_service.search.return_value = [
        KnowledgeFragment(
            source="wiki", 
            content="Python is a language.", 
            identifier="Python", 
            confidence=0.9
        )
    ]
    
    # Real affective system for testing logic
    affective_system = AffectiveSystem()
    
    stage = CognitiveStage(
        node_id="test_node",
        pmflow_store=pmflow,
        graph_store=graph,
        encoder=encoder,
        knowledge_service=knowledge_service,
        affective_system=affective_system
    )
    return stage, knowledge_service

def test_knowledge_gap_filling(advanced_stage):
    stage, mocked_ks = advanced_stage
    
    # Case 1: Grounding finds nothing (force empty return)
    stage._ground_vector_to_concepts = MagicMock(return_value=[])
    
    # Payload is text
    stage.learn("What is Python?")
    
    # Check if Knowledge Service was called
    mocked_ks.search.assert_called_with("What is Python?", {})
    
    # Check if external knowledge made it to thoughts
    assert "Python is a language." in stage.last_thought["external_knowledge"]
    
    # Check logs/info (implicit via last_thought structure)
    assert len(stage.last_thought["external_knowledge"]) == 1

def test_affective_modulation(advanced_stage):
    stage, _ = advanced_stage
    
    # Initial state should be neutral
    assert stage.affective.mood.label == "neutral"
    
    # Simulate a success feedback loop
    # Context with positive feedback
    ctx = {"feedback_score": 1.0}
    stage.learn("Hello", ctx)
    
    # Affect should shift to positive/happy
    # 0.0 + (0.2 * 1.0) * decay... > 0
    assert stage.affective.mood.valence > 0.0
    
    # Check if affect is reported in thought
    assert stage.last_thought["affect"]["mood_valence"] > 0.0

def test_curiosity_trigger(advanced_stage):
    stage, mocked_ks = advanced_stage
    
    # Artificially boost curiosity
    stage.affective.personality.curiosity = 1.0
    stage.affective.mood.arousal = 1.0 
    
    # Grounding return SOME concepts, but few (e.g. 1)
    stage._ground_vector_to_concepts = MagicMock(return_value=[("some_id", 0.5)])
    
    # Learn
    stage.learn("Tell me more.")
    
    # Should trigger knowledge search because curiosity is high (>0.7) and concepts < 3
    mocked_ks.search.assert_called()

def test_autodidact_loop(advanced_stage):
    stage, _ = advanced_stage
    
    # 1. Test Semantic Extraction (Learning)
    # Payload is a factual statement
    stage.learn("A Beagle is a type of dog.")
    
    # Check if extraction happened
    assert "extracted_knowledge" in stage.last_thought
    extracted = stage.last_thought["extracted_knowledge"]
    assert len(extracted) > 0
    assert extracted[0].subject == "beagle"
    assert extracted[0].object == "dog"
    assert extracted[0].predicate == "is_a"

    # Verify Persistence (Mock check)
    # The logic transforms "beagle" -> "beagle" (lowercase/clean)
    # add_node for subject, object
    # add_edge for relation
    stage.graph.add_edge.assert_called_with(
        "beagle", "dog", "is_a", confidence=extracted[0].confidence
    )
    
    # 2. Test Feedback Detection (Feeling)
    # Payload is positive feedback
    # Payload is positive feedback
    # Capture prior valence
    start_valence = stage.affective.mood.valence
    
    stage.learn("Perfect, thanks!")
    
    # Check if valence increased via implicit detection
    assert stage.affective.mood.valence > start_valence
    assert stage.last_thought["affect"]["mood_valence"] > start_valence

def test_generative_loop(advanced_stage):
    stage, _ = advanced_stage
    
    # 1. Teach Semantic Pattern
    # "X is a type of Y" -> Predicate: is_a (Confidence 0.9)
    # Using stronger pattern to pass > 0.8 threshold
    input_text = "A Sparrow is a type of Bird."
    
    stage.learn(input_text)
    
    # Check if a syntax pattern was persisted to the graph
    # We expect graph.add_node to be called for the concepts (sparrow, bird) AND the pattern
    # The pattern likely is "A {subject} is a type of {object}."
    
    # Filter calls to add_node where type="syntax_pattern"
    pattern_found = False
    for call in stage.graph.add_node.call_args_list:
        args, kwargs = call
        if kwargs.get("node_type") == "syntax_pattern":
            pattern_found = True
            # Check if template looks right (case insensitive)
            assert "{subject}" in kwargs["term"]
            assert "{object}" in kwargs["term"]
            break
            
    assert pattern_found, "GenerativeSystem failed to learn syntax pattern from input."
    
    # 2. Check Generation (Reflection)
    # The system should have used the learned (or default) pattern to generate a response
    response = stage.last_thought.get("response")
    assert response is not None
    # Check lowercase presence since extraction normalizes casing
    assert "sparrow" in response.lower()
    assert "bird" in response.lower()


