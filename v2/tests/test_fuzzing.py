import pytest
import random
import string
from unittest.mock import MagicMock
from v2.lilith_v2.cognitive_stage import CognitiveStage
from v2.lilith_v2.knowledge_service import KnowledgeService
from v2.lilith_v2.affective_system import AffectiveSystem

def generate_noise(text: str, error_rate: float = 0.1) -> str:
    """Inject typos, insertions, deletions into text."""
    chars = list(text)
    n_errors = int(len(text) * error_rate)
    
    for _ in range(n_errors):
        op = random.choice(["insert", "delete", "replace", "swap"])
        idx = random.randint(0, len(chars) - 1)
        
        if op == "insert":
            chars.insert(idx, random.choice(string.ascii_letters))
        elif op == "delete" and len(chars) > 1:
            chars.pop(idx)
        elif op == "replace":
            chars[idx] = random.choice(string.ascii_letters)
        elif op == "swap" and idx < len(chars) - 1:
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
            
    return "".join(chars)

@pytest.fixture
def stage():
    pmflow = MagicMock()
    graph = MagicMock()
    encoder = MagicMock()
    encoder.encode.return_value = MagicMock() # Mock tensor
    
    stage = CognitiveStage(
        node_id="fuzz_test",
        pmflow_store=pmflow,
        graph_store=graph,
        encoder=encoder,
        knowledge_service=MagicMock(spec=KnowledgeService),
        affective_system=AffectiveSystem()
    )
    return stage

def test_robustness_crash(stage):
    """Ensure random garbage doesn't crash the loop."""
    for _ in range(50):
        # Generate pure garbage
        garbage = "".join(random.choices(string.printable, k=random.randint(5, 100)))
        try:
            stage.learn(garbage)
        except Exception as e:
            pytest.fail(f"Crashed on input: {repr(garbage)} with error: {e}")

def test_extraction_under_noise(stage):
    """Test if SemanticExtractor can handle minor typos (Fuzzing)."""
    # This expects the system to be robust. 
    # Current Regex implementation is NOT robust to typos in keywords.
    # It SHOULD be robust to typos in the entities themselves (it should just extract the typo'd entity).
    
    base_fact = "A Beagle is a type of Dog."
    
    # 1. Typos in entities shouldn't break extraction mechanism
    # "A Beagle is a type of Dg." -> Should extract (Beagle, Dg, is_a)
    noisy_obj = "A Beagle is a type of Dg." # 'Dog' -> 'Dg'
    stage.learn(noisy_obj)
    
    # Check extraction
    extracted = stage.last_thought.get("extracted_knowledge", [])
    assert len(extracted) > 0
    assert extracted[0].subject.lower() == "beagle"
    # assert rs[0].object.lower() == "dg" # This is fine
    
    # 2. Typos in keywords (Structural logic)
    # "A Beagle is a tyype of Dog." -> Previously failed Regex.
    # Now with FuzzyUtils, it should be auto-corrected to "type" and extracted.
    noisy_keyword = "A Beagle is a tyype of Dog."
    stage.learn(noisy_keyword)
    
    # Check extraction for the second fact
    extracted = stage.last_thought.get("extracted_knowledge", [])
    # We might have the previous extraction in history if we reused stage, 
    # but stage.last_thought is overwritten each learn() call usually.
    
    assert len(extracted) > 0, "Failed to extract fact despite keyword typo correction."
    assert extracted[0].subject.lower() == "beagle"
    assert extracted[0].predicate == "is_a"

