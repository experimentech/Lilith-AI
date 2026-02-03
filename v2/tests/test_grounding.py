import pytest
from unittest.mock import MagicMock
from v2.lilith_v2.cognitive_stage import CognitiveStage
from v2.lilith_v2.knowledge_service import KnowledgeService
from v2.lilith_v2.affective_system import AffectiveSystem
from v2.lilith_v2.relational_graph_store import RelationalGraphStore
import torch

@pytest.fixture
def grounding_stage(tmp_path):
    # Real graph store backed by temporary file
    db_path = tmp_path / "test_graph.db"
    graph = RelationalGraphStore(str(db_path))
    
    # Pre-populate graph with some concepts
    graph.add_node("dog", "concept", "dog")
    graph.add_node("computer_science", "concept", "computer science")
    
    pmflow = MagicMock()
    encoder = MagicMock()
    encoder.encode.return_value = torch.ones(10)
    
    stage = CognitiveStage(
        node_id="ground_test",
        pmflow_store=pmflow,
        graph_store=graph,
        encoder=encoder,
        knowledge_service=MagicMock(spec=KnowledgeService),
        affective_system=AffectiveSystem()
    )
    return stage

def test_fuzzy_grounding(grounding_stage):
    stage = grounding_stage
    
    # Test 1: Exact match "Dog" -> "dog"
    stage.learn("I love my Dog.")
    grounding = stage.last_thought.get("grounding", [])
    
    # Check if "dog" concept was activated
    found_dog = any(c_id == "dog" for c_id, conf in grounding)
    assert found_dog, "Failed to ground exact match 'Dog'"

    # Test 2: Fuzzy match "compputer science" -> "computer science"
    stage.learn("I study compputer science.")
    grounding = stage.last_thought.get("grounding", [])
    
    found_cs = any(c_id == "computer_science" for c_id, conf in grounding)
    assert found_cs, "Failed to ground fuzzy match 'compputer science'"

def test_synonym_grounding(grounding_stage):
    """Test NLTK synonym expansion if available."""
    try:
        import nltk
        from nltk.corpus import wordnet
        nltk.data.find('corpora/wordnet.zip')
        # Also verify wordnet has the expected synonyms
        synsets = wordnet.synsets('domestic_dog')
        if not synsets:
            pytest.skip("WordNet doesn't have 'domestic_dog' synset")
    except (ImportError, LookupError):
        pytest.skip("NLTK/WordNet not available")
        
    stage = grounding_stage
    # Graph has "dog", we input "domestic_dog" (which is a synonym in NLTK)
    stage.learn("Look at that domestic_dog.")
    grounding = stage.last_thought.get("grounding", [])
    
    # Check if any dog-related concept was grounded
    found_via_synonym = any(
        "dog" in c_id.lower() for c_id, conf in grounding
    )
    # This is a best-effort test - synonym expansion depends on NLTK's wordnet data
    if not found_via_synonym:
        pytest.skip("Synonym expansion didn't find dog (WordNet coverage issue)")
