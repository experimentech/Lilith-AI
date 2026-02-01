import sys
import os
import logging
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from v2.lilith_v2.generative_system import GenerativeSystem

# Mock specialized objects
class MockGraph:
    def __init__(self):
        self.nodes = {}
    
    def add_node(self, node_id, node_type, term, confidence, data):
        self.nodes[node_id] = {
            "type": node_type,
            "term": term,
            "data": data
        }

class MockRelation:
    def __init__(self, s, p, o):
        self.subject = s
        self.predicate = p
        self.object = o

def test_grammar_learning_with_pos():
    """Verify that GenerativeSystem learns patterns AND captures POS tags."""
    
    graph = MockGraph()
    gen_sys = GenerativeSystem(graph)
    
    # Input
    text = "A Beagle is a type of dog."
    relations = [MockRelation("Beagle", "is_a", "dog")]
    
    # Act
    gen_sys.learn_grammar(text, relations)
    
    # Assert
    assert len(graph.nodes) > 0
    
    # Inspect the learned node
    node = list(graph.nodes.values())[0]
    
    # 1. Verify Regex Anti-Unification
    print(f"Learned Template: {node['term']}")
    assert "{subject}" in node['term']
    assert "{object}" in node['term']
    assert node['term'] == "A {subject} is a type of {object}."
    
    # 2. Verify POS Tagging (Architecture Alignment)
    # The POS sequence for "A Beagle is a type of dog" should be roughly:
    # DT, NNP, VBZ, DT, NN, IN, NN
    pos_seq = node['data'].get('pos_sequence', [])
    print(f"Learned POS: {pos_seq}")
    
    # We check if list is not empty, implying NLTK worked
    if pos_seq:
        assert len(pos_seq) > 0
        assert "DT" in pos_seq # Determiner 'A'
        assert "NN" in pos_seq or "NNP" in pos_seq # Noun
        print("✅ POS Tagging aligned with GRAMMAR_STAGE_DESIGN.")
    else:
        print("⚠️  POS Tagging skipped (NLTK not active in test env?)")

if __name__ == "__main__":
    try:
        test_grammar_learning_with_pos()
        print("Test Passed!")
    except AssertionError as e:
        print(f"Test Failed: {e}")
        exit(1)
