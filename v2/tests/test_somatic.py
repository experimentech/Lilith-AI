import unittest
from typing import List, Any, Dict
from unittest.mock import MagicMock

from v2.lilith_v2.somatic_layer import SomaticLayer, SomaticConfig
from v2.lilith_v2.io_port import IOPort
from v2.lilith_v2.cognitive_stage import CognitiveStage

class MockPort(IOPort):
    def __init__(self):
        self.inbox = []
        self.outbox = []
    
    def send(self, message, meta=None):
        self.outbox.append((message, meta))
        
    def receive(self, meta=None):
        while self.inbox:
            yield self.inbox.pop(0)

class TestSomaticLayer(unittest.TestCase):
    def setUp(self):
        self.brain = MagicMock(spec=CognitiveStage)
        self.brain.last_interaction = None
        
        self.eye = MockPort()
        self.hand = MockPort()
        
        self.config = SomaticConfig(
            sense_ports=["eye"],
            limb_ports=["hand"]
        )
        
        self.body = SomaticLayer(
            self.brain,
            {"eye": self.eye, "hand": self.hand},
            self.config
        )

    def test_afferent_flow(self):
        """Test Sense -> Brain."""
        # 1. Stimulate Eye
        self.eye.inbox.append("Visual Data")
        
        # 2. Tick Body
        self.body.tick()
        
        # 3. Verify Brain Received
        self.brain.learn.assert_called_with("Visual Data", ctx={"source_sense": "eye"})

    def test_efferent_flow(self):
        """Test Brain -> Limb."""
        # 1. Brain generates impulse
        self.brain.last_interaction = {"response": "Move Hand"}
        
        # 2. Tick Body
        self.body.tick()
        
        # 3. Verify Limb Acted
        self.assertEqual(len(self.hand.outbox), 1)
        self.assertEqual(self.hand.outbox[0][0], "Move Hand")
        
        # 4. Verify Refractory (Impulse cleared)
        self.assertIsNone(self.brain.last_interaction)

if __name__ == "__main__":
    unittest.main()
