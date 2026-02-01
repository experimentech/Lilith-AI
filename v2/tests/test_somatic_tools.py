import unittest
from unittest.mock import MagicMock
from v2.lilith_v2.somatic_layer import SomaticLayer, SomaticConfig
from v2.lilith_v2.vscode_endpoint import VSCodeMCPAdapter, make_descriptor
from v2.lilith_v2.mcp_transport_tools import LocalToolsTransport
from v2.lilith_v2.cognitive_stage import CognitiveStage

class TestSomaticTools(unittest.TestCase):
    def setUp(self):
        # 1. Setup Transport (Muscles)
        self.transport = LocalToolsTransport()
        self.transport.register_tool("add", lambda x, y: x + y)
        
        # 2. Setup Limb (Port)
        self.descriptor = make_descriptor(name="calculator")
        self.hand = VSCodeMCPAdapter(self.descriptor, transport=self.transport)
        
        # 3. Setup Brain
        self.brain = MagicMock(spec=CognitiveStage)
        self.brain.last_interaction = None
        
        # 4. Setup Body
        self.config = SomaticConfig(
            sense_ports=["calculator"],
            limb_ports=["calculator"] 
        )
        self.body = SomaticLayer(
            self.brain,
            {"calculator": self.hand},
            self.config
        )

    def test_tool_execution_loop(self):
        """Verify Brain -> Limb -> Action -> Sense -> Brain loop."""
        
        # 1. Brain generates impulse (Intention to Add)
        # Note: The Brain must output exactly what LocalToolsTransport expects
        action_payload = {"action": "add", "args": {"x": 5, "y": 3}}
        self.brain.last_interaction = action_payload
        
        # 2. Tick Body (Efferent Phase: Action Execution)
        # The body reads last_interaction, sends to 'hand', which calls transport
        # 'hand' (VSCodeMCPAdapter) synchronously puts response into its OWN queue
        self.body.tick()
        
        # Verify Brain reset
        self.assertIsNone(self.brain.last_interaction)
        
        # Verify Port Internal State (Implementation Detail Check)
        # The queue should now contain the result
        # self.assertEqual(len(self.hand._queue), 1) 
        
        # 3. Tick Body Again (Afferent Phase: Sensation)
        # The body reads from 'hand' queue (the result of the action) and feeds brain
        self.body.tick()
        
        # 4. Verify Brain perceived the result
        # Expected response structure from VSCodeMCPAdapter + LocalToolsTransport
        # Adapter wraps result in {"response": ...}
        # Transport returns {"status": "success", "result": 8}
        # InMemoryIOPort wraps in {"message": ..., "meta": ...}
        
        # Check specific call args
        args, kwargs = self.brain.learn.call_args
        sensation = args[0]
        
        # Unpack the InMemoryIOPort wrapper to check the core message
        self.assertIn('message', sensation)
        self.assertEqual(sensation['message']['response']['result'], 8)
        self.assertEqual(sensation['message']['response']['status'], 'success')
        
        self.assertEqual(kwargs['ctx']['source_sense'], 'calculator')

if __name__ == "__main__":
    unittest.main()
