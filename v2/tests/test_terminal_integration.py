import unittest
import os
from unittest.mock import patch
from v2.lilith_v2.app import V2Runtime
from v2.lilith_v2.mcp_router import MCPDescriptor, EndpointType

class TestTerminalIntegration(unittest.TestCase):
    
    @patch.dict(os.environ, {"LILITH_ENABLE_TERMINAL": "true"})
    def test_terminal_enabled(self):
        """Verify terminal works when explicitly enabled."""
        runtime = V2Runtime.default()
        
        # 1. Payload
        payload = {
            "action": "run_command",
            "args": {"command": "echo 'Lilith Terminal Active'"}
        }
        
        digest = MCPDescriptor("term_test", EndpointType.ACTION)
        
        # 2. Dispatch
        runtime.route_and_dispatch(
            descriptor=digest,
            context={"tenant": "test_term"},
            payload=payload
        )
        
        # 3. Check Response
        binding = runtime.bindings["trunk.tools"]
        adapter = binding.ports[0]
        queue = list(adapter._queue)
        
        # Last message should be the response
        last = queue[-1]
        
        # The response structure is wrapped differently depending on the tool result.
        # From error: {'status': 'success', 'result': {'status': 'success', 'stdout': 'Lilith Terminal Active'}}
        # So it's response['result']['stdout']
        
        response = last['message']['response']
        result_payload = response.get('result', {})
        
        self.assertEqual(response.get('status'), 'success', f"Expected success, got {response}")
        self.assertEqual(result_payload.get('stdout'), 'Lilith Terminal Active', f"Stdout mismatch in {response}")

    @patch.dict(os.environ, {"LILITH_ENABLE_TERMINAL": "false"})
    def test_terminal_disabled_by_default(self):
        """Verify terminal is inaccessible by default."""
        runtime = V2Runtime.default()
        
        payload = {
            "action": "run_command",
            "args": {"command": "echo 'Should Fail'"}
        }
        
        digest = MCPDescriptor("term_test", EndpointType.ACTION)
        
        # The tool simply won't be registered, so dispatching a 'run_command' 
        # action to the transport will return "Tool not found" error.
        
        runtime.route_and_dispatch(
            descriptor=digest,
            context={"tenant": "test_term_fail"},
            payload=payload
        )
        
        binding = runtime.bindings["trunk.tools"]
        adapter = binding.ports[0]
        queue = list(adapter._queue)
        last = queue[-1]
        response = last['message']['response']
        
        # Expect error because tool is not registered
        # When tool is not found, LocalToolsTransport returns {"error": "Tool ... not found"}
        # But V2Runtime logic or Adapter might wrap it.
        # Let's inspect failed payload structure if needed.
        # Usually: {"status": "error", "message": "Tool ... not found"} but let's check field names.
        
        self.assertIn("error", str(response).lower())
        self.assertIn("not found", str(response).lower())

if __name__ == "__main__":
    unittest.main()
