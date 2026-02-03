import unittest
import os
from v2.lilith_v2.app import V2Runtime
from v2.lilith_v2.mcp_router import MCPDescriptor, EndpointType

class TestFileSystemIntegration(unittest.TestCase):
    def setUp(self):
        # Ensure clean state
        if "LILITH_ENABLE_FS" in os.environ:
            del os.environ["LILITH_ENABLE_FS"]
    
    def test_fs_disabled_by_default(self):
        """File system operations should be disabled by default for safety."""
        runtime = V2Runtime.default()
        binding = runtime.bindings["trunk.tools"]
        adapter = binding.ports[0]
        transport = adapter._transport
        
        # list_tools should NOT include read_file or write_file
        tools = transport.list_tools()
        tool_names = {t["name"] for t in tools}
        
        self.assertNotIn("read_file", tool_names, "read_file should be disabled by default")
        self.assertNotIn("write_file", tool_names, "write_file should be disabled by default")
    
    def test_fs_tools(self):
        # Enable file system for this test
        os.environ["LILITH_ENABLE_FS"] = "true"
        # 1. Init Runtime (uses default() which wires trunk.tools)
        runtime = V2Runtime.default()
        
        # 2. Get Transport (Muscles)
        # Runtime bindings store the ports.
        # "trunk.tools" binding has ports[0] which is VSCodeMCPAdapter
        binding = runtime.bindings["trunk.tools"]
        adapter = binding.ports[0]
        
        # Check transport is attached
        self.assertIsNotNone(adapter._transport)
        
        # 3. Simulate Brain Impulse (Write File)
        # "trunk.tools" is an Action endpoint.
        # We route an action request to it.
        descriptor = MCPDescriptor("fs_action", EndpointType.ACTION)
        
        payload = {
            "action": "write_file",
            "args": {
                "path": "test_output.txt",
                "content": "Hello World from Lilith"
            }
        }
        
        decision = runtime.route_and_dispatch(
            descriptor=descriptor,
            context={"tenant": "test_fs"},
            payload=payload
        )
        
        self.assertIn("trunk.tools", decision.target_nodes)
        
        # 4. Verify Side Effect (File Written)
        # Because LocalToolsTransport executes synchronously in the adapter,
        # the file should exist now.
        self.assertTrue(os.path.exists("test_output.txt"))
        with open("test_output.txt", "r") as f:
            content = f.read()
            self.assertEqual(content, "Hello World from Lilith")
            
        # 5. Simulate Brain Impulse (Read File)
        payload_read = {
            "action": "read_file",
            "args": {"path": "test_output.txt"}
        }
        
        runtime.route_and_dispatch(
            descriptor=descriptor,
            context={"tenant": "test_fs"},
            payload=payload_read
        )
        
        # 6. Verify Feedback (Proprioception)
        # The adapter should have queued the response (content of file)
        queue = list(adapter._queue)
        # Queue has: [Write_Echo, Write_Result, Read_Echo, Read_Result]
        # We want the last one.
        read_result_msg = queue[-1]['message']
        
        # LocalToolsTransport returns {"status": "success", "result": ...}
        # VSCodeAdapter wraps it in {"response": ...}
        # In runtime.route_and_dispatch, we extract response.
        # Wait, route_and_dispatch pushes the INPUT.
        # The ADAPTER (if transport present) pushes the OUTPUT.
        
        # So queue sequence:
        # 1. route_and_dispatch pushes Write INPUT.
        # 2. Adapter.send() calls Transport -> Transport writes file -> Adapter queues Write OUTPUT.
        # 3. route_and_dispatch pushes Read INPUT.
        # 4. Adapter.send() calls Transport -> Transport reads file -> Adapter queues Read OUTPUT.
        
        last_msg = queue[-1]
        self.assertIn("response", last_msg['message'])
        result_payload = last_msg['message']['response']
        self.assertEqual(result_payload['status'], "success")
        self.assertEqual(result_payload['result'], "Hello World from Lilith")

    def tearDown(self):
        if os.path.exists("test_output.txt"):
            os.remove("test_output.txt")
        # Reset env var
        if "LILITH_ENABLE_FS" in os.environ:
            del os.environ["LILITH_ENABLE_FS"]

if __name__ == "__main__":
    unittest.main()
