import unittest
from unittest.mock import patch, MagicMock
from v2.lilith_v2.weather_tools import get_weather, WeatherTools
from v2.lilith_v2.app import V2Runtime
from v2.lilith_v2.mcp_router import MCPDescriptor, EndpointType

class TestWeatherIntegration(unittest.TestCase):
    
    @patch('v2.lilith_v2.weather_tools.requests.get')
    def test_weather_tool(self, mock_get):
        """Verify weather tool works via trunk.tools."""
        
        # 1. Mock API Responses
        # Geocode response
        mock_geo = MagicMock()
        mock_geo.json.return_value = {"results": [{"latitude": -33.86, "longitude": 151.20, "name": "Sydney"}]}
        mock_geo.raise_for_status.return_value = None
        
        # Forecast response
        mock_wx = MagicMock()
        mock_wx.json.return_value = {
            "current_weather": {
                "temperature": 25.5,
                "weathercode": 1,
                "windspeed": 10.0
            }
        }
        
        # Sequence: Geocode call -> Forecast call
        mock_get.side_effect = [mock_geo, mock_wx]
        
        # 2. Setup Runtime
        runtime = V2Runtime.default()
        
        # 3. Dispatch Request
        payload = {
            "action": "get_weather",
            "args": {"location": "Sydney"}
        }
        
        digest = MCPDescriptor("weather_req", EndpointType.ACTION)
        
        runtime.route_and_dispatch(
            descriptor=digest,
            context={"tenant": "test_wx"},
            payload=payload
        )
        
        # 4. Check Response
        binding = runtime.bindings["trunk.tools"]
        adapter = binding.ports[0]
        queue = list(adapter._queue)
        
        last = queue[-1]
        response = last['message']['response']
        
        # We expect success structure from LocalToolsTransport -> Tool result
        self.assertEqual(response['status'], 'success')
        
        result = response['result']
        self.assertEqual(result['location'], "Sydney")
        self.assertEqual(result['temperature_c'], 25.5)
        self.assertEqual(result['status'], 'success')

if __name__ == "__main__":
    unittest.main()
