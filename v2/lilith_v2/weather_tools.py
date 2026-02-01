from typing import Any, Dict, Optional
import time
import requests

class WeatherTools:
    """
    Weather tool implementation ported from v1 MCPAdapter.
    """
    
    def __init__(self):
        self._last_request: Dict[str, float] = {}
        self.min_interval = 5.0
        self.timeout = 5.0
        
    def get_weather(self, location: str) -> Dict[str, Any]:
        """
        Get weather for location.
        Returns dict with location, summary, temperature_c.
        """
        location = location.lower().strip()
        now = time.time()
        
        # Simple rate limit per location
        last = self._last_request.get(location)
        if last and now - last < self.min_interval:
             return {"location": location, "summary": "rate_limited", "temperature_c": 0.0, "status": "cached"}
             
        self._last_request[location] = now
        
        # Attempt Geocoding + Weather
        try:
            # 1. Geocode
            geo_resp = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1},
                timeout=self.timeout
            )
            geo_resp.raise_for_status()
            results = geo_resp.json().get("results")
            
            if not results:
                 return {"error": f"Location '{location}' not found."}
                 
            lat = results[0]["latitude"]
            lon = results[0]["longitude"]
            name = results[0]["name"]
            
            # 2. Forecast
            wx_resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon, "current_weather": True},
                timeout=self.timeout
            )
            wx_resp.raise_for_status()
            current = wx_resp.json().get("current_weather", {})
            
            return {
                "location": name,
                "temperature_c": current.get("temperature"),
                "summary": f"Code {current.get('weathercode')}",
                "windspeed": current.get("windspeed"),
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e), "status": "failed"}

# Standalone instance
_weather_tools = WeatherTools()

def get_weather(location: str):
    return _weather_tools.get_weather(location)
