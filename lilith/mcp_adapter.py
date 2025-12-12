"""
Minimal MCP adapter skeleton for Lilith.

This module is intentionally light: it exposes thin handler methods that map
MCP-style tool calls to LilithSession operations without altering core logic.
Add your transport/server of choice (e.g., FastAPI, aiohttp, MCP runtime) to
wire these handlers to real endpoints.
"""
import os
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from lilith.session import LilithSession, SessionConfig, SessionResponse


class TTLCache:
    """Minimal TTL cache to reduce external calls."""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._cache: Dict[str, Tuple[float, object]] = {}

    def get(self, key: str) -> Optional[object]:
        if key in self._cache:
            expires, value = self._cache[key]
            if time.time() < expires:
                return value
            self._cache.pop(key, None)
        return None

    def set(self, key: str, value: object) -> None:
        self._cache[key] = (time.time() + self.ttl, value)


# Optional external calls are best-effort; fall back to static placeholders.


@dataclass
class WeatherReport:
    location: str
    summary: str
    temperature_c: float


class WeatherClient:
    def __init__(self, ttl_seconds: int = 300, min_interval_seconds: int = 5, http_timeout_seconds: float = 5.0):
        self.cache = TTLCache(ttl_seconds=ttl_seconds)
        self.min_interval_seconds = min_interval_seconds
        self.http_timeout_seconds = http_timeout_seconds
        self._last_request: Dict[str, float] = {}
        try:
            import requests  # type: ignore

            self._requests = requests
        except Exception:
            self._requests = None

    def get_weather(self, location: str) -> WeatherReport:
        cached = self.cache.get(location.lower())
        if cached:
            return cached  # type: ignore

        now = time.time()
        last = self._last_request.get(location.lower())
        if last and now - last < self.min_interval_seconds:
            # Too soon; return placeholder immediately
            return WeatherReport(location=location, summary="clear skies", temperature_c=20.0)
        self._last_request[location.lower()] = now

        # Default placeholder
        report = WeatherReport(location=location, summary="clear skies", temperature_c=20.0)

        if self._requests:
            try:
                geo_resp = self._requests.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": location, "count": 1},
                    timeout=self.http_timeout_seconds,
                )
                geo_resp.raise_for_status()
                geo_data = geo_resp.json()
                results = geo_data.get("results") or []
                if results:
                    lat = results[0].get("latitude")
                    lon = results[0].get("longitude")
                    if lat is not None and lon is not None:
                        wx_resp = self._requests.get(
                            "https://api.open-meteo.com/v1/forecast",
                            params={"latitude": lat, "longitude": lon, "current_weather": True},
                            timeout=self.http_timeout_seconds,
                        )
                        wx_resp.raise_for_status()
                        wx = wx_resp.json().get("current_weather") or {}
                        temp = wx.get("temperature")
                        summary = wx.get("weathercode")
                        if temp is not None:
                            report.temperature_c = float(temp)
                        if summary is not None:
                            report.summary = f"weather code {summary}"
            except Exception:
                pass

        self.cache.set(location.lower(), report)
        return report


class NewsClient:
    def __init__(self, ttl_seconds: int = 300, min_interval_seconds: int = 5, http_timeout_seconds: float = 5.0):
        self.cache = TTLCache(ttl_seconds=ttl_seconds)
        self.min_interval_seconds = min_interval_seconds
        self.http_timeout_seconds = http_timeout_seconds
        self._last_request: Dict[str, float] = {}
        try:
            import requests  # type: ignore

            self._requests = requests
        except Exception:
            self._requests = None

    def get_news(self, topic: str) -> str:
        key = topic.lower()
        cached = self.cache.get(key)
        if cached:
            return cached  # type: ignore

        now = time.time()
        last = self._last_request.get(key)
        if last and now - last < self.min_interval_seconds:
            return f"Top headline for {topic}: placeholder headline."
        self._last_request[key] = now

        headline = f"Top headline for {topic}: placeholder headline."

        if self._requests:
            try:
                resp = self._requests.get(
                    "https://hn.algolia.com/api/v1/search",
                    params={"query": topic, "tags": "story", "hitsPerPage": 1},
                    timeout=self.http_timeout_seconds,
                )
                resp.raise_for_status()
                hits = resp.json().get("hits") or []
                if hits:
                    title = hits[0].get("title") or hits[0].get("story_title")
                    url = hits[0].get("url")
                    if title:
                        headline = title if not url else f"{title} ({url})"
            except Exception:
                pass

        self.cache.set(key, headline)
        return headline


class MCPAdapter:
    """
    MCP-facing adapter that keeps LilithSession untouched.

    - One session per (client_id, context_id) to preserve isolation.
    - No changes to Lilith core; all state remains in existing stores.
    - Transient tools (weather/news) intentionally avoid persisting to the DB.
    """

    def __init__(
        self,
        session_config: Optional[SessionConfig] = None,
        session_factory: Optional[Callable[[str, Optional[str], SessionConfig], LilithSession]] = None,
        weather_client: Optional[WeatherClient] = None,
        news_client: Optional[NewsClient] = None,
        max_sessions: int = 128,
        session_ttl_seconds: int = 3600,
        weather_timeout_seconds: float = 5.0,
        news_timeout_seconds: float = 5.0,
    ) -> None:
        self.session_config = session_config or self._build_session_config_from_env()
        self.session_factory = session_factory or self._default_session_factory
        self._sessions: "OrderedDict[Tuple[str, str], Tuple[float, LilithSession]]" = OrderedDict()
        self._lock = threading.Lock()
        self.max_sessions = max_sessions
        self.session_ttl_seconds = session_ttl_seconds
        self.weather_client = weather_client or WeatherClient(http_timeout_seconds=weather_timeout_seconds)
        self.news_client = news_client or NewsClient(http_timeout_seconds=news_timeout_seconds)
        # Cache stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions_expired = 0
        self.cache_evictions_lru = 0

    def _build_session_config_from_env(self) -> SessionConfig:
        """Create SessionConfig, honoring env toggles for personality/mood."""

        def _truthy(name: str) -> bool:
            return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}

        cfg = SessionConfig()
        if _truthy("LILITH_PERSONALITY_ENABLE"):
            cfg.enable_personality = True
        if _truthy("LILITH_MOOD_ENABLE"):
            cfg.enable_mood = True
        return cfg

    def _default_session_factory(self, client_id: str, context_id: Optional[str], config: SessionConfig) -> LilithSession:
        return LilithSession(user_id=client_id, context_id=context_id, config=config)

    def _get_session(self, client_id: str, context_id: Optional[str] = None) -> LilithSession:
        key = (client_id, context_id or "default")
        now = time.time()
        with self._lock:
            # Drop expired sessions
            self._prune_locked(now)

            if key in self._sessions:
                expires, session = self._sessions.pop(key)
                self._sessions[key] = (expires, session)  # mark as most recently used
                self.cache_hits += 1
                return session

            # Evict oldest if at capacity
            if len(self._sessions) >= self.max_sessions:
                self._sessions.popitem(last=False)
                self.cache_evictions_lru += 1

            session = self.session_factory(client_id, context_id, self.session_config)
            self._sessions[key] = (now + self.session_ttl_seconds, session)
            self.cache_misses += 1
            return session

    def _prune_locked(self, now: float) -> None:
        expired_keys = [k for k, (expires, _) in self._sessions.items() if expires < now]
        for key in expired_keys:
            self._sessions.pop(key, None)
            self.cache_evictions_expired += 1

    def get_cache_stats(self) -> Dict[str, float]:
        with self._lock:
            return {
                "size": float(len(self._sessions)),
                "hits": float(self.cache_hits),
                "misses": float(self.cache_misses),
                "evicted_expired": float(self.cache_evictions_expired),
                "evicted_lru": float(self.cache_evictions_lru),
                "ttl_seconds": float(self.session_ttl_seconds),
                "max_sessions": float(self.max_sessions),
            }

    # Core chat entry point
    def handle_chat(self, client_id: str, message: str, context_id: Optional[str] = None) -> SessionResponse:
        session = self._get_session(client_id, context_id)
        return session.process_message(message)

    # Teaching / pattern injection (uses existing store logic)
    def handle_teach(self, client_id: str, trigger: str, response: str, intent: str = "general", context_id: Optional[str] = None) -> str:
        session = self._get_session(client_id, context_id)
        if hasattr(session, "teach"):
            return session.teach(trigger, response, intent=intent)
        # Fallback for stub sessions in tests
        return session.store.add_pattern(trigger_context=trigger, response_text=response, intent=intent)

    # Feedback wiring
    def handle_upvote(self, client_id: str, pattern_id: str, strength: float = 1.0, context_id: Optional[str] = None) -> None:
        session = self._get_session(client_id, context_id)
        session.upvote(pattern_id, strength=strength)

    def handle_downvote(self, client_id: str, pattern_id: str, strength: float = 1.0, context_id: Optional[str] = None) -> None:
        session = self._get_session(client_id, context_id)
        session.downvote(pattern_id, strength=strength)

    # Lightweight stats surface (user store only)
    def handle_stats(self, client_id: str, context_id: Optional[str] = None, include_cache: bool = False) -> Dict[str, float]:
        session = self._get_session(client_id, context_id)
        stats = {}
        if session.store.user_store and hasattr(session.store.user_store, "get_stats"):
            stats = session.store.user_store.get_stats()
        if include_cache:
            stats["mcp_cache"] = self.get_cache_stats()
        return stats

    # Transient tools: do not persist; return data directly
    def handle_weather(self, client_id: str, location: str, context_id: Optional[str] = None) -> WeatherReport:
        _ = self._get_session(client_id, context_id)  # ensures isolation; no storage used
        return self.weather_client.get_weather(location)

    def handle_news(self, client_id: str, topic: str, context_id: Optional[str] = None) -> str:
        _ = self._get_session(client_id, context_id)  # ensures isolation; no storage used
        return self.news_client.get_news(topic)

    # Optional: allow callers to reset their own user data (delegates to store helpers)
    def handle_reset_user(self, client_id: str, keep_backup: bool = True, bootstrap: bool = False, context_id: Optional[str] = None) -> Optional[str]:
        session = self._get_session(client_id, context_id)
        if hasattr(session.store, "user_store") and session.store.user_store and hasattr(session.store.user_store, "reset_user_data"):
            return session.store.user_store.reset_user_data(keep_backup=keep_backup, bootstrap=bootstrap)
        return None


__all__ = [
    "MCPAdapter",
    "WeatherReport",
    "WeatherClient",
    "NewsClient",
]
