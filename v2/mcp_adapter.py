"""
V2 MCP Adapter - CognitiveStage wrapper for MCP-style endpoints.

This is the V2 equivalent of lilith/mcp_adapter.py, but uses CognitiveStage
instead of LilithSession. The interface is intentionally similar to minimize
porting effort for existing MCP consumers.

Key differences from V1:
- Uses CognitiveStage.learn() instead of LilithSession.process_message()
- Response comes from stage.last_thought["response"]
- Multi-tenant via stores passed to CognitiveStage
- ResponseComposer handles pattern matching/learning internally
"""
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MCPResponse:
    """Response from MCP chat endpoint."""
    text: str
    confidence: float = 0.0
    source: str = "cognitive"
    is_fallback: bool = False
    is_low_confidence: bool = False
    pattern_id: Optional[str] = None
    learned_fact: Optional[str] = None
    mood: Optional[Any] = None
    reasoning_confidence: Optional[float] = None


@dataclass
class WeatherReport:
    """Weather data structure (mirrors V1)."""
    location: str
    summary: str
    temperature_c: float


class V2MCPAdapter:
    """
    MCP-facing adapter that wraps CognitiveStage.

    - One CognitiveStage per (client_id, context_id) for isolation.
    - Transient tools (weather/news) don't pollute the learning stores.
    - Thread-safe session cache with TTL and LRU eviction.
    """

    def __init__(
        self,
        data_path: Optional[Path] = None,
        max_sessions: int = 128,
        session_ttl_seconds: int = 3600,
    ) -> None:
        self.data_path = data_path or Path("data/v2_mcp")
        self.max_sessions = max_sessions
        self.session_ttl_seconds = session_ttl_seconds
        
        # Session cache: (client_id, context_id) -> (expires, CognitiveStage)
        self._sessions: "OrderedDict[Tuple[str, str], Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        
        # Cache stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions_expired = 0
        self.cache_evictions_lru = 0
        
        # Lazy imports for components
        self._CognitiveStage = None
        self._MultiTenantPMFlowManager = None
        self._MultiTenantGraphManager = None
        self._RelationalStore = None
        self._encoder = None
        self._pmflow = None
        self._graph = None
        
        # Initialize stores
        self._init_stores()
    
    def _init_stores(self) -> None:
        """Initialize shared stores (lazy import to avoid circular deps)."""
        try:
            from v2.lilith_v2.cognitive_stage import CognitiveStage
            from v2.lilith_v2.multi_tenant_store import MultiTenantPMFlowManager, MultiTenantGraphManager
            from v2.lilith_v2.relational_store import RelationalStore
            
            self._CognitiveStage = CognitiveStage
            self._MultiTenantPMFlowManager = MultiTenantPMFlowManager
            self._MultiTenantGraphManager = MultiTenantGraphManager
            self._RelationalStore = RelationalStore
            
            # Create data directories
            base_root = self.data_path / "base"
            users_root = self.data_path / "users"
            base_root.mkdir(parents=True, exist_ok=True)
            users_root.mkdir(parents=True, exist_ok=True)
            
            self._pmflow = MultiTenantPMFlowManager(str(base_root), str(users_root))
            self._graph = MultiTenantGraphManager(str(base_root), str(users_root))
            self._users_root = users_root
            
            # Create encoder
            self._encoder = self._create_encoder()
            
            logger.info(f"V2MCPAdapter initialized with data_path={self.data_path}")
        except Exception as e:
            logger.error(f"Failed to initialize V2MCPAdapter: {e}")
            raise
    
    def _create_encoder(self):
        """Create the best available encoder."""
        try:
            from pmflow import PMFlowEmbeddingEncoder
            return PMFlowEmbeddingEncoder(
                dimension=96,
                latent_dim=48,
                enable_flow=True,
            )
        except ImportError:
            logger.warning("PMFlowEmbeddingEncoder not available, using fallback")
            import torch
            class SimpleEncoder:
                def __init__(self):
                    self.embedding_dim = 64
                    self.enable_flow = False
                def encode(self, text):
                    seed = sum(ord(c) for c in str(text))
                    torch.manual_seed(seed)
                    return torch.randn(64)
            return SimpleEncoder()
    
    def _get_session(self, client_id: str, context_id: Optional[str] = None) -> Any:
        """Get or create CognitiveStage for a client."""
        key = (client_id, context_id or "default")
        now = time.time()
        
        with self._lock:
            # Prune expired sessions
            self._prune_locked(now)
            
            if key in self._sessions:
                expires, stage = self._sessions.pop(key)
                self._sessions[key] = (expires, stage)  # LRU refresh
                self.cache_hits += 1
                return stage
            
            # Evict oldest if at capacity
            if len(self._sessions) >= self.max_sessions:
                self._sessions.popitem(last=False)
                self.cache_evictions_lru += 1
            
            # Create new CognitiveStage
            stage = self._create_stage(client_id, context_id)
            self._sessions[key] = (now + self.session_ttl_seconds, stage)
            self.cache_misses += 1
            return stage
    
    def _create_stage(self, client_id: str, context_id: Optional[str]) -> Any:
        """Create a new CognitiveStage for the client."""
        node_id = f"{client_id}:{context_id or 'default'}"
        
        # Create user-specific response store
        user_path = self._users_root / client_id
        user_path.mkdir(parents=True, exist_ok=True)
        response_store = self._RelationalStore(str(user_path / "responses.db"))
        
        stage = self._CognitiveStage(
            node_id=node_id,
            pmflow_store=self._pmflow,
            graph_store=self._graph,
            encoder=self._encoder,
            config={
                "knowledge_enabled": True,
                "response_store": response_store,
                "composition_mode": "adaptive",
                "enable_blending": True,
                "enable_learning": True,
                "enable_reasoning": True,
                "deliberation_steps": 10,
            }
        )
        
        logger.debug(f"Created CognitiveStage for {node_id}")
        return stage
    
    def _prune_locked(self, now: float) -> None:
        """Prune expired sessions (must hold lock)."""
        expired_keys = [k for k, (expires, _) in self._sessions.items() if expires < now]
        for key in expired_keys:
            self._sessions.pop(key, None)
            self.cache_evictions_expired += 1
    
    def get_cache_stats(self) -> Dict[str, float]:
        """Get session cache statistics."""
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
    
    # ===== Core Handlers =====
    
    def handle_chat(
        self,
        client_id: str,
        message: str,
        context_id: Optional[str] = None
    ) -> MCPResponse:
        """Process a chat message through CognitiveStage."""
        stage = self._get_session(client_id, context_id)
        
        # Process message
        ctx = {"tenant": client_id}
        stage.learn(message, ctx)
        
        # Extract response
        response_text = stage.last_thought.get("response", "")
        composed = stage.last_thought.get("composed_response")
        
        # Build response object
        response = MCPResponse(text=response_text)
        
        if composed:
            response.confidence = composed.confidence
            response.source = composed.source
            response.is_fallback = composed.is_fallback
            response.is_low_confidence = composed.confidence < 0.5
            if composed.fragment_ids:
                response.pattern_id = composed.fragment_ids[0]
        else:
            response.source = stage.last_thought.get("composition_source", "generative")
            # Use reasoning confidence if available
            reasoning_conf = stage.last_thought.get("reasoning_confidence")
            if reasoning_conf is not None:
                response.confidence = reasoning_conf
                response.reasoning_confidence = reasoning_conf
        
        # Check for learned facts
        extracted = stage.last_thought.get("extracted_knowledge", [])
        if extracted:
            rel = extracted[0]
            response.learned_fact = f"{rel.subject} {rel.predicate} {rel.object}"
        
        # Mood (if affective system provides it)
        if hasattr(stage, 'affective'):
            mood = stage.affective.get_state_vector()
            if mood:
                response.mood = type('Mood', (), {
                    'label': mood.get('current_mood', 'neutral'),
                    'emoji': '😊' if mood.get('energy', 0) > 0.5 else '😐'
                })()
        
        return response
    
    def handle_teach(
        self,
        client_id: str,
        trigger: str,
        response: str,
        intent: str = "general",
        context_id: Optional[str] = None
    ) -> str:
        """Teach a new response pattern."""
        stage = self._get_session(client_id, context_id)
        
        # Add to response composer's pattern store
        if stage._response_composer and stage._response_composer.patterns:
            pattern_id = stage._response_composer.patterns.add_pattern(
                fragment_id=None,  # Auto-generate ID
                trigger_context=trigger,
                response_text=response,
                intent=intent,
                success_score=0.7,  # Start with moderate confidence
            )
            logger.info(f"Taught pattern {pattern_id}: '{trigger}' -> '{response}'")
            return pattern_id or f"taught_{trigger[:20]}"
        
        return f"no_store_{trigger[:20]}"
    
    def handle_forget(
        self,
        client_id: str,
        pattern_id: str,
        context_id: Optional[str] = None
    ) -> bool:
        """Remove a learned pattern."""
        stage = self._get_session(client_id, context_id)
        
        if stage._response_composer and stage._response_composer.patterns:
            # PatternStore doesn't have delete, but we can set confidence to 0
            # which effectively removes it from consideration
            try:
                stage._response_composer.patterns.update_pattern(
                    pattern_id=pattern_id,
                    success_score=0.0,
                )
                return True
            except Exception as e:
                logger.warning(f"Failed to forget pattern {pattern_id}: {e}")
        return False
    
    def handle_upvote(
        self,
        client_id: str,
        pattern_id: str,
        strength: float = 1.0,
        context_id: Optional[str] = None
    ) -> None:
        """Upvote a response pattern."""
        stage = self._get_session(client_id, context_id)
        stage.record_response_outcome(True)
    
    def handle_downvote(
        self,
        client_id: str,
        pattern_id: str,
        strength: float = 1.0,
        context_id: Optional[str] = None
    ) -> None:
        """Downvote a response pattern."""
        stage = self._get_session(client_id, context_id)
        stage.record_response_outcome(False)
    
    def handle_stats(
        self,
        client_id: str,
        context_id: Optional[str] = None,
        include_cache: bool = True
    ) -> Dict[str, Any]:
        """Get stats for a client's session."""
        stage = self._get_session(client_id, context_id)
        
        stats = stage.stats()
        if include_cache:
            stats["cache"] = self.get_cache_stats()
        
        return stats
    
    def handle_weather(
        self,
        client_id: str,
        location: str,
        context_id: Optional[str] = None
    ) -> WeatherReport:
        """Get weather (transient, no learning)."""
        # Placeholder - would integrate with weather API
        return WeatherReport(
            location=location,
            summary="clear skies",
            temperature_c=20.0
        )
    
    def handle_news(
        self,
        client_id: str,
        topic: str,
        context_id: Optional[str] = None
    ) -> str:
        """Get news (transient, no learning)."""
        # Placeholder - would integrate with news API
        return f"Top headline for {topic}: placeholder headline."


__all__ = ["V2MCPAdapter", "MCPResponse", "WeatherReport"]
