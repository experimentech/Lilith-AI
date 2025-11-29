"""
Layered Fragment Store - Flexible knowledge layer stacking.

This module provides a generic API for stacking multiple fragment stores
with configurable read/write policies. Each layer can have different
permissions and priority.

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                    LAYER STACK                       │
    ├─────────────────────────────────────────────────────┤
    │  Layer N (highest priority) - e.g., User personal   │
    │  Layer N-1                  - e.g., Server shared   │
    │  ...                                                 │
    │  Layer 1                    - e.g., Base knowledge  │
    │  Layer 0 (lowest priority)  - e.g., Fallback       │
    └─────────────────────────────────────────────────────┘

Read Policy:
    - Queries all readable layers from highest to lowest priority
    - Returns best match across all layers (configurable: first-match or best-score)

Write Policy:
    - Writes to the designated writable layer
    - Can have multiple writable layers with different intents

Use Cases:
    - DM context: base (read) → user (read/write)
    - Server context: base (read) → server (read/write) → user (read for prefs)
    - Teacher mode: base (read/write)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Callable
from enum import Enum, auto
from pathlib import Path


class LayerPermission(Enum):
    """Permission levels for layers."""
    READ_ONLY = auto()      # Can only read from this layer
    READ_WRITE = auto()     # Can read and write
    WRITE_ONLY = auto()     # Can only write (rare, for logging)
    DISABLED = auto()       # Layer is disabled in this context


class MergeStrategy(Enum):
    """How to merge results from multiple layers."""
    FIRST_MATCH = auto()    # Return first layer that has a match
    BEST_SCORE = auto()     # Return best score across all layers
    MERGE_ALL = auto()      # Merge results from all layers


@dataclass
class LayerConfig:
    """Configuration for a single layer in the stack."""
    name: str                           # Human-readable name
    store: Any                          # The actual fragment store
    permission: LayerPermission         # Read/write permission
    priority: int = 0                   # Higher = checked first for reads
    weight: float = 1.0                 # Score multiplier for this layer
    
    def can_read(self) -> bool:
        return self.permission in (LayerPermission.READ_ONLY, LayerPermission.READ_WRITE)
    
    def can_write(self) -> bool:
        return self.permission in (LayerPermission.WRITE_ONLY, LayerPermission.READ_WRITE)


@dataclass
class LayerResult:
    """Result from a layer query."""
    layer_name: str
    response: Any               # The response object from the store
    score: float                # Confidence/match score
    source_id: Optional[str]    # Pattern ID if available


@dataclass
class PatternResponse:
    """Wrapper for pattern retrieval results to match response interface."""
    text: str
    confidence: float
    pattern_id: Optional[str] = None


class LayeredFragmentStore:
    """
    A fragment store that stacks multiple underlying stores.
    
    Provides unified read/write interface while respecting layer
    permissions and priorities.
    
    Example:
        # Create layers
        base_layer = LayerConfig(
            name="base",
            store=base_store,
            permission=LayerPermission.READ_ONLY,
            priority=0
        )
        server_layer = LayerConfig(
            name="server",
            store=server_store,
            permission=LayerPermission.READ_WRITE,
            priority=10
        )
        user_layer = LayerConfig(
            name="user",
            store=user_store,
            permission=LayerPermission.READ_WRITE,
            priority=20
        )
        
        # Create layered store
        layered = LayeredFragmentStore([base_layer, server_layer, user_layer])
        
        # Query searches all readable layers
        result = layered.find_response("hello")
        
        # Write goes to highest-priority writable layer
        layered.add_pattern("hi", "Hello there!")
    """
    
    def __init__(
        self,
        layers: List[LayerConfig],
        merge_strategy: MergeStrategy = MergeStrategy.BEST_SCORE,
        default_write_layer: Optional[str] = None
    ):
        """
        Initialize layered store.
        
        Args:
            layers: List of layer configurations
            merge_strategy: How to merge results from multiple layers
            default_write_layer: Name of layer to write to (if None, uses highest priority writable)
        """
        self.layers = sorted(layers, key=lambda l: l.priority, reverse=True)
        self.merge_strategy = merge_strategy
        self.default_write_layer = default_write_layer
        
        # Build quick lookup
        self._layer_map: Dict[str, LayerConfig] = {l.name: l for l in self.layers}
        
        # Validate
        if not any(l.can_read() for l in self.layers):
            raise ValueError("At least one layer must be readable")
    
    def get_layer(self, name: str) -> Optional[LayerConfig]:
        """Get a layer by name."""
        return self._layer_map.get(name)
    
    def get_readable_layers(self) -> List[LayerConfig]:
        """Get all readable layers in priority order."""
        return [l for l in self.layers if l.can_read()]
    
    def get_writable_layers(self) -> List[LayerConfig]:
        """Get all writable layers in priority order."""
        return [l for l in self.layers if l.can_write()]
    
    def get_write_layer(self, layer_name: Optional[str] = None) -> Optional[LayerConfig]:
        """
        Get the layer to write to.
        
        Args:
            layer_name: Specific layer name, or None for default
            
        Returns:
            Layer config or None if no writable layer
        """
        if layer_name:
            layer = self._layer_map.get(layer_name)
            if layer and layer.can_write():
                return layer
            return None
        
        if self.default_write_layer:
            return self.get_write_layer(self.default_write_layer)
        
        # Return highest priority writable layer
        writable = self.get_writable_layers()
        return writable[0] if writable else None
    
    def find_response(
        self,
        query: str,
        context: Optional[str] = None,
        **kwargs
    ) -> Tuple[Optional[Any], Optional[LayerResult]]:
        """
        Find best response across all readable layers.
        
        Args:
            query: The query text
            context: Optional context
            **kwargs: Additional args passed to underlying stores
            
        Returns:
            Tuple of (response, layer_result) or (None, None)
        """
        results: List[LayerResult] = []
        
        for layer in self.get_readable_layers():
            try:
                # Try to get response from this layer
                store = layer.store
                response = None
                score = 0.0
                pattern_id = None
                
                # Handle different store interfaces
                if hasattr(store, 'find_response'):
                    response = store.find_response(query, context=context, **kwargs)
                elif hasattr(store, 'get_best_response'):
                    response = store.get_best_response(query, **kwargs)
                elif hasattr(store, 'retrieve_patterns'):
                    # ResponseFragmentStore API
                    patterns = store.retrieve_patterns(query, topk=1, min_score=0.3)
                    if patterns:
                        pattern_obj, similarity = patterns[0]
                        # Create a response-like object
                        response = PatternResponse(
                            text=pattern_obj.response_text,
                            confidence=similarity,
                            pattern_id=pattern_obj.id if hasattr(pattern_obj, 'id') else None
                        )
                        score = similarity * layer.weight
                        pattern_id = response.pattern_id
                
                if response is None:
                    continue
                
                # Extract score if not already set
                if score == 0.0:
                    score = self._extract_score(response) * layer.weight
                
                # Extract pattern ID if not already set
                if pattern_id is None:
                    pattern_id = self._extract_pattern_id(response)
                
                result = LayerResult(
                    layer_name=layer.name,
                    response=response,
                    score=score,
                    source_id=pattern_id
                )
                
                # For FIRST_MATCH, return immediately if we have a good result
                if self.merge_strategy == MergeStrategy.FIRST_MATCH and score > 0.5:
                    return response, result
                
                results.append(result)
                
            except Exception as e:
                print(f"  ⚠️ Layer {layer.name} query failed: {e}")
                continue
        
        if not results:
            return None, None
        
        # Select best result
        if self.merge_strategy == MergeStrategy.BEST_SCORE:
            best = max(results, key=lambda r: r.score)
            return best.response, best
        
        elif self.merge_strategy == MergeStrategy.FIRST_MATCH:
            # We get here if no high-confidence match was found
            # Return the best of what we have
            best = max(results, key=lambda r: r.score)
            return best.response, best
        
        elif self.merge_strategy == MergeStrategy.MERGE_ALL:
            # Return all results (caller handles merging)
            return results, results[0] if results else None
        
        return None, None
    
    def _extract_score(self, response: Any) -> float:
        """Extract confidence score from various response formats."""
        if response is None:
            return 0.0
        
        # Try common attribute names
        for attr in ['confidence', 'score', 'similarity', 'match_score']:
            if hasattr(response, attr):
                val = getattr(response, attr)
                if isinstance(val, (int, float)):
                    return float(val)
        
        # Try dict access
        if isinstance(response, dict):
            for key in ['confidence', 'score', 'similarity']:
                if key in response:
                    return float(response[key])
        
        # Default: assume it's a valid response
        return 0.5
    
    def _extract_pattern_id(self, response: Any) -> Optional[str]:
        """Extract pattern ID from various response formats."""
        if response is None:
            return None
        
        # Try common attribute names
        for attr in ['pattern_id', 'id', 'fragment_id', 'source_id']:
            if hasattr(response, attr):
                return getattr(response, attr)
        
        # Try fragment_ids list
        if hasattr(response, 'fragment_ids') and response.fragment_ids:
            return response.fragment_ids[0]
        
        # Try dict access
        if isinstance(response, dict):
            for key in ['pattern_id', 'id', 'fragment_id']:
                if key in response:
                    return response[key]
        
        return None
    
    def add_pattern(
        self,
        pattern: str,
        response: str,
        layer_name: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Add a pattern to a writable layer.
        
        Args:
            pattern: The pattern/query text
            response: The response text
            layer_name: Specific layer to write to (or default)
            **kwargs: Additional args (intent, success_rate, etc.)
            
        Returns:
            Pattern ID if successful, None otherwise
        """
        layer = self.get_write_layer(layer_name)
        
        if not layer:
            print(f"  ⚠️ No writable layer available")
            return None
        
        try:
            store = layer.store
            
            # Handle different store interfaces
            if hasattr(store, 'add_pattern'):
                # Try ResponseFragmentStore API first (trigger_context, response_text)
                import inspect
                sig = inspect.signature(store.add_pattern)
                params = list(sig.parameters.keys())
                
                if 'trigger_context' in params:
                    # ResponseFragmentStore API
                    intent = kwargs.pop('intent', 'general')
                    success_rate = kwargs.pop('success_rate', 0.5)
                    return store.add_pattern(
                        trigger_context=pattern, 
                        response_text=response,
                        intent=intent,
                        success_score=success_rate
                    )
                else:
                    # Generic API with pattern/response
                    return store.add_pattern(pattern=pattern, response=response, **kwargs)
            elif hasattr(store, 'add'):
                return store.add(pattern, response, **kwargs)
            else:
                print(f"  ⚠️ Layer {layer.name} doesn't support adding patterns")
                return None
                
        except Exception as e:
            print(f"  ⚠️ Failed to add pattern to {layer.name}: {e}")
            return None
    
    def upvote(self, pattern_id: str, strength: float = 0.1, layer_name: Optional[str] = None):
        """Upvote a pattern (finds it in the appropriate layer)."""
        # Try to find which layer has this pattern
        for layer in self.get_writable_layers():
            if layer_name and layer.name != layer_name:
                continue
            try:
                store = layer.store
                if hasattr(store, 'upvote'):
                    store.upvote(pattern_id, strength=strength)
                    return
            except:
                continue
    
    def downvote(self, pattern_id: str, strength: float = 0.1, layer_name: Optional[str] = None):
        """Downvote a pattern."""
        for layer in self.get_writable_layers():
            if layer_name and layer.name != layer_name:
                continue
            try:
                store = layer.store
                if hasattr(store, 'downvote'):
                    store.downvote(pattern_id, strength=strength)
                    return
            except:
                continue
    
    def get_pattern_count(self) -> Dict[str, int]:
        """Get pattern counts from all layers."""
        counts = {}
        total = 0
        
        for layer in self.layers:
            try:
                store = layer.store
                if hasattr(store, 'get_pattern_count'):
                    layer_counts = store.get_pattern_count()
                    if isinstance(layer_counts, dict):
                        counts[layer.name] = layer_counts.get('total', 0)
                    else:
                        counts[layer.name] = layer_counts
                elif hasattr(store, '__len__'):
                    counts[layer.name] = len(store)
                else:
                    counts[layer.name] = 0
                total += counts[layer.name]
            except:
                counts[layer.name] = 0
        
        counts['total'] = total
        return counts
    
    def get_layer_info(self) -> List[Dict[str, Any]]:
        """Get information about all layers."""
        info = []
        for layer in self.layers:
            info.append({
                'name': layer.name,
                'priority': layer.priority,
                'permission': layer.permission.name,
                'can_read': layer.can_read(),
                'can_write': layer.can_write(),
                'weight': layer.weight
            })
        return info


class LayerStackBuilder:
    """
    Fluent builder for creating layer stacks.
    
    Example:
        stack = (LayerStackBuilder()
            .add_base(base_store)
            .add_server(server_store, guild_id="123")
            .add_user(user_store, user_id="456")
            .build())
    """
    
    def __init__(self):
        self._layers: List[LayerConfig] = []
        self._merge_strategy = MergeStrategy.BEST_SCORE
        self._default_write_layer: Optional[str] = None
    
    def add_layer(
        self,
        name: str,
        store: Any,
        permission: LayerPermission = LayerPermission.READ_ONLY,
        priority: int = 0,
        weight: float = 1.0
    ) -> 'LayerStackBuilder':
        """Add a generic layer."""
        self._layers.append(LayerConfig(
            name=name,
            store=store,
            permission=permission,
            priority=priority,
            weight=weight
        ))
        return self
    
    def add_base(
        self,
        store: Any,
        priority: int = 0,
        weight: float = 1.0
    ) -> 'LayerStackBuilder':
        """Add base knowledge layer (read-only)."""
        return self.add_layer(
            name="base",
            store=store,
            permission=LayerPermission.READ_ONLY,
            priority=priority,
            weight=weight
        )
    
    def add_server(
        self,
        store: Any,
        guild_id: Optional[str] = None,
        priority: int = 10,
        weight: float = 1.1  # Slight preference for server knowledge
    ) -> 'LayerStackBuilder':
        """Add server/guild layer (read-write)."""
        name = f"server:{guild_id}" if guild_id else "server"
        return self.add_layer(
            name=name,
            store=store,
            permission=LayerPermission.READ_WRITE,
            priority=priority,
            weight=weight
        )
    
    def add_user(
        self,
        store: Any,
        user_id: Optional[str] = None,
        priority: int = 20,
        weight: float = 1.2,  # Preference for user-specific knowledge
        read_only: bool = False
    ) -> 'LayerStackBuilder':
        """Add user layer."""
        name = f"user:{user_id}" if user_id else "user"
        permission = LayerPermission.READ_ONLY if read_only else LayerPermission.READ_WRITE
        return self.add_layer(
            name=name,
            store=store,
            permission=permission,
            priority=priority,
            weight=weight
        )
    
    def with_merge_strategy(self, strategy: MergeStrategy) -> 'LayerStackBuilder':
        """Set merge strategy."""
        self._merge_strategy = strategy
        return self
    
    def with_default_write_layer(self, layer_name: str) -> 'LayerStackBuilder':
        """Set default layer for writes."""
        self._default_write_layer = layer_name
        return self
    
    def build(self) -> LayeredFragmentStore:
        """Build the layered store."""
        return LayeredFragmentStore(
            layers=self._layers,
            merge_strategy=self._merge_strategy,
            default_write_layer=self._default_write_layer
        )


# Convenience function for common configurations
def create_dm_stack(base_store: Any, user_store: Any) -> LayeredFragmentStore:
    """Create a stack for DM context: base → user."""
    return (LayerStackBuilder()
        .add_base(base_store)
        .add_user(user_store)
        .with_default_write_layer("user")
        .build())


def create_server_stack(
    base_store: Any,
    server_store: Any,
    user_store: Optional[Any] = None,
    guild_id: str = "",
    user_id: str = ""
) -> LayeredFragmentStore:
    """Create a stack for server context: base → server → user(read-only)."""
    builder = (LayerStackBuilder()
        .add_base(base_store)
        .add_server(server_store, guild_id=guild_id)
        .with_default_write_layer(f"server:{guild_id}" if guild_id else "server"))
    
    if user_store:
        # User layer is read-only in server context (for preferences)
        builder.add_user(user_store, user_id=user_id, read_only=True)
    
    return builder.build()
