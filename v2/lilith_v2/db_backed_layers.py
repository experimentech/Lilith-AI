"""
Database-Backed Plastic Layers

These layers follow the v1 principle:
  - Databases store WHAT we know (salience values, patterns, intents)
  - Neural Networks determine HOW to find and use that information

Benefits:
  - NNs stay lightweight (only navigation/attention weights)
  - Information is infinitely scalable (database storage)
  - Queryable (can inspect what the system has learned)
  - Persistent by default (no separate save/load)

Tradeoff:
  - Lookup latency for database queries (acceptable for conversational AI)

Architecture:
    Input → SalienceLayer (queries DB for word salience) 
          → PatternLayer (queries DB for pattern matches)
          → IntentLayer (queries DB for intent prototypes)
          → Output
    
    Learning updates go to the database, not NN weights.
"""

import torch
import logging
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LayerActivation:
    """Activation state passed between layers."""
    data: Any
    activations: torch.Tensor
    confidence: float = 1.0
    source_layer: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


class DBBackedLayer(ABC):
    """
    Base class for database-backed plastic layers.
    
    The NN weights only control:
      - Attention/weighting for combining DB results
      - Threshold biases
      - Context blending
    
    All learned information goes to the database.
    """
    
    def __init__(
        self,
        layer_id: str,
        graph_store: Any,  # RelationalGraphStore or MultiTenantGraphManager
        learning_rate: float = 0.01,
        device: str = "cpu"
    ):
        self.id = layer_id
        self.graph = graph_store
        self.learning_rate = learning_rate
        self.device = device
        
        # Lightweight NN: only for combining/weighting, not storing info
        # Attention weights: how much to trust different sources
        self.attention_weights = torch.ones(4, device=device)  # [db_result, context, recency, frequency]
        
        # Threshold bias: shifts decision boundaries
        self.threshold_bias = torch.tensor(0.0, device=device)
        
        # Statistics (not learned info)
        self._forward_count = 0
        self._learn_count = 0
        
        # Cache for learning
        self._last_tokens: List[str] = []
        
    @abstractmethod
    def forward(self, activation: LayerActivation, tenant_id: Optional[str] = None) -> LayerActivation:
        """Process input through the layer, querying DB as needed."""
        pass
    
    @abstractmethod
    def learn(self, success: float, tenant_id: Optional[str] = None) -> None:
        """Update database with learned information."""
        pass
    
    def stats(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'forward_count': self._forward_count,
            'learn_count': self._learn_count,
            'attention_weights': self.attention_weights.tolist(),
            'threshold_bias': float(self.threshold_bias),
        }


class DBSalienceLayer(DBBackedLayer):
    """
    Salience Layer backed by database storage.
    
    Stores per-word salience values in the graph as node properties.
    NN only learns attention weights for combining salience signals.
    
    Database schema:
        Node: word:{word}
        Properties: salience (float), frequency (int), last_seen (timestamp)
    """
    
    # Bootstrap low salience words (stored in DB on first encounter)
    BOOTSTRAP_LOW_SALIENCE = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "of", "to", "in", "for", "on", "with", "at", "by", "from",
        "it", "this", "that", "these", "those",
        "what", "when", "where", "which", "who", "whose", "whom",
        "how", "why", "whether", "if",
        "i", "me", "my", "mine", "you", "your", "yours", 
        "he", "him", "his", "she", "her", "hers",
        "we", "us", "our", "ours", "they", "them", "their", "theirs",
        "do", "does", "did", "have", "has", "had", "can", "could",
        "will", "would", "shall", "should", "may", "might", "must",
        "know", "think", "find", "tell", "make", "take", "give",
        "want", "need", "like", "feel", "see", "look", "seem",
        "get", "got", "going", "come", "came", "went", "been",
        "about", "just", "only", "also", "very", "really", "too",
        "yes", "no", "not", "okay", "please", "thank", "sorry",
    }
    
    def __init__(
        self,
        layer_id: str = "layer.salience",
        graph_store: Any = None,
        learning_rate: float = 0.01,
        device: str = "cpu"
    ):
        super().__init__(layer_id, graph_store, learning_rate, device)
        
        # Small cache to avoid repeated DB lookups in same batch
        self._salience_cache: Dict[str, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
    def _get_word_salience(self, word: str, tenant_id: Optional[str] = None) -> float:
        """
        Get salience for a word from the database.
        Creates the node with bootstrap value if it doesn't exist.
        """
        word_lower = word.lower().rstrip("?!.,;:'\"")
        
        # Check cache first
        cache_key = f"{tenant_id}:{word_lower}"
        if cache_key in self._salience_cache:
            self._cache_hits += 1
            return self._salience_cache[cache_key]
        
        self._cache_misses += 1
        
        node_id = f"word:{word_lower}"
        
        # Try to get from database
        if self.graph:
            try:
                node = self.graph.get_node(node_id, tenant_id=tenant_id)
                if node:
                    # Node exists, get salience from data
                    data = node.get('data', {})
                    if isinstance(data, str):
                        import json
                        try:
                            data = json.loads(data)
                        except:
                            data = {}
                    salience = data.get('salience', 0.5)
                    self._salience_cache[cache_key] = salience
                    return salience
            except Exception as e:
                logger.debug(f"DB lookup failed for {word_lower}: {e}")
        
        # Node doesn't exist - create with bootstrap value
        if word_lower in self.BOOTSTRAP_LOW_SALIENCE:
            salience = 0.1  # Low salience for function words
        else:
            salience = 0.7  # Default high salience for content words
        
        # Store in database
        if self.graph:
            try:
                self.graph.add_node(
                    node_id, "word", word_lower,
                    confidence=salience,
                    data={'salience': salience, 'frequency': 1},
                    tenant_id=tenant_id
                )
            except Exception as e:
                logger.debug(f"DB write failed for {word_lower}: {e}")
        
        self._salience_cache[cache_key] = salience
        return salience
    
    def _update_word_salience(self, word: str, delta: float, tenant_id: Optional[str] = None) -> None:
        """
        Update salience for a word in the database.
        
        This is the learning step - information goes TO the database.
        """
        word_lower = word.lower().rstrip("?!.,;:'\"")
        node_id = f"word:{word_lower}"
        cache_key = f"{tenant_id}:{word_lower}"
        
        if not self.graph:
            return
            
        try:
            import json
            
            # Get current value
            node = self.graph.get_node(node_id, tenant_id=tenant_id)
            if node:
                data = node.get('data', {})
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except:
                        data = {}
                current_salience = data.get('salience', 0.5)
                frequency = data.get('frequency', 1)
            else:
                current_salience = 0.5
                frequency = 0
            
            # Apply update with decay toward neutral
            new_salience = current_salience + self.learning_rate * delta
            new_salience = max(0.0, min(1.0, new_salience))  # Clamp to [0, 1]
            
            # Store updated value
            self.graph.add_node(
                node_id, "word", word_lower,
                confidence=new_salience,
                data={
                    'salience': new_salience,
                    'frequency': frequency + 1
                },
                tenant_id=tenant_id
            )
            
            # Update cache
            self._salience_cache[cache_key] = new_salience
            
        except Exception as e:
            logger.debug(f"Salience update failed for {word_lower}: {e}")
    
    def forward(self, activation: LayerActivation, tenant_id: Optional[str] = None) -> LayerActivation:
        """
        Compute salience for each token by querying the database.
        """
        self._forward_count += 1
        
        tokens = activation.data
        if isinstance(tokens, str):
            tokens = tokens.split()
        
        # Query database for each token's salience
        saliences = []
        for token in tokens:
            sal = self._get_word_salience(token, tenant_id)
            saliences.append(sal)
        
        salience_tensor = torch.tensor(saliences, device=self.device)
        
        # Apply NN threshold bias (lightweight modification)
        adjusted_salience = salience_tensor + self.threshold_bias
        
        # Store for learning
        self._last_tokens = [t.lower().rstrip("?!.,;:'\"") for t in tokens]
        self._last_saliences = saliences
        
        # Extract salient words for downstream layers
        salient_words = [t for t, s in zip(self._last_tokens, saliences) if s > 0.5]
        
        return LayerActivation(
            data=tokens,
            activations=torch.sigmoid(adjusted_salience * 4),  # Scale for sharper decisions
            confidence=activation.confidence,
            source_layer=self.id,
            context={**activation.context, 'raw_saliences': saliences, 'salient_words': salient_words}
        )
    
    def learn(self, success: float, tenant_id: Optional[str] = None) -> None:
        """
        Update salience values in the database based on feedback.
        
        Positive feedback → increase salience for tokens that were used
        Negative feedback → decrease salience
        """
        self._learn_count += 1
        
        if not self._last_tokens:
            return
        
        # Update salience for each token that was processed
        for token in self._last_tokens:
            # Hebbian: words that led to good outcomes become more salient
            delta = success * 0.1  # Gentle updates
            self._update_word_salience(token, delta, tenant_id)
    
    def get_salient_words(self, text: str, threshold: float = 0.5, tenant_id: Optional[str] = None) -> List[str]:
        """
        Filter text to only salient words.
        
        This is the main API for replacing stop word filtering.
        """
        tokens = text.split()
        salient = []
        
        for token in tokens:
            clean = token.lower().rstrip("?!.,;:'\"")
            if len(clean) > 2:
                salience = self._get_word_salience(token, tenant_id)
                if salience > threshold:
                    salient.append(clean)
        
        return salient
    
    def is_salient(self, word: str, threshold: float = 0.5, tenant_id: Optional[str] = None) -> bool:
        """
        Check if a single word is salient.
        
        This is the drop-in replacement for:
            if word not in stop_words:  # OLD
            if layer.is_salient(word):  # NEW
        """
        clean = word.lower().rstrip("?!.,;:'\"")
        if len(clean) <= 2:
            return False
        salience = self._get_word_salience(clean, tenant_id)
        return salience > threshold
    
    def teach_salience(self, word: str, is_salient: bool, tenant_id: Optional[str] = None) -> None:
        """
        Explicitly teach that a word is/isn't salient.
        
        Goes directly to database.
        """
        target = 0.9 if is_salient else 0.1
        word_lower = word.lower()
        node_id = f"word:{word_lower}"
        
        if self.graph:
            import json
            self.graph.add_node(
                node_id, "word", word_lower,
                confidence=target,
                data={'salience': target, 'frequency': 1, 'taught': True},
                tenant_id=tenant_id
            )
            
            # Update cache
            cache_key = f"{tenant_id}:{word_lower}"
            self._salience_cache[cache_key] = target
            
        logger.debug(f"Taught: '{word}' salience = {target}")
    
    def clear_cache(self) -> None:
        """Clear the salience cache (e.g., when switching tenants)."""
        self._salience_cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        base = super().stats()
        base.update({
            'cache_size': len(self._salience_cache),
            'cache_hit_rate': self._cache_hits / max(1, self._cache_hits + self._cache_misses),
        })
        return base


class DBPatternLayer(DBBackedLayer):
    """
    Pattern Layer backed by database storage.
    
    Stores learned patterns as graph edges:
        pattern:{pattern_id} --[matches]--> word:{word}
    
    NN only learns attention weights for combining pattern matches.
    """
    
    def __init__(
        self,
        layer_id: str = "layer.pattern",
        graph_store: Any = None,
        learning_rate: float = 0.01,
        device: str = "cpu"
    ):
        super().__init__(layer_id, graph_store, learning_rate, device)
        self._pattern_count = 0
        
    def forward(self, activation: LayerActivation, tenant_id: Optional[str] = None) -> LayerActivation:
        """
        Match input against patterns in the database.
        """
        self._forward_count += 1
        
        # For now, pass through with pattern matching placeholder
        # Full implementation would query graph for pattern nodes
        
        tokens = activation.data
        if isinstance(tokens, str):
            tokens = tokens.split()
            
        self._last_tokens = tokens
        
        return LayerActivation(
            data={'tokens': tokens, 'patterns': []},
            activations=activation.activations,
            confidence=activation.confidence * 0.9,
            source_layer=self.id,
            context=activation.context
        )
    
    def learn(self, success: float, tenant_id: Optional[str] = None) -> None:
        """Store successful token sequences as patterns."""
        self._learn_count += 1
        
        if not self._last_tokens or len(self._last_tokens) < 2:
            return
            
        if success > 0.5 and self.graph:
            # Store the token sequence as a pattern
            pattern_id = f"pattern:{self._pattern_count}"
            try:
                import json
                # Create pattern node
                self.graph.add_node(
                    pattern_id, "pattern", " ".join(self._last_tokens[:5]),
                    confidence=success,
                    data={'tokens': self._last_tokens[:5], 'success': success},
                    tenant_id=tenant_id
                )
                self._pattern_count += 1
            except Exception as e:
                logger.debug(f"Pattern storage failed: {e}")


class DBIntentLayer(DBBackedLayer):
    """
    Intent Layer backed by database storage.
    
    Stores intent prototypes as graph nodes:
        intent:{intent_name}
    
    With edges to associated patterns.
    """
    
    # Bootstrap intents
    BOOTSTRAP_INTENTS = [
        "question", "statement", "command", "teaching",
        "feedback_positive", "feedback_negative", "greeting", "farewell"
    ]
    
    def __init__(
        self,
        layer_id: str = "layer.intent",
        graph_store: Any = None,
        learning_rate: float = 0.01,
        device: str = "cpu"
    ):
        super().__init__(layer_id, graph_store, learning_rate, device)
        self._ensure_bootstrap_intents()
        
    def _ensure_bootstrap_intents(self) -> None:
        """Create bootstrap intent nodes if they don't exist."""
        if not self.graph:
            return
            
        for intent in self.BOOTSTRAP_INTENTS:
            node_id = f"intent:{intent}"
            try:
                import json
                self.graph.add_node(
                    node_id, "intent", intent,
                    confidence=0.5,
                    data={'bootstrap': True},
                    tenant_id=None  # Base tenant
                )
            except:
                pass  # Already exists
    
    def forward(self, activation: LayerActivation, tenant_id: Optional[str] = None) -> LayerActivation:
        """
        Detect intent by matching against database prototypes.
        """
        self._forward_count += 1
        
        # Simple heuristics for now - full implementation would use
        # stored intent-pattern associations from the graph
        data = activation.data
        tokens = data.get('tokens', []) if isinstance(data, dict) else data
        
        if isinstance(tokens, str):
            tokens = tokens.split()
        
        text = " ".join(tokens).lower()
        
        # Simple classification (to be replaced by learned associations)
        if "?" in text or text.startswith(("what", "how", "why", "when", "where", "who")):
            intent = "question"
            confidence = 0.7
        elif text.startswith(("please", "can you", "could you", "would you")):
            intent = "command"
            confidence = 0.6
        elif any(w in text for w in ["thanks", "thank", "good", "great", "nice"]):
            intent = "feedback_positive"
            confidence = 0.6
        elif any(w in text for w in ["no", "wrong", "bad", "not right"]):
            intent = "feedback_negative"
            confidence = 0.6
        elif any(w in text for w in ["is a", "means", "is the", "are"]) and "?" not in text:
            intent = "teaching"
            confidence = 0.5
        else:
            intent = "statement"
            confidence = 0.4
        
        self._last_intent = intent
        
        # Get salient words from context (passed from salience layer)
        salient_words = activation.context.get('salient_words', [])
        
        return LayerActivation(
            data={
                'tokens': tokens,
                'intent': intent,
                'salient_words': salient_words,
            },
            activations=torch.tensor([confidence], device=self.device),
            confidence=confidence,
            source_layer=self.id,
            context=activation.context
        )
    
    def learn(self, success: float, tenant_id: Optional[str] = None) -> None:
        """Update intent associations based on feedback."""
        self._learn_count += 1
        
        if not hasattr(self, '_last_intent') or not self.graph:
            return
            
        # Could store token-intent associations for future matching
        # For now, just track statistics


class DBSynthesisLayer(DBBackedLayer):
    """
    Synthesis Layer backed by database storage.
    
    Stores response templates/patterns in the graph.
    NN only learns selection weights for choosing responses.
    
    Database schema:
        Node: template:{hash}
        Properties: text, intent, trigger, success_score, usage_count
        
        Edge: template:{hash} --[triggered_by]--> intent:{intent}
        Edge: template:{hash} --[responds_to]--> word:{salient_word}
    """
    
    # Bootstrap templates for common intents
    BOOTSTRAP_TEMPLATES = {
        "question": [
            "I'm still learning about that. Can you tell me more?",
            "That's interesting! What made you curious about {topic}?",
            "I don't know much about {topic} yet. Could you explain?",
        ],
        "teaching": [
            "I see! So {topic} is important. I'll remember that.",
            "Thank you for teaching me about {topic}.",
            "Got it! {topic} - I'm adding that to my understanding.",
        ],
        "statement": [
            "I understand. You mentioned {topic}.",
            "Interesting point about {topic}.",
            "I see what you mean about {topic}.",
        ],
        "greeting": [
            "Hello! How can I help you?",
            "Hi there! What would you like to talk about?",
            "Hey! What's on your mind?",
        ],
        "feedback_positive": [
            "That's great! I'm glad I could help.",
            "Wonderful! I'm learning!",
            "Excellent! Thank you for the feedback.",
        ],
        "feedback_negative": [
            "I see. I'll try to do better.",
            "Sorry about that. Let me try again.",
            "Thank you for the correction. I'm learning.",
        ],
        "unknown": [
            "I'm not sure I understand. Could you rephrase that?",
            "Hmm, I don't quite follow. Can you explain?",
            "I'm still learning. Could you help me understand?",
        ],
    }
    
    def __init__(
        self,
        layer_id: str = "layer.synthesis",
        graph_store: Any = None,
        learning_rate: float = 0.02,
        device: str = "cpu"
    ):
        super().__init__(layer_id, graph_store, learning_rate, device)
        
        # Temperature for response selection (higher = more random)
        self.temperature = torch.nn.Parameter(torch.tensor(1.0))
        
        # Cache for templates
        self._template_cache: Dict[str, List[Dict]] = {}
        
        # Last selection for learning
        self._last_selected_template: Optional[str] = None
        self._last_intent: Optional[str] = None
        self._last_salient_words: List[str] = []
        
        # Bootstrap templates on first use
        self._bootstrapped: Set[str] = set()
        
    def _get_template_id(self, text: str) -> str:
        """Generate unique ID for a template."""
        return f"template:{hashlib.md5(text.encode()).hexdigest()[:12]}"
    
    def _bootstrap_intent(self, intent: str, tenant_id: Optional[str] = None) -> None:
        """Bootstrap templates for an intent if not already done."""
        cache_key = f"{tenant_id}:{intent}"
        if cache_key in self._bootstrapped:
            return
            
        templates = self.BOOTSTRAP_TEMPLATES.get(intent, self.BOOTSTRAP_TEMPLATES["unknown"])
        
        if self.graph:
            # First, ensure the intent node exists (required for edge FK constraint)
            try:
                self.graph.add_node(
                    f"intent:{intent}", "intent", intent,
                    confidence=0.5,
                    tenant_id=tenant_id
                )
            except Exception as e:
                logger.debug(f"Intent node create: {e}")  # May already exist
            
            for text in templates:
                template_id = self._get_template_id(text)
                try:
                    self.graph.add_node(
                        template_id, "template", text[:50],
                        confidence=0.5,
                        data={
                            'text': text,
                            'intent': intent,
                            'success_score': 0.5,
                            'usage_count': 0,
                        },
                        tenant_id=tenant_id
                    )
                    # Link to intent
                    self.graph.add_edge(
                        template_id, f"intent:{intent}",
                        "triggered_by", confidence=0.7,
                        tenant_id=tenant_id
                    )
                except Exception as e:
                    logger.debug(f"Bootstrap template failed: {e}")
            
            # Commit bootstrap data so it's visible to subsequent queries
            if hasattr(self.graph, '_conn'):
                self.graph._conn.commit()
                    
        self._bootstrapped.add(cache_key)
    
    def _get_templates_for_intent(
        self, 
        intent: str, 
        tenant_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get response templates for an intent from database.
        """
        # Ensure bootstrapped
        self._bootstrap_intent(intent, tenant_id)
        
        if not self.graph:
            return []
            
        try:
            # Query templates linked to this intent
            # Get edges pointing to intent
            store = self.graph._get_store(tenant_id) if hasattr(self.graph, '_get_store') else self.graph
            
            rows = store._conn.execute("""
                SELECT n.id, n.term, n.confidence, n.data
                FROM edges e
                JOIN nodes n ON e.source = n.id
                WHERE e.target = ? AND e.type = 'triggered_by'
                ORDER BY n.confidence DESC
                LIMIT ?
            """, (f"intent:{intent}", limit)).fetchall()
            
            templates = []
            for row in rows:
                data = row['data']
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except:
                        data = {}
                templates.append({
                    'id': row['id'],
                    'text': data.get('text', row['term']),
                    'success_score': data.get('success_score', row['confidence']),
                    'usage_count': data.get('usage_count', 0),
                })
            return templates
            
        except Exception as e:
            logger.debug(f"Template query failed: {e}")
            return []
    
    def forward(
        self, 
        activation: LayerActivation, 
        tenant_id: Optional[str] = None
    ) -> LayerActivation:
        """
        Select a response template based on intent and context.
        """
        self._forward_count += 1
        
        # Get intent from previous layer
        intent_data = activation.data
        intent = "unknown"
        salient_words = []
        
        if isinstance(intent_data, dict):
            intent = intent_data.get('intent', 'unknown')
            salient_words = intent_data.get('salient_words', [])
        
        # Get candidate templates
        templates = self._get_templates_for_intent(intent, tenant_id)
        
        if not templates:
            # Fallback
            selected_text = f"I hear you. [intent: {intent}]"
            self._last_selected_template = None
        else:
            # Select based on success scores with temperature
            scores = torch.tensor([t['success_score'] for t in templates])
            probs = torch.softmax(scores / float(self.temperature), dim=0)
            
            # Sample or argmax based on temperature
            if float(self.temperature) > 0.5:
                idx = torch.multinomial(probs, 1).item()
            else:
                idx = torch.argmax(probs).item()
                
            selected = templates[idx]
            selected_text = selected['text']
            self._last_selected_template = selected['id']
            
            # Fill in template variables
            if salient_words:
                topic = salient_words[0] if salient_words else "that"
                selected_text = selected_text.replace("{topic}", topic)
        
        self._last_intent = intent
        self._last_salient_words = salient_words
        
        return LayerActivation(
            data={
                'response': selected_text,
                'intent': intent,
                'template_id': self._last_selected_template,
            },
            activations=activation.activations,
            confidence=activation.confidence,
            source_layer=self.id,
            context={**activation.context, 'response': selected_text}
        )
    
    def learn(self, success: float, tenant_id: Optional[str] = None) -> None:
        """
        Update template success scores in database.
        
        Positive feedback → increase score, decrease temperature
        Negative feedback → decrease score, increase temperature (explore more)
        """
        self._learn_count += 1
        
        if not self._last_selected_template or not self.graph:
            return
            
        try:
            # Get current template data
            store = self.graph._get_store(tenant_id) if hasattr(self.graph, '_get_store') else self.graph
            
            row = store._conn.execute(
                "SELECT data, confidence FROM nodes WHERE id = ?",
                (self._last_selected_template,)
            ).fetchone()
            
            if row:
                data = row['data']
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except:
                        data = {}
                
                # Update success score
                old_score = data.get('success_score', 0.5)
                new_score = old_score + self.learning_rate * success
                new_score = max(0.0, min(1.0, new_score))
                
                # Update usage count
                usage = data.get('usage_count', 0) + 1
                
                data['success_score'] = new_score
                data['usage_count'] = usage
                
                # Write back
                store._conn.execute(
                    "UPDATE nodes SET confidence = ?, data = ? WHERE id = ?",
                    (new_score, json.dumps(data), self._last_selected_template)
                )
                store._conn.commit()
                
            # Adjust temperature based on feedback
            if success > 0:
                # Success → lower temperature (exploit)
                self.temperature.data = torch.clamp(self.temperature.data * 0.99, 0.1, 2.0)
            else:
                # Failure → higher temperature (explore)
                self.temperature.data = torch.clamp(self.temperature.data * 1.01, 0.1, 2.0)
                
        except Exception as e:
            logger.debug(f"Template learning failed: {e}")
    
    def add_template(
        self,
        text: str,
        intent: str,
        trigger_words: Optional[List[str]] = None,
        tenant_id: Optional[str] = None
    ) -> str:
        """
        Add a new response template to the database.
        
        This is for explicit teaching:
            "When I ask about X, you could say Y."
        """
        if not self.graph:
            return ""
            
        template_id = self._get_template_id(text)
        
        try:
            self.graph.add_node(
                template_id, "template", text[:50],
                confidence=0.6,  # Start above default
                data={
                    'text': text,
                    'intent': intent,
                    'success_score': 0.6,
                    'usage_count': 0,
                },
                tenant_id=tenant_id
            )
            
            # Link to intent
            self.graph.add_edge(
                template_id, f"intent:{intent}",
                "triggered_by", confidence=0.8,
                tenant_id=tenant_id
            )
            
            # Link to trigger words
            if trigger_words:
                for word in trigger_words:
                    self.graph.add_edge(
                        template_id, f"word:{word.lower()}",
                        "responds_to", confidence=0.7,
                        tenant_id=tenant_id
                    )
                    
            logger.debug(f"Added template: {template_id} for intent {intent}")
            return template_id
            
        except Exception as e:
            logger.debug(f"Add template failed: {e}")
            return ""
    
    def stats(self) -> Dict[str, Any]:
        base = super().stats()
        base['temperature'] = float(self.temperature)
        base['bootstrapped_intents'] = len(self._bootstrapped)
        return base


class DBBackedLayerStack:
    """
    Stack of database-backed layers.
    
    All learned information goes to the database.
    NNs only control navigation/attention.
    """
    
    def __init__(
        self,
        graph_store: Any,
        device: str = "cpu"
    ):
        self.graph = graph_store
        self.device = device
        
        # Create the layer stack
        self.salience = DBSalienceLayer(graph_store=graph_store, device=device)
        self.pattern = DBPatternLayer(graph_store=graph_store, device=device)
        self.intent = DBIntentLayer(graph_store=graph_store, device=device)
        self.synthesis = DBSynthesisLayer(graph_store=graph_store, device=device)
        
        self.layers = [self.salience, self.pattern, self.intent, self.synthesis]
        
    def forward(self, text: str, tenant_id: Optional[str] = None) -> LayerActivation:
        """Process input through all layers."""
        tokens = text.split()
        activation = LayerActivation(
            data=tokens,
            activations=torch.ones(len(tokens), device=self.device),
            confidence=1.0,
            source_layer="input"
        )
        
        for layer in self.layers:
            activation = layer.forward(activation, tenant_id=tenant_id)
            
        return activation
    
    def learn(self, success: float, tenant_id: Optional[str] = None) -> None:
        """Propagate learning to all layers (updates database)."""
        for layer in self.layers:
            layer.learn(success, tenant_id=tenant_id)
    
    def stats(self) -> Dict[str, Any]:
        return {layer.id: layer.stats() for layer in self.layers}


def create_db_backed_processor(
    graph_store: Any,
    device: str = "cpu"
) -> DBBackedLayerStack:
    """
    Create a database-backed processing stack.
    
    This follows the v1 principle:
      - Database stores information
      - NN navigates/combines
      - Infinitely scalable, lightweight
    """
    return DBBackedLayerStack(graph_store=graph_store, device=device)
