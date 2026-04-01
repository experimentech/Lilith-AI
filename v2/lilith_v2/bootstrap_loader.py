"""
Capability Bootstrap Loader

Loads capability_bootstrap.json and applies it to:
1. Response patterns -> ResponseComposer
2. Magic phrases -> Semantic encoder + action planner
3. Self-knowledge -> Knowledge graph
4. Semantic pairs -> Encoder training

Usage:
    from v2.lilith_v2.bootstrap_loader import BootstrapLoader
    loader = BootstrapLoader(cognitive_stage, data_path="data/seed")
    loader.apply()
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BootstrapLoader:
    """
    Loads capability bootstrap data into a CognitiveStage.
    
    This seeds the system with:
    - Self-knowledge about its identity and capabilities
    - Magic phrases that trigger internal abilities  
    - Response patterns for common interactions
    - Action bindings for tool invocations
    - Semantic pairs for encoder training
    """
    
    def __init__(
        self,
        cognitive_stage,  # CognitiveStage instance (duck-typed to avoid circular import)
        data_path: str = "data/seed",
        bootstrap_file: str = "capability_bootstrap.json",
    ):
        self.stage = cognitive_stage
        self.data_path = Path(data_path)
        self.bootstrap_file = self.data_path / bootstrap_file
        self._data: Optional[Dict[str, Any]] = None
        
    def load(self) -> Dict[str, Any]:
        """Load bootstrap data from JSON file."""
        if self._data is not None:
            return self._data
            
        if not self.bootstrap_file.exists():
            logger.warning(f"Bootstrap file not found: {self.bootstrap_file}")
            return {}
            
        try:
            with open(self.bootstrap_file, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
            logger.info(f"Loaded bootstrap data from {self.bootstrap_file}")
            return self._data
        except Exception as e:
            logger.error(f"Failed to load bootstrap data: {e}")
            return {}
    
    def apply(self, tenant_id: Optional[str] = None) -> Dict[str, int]:
        """
        Apply all bootstrap data to the cognitive stage.
        
        Returns:
            Dict with counts of applied items
        """
        data = self.load()
        if not data:
            return {"error": "No bootstrap data loaded"}
            
        stats = {
            "self_knowledge": 0,
            "magic_phrases": 0,
            "response_patterns": 0,
            "action_bindings": 0,
            "semantic_pairs": 0,
            "foundational_statements": 0,
        }
        
        # 1. Apply self-knowledge to graph
        if "self_knowledge" in data:
            stats["self_knowledge"] = self._apply_self_knowledge(
                data["self_knowledge"], tenant_id
            )
            
        # 2. Apply magic phrases (trains encoder + stores mappings)
        if "magic_phrases" in data:
            stats["magic_phrases"] = self._apply_magic_phrases(
                data["magic_phrases"], tenant_id
            )
            
        # 3. Apply response patterns
        if "response_patterns" in data:
            stats["response_patterns"] = self._apply_response_patterns(
                data["response_patterns"], tenant_id
            )
            
        # 4. Apply action bindings (register with planner)
        if "action_bindings" in data:
            stats["action_bindings"] = self._apply_action_bindings(
                data["action_bindings"], tenant_id
            )
            
        # 5. Apply semantic bootstrap (train encoder)
        if "semantic_bootstrap" in data:
            stats["semantic_pairs"] = self._apply_semantic_bootstrap(
                data["semantic_bootstrap"]
            )
            # Also apply foundational statements
            if "foundational_statements" in data["semantic_bootstrap"]:
                stats["foundational_statements"] = self._apply_foundational_statements(
                    data["semantic_bootstrap"]["foundational_statements"]
                )
            
        logger.info(f"Bootstrap applied: {stats}")
        return stats
    
    def _apply_self_knowledge(
        self, 
        knowledge: Dict[str, Any], 
        tenant_id: Optional[str]
    ) -> int:
        """Store self-knowledge in graph."""
        count = 0
        graph = self.stage.graph
        
        # Store identity as a concept node
        identity = knowledge.get("identity", {})
        if identity:
            try:
                # Create or update self-knowledge node
                description = identity.get("description", "")
                graph.add_node(
                    node_id="self:identity",
                    node_type="self_knowledge",
                    term=identity.get("name", "Lilith"),
                    data={
                        "name": identity.get("name", "Lilith"),
                        "type": identity.get("type", "ai"),
                        "description": description,
                    },
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to store identity: {e}")
        
        # Store capabilities as connected nodes
        capabilities = knowledge.get("capabilities", [])
        for i, cap in enumerate(capabilities):
            try:
                cap_id = f"self:capability:{i}"
                graph.add_node(
                    node_id=cap_id,
                    node_type="capability",
                    term=f"capability_{i}",
                    data={"description": cap},
                )
                # Link to identity
                graph.add_edge(
                    source="self:identity",
                    target=cap_id,
                    edge_type="has_capability",
                )
                count += 1
            except Exception as e:
                logger.debug(f"Failed to store capability: {e}")
                
        # Store limitations
        limitations = knowledge.get("limitations", [])
        for i, lim in enumerate(limitations):
            try:
                lim_id = f"self:limitation:{i}"
                graph.add_node(
                    node_id=lim_id,
                    node_type="limitation",
                    term=f"limitation_{i}",
                    data={"description": lim},
                )
                graph.add_edge(
                    source="self:identity",
                    target=lim_id,
                    edge_type="has_limitation",
                )
                count += 1
            except Exception as e:
                logger.debug(f"Failed to store limitation: {e}")
                
        return count
    
    def _apply_magic_phrases(
        self, 
        phrases: Dict[str, Any], 
        tenant_id: Optional[str]
    ) -> int:
        """
        Apply magic phrases - these train the encoder to associate
        trigger phrases with ability concepts.
        """
        count = 0
        encoder = self.stage.encoder
        graph = self.stage.graph
        
        # Process each ability category
        for category, bindings in phrases.items():
            if category.startswith("_"):  # Skip metadata
                continue
                
            if not isinstance(bindings, list):
                continue
                
            for binding in bindings:
                triggers = binding.get("triggers", [])
                ability = binding.get("ability", "")
                action = binding.get("action", "")
                description = binding.get("description", "")
                
                # Store ability as a graph node
                ability_id = f"ability:{ability}:{action}"
                try:
                    graph.add_node(
                        node_id=ability_id,
                        node_type="ability",
                        term=action,
                        data={
                            "ability": ability,
                            "action": action,
                            "description": description,
                            "triggers": triggers,
                        },
                    )
                except Exception as e:
                    logger.debug(f"Failed to store ability node: {e}")
                
                # Train encoder to associate triggers with the ability
                if hasattr(encoder, 'add_words'):
                    try:
                        # Add ability concept to vocabulary
                        encoder.add_words([ability, action] + triggers)
                        
                        # Link trigger to ability via graph edges
                        for trigger in triggers:
                            trigger_id = f"trigger:{trigger.replace(' ', '_')}"
                            graph.add_node(
                                node_id=trigger_id,
                                node_type="trigger",
                                term=trigger,
                                data={"category": category},
                            )
                            graph.add_edge(
                                source=trigger_id,
                                target=ability_id,
                                edge_type="triggers",
                            )
                        count += 1
                    except Exception as e:
                        logger.debug(f"Failed to train trigger: {e}")
                        
        return count
    
    def _apply_response_patterns(
        self, 
        patterns: Dict[str, Any], 
        tenant_id: Optional[str]
    ) -> int:
        """
        Store response patterns for pattern-based response generation.
        """
        count = 0
        
        # Access response store through stage config or direct attribute
        response_store = self.stage.config.get("response_store")
        if response_store is None and hasattr(self.stage, 'response_store'):
            response_store = self.stage.response_store
            
        if response_store is None:
            logger.debug("No response store available for pattern loading")
            return 0
            
        for category, items in patterns.items():
            if category.startswith("_"):  # Skip metadata
                continue
                
            if not isinstance(items, list):
                continue
                
            for item in items:
                input_text = item.get("input", "")
                responses = item.get("responses", [])
                context = item.get("context", category)
                
                for response in responses:
                    try:
                        # Store as a response pattern
                        # The RelationalStore should have a method for this
                        response_store.execute("""
                            INSERT OR IGNORE INTO response_patterns 
                            (input_text, response_text, context, score)
                            VALUES (?, ?, ?, 1.0)
                        """, (input_text, response, context))
                        count += 1
                    except Exception as e:
                        # Try alternative storage method
                        try:
                            response_store.store({
                                "type": "response_pattern",
                                "input": input_text,
                                "response": response,
                                "context": context,
                            })
                            count += 1
                        except Exception:
                            logger.debug(f"Failed to store response pattern: {e}")
                            
        return count
    
    def _apply_action_bindings(
        self, 
        bindings: Dict[str, Any], 
        tenant_id: Optional[str]
    ) -> int:
        """
        Register action bindings with the action planner.
        """
        count = 0
        actions = bindings.get("actions", [])
        
        if not hasattr(self.stage, 'register_action'):
            logger.debug("Stage doesn't support action registration")
            return 0
            
        for action in actions:
            name = action.get("name", "")
            description = action.get("description", "")
            tool_binding = action.get("tool_binding", "")
            arg_template = action.get("arg_template", {})
            triggers = action.get("triggers", [])
            
            if not name or not tool_binding:
                continue
                
            try:
                # Register with action planner
                result = self.stage.register_action(
                    name=name,
                    description=description + f" Triggers: {', '.join(triggers)}",
                    tool_binding=tool_binding,
                    arg_template=arg_template,
                    tenant_id=tenant_id,
                )
                if result:
                    count += 1
            except Exception as e:
                logger.debug(f"Failed to register action {name}: {e}")
                
        return count
    
    def _apply_semantic_bootstrap(self, semantic: Dict[str, Any]) -> int:
        """
        Train encoder on semantic relationships.
        """
        count = 0
        encoder = self.stage.encoder
        
        # Get base encoder if wrapped
        base_encoder = getattr(encoder, 'base_encoder', encoder)
        
        # First, add all words to vocabulary
        all_words = set()
        
        # Collect words from synonyms
        synonyms = semantic.get("synonyms", [])
        for group in synonyms:
            all_words.update(group)
        
        # Collect words from antonyms
        antonyms = semantic.get("antonyms", [])
        for pair in antonyms:
            all_words.update(pair)
            
        # Add to vocabulary
        if hasattr(encoder, 'add_words'):
            encoder.add_words(list(all_words))
            count = len(all_words)
            
        # If encoder supports word pair loss, do a few training steps
        if hasattr(base_encoder, 'compute_word_pair_loss'):
            import torch.optim as optim
            import torch
            
            # Build training pairs
            pairs = []
            for group in synonyms:
                if len(group) >= 2:
                    for i in range(len(group)):
                        for j in range(i + 1, len(group)):
                            pairs.append((group[i], group[j], 1.0))
                            
            for pair in antonyms:
                if len(pair) == 2:
                    pairs.append((pair[0], pair[1], -1.0))
            
            if pairs:
                params = list(base_encoder.embedding.parameters())
                if params:
                    optimizer = optim.Adam(params, lr=0.01)
                    
                    # Few training steps
                    for _ in range(10):
                        optimizer.zero_grad()
                        total_loss = torch.tensor(0.0)
                        
                        for w1, w2, target in pairs[:20]:  # Limit per step
                            try:
                                target_sim = 0.9 if target > 0 else -0.5
                                loss = base_encoder.compute_word_pair_loss(
                                    w1, w2, torch.tensor(target_sim)
                                )
                                total_loss = total_loss + loss
                            except Exception:
                                pass
                        
                        if total_loss.requires_grad:
                            total_loss.backward()
                            optimizer.step()
                    
                    count += len(pairs)
            
        return count
    
    def _apply_foundational_statements(self, statements: List[Dict[str, Any]]) -> int:
        """
        Apply foundational statements that teach semantic relationships.
        
        These are natural language statements like:
        - "hot is the opposite of cold" → antonym
        - "a dog is an animal" → hypernym
        
        This bootstraps the core contrastive learning knowledge.
        """
        count = 0
        encoder = self.stage.encoder
        base_encoder = getattr(encoder, 'base_encoder', encoder)
        
        # Map relation types to training targets
        target_map = {
            "antonym": -0.7,
            "negative": -0.3,
            "synonym": 0.9,
            "hypernym": 0.9,
            "property": 0.6,
            "causal": 0.5,
        }
        
        # Add all terms to vocabulary first
        all_terms = set()
        for stmt in statements:
            terms = stmt.get("terms", [])
            all_terms.update(terms)
            
        if hasattr(encoder, 'add_words') and all_terms:
            encoder.add_words(list(all_terms))
            
        # Train on pairs if encoder supports it
        if hasattr(base_encoder, 'compute_word_pair_loss'):
            import torch
            import torch.optim as optim
            
            params = []
            if hasattr(base_encoder, 'embedding'):
                params = list(base_encoder.embedding.parameters())
            elif hasattr(encoder, 'get_trainable_parameters'):
                params = encoder.get_trainable_parameters()
                
            if params:
                optimizer = optim.Adam(params, lr=0.01)
                
                for stmt in statements:
                    terms = stmt.get("terms", [])
                    rel_type = stmt.get("type", "")
                    
                    if len(terms) < 2:
                        continue
                        
                    term1, term2 = terms[0], terms[1]
                    target = target_map.get(rel_type, 0.0)
                    
                    if target == 0.0:
                        continue
                        
                    try:
                        # A few gradient steps per pair
                        for _ in range(3):
                            optimizer.zero_grad()
                            loss = base_encoder.compute_word_pair_loss(
                                term1, term2, torch.tensor(target)
                            )
                            if loss.requires_grad:
                                loss.backward()
                                optimizer.step()
                        count += 1
                    except Exception as e:
                        logger.debug(f"Failed to train '{terms}': {e}")
                        
        return count


def bootstrap_cognitive_stage(
    stage,
    data_path: str = "data/seed",
    tenant_id: Optional[str] = None,
) -> Dict[str, int]:
    """
    Convenience function to bootstrap a cognitive stage.
    
    Args:
        stage: CognitiveStage instance
        data_path: Path to seed data directory
        tenant_id: Optional tenant ID
        
    Returns:
        Statistics about applied bootstrap data
    """
    loader = BootstrapLoader(stage, data_path=data_path)
    return loader.apply(tenant_id=tenant_id)
