#!/usr/bin/env python3
"""
Seeding Utility for Lilith V2.
Ingests base data (patterns, QA, grammar) into the Multi-Tenant 'Base' (Teacher) Layer.
"""

import sys
import os
import json
import hashlib
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from v2.lilith_v2.multi_tenant_store import MultiTenantGraphManager
from v2.lilith_v2.semantic_extractor import SemanticExtractor
from v2.lilith_v2.generative_system import GenerativeSystem
from v2.lilith_v2.relational_store import RelationalStore
from v2.lilith_v2.pattern_store_adapter import PatternStoreAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("seed")

DATA_ROOT = Path("data")
SEED_ROOT = DATA_ROOT / "seed"

def get_hash(text: str) -> str:
    """Generate a stable ID for text chunks."""
    return hashlib.md5(text.lower().strip().encode("utf-8")).hexdigest()

def seed_qa(graph: MultiTenantGraphManager):
    """Ingest Q&A Bootstrap pairs."""
    qa_file = SEED_ROOT / "qa_bootstrap.txt"
    if not qa_file.exists():
        logger.warning(f"QA file not found: {qa_file}")
        return

    logger.info("Seeding QA pairs...")
    with open(qa_file, "r") as f:
        content = f.read()

    # Simple parse based on Q: / A: pattern
    # We split by "Q:" but need to be careful.
    
    lines = content.split('\n')
    current_q = None
    
    count = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        if line.startswith("Q:"):
            current_q = line[2:].strip()
        elif line.startswith("A:") and current_q:
            answer = line[2:].strip()
            
            # Persist Q
            q_id = f"q_{get_hash(current_q)}"
            graph.add_node(
                node_id=q_id,
                node_type="utterance",
                term=current_q,
                confidence=1.0,
                data={"source": "qa_seed", "role": "question"},
                tenant_id="teacher" # Explicitly base
            )
            
            # Persist A
            a_id = f"a_{get_hash(answer)}"
            graph.add_node(
                node_id=a_id,
                node_type="utterance",
                term=answer,
                confidence=1.0,
                data={"source": "qa_seed", "role": "answer"},
                tenant_id="teacher"
            )
            
            # Link
            graph.add_edge(q_id, a_id, "elicits_response", confidence=1.0, tenant_id="teacher")
            
            # Optional: Self-link for finding the q from the text
            # (handled by find_nodes_by_term)
            
            current_q = None
            count += 1
            
    logger.info(f"Ingested {count} QA pairs.")

def seed_patterns(graph: MultiTenantGraphManager):
    """Ingest Base Patterns (Classic Intent/Response)."""
    pat_file = SEED_ROOT / "base_patterns.json"
    if not pat_file.exists():
        logger.warning(f"Pattern file not found: {pat_file}")
        return
        
    logger.info("Seeding Patterns...")
    with open(pat_file, "r") as f:
        try:
            patterns = json.load(f)
        except json.JSONDecodeError:
            logger.error("Failed to parse base_patterns.json")
            return

    count = 0
    for p in patterns:
        trigger = p.get("trigger_context")
        response = p.get("response_text")
        intent = p.get("intent")
        
        if not trigger or not response:
            continue
            
        # Trigger Node
        t_id = f"trig_{get_hash(trigger)}"
        graph.add_node(
            node_id=t_id,
            node_type="pattern_trigger",
            term=trigger,
            confidence=1.0,
            data={"intent": intent, "source": "base_patterns"},
            tenant_id="teacher"
        )
        
        # Response Node
        r_id = f"resp_{get_hash(response)}"
        graph.add_node(
            node_id=r_id,
            node_type="utterance",
            term=response,
            confidence=1.0,
            data={"source": "base_patterns"},
            tenant_id="teacher"
        )
        
        # Link
        graph.add_edge(t_id, r_id, "elicits_response", confidence=1.0, tenant_id="teacher")
        count += 1
        
    logger.info(f"Ingested {count} conversation patterns.")

def seed_grammar(graph: MultiTenantGraphManager):
    """Learn Syntax Templates from Grammar Bootstrap."""
    gram_file = SEED_ROOT / "grammar_bootstrap.txt"
    if not gram_file.exists():
        logger.warning("Grammar file not found.")
        return

    logger.info("Seeding Grammar Templates...")
    
    extractor = SemanticExtractor()
    # We need to hack GenerativeSystem to use our graph manager
    gen = GenerativeSystem(graph)
    
    with open(gram_file, "r") as f:
        lines = f.readlines()
        
    count = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("="):
            continue
            
        # Extract relations
        relations = extractor.extract(line)
        if relations:
            # Learn (Persists to graph via gen.learn_grammar -> graph.add_node)
            # Must ensure GenerativeSystem uses correct tenant (None/Default) 
            # The current GenerativeSystem just calls graph.add_node.
            # MultiTenantGraphManager.add_node defaults to tenant_id=None (Teacher) if not supplied?
            # Let's check MultiTenantGraphManager signature: add_node(..., tenant_id=None).
            # Yes, default None maps to 'teacher' in our logic.
            gen.learn_grammar(line, relations)
            count += 1
            
    logger.info(f"Processed {count} grammar examples.")


def seed_response_patterns(pattern_store: PatternStoreAdapter):
    """
    Seed the ResponseComposer's pattern store with base patterns.
    These are used for pattern-based response generation with blending and learning.
    """
    pat_file = SEED_ROOT / "base_patterns.json"
    if not pat_file.exists():
        logger.warning(f"Pattern file not found for response composer: {pat_file}")
        return
        
    logger.info("Seeding ResponseComposer patterns...")
    with open(pat_file, "r") as f:
        try:
            patterns = json.load(f)
        except json.JSONDecodeError:
            logger.error("Failed to parse base_patterns.json")
            return

    count = 0
    for p in patterns:
        trigger = p.get("trigger_context")
        response = p.get("response_text")
        intent = p.get("intent", "general")
        
        if not trigger or not response:
            continue
        
        # Add to ResponseComposer's pattern store
        fragment_id = f"resp_{get_hash(trigger)}"
        pattern_store.add_pattern(
            fragment_id=fragment_id,
            trigger_context=trigger,
            response_text=response,
            intent=intent,
            success_score=0.75,  # Start with moderate confidence
            embedding=None,     # Will be computed at runtime if encoder available
        )
        count += 1
        
    logger.info(f"Ingested {count} response patterns for ResponseComposer.")


def seed_qa_response_patterns(pattern_store: PatternStoreAdapter):
    """
    Seed Q&A pairs into ResponseComposer for direct pattern matching.
    """
    qa_file = SEED_ROOT / "qa_bootstrap.txt"
    if not qa_file.exists():
        logger.warning(f"QA file not found for response composer: {qa_file}")
        return

    logger.info("Seeding QA into ResponseComposer...")
    with open(qa_file, "r") as f:
        content = f.read()

    lines = content.split('\n')
    current_q = None
    
    count = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        if line.startswith("Q:"):
            current_q = line[2:].strip()
        elif line.startswith("A:") and current_q:
            answer = line[2:].strip()
            
            # Add as response pattern
            fragment_id = f"qa_{get_hash(current_q)}"
            pattern_store.add_pattern(
                fragment_id=fragment_id,
                trigger_context=current_q,
                response_text=answer,
                intent="qa",
                success_score=0.9,  # QA patterns start with high confidence
                embedding=None,
            )
            
            current_q = None
            count += 1
            
    logger.info(f"Ingested {count} QA pairs into ResponseComposer.")


def main():
    print("Initializing Lilith V2 Seed Injector...")
    
    base_root = DATA_ROOT / "base"
    users_root = DATA_ROOT / "users"
    
    # Init Graph Manager
    graph_manager = MultiTenantGraphManager(str(base_root), str(users_root))
    
    # Init shared response pattern store (base layer)
    base_response_db = base_root / "responses.db"
    os.makedirs(base_root, exist_ok=True)
    response_store = RelationalStore(str(base_response_db))
    pattern_store = PatternStoreAdapter(response_store, "base_responses")
    
    # Run Seeds
    seed_qa(graph_manager)
    seed_patterns(graph_manager)
    seed_grammar(graph_manager)
    
    # Seed ResponseComposer patterns
    seed_response_patterns(pattern_store)
    seed_qa_response_patterns(pattern_store)
    
    # RelationalStore handles cleanup automatically (SQLite)
    graph_manager.close()
    print("Seeding Complete. 'Teacher' layer populated.")

if __name__ == "__main__":
    main()
