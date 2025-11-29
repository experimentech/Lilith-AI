#!/usr/bin/env python3
"""
Simple tests for base patterns retrieval to ensure better OOTB comprehension.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.multi_tenant_store import MultiTenantFragmentStore
from lilith.user_auth import UserIdentity, AuthMode


def test_base_patterns_retrieval():
    """Test retrieval of a few new base patterns we recently added"""
    encoder = PMFlowEmbeddingEncoder(dimension=96, latent_dim=32)
    
    # Use a non-teacher user so we get base + user stores merged
    identity = UserIdentity(user_id="testuser", auth_mode=AuthMode.SIMPLE, display_name="Test User")
    store = MultiTenantFragmentStore(encoder, identity, base_data_path="data")
    
    # Lower min_score to be permissive for semantic retrieval
    matches = store.retrieve_patterns("how are you", topk=3, min_score=0.0)
    assert len(matches) > 0, "Expected at least one match for 'how are you'"
    intents = [p.intent for p, _ in matches]
    assert any(i in ("greeting", "smalltalk", "wellbeing") for i in intents), f"No greeting or smalltalk intent in {intents}"

    matches = store.retrieve_patterns("who made you", topk=3, min_score=0.0)
    assert len(matches) > 0, "Expected at least one match for 'who made you'"
    intents = [p.intent for p, _ in matches]
    assert any(i in ("capability", "identity") for i in intents), f"No capability / identity intent in {intents}"

    matches = store.retrieve_patterns("help me", topk=3, min_score=0.0)
    assert len(matches) > 0, "Expected a match for 'help me'"
    intents = [p.intent for p, _ in matches]
    assert any(i in ("help","capability","clarify") for i in intents), f"No help/capability/clarify intent in {intents}"

    matches = store.retrieve_patterns("what's up", topk=3, min_score=0.0)
    assert len(matches) > 0
    intents = [p.intent for p, _ in matches]
    assert any(i in ("smalltalk", "greeting", "wellbeing", "knowledge", "identity") for i in intents), f"No smalltalk/greeting/wellbeing intent in {intents}"


if __name__ == "__main__":
    test_base_patterns_retrieval()
