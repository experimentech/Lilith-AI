#!/usr/bin/env python
"""Debug script to trace compose_response"""

from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.response_composer import ResponseComposer
from lilith.conversation_state import ConversationState
from lilith.multi_tenant_store import MultiTenantFragmentStore
from lilith.user_auth import UserIdentity, AuthMode

encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
user = UserIdentity(user_id='debugtest', auth_mode=AuthMode.SIMPLE, display_name='Debug Test')
store = MultiTenantFragmentStore(encoder, user, 'data')
state = ConversationState(encoder)
composer = ResponseComposer(store, state, semantic_encoder=encoder, enable_modal_routing=True, enable_knowledge_augmentation=True)

# Test inputs
test_inputs = ['hello', 'Hello', 'Hello.', 'hi', 'quit smoking is hard']

for inp in test_inputs:
    print(f'\n{"="*60}')
    print(f'Testing: "{inp}"')
    print("="*60)
    
    # Check retrieval
    patterns = store.retrieve_patterns(inp, topk=3)
    print(f'\nRetrieved patterns:')
    for p, score in patterns:
        print(f'  {score:.3f}: "{p.trigger_context}" -> "{p.response_text[:40]}..."')
        print(f'         (success_score={p.success_score}, usage_count={p.usage_count})')
    
    # Check composition
    response = composer.compose_response(inp)
    print(f'\nComposed response:')
    print(f'  Text: "{response.text}"')
    print(f'  Confidence: {response.confidence:.3f}')
    print(f'  Is fallback: {response.is_fallback}')
    print(f'  Is low confidence: {response.is_low_confidence}')
