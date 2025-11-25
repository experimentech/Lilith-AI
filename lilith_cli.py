"""
Multi-tenant CLI for Lilith with user authentication.

Supports different auth modes:
- Teacher mode: No username, writes to base
- User mode: Prompts for username, isolated storage
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.response_composer import ResponseComposer
from lilith.conversation_state import ConversationState
from lilith.multi_tenant_store import MultiTenantFragmentStore
from lilith.user_auth import UserAuthenticator, AuthMode


def main():
    """Run multi-tenant Lilith CLI"""
    
    print("=" * 60)
    print("LILITH - Multi-Tenant Conversational AI")
    print("=" * 60)
    print()
    
    # Prompt for mode
    print("Select mode:")
    print("  1. Teacher mode (writes to base knowledge)")
    print("  2. User mode (isolated personal storage)")
    print()
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            auth_mode = AuthMode.NONE
            break
        elif choice == "2":
            auth_mode = AuthMode.SIMPLE
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    print()
    
    # Authenticate user
    authenticator = UserAuthenticator(auth_mode)
    user_identity = authenticator.authenticate()
    
    print()
    print(f"Welcome, {user_identity.display_name}!")
    print()
    
    # Create encoder
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    
    # Create multi-tenant fragment store
    fragment_store = MultiTenantFragmentStore(
        encoder,
        user_identity,
        base_data_path="data"
    )
    
    # Create conversation state
    state = ConversationState(encoder)
    
    # Create composer
    composer = ResponseComposer(
        fragment_store,
        state,
        semantic_encoder=encoder,
        enable_knowledge_augmentation=True,
        enable_modal_routing=True
    )
    
    print("🤖 Lilith initialized")
    print()
    
    # Show pattern counts
    counts = fragment_store.get_pattern_count()
    if user_identity.is_teacher():
        print(f"   Base patterns: {counts['base']}")
    else:
        print(f"   Your patterns: {counts['user']}")
        print(f"   Base patterns: {counts['base']}")
        print(f"   Total accessible: {counts['total']}")
    
    print()
    print("Type 'quit' to exit, 'stats' for statistics")
    print("=" * 60)
    print()
    
    # Conversation loop
    turn = 0
    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'stats':
            counts = fragment_store.get_pattern_count()
            print(f"\n📊 Statistics:")
            if not user_identity.is_teacher():
                print(f"   Your patterns: {counts['user']}")
            print(f"   Base patterns: {counts['base']}")
            print(f"   Total: {counts['total']}")
            print()
            continue
        
        # Generate response
        turn += 1
        response = composer.generate_response(user_input)
        
        print(f"Lilith: {response['text']}")
        
        if response.get('modality'):
            print(f"   [Modality: {response['modality']}]")
        
        print()


if __name__ == "__main__":
    main()
