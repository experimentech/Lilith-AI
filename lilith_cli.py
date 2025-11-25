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
    print("Type 'quit' to exit")
    print("Commands: '/stats', '/reset', '/help'")
    print("Feedback: '/+' (upvote), '/-' (downvote), '/?' (show last pattern ID)")
    print("=" * 60)
    print()
    
    # Conversation loop
    turn = 0
    last_pattern_id = None
    
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
        
        # Commands must start with '/'
        if user_input.startswith('/'):
            command = user_input[1:].lower()
            
            if command == 'stats':
                counts = fragment_store.get_pattern_count()
                print(f"\n📊 Statistics:")
                if not user_identity.is_teacher():
                    print(f"   Your patterns: {counts['user']}")
                print(f"   Base patterns: {counts['base']}")
                print(f"   Total: {counts['total']}")
                print()
                continue
            
            elif command == 'help':
                print("\n📖 Commands:")
                print("   quit     - Exit the program")
                print("   /stats   - Show pattern statistics")
                print("   /reset   - Reset your data (with backup)")
                print("   /+       - Upvote last response")
                print("   /-       - Downvote last response")
                print("   /?       - Show last pattern ID")
                print()
                continue
            
            elif command == 'reset':
                if user_identity.is_teacher():
                    print("\n⚠️  Cannot reset in teacher mode")
                    print()
                    continue
                
                print("\n⚠️  This will reset your personal patterns")
                confirm = input("Create backup and reset? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    backup = fragment_store.reset_user_data(keep_backup=True)
                    print(f"✅ Reset complete. Backup: {backup.name if backup else 'none'}")
                else:
                    print("❌ Reset cancelled")
                print()
                continue
            
            elif command == '+':
                if last_pattern_id:
                    fragment_store.upvote(last_pattern_id)
                    print("\n👍 Upvoted!")
                else:
                    print("\n⚠️  No recent response to upvote")
                print()
                continue
            
            elif command == '-':
                if last_pattern_id:
                    fragment_store.downvote(last_pattern_id)
                    print("\n👎 Downvoted!")
                else:
                    print("\n⚠️  No recent response to downvote")
                print()
                continue
            
            elif command == '?':
                if last_pattern_id:
                    print(f"\n📋 Last pattern ID: {last_pattern_id}")
                else:
                    print("\n⚠️  No recent response")
                print()
                continue
            
            else:
                print(f"\n⚠️  Unknown command: /{command}")
                print("   Type '/help' for available commands")
                print()
                continue
        
        # Generate response
        turn += 1
        response = composer.compose_response(user_input)
        
        # Track pattern ID if response came from learned patterns
        if response.fragment_ids:
            last_pattern_id = response.fragment_ids[0]  # First (primary) pattern
        else:
            last_pattern_id = None
        
        print(f"Lilith: {response.text}")
        
        if response.modality:
            print(f"   [Modality: {response.modality}]")
        if response.is_fallback:
            print(f"   [Fallback response - teach me!]")
        
        print()


if __name__ == "__main__":
    main()
