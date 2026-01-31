"""
Multi-tenant CLI for Lilith with user authentication.

Supports different auth modes:
- Teacher mode: No username, writes to base
- User mode: Prompts for username, isolated storage
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.multi_tenant_store import MultiTenantFragmentStore
from lilith.user_auth import UserAuthenticator, AuthMode
from lilith.session import LilithSession, SessionConfig


def _truthy(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def _parse_mcp_endpoints(raw: str):
    """Parse MCP endpoints from either JSON or a simple name=url list."""

    raw = (raw or "").strip()
    if not raw:
        return None

    # Preferred: JSON list of {name,url}
    if raw.startswith("["):
        try:
            import json

            data = json.loads(raw)
            if isinstance(data, list):
                out = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or "").strip()
                    url = str(item.get("url") or "").strip()
                    if name and url:
                        out.append({"name": name, "url": url})
                return out or None
        except Exception:
            return None

    # Fallback: comma-separated name=url pairs
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        name, url = part.split("=", 1)
        name = name.strip()
        url = url.strip()
        if name and url:
            out.append({"name": name, "url": url})
    return out or None


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
    
    # Create session with unified config
    config = SessionConfig(
        data_path="data",
        learning_enabled=True,
        enable_declarative_learning=True,
        enable_feedback_detection=True,
        plasticity_enabled=True,
    )

    if _truthy("LILITH_PERSONALITY_ENABLE"):
        config.enable_personality = True
    if _truthy("LILITH_MOOD_ENABLE"):
        config.enable_mood = True
    if _truthy("LILITH_PREFERENCES_ENABLE"):
        config.enable_preferences = True
    if _truthy("LILITH_DISABLE_KA"):
        config.enable_knowledge_augmentation = False
    if _truthy("LILITH_DISABLE_MODAL"):
        config.enable_modal_routing = False
    if _truthy("LILITH_DISABLE_COMPOSITIONAL"):
        config.enable_compositional = False
    if _truthy("LILITH_DISABLE_PRAGMATIC"):
        config.enable_pragmatic_templates = False
    
    # Reasoning stage configuration
    if _truthy("LILITH_DISABLE_REASONING"):
        config.enable_reasoning = False
    elif os.getenv("LILITH_REASONING_ENABLE"):
        # Explicit enable if var exists and not falsy (or truthy check)
        config.enable_reasoning = _truthy("LILITH_REASONING_ENABLE")

    # Optional MCP Tool Stage configuration.
    # Enable by providing endpoints via env var.
    endpoints = _parse_mcp_endpoints(os.getenv("LILITH_MCP_ENDPOINTS", ""))
    if endpoints:
        config.enable_mcp_tool_stream = True
        config.mcp_endpoints = endpoints
        config.mcp_stage_mode = (os.getenv("LILITH_MCP_MODE", "") or config.mcp_stage_mode).strip() or config.mcp_stage_mode
        try:
            if os.getenv("LILITH_MCP_MIN_SCORE"):
                config.mcp_min_tool_score = float(os.getenv("LILITH_MCP_MIN_SCORE"))
        except Exception:
            pass
        try:
            if os.getenv("LILITH_MCP_TOPK"):
                config.mcp_max_tools_per_turn = int(os.getenv("LILITH_MCP_TOPK"))
        except Exception:
            pass
    
    session = LilithSession(
        user_id=user_identity.user_id,
        context_id="cli",
        config=config,
        store=fragment_store,
        display_name=user_identity.display_name or "User"
    )

    print("🤖 Lilith initialized")
    if session.feedback_tracker:
        print("   🎯 Auto-feedback detection enabled")
    if session.auto_learner:
        print("   📚 Auto-learning from conversations enabled")
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
    print("Type '/quit' or '/exit' to exit")
    print("Commands: '/teach', '/think', '/correct', '/weather', '/mcp', '/stats', '/reset', '/help'")
    print("Feedback: '/+' (upvote), '/-' (downvote), '/?' (show last pattern ID)")
    print("=" * 60)
    print()
    
    # Conversation loop
    turn = 0
    
    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Save session state before exit
            session.cleanup()
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Commands must start with '/'
        # Be forgiving about accidental double-slashes (e.g. "//+").
        if user_input.startswith('//') and not user_input.startswith('///'):
            candidate = user_input[1:]
            # Only normalize if it looks like a known command/feedback.
            head = candidate[1:].strip().split(maxsplit=1)[0].lower() if candidate.startswith('/') else ""
            if head in {"quit", "exit", "stats", "help", "teach", "feedback", "reset", "+", "-", "?", "correct", "weather", "mcp"}:
                user_input = candidate

        if user_input.startswith('/'):
            raw = user_input[1:].strip()
            if not raw:
                continue
            parts = raw.split(maxsplit=1)
            command = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            
            if command in ['quit', 'exit']:
                # Save session state before exit
                session.cleanup()
                stats = session.get_stats()
                if session.auto_learner and 'auto_learning' in stats:
                    auto_stats = stats['auto_learning']
                    if auto_stats.get('pairs_added', 0) > 0:
                        print(f"\n📚 Session learning: {auto_stats['pairs_added']} semantic pairs extracted")
                print("\nGoodbye!")
                break
            
            elif command == 'stats':
                stats = session.get_stats()
                counts = stats['pattern_counts']
                print(f"\n📊 Statistics:")
                if not user_identity.is_teacher():
                    print(f"   Your patterns: {counts['user']}")
                print(f"   Base patterns: {counts['base']}")
                print(f"   Total: {counts['total']}")
                
                # Show vocabulary stats if available
                if 'vocabulary' in stats and stats['vocabulary']:
                    vocab_stats = stats['vocabulary']
                    print(f"\n📖 Vocabulary:")
                    print(f"   Total terms: {vocab_stats['total_terms']}")
                    print(f"   Technical terms: {vocab_stats['technical_terms']}")
                    print(f"   Common terms: {vocab_stats['common_terms']}")
                    
                    if vocab_stats['top_technical']:
                        print(f"\n   Top technical terms:")
                        for term, freq in vocab_stats['top_technical'][:5]:
                            print(f"     - {term} ({freq})")
                
                # Show pattern stats if available
                if 'patterns' in stats and stats['patterns']:
                    pattern_stats = stats['patterns']
                    print(f"\n📝 Syntactic Patterns:")
                    print(f"   Total patterns: {pattern_stats['total_patterns']}")
                    print(f"   Total matches: {pattern_stats['total_matches']}")
                    
                    if pattern_stats['by_intent']:
                        print(f"\n   By intent:")
                        for intent, count in pattern_stats['by_intent'].items():
                            print(f"     - {intent}: {count}")
                    
                    if pattern_stats['top_patterns']:
                        print(f"\n   Most frequent:")
                        for template, freq, intent in pattern_stats['top_patterns'][:3]:
                            print(f"     - {template} ({freq}x)")
                
                print()
                continue
            
            elif command == 'help':
                print("\n📖 Commands:")
                print("   /quit    - Exit the program")
                print("   /exit    - Exit the program")
                print("   /teach   - Teach a new question/answer pair")
                print("   /think   - Submit a query with visible reasoning trace")
                print("   /correct - Teach a correction for my last reply")
                print("   /weather - Get current weather for a location")
                print("   /mcp     - Show/toggle MCP tool stage")
                print("   /stats   - Show pattern statistics")
                print("   /feedback- Show auto-feedback stats")
                print("   /reset   - Reset your data (with backup)")
                print("   /+       - Upvote last response")
                print("   /-       - Downvote last response")
                print("   /?       - Show last pattern ID")
                print()
                continue
            
            elif command == 'think':
                # Explicit thinking command
                query = arg.strip()
                if not query:
                    query = input("Think about: ").strip()
                
                if not query:
                    print("❌ cancelled - no query provided")
                    print()
                    continue
                
                print(f"\n🧠 Thinking about: '{query}'")
                print("=" * 40)
                
                # Process normally - reasoning stage prints to stdout
                # We mainly provide the visual framing here
                turn += 1
                response = session.process_message(query)
                
                print("=" * 40)
                print(f"Lilith: {response.text}")
                print()
                continue

            elif command == 'mcp':
                # Runtime toggle/config for the MCP tool stage.
                # Usage:
                #   /mcp
                #   /mcp status
                #   /mcp mode <off|gated|always_sense|always_call>
                #   /mcp on|off
                #   /mcp score <float>
                #   /mcp topk <int>

                sub = (arg or "").strip()
                parts2 = sub.split(maxsplit=1) if sub else []
                subcmd = (parts2[0].lower() if parts2 else "status")
                subarg = (parts2[1].strip() if len(parts2) > 1 else "")

                stage = getattr(session, "mcp_stage", None)

                if subcmd in {"status", "show"}:
                    enabled = bool(getattr(session.config, "enable_mcp_tool_stream", False))
                    eps = list(getattr(session.config, "mcp_endpoints", []) or [])
                    print("\n🔌 MCP Tool Stage")
                    print(f"   Enabled: {enabled}")
                    print(f"   Endpoints: {len(eps)}")
                    for e in eps[:5]:
                        try:
                            print(f"     - {e.get('name')}: {e.get('url')}")
                        except Exception:
                            continue
                    if stage is None:
                        print("   Stage: not initialized")
                        print("\n   To enable, set LILITH_MCP_ENDPOINTS and restart the CLI.")
                        print("   Example:")
                        print("     export LILITH_MCP_ENDPOINTS='[{\"name\":\"stub\",\"url\":\"ws://127.0.0.1:8765\"}]'")
                    else:
                        print(f"   Mode: {getattr(stage, 'mode', 'unknown')}")
                        print(f"   Min score: {getattr(stage, 'min_score', None)}")
                        print(f"   TopK: {getattr(stage, 'topk', None)}")
                    print()
                    continue

                if stage is None:
                    print("\n⚠️  MCP stage is not initialized.")
                    print("   Set LILITH_MCP_ENDPOINTS and restart the CLI.")
                    print()
                    continue

                if subcmd in {"on", "enable"}:
                    stage.mode = "gated"
                    session.config.mcp_stage_mode = "gated"
                    print("\n✅ MCP stage enabled (mode=gated)")
                    print()
                    continue

                if subcmd in {"off", "disable"}:
                    stage.mode = "off"
                    session.config.mcp_stage_mode = "off"
                    print("\n✅ MCP stage disabled (mode=off)")
                    print()
                    continue

                if subcmd == "mode":
                    mode = (subarg or "").strip().lower()
                    if mode not in {"off", "gated", "always_sense", "always_call"}:
                        print("\n❌ Invalid mode. Use one of: off, gated, always_sense, always_call")
                        print()
                        continue
                    stage.mode = mode
                    session.config.mcp_stage_mode = mode
                    print(f"\n✅ MCP stage mode set to: {mode}")
                    print()
                    continue

                if subcmd == "score":
                    try:
                        v = float(subarg)
                        if v < 0.0 or v > 1.0:
                            raise ValueError("score out of range")
                        stage.min_score = v
                        session.config.mcp_min_tool_score = v
                        print(f"\n✅ MCP min tool score set to: {v:.3f}")
                    except Exception:
                        print("\n❌ Usage: /mcp score <float between 0 and 1>")
                    print()
                    continue

                if subcmd == "topk":
                    try:
                        v = int(subarg)
                        if v < 0:
                            raise ValueError("topk must be >= 0")
                        stage.topk = v
                        session.config.mcp_max_tools_per_turn = v
                        print(f"\n✅ MCP topk set to: {v}")
                    except Exception:
                        print("\n❌ Usage: /mcp topk <int>")
                    print()
                    continue

                print(f"\n⚠️  Unknown /mcp subcommand: {subcmd}")
                print("   Try: /mcp, /mcp mode gated, /mcp off")
                print()
                continue
            
            elif command == 'teach':
                print("\n📚 Teaching Mode")
                print("=" * 50)
                
                # Get question
                question = input("Question: ").strip()
                if not question:
                    print("❌ Teaching cancelled - no question provided")
                    print()
                    continue
                
                # Get answer
                answer = input("Answer: ").strip()
                if not answer:
                    print("❌ Teaching cancelled - no answer provided")
                    print()
                    continue
                
                # Add the pattern using session
                try:
                    pattern_id = session.teach(question, answer, intent="taught")
                    
                    if pattern_id:
                        location = "base knowledge" if user_identity.is_teacher() else "your personal knowledge"
                        print(f"\n✅ Learned!")
                        print(f"   Q: {question}")
                        print(f"   A: {answer}")
                        print(f"   Stored in: {location}")
                        print(f"   Pattern ID: {pattern_id}")
                        print(f"\n   Try asking me now! 🎓")
                    else:
                        print(f"\n❌ Failed to store pattern (no writable layer)")
                except Exception as e:
                    print(f"\n❌ Failed to learn pattern: {e}")
                
                print()
                continue

            elif command == 'correct':
                # Two forms:
                #  1) /correct <corrected text>   (uses last bot reply as the incorrect source)
                #  2) /correct                    (prompts for corrected text; uses last bot reply)
                # If we have no last reply, prompt for the incorrect text.

                print("\n📝 Correction Mode")
                print("=" * 50)

                incorrect = (session.last_response_text or "").strip()
                if not incorrect:
                    incorrect = input("Incorrect text (what I said): ").strip()
                    if not incorrect:
                        print("❌ Correction cancelled - no incorrect text provided")
                        print()
                        continue

                if arg:
                    correct = arg
                else:
                    correct = input("Corrected text (what I should say): ").strip()

                if not correct:
                    print("❌ Correction cancelled - no corrected text provided")
                    print()
                    continue

                ok = session.learn_syntax_correction(incorrect=None, correct=correct, use_last_response=True)
                if ok:
                    print("\n✅ Learned correction for future responses.")
                else:
                    print("\n⚠️  Could not learn correction (syntax stage may be disabled).")

                print()
                continue

            elif command == 'weather':
                location = arg.strip()
                if not location:
                    location = input("Location (e.g., Sydney, NSW): ").strip()
                if not location:
                    print("❌ Weather cancelled - no location provided")
                    print()
                    continue

                try:
                    from lilith.mcp_adapter import WeatherClient

                    client = WeatherClient()
                    report = client.get_weather(location)
                    print(f"\n🌦️  Weather for {report.location}: {report.summary}, {report.temperature_c:.1f}°C")
                except Exception as e:
                    print(f"\n⚠️  Weather lookup failed: {e}")

                print()
                continue
            
            elif command == 'feedback':
                stats = session.get_feedback_stats()
                if stats:
                    print("\n📊 Auto-Feedback Statistics:")
                    print(f"   Total interactions: {stats['total_interactions']}")
                    print(f"   Positive signals: {stats['positive']} ({stats['positive_rate']:.1%})")
                    print(f"   Negative signals: {stats['negative']} ({stats['negative_rate']:.1%})")
                    print(f"   Neutral: {stats['neutral']}")
                    
                    if session.feedback_tracker:
                        recent = session.feedback_tracker.get_recent_feedback(5)
                        if recent:
                            print("\n   Recent feedback:")
                            for r in recent:
                                signal = r['feedback']
                                print(f"     - '{r['input']}' → {signal}")
                else:
                    print("\n⚠️  Feedback detection not available")
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
                if session.last_pattern_id:
                    success = session.upvote()
                    if success:
                        if session.last_pattern_id.startswith('external_'):
                            print(f"\n📚 Learned from external knowledge!")
                            print(f"   Next time you ask '{session.last_user_input[:50] if session.last_user_input else '...'}...', I'll remember!")
                        else:
                            print(f"\n👍 Upvoted pattern: {session.last_pattern_id}")
                            print(f"   Pattern reinforced - I'll be more confident using it!")
                else:
                    print("\n⚠️  No recent response to upvote")
                print()
                continue
            
            elif command == '-':
                if session.last_pattern_id:
                    session.downvote()
                    print(f"\n👎 Downvoted pattern: {session.last_pattern_id}")
                    print(f"   Pattern weakened - I'll use it less often")
                else:
                    print("\n⚠️  No recent response to downvote")
                print()
                continue
            
            elif command == '?':
                if session.last_pattern_id:
                    print(f"\n📋 Last pattern ID: {session.last_pattern_id}")
                else:
                    print("\n⚠️  No recent response")
                print()
                continue
            
            else:
                print(f"\n⚠️  Unknown command: /{command}")
                print("   Type '/help' for available commands")
                print()
                continue
        
        # Process message through session
        turn += 1
        response = session.process_message(user_input)
        
        # Show learned fact if declarative learning occurred
        if response.learned_fact:
            print(f"   � Learned: {response.learned_fact}")
        
        # Prefix mood emoji if available and enabled
        prefix = ""
        if getattr(response, "mood", None) is not None and response.mood.emoji:
            prefix = f"{response.mood.emoji} "
        suffix = ""
        if getattr(response, "personality", None) is not None and response.personality.tone != "neutral":
            suffix = f" (tone: {response.personality.tone})"
        print(f"Lilith: {prefix}{response.text}{suffix}")
        
        # Enhanced fallback feedback
        if response.is_fallback:
            if response.is_low_confidence:
                # No knowledge found at all
                print(f"   💡 I don't know about this yet. You can teach me:")
                print(f"      Option 1: Use '/teach' command for direct teaching")
                print(f"      Option 2: Provide the answer, then I'll respond to it,")
                print(f"                and you can upvote MY answer with '/+'")
            else:
                # Wikipedia or external knowledge found
                print(f"   📚 This is from external knowledge - not yet in my learned patterns")
                print(f"      Upvote with '/+' to save this for faster recall next time!")
        
        print()


if __name__ == "__main__":
    main()
