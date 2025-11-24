#!/usr/bin/env python3
"""
Train from Multi-Turn Conversation CSV

Trains the conversation system on the Conversation.csv dataset,
which contains natural multi-turn dialogues with context dependencies.

This will teach the system to:
- Handle pronouns and references
- Maintain topic across turns
- Respond to context-dependent questions
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from conversation_loop import ConversationLoop
from pipeline.response_fragments import ResponsePattern


def load_conversations_from_csv(csv_path: str, max_turns: int = None):
    """
    Load conversations from CSV file.
    
    Args:
        csv_path: Path to CSV file with columns: index, question, answer
        max_turns: Maximum number of turns to load (None = all)
        
    Returns:
        List of (question, answer) tuples
    """
    conversations = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_turns and i >= max_turns:
                break
            
            question = row['question'].strip()
            answer = row['answer'].strip()
            
            if question and answer:
                conversations.append((question, answer))
    
    return conversations


def train_from_csv(
    csv_path: str,
    max_turns: int = 500,  # Start with 500 for faster training
    learn_from_bot: bool = True,
    use_grammar: bool = True
):
    """
    Train conversation system from CSV dataset.
    
    Args:
        csv_path: Path to conversation CSV
        max_turns: Maximum turns to train on
        learn_from_bot: Whether to learn bot response patterns
        use_grammar: Whether to learn syntax patterns
    """
    print("\n🧠 Initializing Pure Neuro-Symbolic System...")
    conv = ConversationLoop(use_grammar=use_grammar)
    
    print(f"\n✓ Loading conversations from {csv_path}")
    conversations = load_conversations_from_csv(csv_path, max_turns=max_turns)
    print(f"✓ Loaded {len(conversations)} turns")
    
    print("\n" + "="*70)
    print("🎓 TRAINING ON MULTI-TURN CONVERSATIONS")
    print("="*70)
    print(f"  Dataset: {csv_path}")
    print(f"  Turns: {len(conversations)}")
    print(f"  Learn from bot: {learn_from_bot}")
    print(f"  Grammar learning: {use_grammar}")
    print()
    
    # Track what we learn
    patterns_learned = 0
    syntax_learned = 0
    
    # Process conversations sequentially to maintain context
    for i, (user_text, bot_response) in enumerate(conversations):
        # Process the user input through pipeline
        response_text = conv.process_user_input(user_text)
        
        # Learn the correct bot response if enabled
        if learn_from_bot:
            # Check if this is a new pattern worth learning
            # (avoid learning every single variant)
            existing_pattern = None
            for pattern in conv.fragment_store.patterns.values():
                # Check for very similar responses
                if pattern.response_text.lower() == bot_response.lower():
                    existing_pattern = pattern
                    break
            
            if not existing_pattern:
                # Learn new response pattern
                fragment_id = f"learned_csv_{i}"
                
                # Determine intent from response
                bot_lower = bot_response.lower()
                if any(w in bot_lower for w in ['what', 'how', 'why', 'when', 'where']):
                    intent = "question_info"
                elif any(w in bot_lower for w in ['hello', 'hi', 'hey']):
                    intent = "greeting"
                elif any(w in bot_lower for w in ['bye', 'goodbye', 'see you']):
                    intent = "farewell"
                elif any(w in bot_lower for w in ['thank', 'thanks']):
                    intent = "acknowledgment"
                elif any(w in bot_lower for w in ['yes', 'yeah', 'yep', 'sure', 'okay']):
                    intent = "agreement"
                elif any(w in bot_lower for w in ['no', 'nope', 'not']):
                    intent = "disagreement"
                else:
                    intent = "statement"
                
                new_pattern = ResponsePattern(
                    fragment_id=fragment_id,
                    trigger_context=user_text,
                    response_text=bot_response,
                    success_score=0.6,  # Start with moderate confidence
                    intent=intent,
                    usage_count=0
                )
                
                conv.fragment_store.patterns[fragment_id] = new_pattern
                patterns_learned += 1
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(conversations)} turns...")
    
    print()
    print("="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"  Turns processed: {len(conversations)}")
    print(f"  Response patterns learned: {patterns_learned}")
    
    # Get final stats
    stats = conv.fragment_store.get_stats()
    print(f"  Total response patterns: {stats['total_patterns']}")
    print(f"  Total syntax patterns: {len(conv.composer.syntax_stage.patterns) if use_grammar and hasattr(conv.composer, 'syntax_stage') and conv.composer.syntax_stage else 0}")
    print("="*70)
    print()
    
    # Save learned patterns
    conv.fragment_store._save_patterns()
    print(f"💾 Saved {stats['total_patterns']} patterns to conversation_patterns.json")
    
    if use_grammar and hasattr(conv.composer, 'syntax_stage') and conv.composer.syntax_stage:
        if hasattr(conv.composer.syntax_stage, 'save_patterns'):
            conv.composer.syntax_stage.save_patterns()
            print(f"💾 Saved syntax patterns to syntax_patterns.json")
        else:
            print(f"  (Syntax stage doesn't support saving)")
    
    # Build BNN intent clusters for faster retrieval
    print()
    conv.build_intent_clusters()
    
    print()
    print("🎉 Training complete! System ready for conversation.")
    print()
    print("To test learned patterns, run:")
    print("  python minimal_conversation_demo.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train from multi-turn conversation CSV")
    parser.add_argument("csv_path", nargs='?', default="../datasets/Conversation.csv",
                       help="Path to conversation CSV file")
    parser.add_argument("--max-turns", type=int, default=500,
                       help="Maximum number of turns to train on (default: 500)")
    parser.add_argument("--no-bot-learning", action="store_true",
                       help="Don't learn from bot responses")
    parser.add_argument("--no-grammar", action="store_true",
                       help="Don't learn syntax patterns")
    
    args = parser.parse_args()
    
    # Check if file exists
    csv_file = Path(args.csv_path)
    if not csv_file.exists():
        print(f"❌ Error: File not found: {args.csv_path}")
        print()
        print("Usage examples:")
        print("  python train_from_csv.py")
        print("  python train_from_csv.py ../datasets/Conversation.csv --max-turns 1000")
        sys.exit(1)
    
    train_from_csv(
        csv_path=args.csv_path,
        max_turns=args.max_turns,
        learn_from_bot=not args.no_bot_learning,
        use_grammar=not args.no_grammar
    )
