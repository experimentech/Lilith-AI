#!/usr/bin/env python3
"""
Bootstrap Lilith's grammar capabilities with comprehensive training dataset.

This script loads the grammar_bootstrap.txt file and teaches Lilith
various sentence structures, enabling better grammatical pattern recognition
and response generation from the start.

Usage:
    python bootstrap_grammar.py [--data-dir DATA_DIR]
"""

import argparse
import re
from pathlib import Path
from tqdm import tqdm

from lilith.session import LilithSession, SessionConfig


def load_grammar_dataset(filepath: Path) -> list[str]:
    """
    Load grammar training sentences from file.
    
    Args:
        filepath: Path to grammar_bootstrap.txt
        
    Returns:
        List of training sentences (comments and blank lines removed)
    """
    sentences = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and blank lines
            if not line or line.startswith('#'):
                continue
            
            # Skip section dividers
            if line.startswith('='):
                continue
            
            sentences.append(line)
    
    return sentences


def bootstrap_grammar(data_dir: str = "data", user_id: str = "bootstrap"):
    """
    Bootstrap Lilith's grammar by teaching it the training dataset.
    
    Args:
        data_dir: Directory for Lilith's data storage
        user_id: User ID for the bootstrap session
    """
    # Load grammar dataset
    grammar_file = Path("data/grammar_bootstrap.txt")
    if not grammar_file.exists():
        print(f"❌ Grammar dataset not found at {grammar_file}")
        print("Make sure grammar_bootstrap.txt exists in the data/ directory.")
        return
    
    print("📖 Loading grammar training dataset...")
    sentences = load_grammar_dataset(grammar_file)
    print(f"✓ Loaded {len(sentences)} training sentences")
    print()
    
    # Create session with grammar enabled
    print("🚀 Initializing Lilith session...")
    config = SessionConfig(
        data_path=data_dir,
        use_grammar=True,           # Enable syntax stage
        enable_knowledge_augmentation=False,  # Don't fetch Wikipedia during bootstrap
        enable_modal_routing=False,  # Keep it simple
        learning_enabled=True,
        plasticity_enabled=True,    # Enable neuroplasticity
    )
    
    session = LilithSession(
        user_id=user_id,
        context_id="grammar_bootstrap",
        config=config
    )
    
    # Verify components are enabled
    print("\n✓ Session initialized")
    if session.composer.syntax_stage:
        print("✓ Syntax stage enabled")
    if session.composer.reasoning_stage:
        print("✓ Reasoning stage enabled")
    print()
    
    # Process each sentence
    print("📚 Teaching grammar patterns...")
    print("=" * 70)
    
    successful = 0
    failed = 0
    
    # Use tqdm for progress bar
    for sentence in tqdm(sentences, desc="Training", unit="sentence"):
        try:
            # Process the sentence (Lilith learns from it)
            response = session.process_message(sentence, passive_mode=True)
            successful += 1
            
        except Exception as e:
            failed += 1
            tqdm.write(f"⚠️  Failed on: {sentence}")
            tqdm.write(f"    Error: {e}")
    
    print("=" * 70)
    print()
    
    # Summary
    print("🎯 Bootstrap Complete!")
    print("=" * 70)
    print(f"✓ Successfully processed: {successful} sentences")
    if failed > 0:
        print(f"⚠️  Failed: {failed} sentences")
    print()
    
    # Show statistics
    print("📊 Statistics:")
    print(f"  Session messages: {successful}")
    
    # Try to get pattern count if available
    try:
        if hasattr(session.store, 'patterns'):
            print(f"  Patterns learned: {len(session.store.patterns)} total")
        elif hasattr(session.store, 'get_all_patterns'):
            patterns = session.store.get_all_patterns()
            print(f"  Patterns learned: {len(patterns)} total")
    except:
        pass
    
    print()
    
    # Test with example queries
    print("🧪 Testing learned grammar...")
    print("=" * 70)
    
    test_queries = [
        "What is machine learning?",
        "How does this work?",
        "Can you explain neural networks?",
        "I want to learn about AI.",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = session.process_message(query)
        print(f"Response: {response.text}")
        print(f"Confidence: {response.confidence:.3f}")
    
    print()
    print("=" * 70)
    print("✅ Grammar bootstrapping complete!")
    print(f"\nLilith now has a foundation of {successful} grammatical patterns.")
    print("These patterns will help with:")
    print("  - Better sentence structure recognition")
    print("  - More grammatical response generation")
    print("  - Improved pattern matching and adaptation")
    print("  - Stronger syntax learning from future conversations")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap Lilith's grammar capabilities with training dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory for Lilith's data storage (default: data)"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="bootstrap",
        help="User ID for bootstrap session (default: bootstrap)"
    )
    
    args = parser.parse_args()
    
    print("🔤 Lilith Grammar Bootstrap Tool")
    print("=" * 70)
    print()
    
    bootstrap_grammar(args.data_dir, args.user_id)


if __name__ == "__main__":
    main()
