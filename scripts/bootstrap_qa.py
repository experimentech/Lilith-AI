#!/usr/bin/env python3
"""
Bootstrap Lilith with Q&A pairs for effective conversational knowledge.

Unlike the grammar bootstrap which stores grammatical patterns,
this creates actual question→answer associations that can be retrieved
when users ask similar questions.

Usage:
    python bootstrap_qa.py [--data-dir DATA_DIR] [--user-id USER_ID]
"""

import argparse
from pathlib import Path
from tqdm import tqdm

from lilith.session import LilithSession, SessionConfig


def load_qa_pairs(filepath: Path) -> list[tuple[str, str]]:
    """
    Load Q&A pairs from file.
    
    Format:
        Q: question text
        A: answer text
    
    Args:
        filepath: Path to qa_bootstrap.txt
        
    Returns:
        List of (question, answer) tuples
    """
    qa_pairs = []
    current_question = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments, blank lines, and section dividers
            if not line or line.startswith('#') or line.startswith('='):
                continue
            
            # Parse Q: and A: lines
            if line.startswith('Q: '):
                current_question = line[3:].strip()
            elif line.startswith('A: '):
                if current_question:
                    answer = line[3:].strip()
                    qa_pairs.append((current_question, answer))
                    current_question = None
    
    return qa_pairs


def bootstrap_qa(data_dir: str = "data", user_id: str = "bootstrap"):
    """
    Bootstrap Lilith's Q&A knowledge base.
    
    Args:
        data_dir: Directory for Lilith's data storage
        user_id: User ID for the bootstrap session
    """
    # Load Q&A dataset
    qa_file = Path("data/qa_bootstrap.txt")
    if not qa_file.exists():
        print(f"❌ Q&A dataset not found at {qa_file}")
        print("Make sure qa_bootstrap.txt exists in the data/ directory.")
        return
    
    print("📖 Loading Q&A training dataset...")
    qa_pairs = load_qa_pairs(qa_file)
    print(f"✓ Loaded {len(qa_pairs)} Q&A pairs")
    print()
    
    # Create session
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
        context_id="qa_bootstrap",
        config=config
    )
    
    # Verify components are enabled
    print("\n✓ Session initialized")
    if session.composer.syntax_stage:
        print("✓ Syntax stage enabled")
    if session.composer.reasoning_stage:
        print("✓ Reasoning stage enabled")
    print()
    
    # Teach each Q&A pair
    print("📚 Teaching Q&A pairs...")
    print("=" * 70)
    
    successful = 0
    failed = 0
    pattern_ids = []
    
    # Use tqdm for progress bar
    for question, answer in tqdm(qa_pairs, desc="Teaching", unit="pair"):
        try:
            # Use teach() method to create Q→A association
            pattern_id = session.teach(question, answer, intent="bootstrap_qa")
            pattern_ids.append(pattern_id)
            successful += 1
            
        except Exception as e:
            failed += 1
            tqdm.write(f"⚠️  Failed on Q: {question}")
            tqdm.write(f"    Error: {e}")
    
    print("=" * 70)
    print()
    
    # Summary
    print("🎯 Bootstrap Complete!")
    print("=" * 70)
    print(f"✓ Successfully taught: {successful} Q&A pairs")
    if failed > 0:
        print(f"⚠️  Failed: {failed} pairs")
    print()
    
    # Test with example queries
    print("🧪 Testing learned knowledge...")
    print("=" * 70)
    
    test_queries = [
        "What is Python?",
        "How does machine learning work?",
        "What are neural networks?",
        "Can you help me?",
        "What is AI?",
        "How do I learn programming?",
    ]
    
    high_confidence = 0
    total_confidence = 0.0
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = session.process_message(query)
        total_confidence += response.confidence
        
        is_good = response.confidence > 0.6 and not response.is_fallback
        if is_good:
            high_confidence += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{status} Response: {response.text}")
        print(f"  Confidence: {response.confidence:.3f}")
        print(f"  Source: {response.source}")
    
    avg_confidence = total_confidence / len(test_queries)
    
    print()
    print("=" * 70)
    print("📊 Test Results:")
    print(f"  High confidence responses: {high_confidence}/{len(test_queries)}")
    print(f"  Average confidence: {avg_confidence:.3f}")
    print("=" * 70)
    print()
    
    if avg_confidence > 0.6:
        print("✅ Excellent! Q&A bootstrap working well!")
    elif avg_confidence > 0.4:
        print("✓ Good! Q&A bootstrap created useful knowledge.")
    else:
        print("⚠️  Bootstrap complete, but confidence is lower than expected.")
        print("   This may improve with more training data or threshold tuning.")
    
    print()
    print(f"Lilith now knows {successful} Q&A pairs!")
    print("These create direct question→answer associations for better retrieval.")
    print()
    print("Next steps:")
    print("  1. Use Lilith normally - she can now answer these questions")
    print("  2. Add more Q&A pairs to qa_bootstrap.txt for your domain")
    print("  3. Provide feedback (upvotes/downvotes) to refine responses")
    print("  4. She'll continue learning from your conversations")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap Lilith with Q&A pairs for conversational knowledge"
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
    
    print("🔤 Lilith Q&A Bootstrap Tool")
    print("=" * 70)
    print()
    
    bootstrap_qa(args.data_dir, args.user_id)


if __name__ == "__main__":
    main()
