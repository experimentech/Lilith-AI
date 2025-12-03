#!/usr/bin/env python3
"""
Bootstrap Lilith's BASE knowledge with Q&A pairs.

Unlike user-specific training, this populates the SHARED BASE knowledge
that is available to ALL users (both Guild and DM contexts).

This is ideal for:
- General knowledge from datasets like SQuAD
- Domain-specific knowledge you want ALL users to access
- Foundation knowledge for your Discord bot

Usage:
    python bootstrap_base.py --qa-file data/generated/squad_training.txt
    python bootstrap_base.py --qa-file data/seed/qa_bootstrap.txt
    
The script runs in Teacher mode, so all data goes to data/base/
"""

import argparse
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from lilith.session import LilithSession, SessionConfig
from lilith.user_auth import UserIdentity


def load_qa_pairs(filepath: Path) -> list[tuple[str, str]]:
    """
    Load Q&A pairs from file.
    
    Format:
        Q: question text
        A: answer text
    
    Args:
        filepath: Path to Q&A file
        
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


def bootstrap_base(qa_file: str, data_dir: str = "data", dry_run: bool = False):
    """
    Bootstrap Lilith's BASE knowledge with Q&A pairs.
    
    Args:
        qa_file: Path to Q&A file
        data_dir: Directory for Lilith's data storage
        dry_run: If True, just load and validate without writing
    """
    qa_path = Path(qa_file)
    
    if not qa_path.exists():
        print(f"❌ Q&A file not found: {qa_path}")
        return
    
    print(f"📖 Loading Q&A dataset from {qa_path}...")
    qa_pairs = load_qa_pairs(qa_path)
    print(f"✓ Loaded {len(qa_pairs)} Q&A pairs")
    print()
    
    if dry_run:
        print("🔍 Dry run mode - showing sample data:")
        print("=" * 70)
        for q, a in qa_pairs[:5]:
            print(f"Q: {q}")
            print(f"A: {a}")
            print()
        print(f"... and {len(qa_pairs) - 5} more pairs")
        print("=" * 70)
        print()
        print("✓ Dry run complete. Use without --dry-run to actually bootstrap.")
        return
    
    # Create a Teacher identity for base knowledge access
    # is_teacher() returns True when user_id == "teacher" or auth_mode == NONE
    from lilith.user_auth import AuthMode
    from lilith.embedding import PMFlowEmbeddingEncoder
    from lilith.multi_tenant_store import MultiTenantFragmentStore
    
    teacher_identity = UserIdentity(
        user_id="teacher",  # "teacher" user_id enables base knowledge writing
        auth_mode=AuthMode.NONE,
        display_name="Base Bootstrap"
    )
    
    print("🚀 Initializing Lilith session in TEACHER mode...")
    print("   (All data will go to data/base/ - shared across all users)")
    print()
    
    config = SessionConfig(
        data_path=data_dir,
        use_grammar=True,
        enable_knowledge_augmentation=False,  # Don't fetch Wikipedia
        enable_modal_routing=False,
        learning_enabled=True,
        plasticity_enabled=True,
    )
    
    # Create encoder and store with teacher identity
    encoder = PMFlowEmbeddingEncoder()
    teacher_store = MultiTenantFragmentStore(
        encoder=encoder,
        user_identity=teacher_identity,
        base_data_path=data_dir
    )
    
    session = LilithSession(
        user_id=teacher_identity.user_id,
        context_id="base_bootstrap",
        config=config,
        store=teacher_store  # Pass pre-configured teacher store
    )
    
    print()
    print("✓ Session initialized in Teacher mode")
    if session.composer.syntax_stage:
        print("✓ Syntax stage enabled")
    if session.composer.reasoning_stage:
        print("✓ Reasoning stage enabled")
    print()
    
    # Teach each Q&A pair
    print("📚 Teaching Q&A pairs to BASE knowledge...")
    print("=" * 70)
    
    successful = 0
    failed = 0
    
    for question, answer in tqdm(qa_pairs, desc="Teaching", unit="pair"):
        try:
            pattern_id = session.teach(question, answer, intent="base_knowledge")
            successful += 1
            
        except Exception as e:
            failed += 1
            tqdm.write(f"⚠️  Failed: {question[:50]}...")
            tqdm.write(f"    Error: {e}")
    
    print("=" * 70)
    print()
    
    # Summary
    print("🎯 Base Bootstrap Complete!")
    print("=" * 70)
    print(f"✓ Successfully taught: {successful} Q&A pairs")
    if failed > 0:
        print(f"⚠️  Failed: {failed} pairs")
    print()
    print("📁 Data stored in:")
    print(f"   • data/base/response_patterns.db")
    print(f"   • data/base/concepts.db")
    print(f"   • data/base/patterns.db")
    print(f"   • data/base/vocabulary.db")
    print()
    
    # Test with example queries (using a regular user session)
    print("🧪 Testing with a USER session (to verify base data is accessible)...")
    print("=" * 70)
    
    # Create a regular user to test (uses TRUSTED auth mode, not teacher)
    test_identity = UserIdentity(
        user_id="test_user",
        auth_mode=AuthMode.TRUSTED,
        display_name="Test User"
    )
    
    test_store = MultiTenantFragmentStore(
        encoder=encoder,
        user_identity=test_identity,
        base_data_path=data_dir
    )
    
    test_session = LilithSession(
        user_id=test_identity.user_id,
        context_id="test",
        config=config,
        store=test_store  # Pass pre-configured user store
    )
    
    test_queries = [
        "What is Python?",
        "How does machine learning work?",
        "What are neural networks?",
    ]
    
    high_confidence = 0
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = test_session.process_message(query)
        
        is_good = response.confidence > 0.6 and not response.is_fallback
        if is_good:
            high_confidence += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{status} Response: {response.text[:100]}...")
        print(f"  Confidence: {response.confidence:.3f}")
        print(f"  Source: {response.source}")
    
    print()
    print("=" * 70)
    print(f"📊 User can access {high_confidence}/{len(test_queries)} queries from base knowledge")
    print("=" * 70)
    print()
    
    print("✅ Base knowledge is now available to ALL users!")
    print()
    print("Next steps:")
    print("  1. Start your Discord bot - all users will have access to this knowledge")
    print("  2. Guild members and DM users both see the same base knowledge")
    print("  3. User-specific learning will be stored separately (won't affect base)")
    print("  4. Use /teach in Discord to add more base knowledge (if you're a teacher)")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap Lilith's BASE knowledge (shared across all users)"
    )
    parser.add_argument(
        "qa_file",
        nargs="?",
        default=None,
        help="Path to Q&A file (e.g., data/generated/squad_training.txt)"
    )
    parser.add_argument(
        "--qa-file",
        dest="qa_file_opt",
        type=str,
        default=None,
        help="Path to Q&A file (alternative syntax)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory for Lilith's data storage (default: data)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate data without writing to database"
    )
    
    args = parser.parse_args()
    
    # Support both positional and --qa-file argument
    qa_file = args.qa_file or args.qa_file_opt
    
    if not qa_file:
        print("❌ Please provide a Q&A file path")
        print()
        print("Usage:")
        print("  python bootstrap_base.py data/generated/squad_training.txt")
        print("  python bootstrap_base.py --qa-file data/seed/qa_bootstrap.txt")
        print("  python bootstrap_base.py data/generated/squad_training.txt --dry-run")
        return
    
    print("🔤 Lilith BASE Knowledge Bootstrap")
    print("=" * 70)
    print("⚠️  This populates SHARED knowledge accessible to ALL users")
    print("=" * 70)
    print()
    
    bootstrap_base(qa_file, args.data_dir, args.dry_run)


if __name__ == "__main__":
    main()
