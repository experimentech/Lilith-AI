#!/usr/bin/env python3
"""
Migrate existing Q&A patterns to compositional concepts.

This is Phase 2C of the Layer 4 restructuring:
- Extract concepts from existing patterns (1,235+ Q&A pairs)
- Store as semantic concepts with properties in concept_store.db
- Remove verbatim patterns (keep only compositional templates)
- Result: ~50 templates + N concepts instead of 1,235+ verbatim patterns
"""

import sys
import os
import sqlite3
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.production_concept_store import ProductionConceptStore


def extract_concept_from_pattern(question: str, answer: str) -> Optional[Tuple[str, List[str]]]:
    """
    Extract concept and properties from a Q&A pattern.
    
    Args:
        question: User query (e.g., "What is Python?")
        answer: Bot response (e.g., "Python is a high-level programming language...")
        
    Returns:
        Tuple of (concept_term, properties) or None if not extractable
    """
    # Pattern: "What is X?" → extract X as concept
    what_is_match = re.search(r'what\s+(?:is|are)\s+(.+?)\?', question.lower())
    if what_is_match:
        concept_term = what_is_match.group(1).strip()
        
        # Extract properties from answer
        # Common patterns: "X is Y", "X are Y", "X refers to Y"
        properties = []
        
        # Split answer into sentences
        sentences = re.split(r'[.!?]\s+', answer)
        for sentence in sentences[:3]:  # First 3 sentences as properties
            sentence = sentence.strip()
            if sentence:
                properties.append(sentence)
        
        if properties:
            return (concept_term, properties)
    
    # Pattern: "Tell me about X" → extract X
    tell_me_match = re.search(r'tell\s+me\s+about\s+(.+?)[\?.]?$', question.lower())
    if tell_me_match:
        concept_term = tell_me_match.group(1).strip()
        properties = [s.strip() for s in re.split(r'[.!?]\s+', answer)[:3] if s.strip()]
        if properties:
            return (concept_term, properties)
    
    # Pattern: "How does X work?" → extract X
    how_match = re.search(r'how\s+(?:does|do)\s+(.+?)\s+work', question.lower())
    if how_match:
        concept_term = how_match.group(1).strip()
        properties = [s.strip() for s in re.split(r'[.!?]\s+', answer)[:3] if s.strip()]
        if properties:
            return (concept_term, properties)
    
    return None


def migrate_patterns_to_concepts(
    patterns_db_path: str,
    concept_store: ProductionConceptStore,
    dry_run: bool = True
) -> Dict[str, int]:
    """
    Migrate patterns from pattern database to concept store.
    
    Args:
        patterns_db_path: Path to patterns.db file
        concept_store: ConceptStore to add concepts to
        dry_run: If True, don't actually add concepts (just analyze)
        
    Returns:
        Statistics dict
    """
    stats = {
        "total_patterns": 0,
        "extractable_concepts": 0,
        "concepts_added": 0,
        "skipped": 0
    }
    
    # Connect to patterns database
    conn = sqlite3.connect(patterns_db_path)
    cursor = conn.cursor()
    
    # Query all patterns
    cursor.execute("""
        SELECT fragment_id, trigger_context, response_text, intent
        FROM response_patterns
        ORDER BY fragment_id
    """)
    
    patterns = cursor.fetchall()
    stats["total_patterns"] = len(patterns)
    
    print(f"\nAnalyzing {len(patterns)} patterns from {patterns_db_path}")
    print("="*60)
    
    for fragment_id, trigger_context, response_text, intent in patterns:
        # Try to extract concept
        result = extract_concept_from_pattern(trigger_context, response_text)
        
        if result:
            concept_term, properties = result
            stats["extractable_concepts"] += 1
            
            if not dry_run:
                try:
                    # Add to concept store
                    concept_id = concept_store.add_concept(
                        term=concept_term,
                        properties=properties,
                        relations=[],
                        source=f"migrated_from_pattern_{fragment_id}"
                    )
                    stats["concepts_added"] += 1
                    
                    if stats["concepts_added"] <= 5:  # Show first 5
                        print(f"  ✅ Migrated: {concept_term}")
                        print(f"     Properties: {properties[0][:60]}...")
                
                except Exception as e:
                    print(f"  ⚠️  Failed to add {concept_term}: {e}")
                    stats["skipped"] += 1
            else:
                if stats["extractable_concepts"] <= 10:  # Show first 10 in dry run
                    print(f"  📝 Would migrate: {concept_term}")
                    print(f"     Properties: {properties[0][:60]}...")
        else:
            stats["skipped"] += 1
    
    conn.close()
    
    return stats


def main():
    """Run pattern migration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate patterns to concepts")
    parser.add_argument("--user", default="bootstrap", help="User ID to migrate")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually migrate, just analyze")
    parser.add_argument("--data-path", default="data", help="Base data path")
    
    args = parser.parse_args()
    
    print("\n🔄 Pattern to Concept Migration (Phase 2C)")
    print("="*60)
    print(f"User: {args.user}")
    print(f"Mode: {'DRY RUN (analysis only)' if args.dry_run else 'LIVE (will migrate)'}")
    print("="*60)
    
    # Paths
    patterns_db = Path(args.data_path) / "users" / args.user / "response_patterns.db"
    concepts_db = Path(args.data_path) / "users" / args.user / "concepts.db"
    
    if not patterns_db.exists():
        print(f"❌ Patterns database not found: {patterns_db}")
        return
    
    # Initialize encoder and concept store
    encoder = PMFlowEmbeddingEncoder()
    concept_store = ProductionConceptStore(
        semantic_encoder=encoder,
        db_path=str(concepts_db)
    )
    
    # Run migration
    stats = migrate_patterns_to_concepts(
        str(patterns_db),
        concept_store,
        dry_run=args.dry_run
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Migration Summary")
    print("="*60)
    print(f"  Total patterns:        {stats['total_patterns']}")
    print(f"  Extractable concepts:  {stats['extractable_concepts']}")
    print(f"  Concepts added:        {stats['concepts_added']}")
    print(f"  Skipped:               {stats['skipped']}")
    
    if stats['extractable_concepts'] > 0:
        percentage = (stats['extractable_concepts'] / stats['total_patterns']) * 100
        print(f"\n  Extraction rate: {percentage:.1f}%")
    
    if args.dry_run:
        print("\n💡 This was a dry run. Use --no-dry-run to actually migrate.")
    else:
        print(f"\n✅ Migration complete! Concepts stored in {concepts_db}")
        print("\nNext steps:")
        print("  1. Test responses with compositional mode")
        print("  2. Verify concept retrieval works correctly")
        print("  3. Consider removing verbatim patterns (keep only templates)")
        print(f"  4. Storage reduction: {stats['total_patterns']} patterns → 26 templates + {stats['concepts_added']} concepts")


if __name__ == "__main__":
    main()
