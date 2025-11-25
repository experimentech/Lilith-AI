"""
Migrate response patterns from JSON to SQLite.

Converts existing JSON-based response pattern files to SQLite databases
while preserving all data and metadata.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lilith.response_fragments_sqlite import ResponseFragmentStoreSQLite, ResponsePattern


def migrate_json_to_sqlite(json_path: str, sqlite_path: str, encoder):
    """
    Migrate patterns from JSON file to SQLite database.
    
    Args:
        json_path: Path to JSON file
        sqlite_path: Path to SQLite database (will be created)
        encoder: Semantic encoder for embeddings
    """
    json_file = Path(json_path)
    
    if not json_file.exists():
        print(f"❌ JSON file not found: {json_path}")
        return False
    
    # Load JSON data
    print(f"📖 Loading patterns from {json_path}...")
    with open(json_file, 'r') as f:
        patterns_data = json.load(f)
    
    print(f"   Found {len(patterns_data)} patterns")
    
    # Create SQLite store (will initialize schema)
    print(f"📝 Creating SQLite database at {sqlite_path}...")
    store = ResponseFragmentStoreSQLite(
        encoder,
        storage_path=sqlite_path,
        enable_fuzzy_matching=False  # Skip fuzzy matching during migration
    )
    
    # Migrate each pattern
    print("🔄 Migrating patterns...")
    migrated = 0
    skipped = 0
    
    for pattern_data in patterns_data:
        try:
            # Use direct SQL to preserve IDs and all fields
            conn = store._get_connection()
            
            fragment_id = pattern_data.get('fragment_id', f'pattern_{migrated}')
            trigger_context = pattern_data.get('trigger_context', '')
            response_text = pattern_data.get('response_text', '')
            success_score = pattern_data.get('success_score', 0.5)
            intent = pattern_data.get('intent', 'general')
            usage_count = pattern_data.get('usage_count', 0)
            embedding_cache = pattern_data.get('embedding_cache')
            
            # Convert embedding cache to JSON string
            embedding_json = json.dumps(embedding_cache) if embedding_cache else None
            
            conn.execute("""
                INSERT OR REPLACE INTO response_patterns
                (fragment_id, trigger_context, response_text, success_score, 
                 intent, usage_count, embedding_cache)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (fragment_id, trigger_context, response_text, success_score,
                  intent, usage_count, embedding_json))
            
            conn.commit()
            conn.close()
            
            migrated += 1
            
        except Exception as e:
            print(f"   ⚠️  Error migrating pattern: {e}")
            skipped += 1
    
    print(f"✅ Migration complete!")
    print(f"   Migrated: {migrated}")
    print(f"   Skipped: {skipped}")
    
    # Verify
    stats = store.get_stats()
    print(f"   Database now has {stats['total_patterns']} patterns")
    
    return True


def migrate_directory(directory: str, encoder):
    """
    Migrate all JSON pattern files in a directory to SQLite.
    
    Args:
        directory: Directory containing JSON pattern files
        encoder: Semantic encoder
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"❌ Directory not found: {directory}")
        return
    
    # Find all JSON pattern files
    json_files = list(dir_path.glob("**/response_patterns.json"))
    
    if not json_files:
        print(f"No response_patterns.json files found in {directory}")
        return
    
    print(f"Found {len(json_files)} JSON pattern files to migrate")
    print()
    
    for json_file in json_files:
        # Create SQLite path (same location, .db extension)
        sqlite_file = json_file.with_suffix('.db')
        
        print(f"📦 Migrating: {json_file.relative_to(dir_path)}")
        migrate_json_to_sqlite(str(json_file), str(sqlite_file), encoder)
        print()


def main():
    """Run migration."""
    from lilith.embedding import PMFlowEmbeddingEncoder
    
    print("=" * 60)
    print("RESPONSE PATTERNS: JSON → SQLite MIGRATION")
    print("=" * 60)
    print()
    
    # Create encoder
    print("🔧 Initializing encoder...")
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    print()
    
    # Migrate data directory
    print("🔍 Scanning data directory...")
    migrate_directory("data", encoder)
    
    print("=" * 60)
    print("✅ MIGRATION COMPLETE")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Test the SQLite databases")
    print("  2. Update code to use ResponseFragmentStoreSQLite")
    print("  3. Backup and remove old JSON files")


if __name__ == "__main__":
    main()
