#!/usr/bin/env python3
"""
User database maintenance utility.

Provides tools for managing user databases including:
- Listing users and database sizes
- Archiving and deleting inactive users
- Exporting user data
- Database statistics
"""

import sys
from pathlib import Path
import shutil
from datetime import datetime, timedelta
import sqlite3
import json

# Default base data path
BASE_DATA_PATH = Path("data/production")


def list_users(base_path: Path = BASE_DATA_PATH):
    """List all users and database sizes."""
    users_dir = base_path / "users"
    if not users_dir.exists():
        print("No users directory found")
        return
    
    print("=" * 70)
    print("USER DATABASES")
    print("=" * 70)
    print(f"{'User ID':<20} {'Size (MB)':>10} {'Patterns':>10} {'Last Update':<20}")
    print("-" * 70)
    
    total_size = 0
    total_patterns = 0
    
    for user_dir in sorted(users_dir.iterdir()):
        if not user_dir.is_dir():
            continue
        
        db_path = user_dir / "response_patterns.db"
        if not db_path.exists():
            continue
        
        # Get database size
        size = db_path.stat().st_size
        total_size += size
        size_mb = size / (1024 * 1024)
        
        # Get pattern count and last update
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("SELECT COUNT(*), MAX(updated_at) FROM response_patterns")
            count, last_update = cursor.fetchone()
            conn.close()
            
            total_patterns += count or 0
            last_update = last_update or "Never"
            
        except Exception as e:
            count = "Error"
            last_update = str(e)
        
        print(f"{user_dir.name:<20} {size_mb:>10.2f} {str(count):>10} {str(last_update):<20}")
    
    print("-" * 70)
    print(f"{'Total:':<20} {total_size / (1024 * 1024):>10.2f} {total_patterns:>10}")
    print()


def delete_user(user_id: str, archive: bool = True):
    """Delete user data with optional archival."""
    user_dir = Path(f"data/users/{user_id}")
    
    if not user_dir.exists():
        print(f"❌ User {user_id} not found")
        return
    
    if archive:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = Path("archives/users")
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{user_id}_{timestamp}.tar.gz"
        
        import tarfile
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(user_dir, arcname=user_id)
        
        print(f"✅ Archived to {archive_path}")
    
    shutil.rmtree(user_dir)
    print(f"✅ Deleted user: {user_id}")


def export_user(user_id: str, output_path: str = None):
    """Export user patterns to JSON."""
    user_dir = Path(f"data/users/{user_id}")
    db_path = user_dir / "response_patterns.db"
    
    if not db_path.exists():
        print(f"❌ User {user_id} not found")
        return
    
    if output_path is None:
        output_path = f"{user_id}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Read all patterns from database
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT * FROM response_patterns")
    
    patterns = []
    for row in cursor:
        pattern = dict(row)
        patterns.append(pattern)
    
    conn.close()
    
    # Write to JSON
    with open(output_path, 'w') as f:
        json.dump(patterns, f, indent=2)
    
    print(f"✅ Exported {len(patterns)} patterns to {output_path}")


def cleanup_inactive(days_inactive: int = 90, dry_run: bool = True):
    """Delete users inactive for N days."""
    users_dir = Path("data/users")
    cutoff_date = datetime.now() - timedelta(days=days_inactive)
    
    print(f"Finding users inactive since {cutoff_date.strftime('%Y-%m-%d')}...")
    print()
    
    inactive_users = []
    
    for user_dir in sorted(users_dir.iterdir()):
        if not user_dir.is_dir():
            continue
        
        db_path = user_dir / "response_patterns.db"
        if not db_path.exists():
            continue
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("SELECT MAX(updated_at) FROM response_patterns")
            last_update = cursor.fetchone()[0]
            conn.close()
            
            if last_update:
                # Parse timestamp (SQLite default format: YYYY-MM-DD HH:MM:SS)
                try:
                    last_update_dt = datetime.strptime(last_update, "%Y-%m-%d %H:%M:%S")
                    if last_update_dt < cutoff_date:
                        inactive_users.append((user_dir.name, last_update))
                except ValueError:
                    # Try parsing as date only
                    try:
                        last_update_dt = datetime.strptime(last_update, "%Y-%m-%d")
                        if last_update_dt < cutoff_date:
                            inactive_users.append((user_dir.name, last_update))
                    except ValueError:
                        print(f"⚠️  Could not parse date for {user_dir.name}: {last_update}")
        
        except Exception as e:
            print(f"⚠️  Error checking {user_dir.name}: {e}")
    
    if not inactive_users:
        print("✅ No inactive users found")
        return
    
    print(f"Found {len(inactive_users)} inactive users:")
    print("-" * 50)
    for user_id, last_update in inactive_users:
        print(f"  {user_id:<20} (last: {last_update})")
    print()
    
    if dry_run:
        print("🔍 DRY RUN - No changes made")
        print(f"Run with --execute to delete these users")
    else:
        print(f"⚠️  Deleting {len(inactive_users)} users...")
        for user_id, _ in inactive_users:
            delete_user(user_id, archive=True)
        print(f"✅ Cleanup complete")


def stats():
    """Show overall statistics."""
    users_dir = Path("data/users")
    base_dir = Path("data/base")
    
    print("=" * 70)
    print("LILITH DATABASE STATISTICS")
    print("=" * 70)
    print()
    
    # Base knowledge stats
    base_db = base_dir / "response_patterns.db"
    if base_db.exists():
        conn = sqlite3.connect(str(base_db))
        
        # Total patterns
        cursor = conn.execute("SELECT COUNT(*) FROM response_patterns")
        total = cursor.fetchone()[0]
        
        # By intent
        cursor = conn.execute("""
            SELECT intent, COUNT(*) as count 
            FROM response_patterns 
            GROUP BY intent 
            ORDER BY count DESC
        """)
        intents = cursor.fetchall()
        
        # Average success score
        cursor = conn.execute("SELECT AVG(success_score) FROM response_patterns")
        avg_score = cursor.fetchone()[0]
        
        conn.close()
        
        print("BASE KNOWLEDGE:")
        print(f"  Total patterns: {total}")
        print(f"  Average success score: {avg_score:.3f}")
        print(f"  Intents:")
        for intent, count in intents:
            print(f"    {intent}: {count}")
        print()
    
    # User statistics
    if users_dir.exists():
        user_count = len([d for d in users_dir.iterdir() if d.is_dir()])
        print(f"USERS: {user_count}")
        print()
        
        # Top users by pattern count
        user_stats = []
        for user_dir in users_dir.iterdir():
            if not user_dir.is_dir():
                continue
            
            db_path = user_dir / "response_patterns.db"
            if not db_path.exists():
                continue
            
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute("SELECT COUNT(*) FROM response_patterns")
                count = cursor.fetchone()[0]
                conn.close()
                user_stats.append((user_dir.name, count))
            except:
                pass
        
        if user_stats:
            user_stats.sort(key=lambda x: x[1], reverse=True)
            print("  Top users by patterns:")
            for user_id, count in user_stats[:10]:
                print(f"    {user_id}: {count}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Lilith User Management Utility")
        print()
        print("Usage: python manage_users.py <command> [options]")
        print()
        print("Commands:")
        print("  list                    - List all users and database sizes")
        print("  delete <user_id>        - Delete user (with archive)")
        print("  delete <user_id> --no-archive - Delete user without archiving")
        print("  export <user_id> [path] - Export user patterns to JSON")
        print("  cleanup <days>          - Find users inactive for N days (dry run)")
        print("  cleanup <days> --execute - Delete users inactive for N days")
        print("  stats                   - Show database statistics")
        print()
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list":
        list_users()
    
    elif command == "delete":
        if len(sys.argv) < 3:
            print("Error: user_id required")
            sys.exit(1)
        
        user_id = sys.argv[2]
        archive = "--no-archive" not in sys.argv
        delete_user(user_id, archive=archive)
    
    elif command == "export":
        if len(sys.argv) < 3:
            print("Error: user_id required")
            sys.exit(1)
        
        user_id = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else None
        export_user(user_id, output_path)
    
    elif command == "cleanup":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 90
        execute = "--execute" in sys.argv
        cleanup_inactive(days, dry_run=not execute)
    
    elif command == "stats":
        stats()
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
