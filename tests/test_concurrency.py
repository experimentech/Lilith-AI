"""
Test concurrent writes to SQLite ResponseFragmentStore.

Verifies that multiple threads/processes can safely write to the same database
without data loss or corruption.
"""

import sys
from pathlib import Path
import threading
import time
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.response_fragments_sqlite import ResponseFragmentStoreSQLite


def concurrent_writer(store_path: str, encoder, thread_id: int, patterns_per_thread: int):
    """
    Write patterns concurrently from a thread.
    
    Args:
        store_path: Path to SQLite database
        encoder: Semantic encoder
        thread_id: Unique thread identifier
        patterns_per_thread: Number of patterns to write
    """
    store = ResponseFragmentStoreSQLite(
        encoder,
        storage_path=store_path,
        enable_fuzzy_matching=False,
        bootstrap_if_empty=False
    )
    
    for i in range(patterns_per_thread):
        pattern_id = store.add_pattern(
            trigger_context=f"thread {thread_id} pattern {i}",
            response_text=f"Response from thread {thread_id}, pattern {i}",
            success_score=0.5 + (i / patterns_per_thread) * 0.5,
            intent=f"thread{thread_id}"
        )
        
        # Small delay to simulate real work
        time.sleep(0.001)
    
    print(f"✓ Thread {thread_id} completed {patterns_per_thread} writes")


def test_concurrent_writes():
    """Test that multiple threads can write without data loss."""
    print("\n" + "=" * 60)
    print("TEST: Concurrent Writes")
    print("=" * 60)
    
    # Clean up test database
    test_db = Path("data/test/concurrent_test.db")
    if test_db.exists():
        test_db.unlink()
    test_db.parent.mkdir(parents=True, exist_ok=True)
    
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)

    # Pre-create the schema so concurrent writers don't race on initialization.
    ResponseFragmentStoreSQLite(
        encoder,
        storage_path=str(test_db),
        enable_fuzzy_matching=False,
        bootstrap_if_empty=False,
    )
    
    num_threads = 5
    patterns_per_thread = 10
    expected_total = num_threads * patterns_per_thread
    
    print(f"Starting {num_threads} threads, {patterns_per_thread} patterns each")
    print(f"Expected total: {expected_total} patterns")
    print()
    
    # Create threads
    threads = []
    start_time = time.time()
    
    for thread_id in range(num_threads):
        thread = threading.Thread(
            target=concurrent_writer,
            args=(str(test_db), encoder, thread_id, patterns_per_thread)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    elapsed = time.time() - start_time
    
    print()
    print(f"All threads completed in {elapsed:.2f}s")
    
    # Verify results
    store = ResponseFragmentStoreSQLite(
        encoder,
        storage_path=str(test_db),
        enable_fuzzy_matching=False,
        bootstrap_if_empty=False
    )
    
    stats = store.get_stats()
    actual_total = stats['total_patterns']
    
    print(f"Patterns in database: {actual_total}")
    print(f"Expected: {expected_total}")
    
    assert actual_total == expected_total, (
        f"Data loss detected! Missing {expected_total - actual_total} patterns"
    )
    print("✅ No data loss - all patterns written successfully!")


def test_read_while_write():
    """Test that reads don't block writes (and vice versa)."""
    print("\n" + "=" * 60)
    print("TEST: Concurrent Reads and Writes")
    print("=" * 60)
    
    # Clean up test database
    test_db = Path("data/test/read_write_test.db")
    if test_db.exists():
        test_db.unlink()
    test_db.parent.mkdir(parents=True, exist_ok=True)
    
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    
    # Pre-populate with some patterns
    store = ResponseFragmentStoreSQLite(
        encoder,
        storage_path=str(test_db),
        enable_fuzzy_matching=False,
        bootstrap_if_empty=True
    )
    
    for i in range(10):
        store.add_pattern(
            trigger_context=f"test pattern {i}",
            response_text=f"Response {i}",
            success_score=0.7,
            intent="test"
        )
    
    base_stats = store.get_stats()
    base_pattern_count = base_stats["total_patterns"]
    print(f"Pre-populated database with {base_pattern_count} patterns (including seeds)")
    
    read_count = [0]  # Mutable container for thread communication
    write_count = [0]
    patterns_per_writer = 20
    
    def reader():
        """Continuously read patterns."""
        local_store = ResponseFragmentStoreSQLite(
            encoder,
            storage_path=str(test_db),
            enable_fuzzy_matching=False,
            bootstrap_if_empty=False
        )
        
        for _ in range(50):
            patterns = local_store.retrieve_patterns("test", topk=5)
            read_count[0] += 1
            time.sleep(0.01)
        
        print("✓ Reader completed 50 reads")
    
    def writer():
        """Continuously write patterns."""
        local_store = ResponseFragmentStoreSQLite(
            encoder,
            storage_path=str(test_db),
            enable_fuzzy_matching=False,
            bootstrap_if_empty=False
        )
        
        for i in range(patterns_per_writer):
            local_store.add_pattern(
                trigger_context=f"new pattern {i}",
                response_text=f"New response {i}",
                success_score=0.8,
                intent="new"
            )
            write_count[0] += 1
            time.sleep(0.01)
        
        print(f"✓ Writer completed {patterns_per_writer} writes")

    # Start concurrent readers and writers
    print("Starting 2 readers and 2 writers...")
    print()
    
    threads = []
    start_time = time.time()
    
    # Start readers
    for _ in range(2):
        t = threading.Thread(target=reader)
        threads.append(t)
        t.start()
    
    # Start writers
    for _ in range(2):
        t = threading.Thread(target=writer)
        threads.append(t)
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    elapsed = time.time() - start_time
    
    print()
    print(f"Completed in {elapsed:.2f}s")
    print(f"Total reads: {read_count[0]}")
    print(f"Total writes: {write_count[0]}")
    
    # Verify final state
    final_stats = store.get_stats()
    expected_patterns = base_pattern_count + write_count[0]
    
    print(f"Final pattern count: {final_stats['total_patterns']}")
    print(f"Expected: {expected_patterns}")
    
    assert final_stats['total_patterns'] == expected_patterns, (
        "Unexpected pattern count!"
    )
    print("✅ Concurrent reads and writes worked correctly!")


def main():
    """Run concurrency tests."""
    print("=" * 60)
    print("SQLITE CONCURRENCY TESTS")
    print("=" * 60)
    print()
    print("These tests verify that SQLite with WAL mode handles:")
    print("  - Multiple concurrent writes (thread-safe)")
    print("  - Simultaneous reads and writes (no blocking)")
    print()
    
    try:
        test_concurrent_writes()
        test1 = True
    except AssertionError as exc:
        print(exc)
        test1 = False

    try:
        test_read_while_write()
        test2 = True
    except AssertionError as exc:
        print(exc)
        test2 = False
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Concurrent Writes: {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"Concurrent R/W: {'✅ PASSED' if test2 else '❌ FAILED'}")
    
    if all([test1, test2]):
        print("\n🎉 ALL CONCURRENCY TESTS PASSED!")
        print("\nSQLite with WAL mode provides:")
        print("  ✅ Thread-safe concurrent writes")
        print("  ✅ Non-blocking concurrent reads/writes")
        print("  ✅ ACID guarantees (no data loss)")
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
