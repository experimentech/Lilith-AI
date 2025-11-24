"""
Benchmark: JSON vs Database pattern retrieval

Compare performance of in-memory JSON storage vs indexed database queries.
"""

import time
from pipeline.response_fragments import ResponseFragmentStore
from pipeline.database_fragment_store import DatabaseBackedFragmentStore


class MockEncoder:
    """Mock encoder for testing."""
    def encode(self, text):
        import torch
        return torch.randn(144)


def benchmark_retrieval():
    """Compare retrieval speed: JSON vs Database."""
    
    print("=" * 80)
    print("RETRIEVAL PERFORMANCE BENCHMARK")
    print("=" * 80)
    print()
    
    # Initialize both stores
    print("📦 Initializing stores...")
    json_store = ResponseFragmentStore(
        semantic_encoder=MockEncoder(),
        storage_path="conversation_patterns.json"
    )
    
    db_store = DatabaseBackedFragmentStore(
        semantic_encoder=MockEncoder(),
        storage_path="conversation_patterns.db"
    )
    
    pattern_count = len(json_store.patterns)
    print(f"   Loaded {pattern_count} patterns")
    print()
    
    # Test queries
    test_queries = [
        "hello how are you",
        "what's the weather like today",
        "I'm going to the beach",
        "do you like movies",
        "thanks for chatting",
        "see you later",
        "what did you do today",
        "how was school",
        "what's your favorite food",
        "tell me about yourself"
    ]
    
    print("🏃 Running benchmark (100 queries each)...")
    print()
    
    # Benchmark JSON store
    json_times = []
    for query in test_queries:
        start = time.time()
        for _ in range(10):
            _ = json_store.retrieve_patterns(query, topk=5)
        elapsed = (time.time() - start) / 10
        json_times.append(elapsed)
    
    json_avg = sum(json_times) / len(json_times)
    
    # Benchmark Database store
    db_times = []
    for query in test_queries:
        start = time.time()
        for _ in range(10):
            _ = db_store.retrieve_patterns(query, topk=5)
        elapsed = (time.time() - start) / 10
        db_times.append(elapsed)
    
    db_avg = sum(db_times) / len(db_times)
    
    # Calculate speedup
    speedup = json_avg / db_avg if db_avg > 0 else 1.0
    
    # Print results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"JSON File Storage (in-memory + embedding similarity):")
    print(f"  Average time: {json_avg*1000:.2f}ms per query")
    print(f"  Total time: {sum(json_times)*1000:.2f}ms")
    print()
    print(f"SQLite Database (indexed keyword queries):")
    print(f"  Average time: {db_avg*1000:.2f}ms per query")
    print(f"  Total time: {sum(db_times)*1000:.2f}ms")
    print()
    print(f"⚡ Speedup: {speedup:.2f}x faster with database")
    print()
    
    # Per-query breakdown
    print("=" * 80)
    print("PER-QUERY BREAKDOWN")
    print("=" * 80)
    print()
    print(f"{'Query':<40} {'JSON (ms)':<12} {'DB (ms)':<12} {'Speedup':<10}")
    print("-" * 80)
    
    for i, query in enumerate(test_queries):
        speedup_q = json_times[i] / db_times[i] if db_times[i] > 0 else 1.0
        print(f"{query:<40} {json_times[i]*1000:>10.2f}  {db_times[i]*1000:>10.2f}  {speedup_q:>8.2f}x")
    
    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()
    
    if speedup > 1.5:
        print(f"✅ Database is significantly faster ({speedup:.1f}x)")
        print("   Indexed keyword queries outperform embedding similarity")
    elif speedup > 1.0:
        print(f"✅ Database is faster ({speedup:.1f}x)")
        print("   Marginal improvement, scales better with more patterns")
    else:
        print(f"⚠️  JSON is faster ({1/speedup:.1f}x)")
        print("   Small dataset - database overhead dominates")
    
    print()
    print(f"Pattern count: {pattern_count}")
    print(f"Database benefits increase with pattern count (O(log n) vs O(n))")
    print()


if __name__ == "__main__":
    benchmark_retrieval()
