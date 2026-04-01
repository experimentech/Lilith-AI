#!/usr/bin/env python3
"""
Lilith V2 Corpus Ingestion Tool

Ingest training data into the production knowledge graph.

Usage:
    python ingest_corpus.py training_data.jsonl
    python ingest_corpus.py conversations.csv --field content
    python ingest_corpus.py books/*.txt --merge
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Ensure we can import lilith_v2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from v2.lilith_v2.corpus_ingester import CorpusIngester, ExtractionConfig, IngestionStats
from v2.lilith_v2.relational_graph_store import RelationalGraphStore

# Try to import encoders - prefer SemanticPMFlowEncoder
try:
    from lilith.learned_vocabulary_encoder import SemanticPMFlowEncoder
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False

try:
    from pmflow import PMFlowEmbeddingEncoder
    HAS_PMFLOW = True
except ImportError:
    HAS_PMFLOW = False


def main():
    parser = argparse.ArgumentParser(
        description="Ingest training data into Lilith's production knowledge graph"
    )
    parser.add_argument("files", nargs="+", help="Corpus files to ingest")
    parser.add_argument("--field", "-f", default="text",
                        help="Text field name for JSON/CSV (default: text)")
    parser.add_argument("--output", "-o", default="data/production",
                        help="Output directory for databases (default: data/production)")
    parser.add_argument("--batch-size", "-b", type=int, default=1000,
                        help="Batch size for commits (default: 1000)")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip computing embeddings (faster)")
    parser.add_argument("--min-word-freq", type=int, default=2,
                        help="Minimum word frequency to include (default: 2)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Setup output directory
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_path.absolute()}")

    # Initialize graph store
    db_path = out_path / "knowledge.sqlite"
    graph = RelationalGraphStore(str(db_path))
    print(f"Knowledge graph: {db_path}")

    # Initialize encoder - prefer semantic for learning capability
    encoder = None
    if not args.no_embeddings:
        if HAS_SEMANTIC:
            print("Loading SemanticPMFlowEncoder for embeddings...")
            encoder = SemanticPMFlowEncoder(
                dimension=96, latent_dim=48, bootstrap_semantics=True
            )
            print("  SemanticPMFlowEncoder ready (trainable word embeddings)")
        elif HAS_PMFLOW:
            print("Loading PMFlow encoder for embeddings...")
            encoder = PMFlowEmbeddingEncoder(dimension=96, latent_dim=48)
            print("  PMFlow encoder ready")
        else:
            print("No encoder available, ingesting without embeddings")
    else:
        print("Skipping embeddings (--no-embeddings)")

    # Configure extraction
    config = ExtractionConfig(
        min_word_frequency=args.min_word_freq,
        extract_vocabulary=True,
        extract_concepts=True,
        extract_relationships=True,
        extract_grammar=True,
    )

    # Create ingester
    ingester = CorpusIngester(
        graph_store=graph,
        encoder=encoder,
        config=config,
        tenant_id="production",
    )

    # Process files
    total_stats = {
        "files": 0,
        "lines": 0,
        "texts": 0,
        "new_words": 0,
        "new_concepts": 0,
        "new_edges": 0,
        "errors": 0,
    }

    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            print(f"  Skipping (not found): {path}")
            continue

        print(f"\nIngesting: {path.name}")
        
        try:
            stats = ingester.ingest_file(
                str(path),
                text_field=args.field,
                batch_size=args.batch_size,
            )
            
            total_stats["files"] += 1
            total_stats["lines"] += stats.lines_processed
            total_stats["texts"] += stats.texts_extracted
            total_stats["new_words"] += stats.new_words
            total_stats["new_concepts"] += stats.new_concepts
            total_stats["new_edges"] += stats.new_edges
            total_stats["errors"] += stats.parse_errors
            
            print(f"  {stats.texts_extracted} texts, {stats.new_words} new words, "
                  f"{stats.new_concepts} new concepts, {stats.new_edges} new edges")
            
        except Exception as e:
            print(f"  Error: {e}")
            total_stats["errors"] += 1

    # Summary
    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)
    print(f"Files processed: {total_stats['files']}")
    print(f"Lines processed: {total_stats['lines']}")
    print(f"Texts extracted: {total_stats['texts']}")
    print(f"New words: {total_stats['new_words']}")
    print(f"New concepts: {total_stats['new_concepts']}")
    print(f"New relationships: {total_stats['new_edges']}")
    print(f"Errors: {total_stats['errors']}")
    
    # Show database size
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"\nDatabase size: {size_mb:.2f} MB")

    # Show extraction stats
    ext_stats = ingester.get_extraction_stats()
    print(f"\nSession stats:")
    print(f"  Unique words seen: {ext_stats['seen_words']}")
    print(f"  Unique concepts: {ext_stats['seen_concepts']}")
    print(f"  Unique edges: {ext_stats['seen_edges']}")

    # Cleanup
    graph.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
