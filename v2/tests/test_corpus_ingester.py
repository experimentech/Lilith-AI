"""
Tests for corpus ingestion.
"""

import os
import sys
import tempfile
import unittest
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2.lilith_v2.corpus_ingester import CorpusIngester, ExtractionConfig, IngestionStats
from v2.lilith_v2.relational_graph_store import RelationalGraphStore


class TestCorpusIngester(unittest.TestCase):
    """Test corpus ingestion functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.sqlite")
        self.graph = RelationalGraphStore(self.db_path)
        
        self.config = ExtractionConfig(
            min_word_frequency=1,
            min_word_length=2,
        )
        self.ingester = CorpusIngester(
            graph_store=self.graph,
            encoder=None,
            config=self.config,
            tenant_id="test",
        )
    
    def tearDown(self):
        """Clean up."""
        self.graph.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ingest_jsonl(self):
        """Test JSONL file ingestion."""
        # Create test file
        corpus_path = os.path.join(self.temp_dir, "corpus.jsonl")
        with open(corpus_path, "w") as f:
            f.write('{"text": "Python is a programming language."}\n')
            f.write('{"text": "Machine learning uses data."}\n')
        
        stats = self.ingester.ingest_file(corpus_path)
        
        self.assertEqual(stats.lines_processed, 2)
        self.assertEqual(stats.texts_extracted, 2)
        self.assertGreater(stats.new_words, 0)
    
    def test_ingest_txt(self):
        """Test plain text file ingestion."""
        corpus_path = os.path.join(self.temp_dir, "corpus.txt")
        with open(corpus_path, "w") as f:
            f.write("Hello world\n")
            f.write("This is a test\n")
            f.write("Learning from text\n")
        
        stats = self.ingester.ingest_file(corpus_path)
        
        self.assertEqual(stats.lines_processed, 3)
        self.assertEqual(stats.texts_extracted, 3)
    
    def test_ingest_csv(self):
        """Test CSV file ingestion."""
        corpus_path = os.path.join(self.temp_dir, "corpus.csv")
        with open(corpus_path, "w") as f:
            f.write("id,text,other\n")
            f.write("1,First sentence here,meta\n")
            f.write("2,Second sentence now,data\n")
        
        stats = self.ingester.ingest_file(corpus_path, text_field="text")
        
        self.assertEqual(stats.lines_processed, 2)
        self.assertEqual(stats.texts_extracted, 2)
    
    def test_vocabulary_extraction(self):
        """Test vocabulary is extracted correctly."""
        texts = [
            "Python programming language",
            "Python is great for data science",
        ]
        
        stats = self.ingester.ingest_texts(texts)
        
        # "python" should appear twice but only be counted once as new
        ext_stats = self.ingester.get_extraction_stats()
        self.assertIn("python", ext_stats["seen_words"])
    
    def test_concept_extraction(self):
        """Test concepts are extracted using SemanticExtractor patterns."""
        # Use text with semantic patterns that SemanticExtractor can match:
        # "X is a Y", "X is a type of Y", "X has Y", etc.
        texts = [
            "Python is a programming language.",
            "A dog is a mammal.",
            "Machine learning is a field of study.",
        ]
        
        stats = self.ingester.ingest_texts(texts)
        
        # Should have extracted concepts from "X is a Y" patterns
        self.assertGreater(stats.new_concepts, 0)
    
    def test_relationship_extraction(self):
        """Test relationships are created using WorldModel."""
        # Use text with entity patterns (determiners + nouns) and
        # spatial/causal markers for WorldModel to extract
        texts = [
            "The cat is on the mat.",
            "The dog runs because it is happy.",
        ]
        
        stats = self.ingester.ingest_texts(texts)
        
        # Should have extracted entity relationships
        self.assertGreater(stats.new_edges, 0)
    
    def test_duplicate_handling(self):
        """Test that duplicates are tracked, not re-added."""
        texts = ["test word repeated"]
        
        stats1 = self.ingester.ingest_texts(texts)
        new_words_1 = stats1.new_words
        
        # Ingest same texts again
        stats2 = self.ingester.ingest_texts(texts)
        
        # Should see them as existing, not new
        self.assertEqual(stats2.new_words, 0)
        self.assertEqual(stats2.existing_words, new_words_1)
    
    def test_min_word_frequency(self):
        """Test minimum word frequency threshold."""
        config = ExtractionConfig(min_word_frequency=2)
        ingester = CorpusIngester(
            self.graph,
            config=config,
            tenant_id="test2",
        )
        
        texts = [
            "unique rare special",
            "common common common",
        ]
        
        stats = ingester.ingest_texts(texts)
        
        # Only "common" should meet the threshold
        ext_stats = ingester.get_extraction_stats()
        self.assertIn("common", ext_stats["seen_words"])
    
    def test_session_reset(self):
        """Test that session can be reset for multiple file ingestion."""
        texts = ["word one"]
        self.ingester.ingest_texts(texts)
        
        ext_before = self.ingester.get_extraction_stats()
        self.assertGreater(len(ext_before["seen_words"]), 0)
        
        self.ingester.reset_session()
        
        ext_after = self.ingester.get_extraction_stats()
        self.assertEqual(len(ext_after["seen_words"]), 0)


class TestMultiTenantWithProduction(unittest.TestCase):
    """Test multi-tenant stores with production layer."""
    
    def setUp(self):
        """Set up test directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = os.path.join(self.temp_dir, "base")
        self.users_dir = os.path.join(self.temp_dir, "users")
        self.prod_dir = os.path.join(self.temp_dir, "production")
        
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.users_dir, exist_ok=True)
        os.makedirs(self.prod_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_production_layer_loading(self):
        """Test that production layer is loaded when present."""
        from v2.lilith_v2.multi_tenant_store import MultiTenantGraphManager
        
        # Create production database with content
        prod_graph = RelationalGraphStore(os.path.join(self.prod_dir, "knowledge.sqlite"))
        prod_graph.add_node("prod_node", "concept", "production_concept", 1.0)
        prod_graph.close()
        
        # Create manager with production layer
        manager = MultiTenantGraphManager(
            self.base_dir,
            self.users_dir,
            self.prod_dir,
        )
        
        # Should find production node
        node = manager.get_node("prod_node", tenant_id="user1")
        self.assertIsNotNone(node)
        self.assertEqual(node["term"], "production_concept")
        
        manager.close()
    
    def test_tenant_overrides_production(self):
        """Test that tenant layer overrides production."""
        from v2.lilith_v2.multi_tenant_store import MultiTenantGraphManager
        
        # Create production with a concept
        prod_graph = RelationalGraphStore(os.path.join(self.prod_dir, "knowledge.sqlite"))
        prod_graph.add_node("shared_node", "concept", "production_version", 1.0)
        prod_graph.close()
        
        # Create manager
        manager = MultiTenantGraphManager(
            self.base_dir,
            self.users_dir,
            self.prod_dir,
        )
        
        # Add tenant override
        manager.add_node("shared_node", "concept", "tenant_version", 1.0, tenant_id="user1")
        
        # Tenant should see their version
        node = manager.get_node("shared_node", tenant_id="user1")
        self.assertEqual(node["term"], "tenant_version")
        
        # Other tenant sees production version
        node2 = manager.get_node("shared_node", tenant_id="user2")
        self.assertEqual(node2["term"], "production_version")
        
        manager.close()


if __name__ == "__main__":
    unittest.main()
