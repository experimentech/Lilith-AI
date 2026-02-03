"""
Corpus Ingester for Lilith V2

Single-pass extraction of vocabulary, grammar, concepts, and relationships
from training files (JSONL, CSV, TXT).

The key insight: most entries share vocabulary and concepts, so the graph
grows sub-linearly with input size (lots of edge additions between existing
nodes rather than exponential node growth).

Usage:
    ingester = CorpusIngester(graph_store, encoder)
    stats = ingester.ingest_file("training_data.jsonl")
"""

import json
import csv
import re
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Generator
from dataclasses import dataclass, field
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Statistics from a corpus ingestion run."""
    
    file_path: str
    lines_processed: int = 0
    texts_extracted: int = 0
    
    # Vocabulary
    new_words: int = 0
    existing_words: int = 0
    
    # Concepts
    new_concepts: int = 0
    existing_concepts: int = 0
    
    # Relationships
    new_edges: int = 0
    strengthened_edges: int = 0
    
    # Grammar
    new_patterns: int = 0
    
    # Errors
    parse_errors: int = 0
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Ingested {self.file_path}:\n"
            f"  Lines: {self.lines_processed}, Texts: {self.texts_extracted}\n"
            f"  Words: {self.new_words} new, {self.existing_words} existing\n"
            f"  Concepts: {self.new_concepts} new, {self.existing_concepts} existing\n"
            f"  Edges: {self.new_edges} new, {self.strengthened_edges} strengthened\n"
            f"  Grammar patterns: {self.new_patterns}\n"
            f"  Errors: {self.parse_errors}"
        )


@dataclass
class ExtractionConfig:
    """Configuration for what to extract from corpus."""
    
    # Extraction toggles
    extract_vocabulary: bool = True
    extract_concepts: bool = True
    extract_relationships: bool = True
    extract_grammar: bool = True
    
    # Thresholds
    min_word_length: int = 2
    max_word_length: int = 50
    min_word_frequency: int = 1  # Require word to appear N times before adding
    
    # Concept extraction
    extract_noun_phrases: bool = True
    extract_named_entities: bool = True  # Capitalized sequences
    
    # Relationship extraction
    cooccurrence_window: int = 5  # Words within N positions are related
    extract_svo_patterns: bool = True  # Subject-Verb-Object
    
    # Grammar
    track_pos_patterns: bool = True  # Part-of-speech patterns
    track_sentence_patterns: bool = True


class CorpusIngester:
    """
    Single-pass corpus ingestion for Lilith's knowledge graph.
    
    Extracts vocabulary, concepts, and relationships from text files
    and populates the graph store incrementally.
    """
    
    def __init__(
        self,
        graph_store,
        encoder=None,
        config: Optional[ExtractionConfig] = None,
        tenant_id: str = "production",
    ):
        """
        Initialize the ingester.
        
        Args:
            graph_store: RelationalGraphStore or MultiTenantGraphManager
            encoder: Optional encoder for embeddings
            config: ExtractionConfig for extraction options
            tenant_id: Tenant ID for writes (default: "production")
        """
        self.graph = graph_store
        self.encoder = encoder
        self.config = config or ExtractionConfig()
        self.tenant_id = tenant_id
        
        # Track what we've seen in this session
        self._seen_words: Set[str] = set()
        self._word_counts: Counter = Counter()
        self._seen_concepts: Set[str] = set()
        self._seen_edges: Set[Tuple[str, str, str]] = set()
        
        # Simple POS patterns (without full NLP library)
        self._pos_patterns = Counter()
        
        # Common function words to skip for concept extraction
        self._stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'this', 'that', 'these', 'those', 'it', 'its', 'i', 'you', 'he', 'she',
            'we', 'they', 'my', 'your', 'his', 'her', 'our', 'their', 'what',
            'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
            'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'also', 'now', 'if', 'then', 'as', 'about', 'after', 'before', 'into',
        }
    
    def ingest_file(
        self,
        file_path: str,
        text_field: Optional[str] = None,
        batch_size: int = 1000,
    ) -> IngestionStats:
        """
        Ingest a file into the knowledge graph.
        
        Supports:
        - .jsonl: JSON Lines, expects {"text": "..."} or configurable field
        - .json: JSON array of objects
        - .csv: CSV with configurable text column
        - .txt: Plain text, one line per entry
        
        Args:
            file_path: Path to the file
            text_field: Field name for text (for JSON/CSV)
            batch_size: Commit every N lines
            
        Returns:
            IngestionStats with counts
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stats = IngestionStats(file_path=str(path))
        
        # Dispatch based on extension
        ext = path.suffix.lower()
        
        if ext == ".jsonl":
            texts = self._read_jsonl(path, text_field or "text", stats)
        elif ext == ".json":
            texts = self._read_json(path, text_field or "text", stats)
        elif ext == ".csv":
            texts = self._read_csv(path, text_field or "text", stats)
        elif ext == ".txt":
            texts = self._read_txt(path, stats)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        # Process texts
        batch_texts = []
        for text in texts:
            if text:
                batch_texts.append(text)
                stats.texts_extracted += 1
                
                if len(batch_texts) >= batch_size:
                    self._process_batch(batch_texts, stats)
                    batch_texts = []
        
        # Process remaining
        if batch_texts:
            self._process_batch(batch_texts, stats)
        
        logger.info(stats.summary())
        return stats
    
    def ingest_texts(
        self,
        texts: List[str],
        source_name: str = "direct",
    ) -> IngestionStats:
        """
        Ingest a list of texts directly.
        
        Args:
            texts: List of text strings
            source_name: Name for stats
            
        Returns:
            IngestionStats
        """
        stats = IngestionStats(file_path=source_name)
        stats.lines_processed = len(texts)
        stats.texts_extracted = len(texts)
        
        self._process_batch(texts, stats)
        
        logger.info(stats.summary())
        return stats
    
    # --- File Readers ---
    
    def _read_jsonl(
        self,
        path: Path,
        text_field: str,
        stats: IngestionStats,
    ) -> Generator[str, None, None]:
        """Read JSON Lines file."""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                stats.lines_processed += 1
                try:
                    obj = json.loads(line.strip())
                    if isinstance(obj, dict):
                        # Try the text field, or common alternatives
                        text = obj.get(text_field) or obj.get("content") or obj.get("message")
                        if text:
                            yield text
                except json.JSONDecodeError:
                    stats.parse_errors += 1
    
    def _read_json(
        self,
        path: Path,
        text_field: str,
        stats: IngestionStats,
    ) -> Generator[str, None, None]:
        """Read JSON array file."""
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for obj in data:
                        stats.lines_processed += 1
                        if isinstance(obj, dict):
                            text = obj.get(text_field) or obj.get("content")
                            if text:
                                yield text
                        elif isinstance(obj, str):
                            yield obj
            except json.JSONDecodeError:
                stats.parse_errors += 1
    
    def _read_csv(
        self,
        path: Path,
        text_field: str,
        stats: IngestionStats,
    ) -> Generator[str, None, None]:
        """Read CSV file."""
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats.lines_processed += 1
                text = row.get(text_field)
                if text:
                    yield text
    
    def _read_txt(
        self,
        path: Path,
        stats: IngestionStats,
    ) -> Generator[str, None, None]:
        """Read plain text file, one line per entry."""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                stats.lines_processed += 1
                text = line.strip()
                if text:
                    yield text
    
    # --- Processing ---
    
    def _process_batch(self, texts: List[str], stats: IngestionStats) -> None:
        """Process a batch of texts."""
        for text in texts:
            self._process_text(text, stats)
    
    def _process_text(self, text: str, stats: IngestionStats) -> None:
        """Extract and store knowledge from a single text."""
        # Tokenize
        words = self._tokenize(text)
        
        if self.config.extract_vocabulary:
            self._extract_vocabulary(words, stats)
        
        if self.config.extract_concepts:
            self._extract_concepts(text, words, stats)
        
        if self.config.extract_relationships:
            self._extract_relationships(words, stats)
        
        if self.config.extract_grammar:
            self._extract_grammar(text, words, stats)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization: split on non-alphanumeric, lowercase
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*[a-zA-Z0-9]|[a-zA-Z]", text)
        return [w.lower() for w in words]
    
    def _extract_vocabulary(self, words: List[str], stats: IngestionStats) -> None:
        """Extract vocabulary words and add to graph."""
        cfg = self.config
        
        for word in words:
            if len(word) < cfg.min_word_length or len(word) > cfg.max_word_length:
                continue
            
            self._word_counts[word] += 1
            
            # Check frequency threshold
            if self._word_counts[word] < cfg.min_word_frequency:
                continue
            
            # Check if already added
            if word in self._seen_words:
                stats.existing_words += 1
                continue
            
            # Check if exists in graph
            node_id = f"word_{word}"
            existing = self.graph.get_node(node_id, tenant_id=self.tenant_id)
            
            if existing:
                stats.existing_words += 1
                self._seen_words.add(word)
            else:
                # Add new word
                embedding = None
                if self.encoder:
                    try:
                        emb = self.encoder.encode([word])
                        if hasattr(emb, 'tolist'):
                            embedding = emb.flatten().tolist()
                    except Exception:
                        pass
                
                self.graph.add_node(
                    node_id=node_id,
                    node_type="word",
                    term=word,
                    confidence=1.0,
                    data={"embedding": embedding} if embedding else None,
                    tenant_id=self.tenant_id,
                )
                stats.new_words += 1
                self._seen_words.add(word)
    
    def _extract_concepts(
        self,
        text: str,
        words: List[str],
        stats: IngestionStats,
    ) -> None:
        """Extract concepts (noun phrases, named entities)."""
        cfg = self.config
        concepts = []
        
        # Extract capitalized sequences (named entities)
        if cfg.extract_named_entities:
            # Match sequences of capitalized words
            entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
            concepts.extend(entities)
        
        # Extract noun phrases (simple heuristic: adj* noun+)
        if cfg.extract_noun_phrases:
            # Very simple: sequences of content words
            content_words = [w for w in words if w not in self._stopwords]
            # Look for 2-3 word sequences
            for i in range(len(content_words) - 1):
                bigram = f"{content_words[i]} {content_words[i+1]}"
                concepts.append(bigram)
                if i < len(content_words) - 2:
                    trigram = f"{bigram} {content_words[i+2]}"
                    concepts.append(trigram)
        
        # Add unique concepts
        for concept in concepts:
            concept_lower = concept.lower()
            concept_id = f"concept_{concept_lower.replace(' ', '_')}"
            
            if concept_id in self._seen_concepts:
                stats.existing_concepts += 1
                continue
            
            existing = self.graph.get_node(concept_id, tenant_id=self.tenant_id)
            
            if existing:
                stats.existing_concepts += 1
                self._seen_concepts.add(concept_id)
            else:
                self.graph.add_node(
                    node_id=concept_id,
                    node_type="concept",
                    term=concept_lower,
                    confidence=1.0,
                    tenant_id=self.tenant_id,
                )
                stats.new_concepts += 1
                self._seen_concepts.add(concept_id)
    
    def _extract_relationships(self, words: List[str], stats: IngestionStats) -> None:
        """Extract co-occurrence relationships."""
        cfg = self.config
        content_words = [w for w in words if w not in self._stopwords and len(w) >= cfg.min_word_length]
        
        # Co-occurrence within window
        window = cfg.cooccurrence_window
        for i, word1 in enumerate(content_words):
            for j in range(i + 1, min(i + window + 1, len(content_words))):
                word2 = content_words[j]
                if word1 == word2:
                    continue
                
                # Canonical edge order
                if word1 > word2:
                    word1, word2 = word2, word1
                
                source_id = f"word_{word1}"
                target_id = f"word_{word2}"
                edge_key = (source_id, target_id, "cooccurs")
                
                if edge_key in self._seen_edges:
                    stats.strengthened_edges += 1
                    continue
                
                # Only add edge if both nodes exist (were added as vocabulary)
                if word1 not in self._seen_words or word2 not in self._seen_words:
                    continue
                
                # Add edge
                try:
                    self.graph.add_edge(
                        source=source_id,
                        target=target_id,
                        edge_type="cooccurs",
                        confidence=0.5,  # Base confidence for co-occurrence
                        tenant_id=self.tenant_id,
                    )
                    stats.new_edges += 1
                    self._seen_edges.add(edge_key)
                except Exception:
                    # Edge creation failed (likely FK constraint)
                    pass
    
    def _extract_grammar(
        self,
        text: str,
        words: List[str],
        stats: IngestionStats,
    ) -> None:
        """Extract grammar patterns (simple heuristics)."""
        # Simple sentence pattern: count word count buckets
        word_count = len(words)
        
        if word_count <= 5:
            pattern = "short"
        elif word_count <= 15:
            pattern = "medium"
        elif word_count <= 30:
            pattern = "long"
        else:
            pattern = "very_long"
        
        self._pos_patterns[pattern] += 1
        
        # Question pattern
        if text.strip().endswith("?"):
            self._pos_patterns["question"] += 1
        
        # Command pattern (starts with verb-like word)
        if words and words[0] in {'go', 'run', 'make', 'do', 'get', 'find', 'show', 'list', 'create', 'delete', 'read', 'write'}:
            self._pos_patterns["imperative"] += 1
        
        # Could add more sophisticated patterns here
        # For now, count as new if first time seeing this sentence structure
        if self._pos_patterns[pattern] == 1:
            stats.new_patterns += 1
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get current extraction statistics."""
        return {
            "seen_words": self._seen_words.copy(),
            "word_counts": dict(self._word_counts),
            "seen_concepts": self._seen_concepts.copy(),
            "seen_edges": len(self._seen_edges),
            "pos_patterns": dict(self._pos_patterns),
        }
    
    def reset_session(self) -> None:
        """Reset session tracking (for multiple file ingestion)."""
        self._seen_words.clear()
        self._word_counts.clear()
        self._seen_concepts.clear()
        self._seen_edges.clear()
        self._pos_patterns.clear()


def ingest_corpus(
    corpus_path: str,
    data_dir: str = "data/production",
    text_field: str = "text",
) -> IngestionStats:
    """
    Convenience function to ingest a corpus into production databases.
    
    Args:
        corpus_path: Path to corpus file (JSONL, CSV, TXT)
        data_dir: Directory for production databases
        text_field: Field name for text content
        
    Returns:
        IngestionStats
    """
    from .relational_graph_store import RelationalGraphStore
    
    # Ensure production directory exists
    prod_path = Path(data_dir)
    prod_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize stores
    graph = RelationalGraphStore(str(prod_path / "knowledge.sqlite"))
    
    # Try to get encoder
    encoder = None
    try:
        from pmflow import PMFlowEmbeddingEncoder
        encoder = PMFlowEmbeddingEncoder(dimension=96, latent_dim=48)
    except ImportError:
        logger.warning("PMFlow not available, ingesting without embeddings")
    
    # Create ingester
    ingester = CorpusIngester(
        graph_store=graph,
        encoder=encoder,
        tenant_id="production",
    )
    
    # Ingest
    stats = ingester.ingest_file(corpus_path, text_field=text_field)
    
    # Cleanup
    graph.close()
    
    return stats
