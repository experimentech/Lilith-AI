#!/usr/bin/env python3
"""
Train Lilith from Documents

Ingest unstructured documents (text, markdown, PDFs) and extract semantic knowledge.
This is similar to how Wikipedia lookup works, but for your own documents.

Usage:
    # Single file
    python train_from_document.py --file docs/manual.txt
    
    # Directory of files
    python train_from_document.py --dir docs/knowledge_base/
    
    # URL
    python train_from_document.py --url "https://example.com/faq"
    
    # For a specific server
    python train_from_document.py --file manual.txt --server-id 123456789
    
    # With custom chunk size
    python train_from_document.py --file large_doc.txt --chunk-size 500

Supported formats:
    - Plain text (.txt)
    - Markdown (.md)
    - HTML (.html)
    - PDF (.pdf) - requires pypdf
    - Word (.docx) - requires python-docx
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lilith.session import LilithSession, SessionConfig
from lilith.semantic_extractor import SemanticExtractor


def load_text_file(filepath: Path) -> str:
    """Load plain text file."""
    return filepath.read_text(encoding='utf-8')


def load_markdown_file(filepath: Path) -> str:
    """Load markdown file, preserving structure."""
    return filepath.read_text(encoding='utf-8')


def load_html_file(filepath: Path) -> str:
    """Load HTML file, extracting text content."""
    try:
        from bs4 import BeautifulSoup
        html = filepath.read_text(encoding='utf-8')
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        return soup.get_text(separator='\n')
    except ImportError:
        print("⚠️  beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
        # Fallback: basic HTML tag stripping
        html = filepath.read_text(encoding='utf-8')
        clean = re.sub(r'<[^>]+>', ' ', html)
        return re.sub(r'\s+', ' ', clean)


def load_pdf_file(filepath: Path) -> str:
    """Load PDF file, extracting text content."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text())
        return '\n\n'.join(text_parts)
    except ImportError:
        raise ImportError("pypdf not installed. Install with: pip install pypdf")


def load_docx_file(filepath: Path) -> str:
    """Load Word document, extracting text content."""
    try:
        from docx import Document
        doc = Document(filepath)
        paragraphs = [p.text for p in doc.paragraphs]
        return '\n\n'.join(paragraphs)
    except ImportError:
        raise ImportError("python-docx not installed. Install with: pip install python-docx")


def load_url(url: str) -> str:
    """Fetch and extract text from URL."""
    import requests
    
    response = requests.get(url, headers={'User-Agent': 'Lilith/1.0'}, timeout=30)
    response.raise_for_status()
    
    content_type = response.headers.get('content-type', '')
    
    if 'html' in content_type:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            return soup.get_text(separator='\n')
        except ImportError:
            # Basic fallback
            clean = re.sub(r'<[^>]+>', ' ', response.text)
            return re.sub(r'\s+', ' ', clean)
    else:
        return response.text


def load_document(source: str) -> Tuple[str, str]:
    """
    Load document from file path or URL.
    
    Returns:
        Tuple of (content, source_name)
    """
    if source.startswith(('http://', 'https://')):
        return load_url(source), source
    
    filepath = Path(source)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {source}")
    
    suffix = filepath.suffix.lower()
    
    loaders = {
        '.txt': load_text_file,
        '.md': load_markdown_file,
        '.markdown': load_markdown_file,
        '.html': load_html_file,
        '.htm': load_html_file,
        '.pdf': load_pdf_file,
        '.docx': load_docx_file,
    }
    
    if suffix not in loaders:
        print(f"⚠️  Unknown file type {suffix}, treating as plain text")
        return load_text_file(filepath), filepath.name
    
    return loaders[suffix](filepath), filepath.name


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into semantic chunks.
    
    Tries to split on paragraph boundaries, falling back to sentence boundaries,
    then word boundaries.
    
    Args:
        text: Full document text
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks for context continuity
        
    Returns:
        List of text chunks
    """
    # First, split on double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If paragraph fits in current chunk, add it
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        else:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk)
            
            # If paragraph is too long, split it further
            if len(para) > chunk_size:
                # Split on sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence
            else:
                current_chunk = para
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def extract_qa_pairs(text: str, extractor: SemanticExtractor) -> List[Tuple[str, str]]:
    """
    Extract question-answer pairs from text using semantic extraction.
    
    Generates questions from extracted concepts and uses the source text
    as the answer.
    
    Args:
        text: Text chunk to extract from
        extractor: SemanticExtractor instance
        
    Returns:
        List of (question, answer) tuples
    """
    qa_pairs = []
    
    # Extract concepts from text
    concepts = extractor.extract_concepts("query", text)
    
    for concept in concepts:
        # Generate "What is X?" questions for type relations
        if concept.type_relations:
            question = f"What is {concept.term}?"
            # Build answer from type relations and properties
            answer_parts = []
            for relation in concept.type_relations:
                answer_parts.append(f"{concept.term} is {relation}")
            if concept.properties:
                answer_parts.append(f"It is characterized by: {', '.join(concept.properties[:3])}")
            
            if answer_parts:
                qa_pairs.append((question, ". ".join(answer_parts) + "."))
        
        # Generate "What is X used for?" questions for usage relations
        if concept.usage_relations:
            question = f"What is {concept.term} used for?"
            answer = f"{concept.term} is used for {', '.join(concept.usage_relations)}."
            qa_pairs.append((question, answer))
        
        # Generate "What does X have?" questions for has relations
        if concept.has_relations:
            question = f"What features does {concept.term} have?"
            answer = f"{concept.term} has {', '.join(concept.has_relations)}."
            qa_pairs.append((question, answer))
        
        # Generate "Who created X?" questions
        if concept.created_by:
            question = f"Who created {concept.term}?"
            answer = f"{concept.term} was created by {concept.created_by}."
            qa_pairs.append((question, answer))
    
    return qa_pairs


def train_from_chunks(
    session: LilithSession,
    chunks: List[str],
    source_name: str,
    extract_qa: bool = True
) -> Tuple[int, int]:
    """
    Train session from text chunks.
    
    Args:
        session: LilithSession instance
        chunks: List of text chunks
        source_name: Name of source document (for logging)
        extract_qa: If True, extract Q&A pairs. If False, store raw chunks.
        
    Returns:
        Tuple of (concepts_stored, qa_pairs_stored)
    """
    extractor = SemanticExtractor()
    concepts_stored = 0
    qa_pairs_stored = 0
    
    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i+1}/{len(chunks)}...", end='\r')
        
        if extract_qa:
            # Extract Q&A pairs and teach them
            qa_pairs = extract_qa_pairs(chunk, extractor)
            
            for question, answer in qa_pairs:
                try:
                    session.teach(question, answer, intent="document_knowledge")
                    qa_pairs_stored += 1
                except Exception as e:
                    print(f"\n  ⚠️  Failed to store Q&A: {e}")
        
        # Also store concepts directly
        concepts = extractor.extract_concepts("query", chunk)
        
        for concept in concepts:
            try:
                # Store in concept store if available
                if hasattr(session, 'fragments') and hasattr(session.fragments, 'concept_store'):
                    if session.fragments.concept_store:
                        session.fragments.concept_store.add_concept(
                            term=concept.term,
                            embedding=None,  # Will be computed
                            properties=concept.type_relations + concept.properties,
                            source=f"document:{source_name}"
                        )
                        concepts_stored += 1
            except Exception as e:
                pass  # Concept storage is optional
    
    print()  # Clear the progress line
    return concepts_stored, qa_pairs_stored


def main():
    parser = argparse.ArgumentParser(
        description="Train Lilith from documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input sources (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--file', '-f', help="Path to document file")
    source_group.add_argument('--dir', '-d', help="Path to directory of documents")
    source_group.add_argument('--url', '-u', help="URL to fetch and train from")
    
    # Target specification
    parser.add_argument('--server-id', help="Discord server ID (stores to server knowledge)")
    parser.add_argument('--user-id', default="document_training", help="User ID for storage")
    parser.add_argument('--data-dir', default="data", help="Base data directory")
    
    # Processing options
    parser.add_argument('--chunk-size', type=int, default=500, help="Target chunk size in characters")
    parser.add_argument('--no-qa', action='store_true', help="Don't extract Q&A pairs, only concepts")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be extracted without storing")
    
    args = parser.parse_args()
    
    print("📚 Lilith Document Training")
    print("=" * 60)
    
    # Collect documents to process
    documents = []
    
    if args.file:
        documents.append(args.file)
    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"❌ Directory not found: {args.dir}")
            sys.exit(1)
        
        # Find all supported files
        extensions = ['.txt', '.md', '.markdown', '.html', '.htm', '.pdf', '.docx']
        for ext in extensions:
            documents.extend(str(p) for p in dir_path.glob(f'**/*{ext}'))
        
        if not documents:
            print(f"❌ No supported documents found in {args.dir}")
            sys.exit(1)
        
        print(f"📁 Found {len(documents)} documents in {args.dir}")
    elif args.url:
        documents.append(args.url)
    
    # Configure session
    if args.server_id:
        data_path = f"{args.data_dir}/servers/{args.server_id}"
        print(f"📍 Storing to server: {args.server_id}")
    else:
        data_path = f"{args.data_dir}/users/{args.user_id}"
        print(f"📍 Storing to user: {args.user_id}")
    
    if args.dry_run:
        print("🔍 DRY RUN - No data will be stored")
        session = None
    else:
        print(f"🚀 Initializing Lilith session...")
        config = SessionConfig(
            data_path=data_path,
            use_grammar=True,
            enable_concepts=True,
            enable_reasoning=True
        )
        session = LilithSession(config)
    
    # Process each document
    total_concepts = 0
    total_qa_pairs = 0
    
    for doc_source in documents:
        print(f"\n📄 Processing: {doc_source}")
        
        try:
            content, source_name = load_document(doc_source)
            print(f"   Loaded {len(content):,} characters")
            
            chunks = chunk_text(content, chunk_size=args.chunk_size)
            print(f"   Split into {len(chunks)} chunks")
            
            if args.dry_run:
                # Just show what would be extracted
                extractor = SemanticExtractor()
                for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                    print(f"\n   --- Chunk {i+1} Preview ---")
                    concepts = extractor.extract_concepts("query", chunk)
                    for c in concepts[:2]:  # Show first 2 concepts per chunk
                        print(f"   Term: {c.term}")
                        if c.type_relations:
                            print(f"   Types: {c.type_relations[:2]}")
                if len(chunks) > 3:
                    print(f"\n   ... and {len(chunks) - 3} more chunks")
            else:
                concepts, qa_pairs = train_from_chunks(
                    session, chunks, source_name, 
                    extract_qa=not args.no_qa
                )
                total_concepts += concepts
                total_qa_pairs += qa_pairs
                print(f"   ✓ Stored {concepts} concepts, {qa_pairs} Q&A pairs")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 60)
    if args.dry_run:
        print("🔍 Dry run complete. No data was stored.")
    else:
        print(f"✅ Training complete!")
        print(f"   Total concepts stored: {total_concepts}")
        print(f"   Total Q&A pairs stored: {total_qa_pairs}")
        print(f"   Data location: {data_path}")


if __name__ == "__main__":
    main()
