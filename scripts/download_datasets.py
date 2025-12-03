#!/usr/bin/env python3
"""
Download and convert open datasets for Lilith training.

Supported datasets:
- squad: Stanford Question Answering Dataset
- eli5: Explain Like I'm 5 (long-form answers)
- quora: Quora Question Pairs (paraphrases)
- trivia: TriviaQA (factual Q&A)

Usage:
    python download_datasets.py squad --output data/squad_training.txt
    python download_datasets.py eli5 --output data/eli5_training.txt --limit 1000
    python download_datasets.py all --output-dir data/datasets/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

# Try to import datasets library
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


def check_datasets_library():
    """Check if datasets library is available."""
    if not DATASETS_AVAILABLE:
        print("❌ The 'datasets' library is required.")
        print("   Install with: pip install datasets")
        sys.exit(1)


def download_squad(limit: int = None) -> List[Tuple[str, str]]:
    """
    Download SQuAD dataset and extract Q&A pairs.
    
    Returns list of (question, answer) tuples.
    """
    print("📥 Downloading SQuAD dataset...")
    dataset = load_dataset("squad", split="train")
    
    qa_pairs = []
    seen_questions = set()
    
    for item in dataset:
        question = item['question'].strip()
        
        # Skip duplicates
        if question.lower() in seen_questions:
            continue
        seen_questions.add(question.lower())
        
        # Get answer - SQuAD has answer spans, we use the text
        answers = item['answers']['text']
        if not answers:
            continue
        
        answer = answers[0].strip()
        
        # Skip very short answers (not good for Lilith)
        if len(answer) < 10:
            continue
        
        # Create a more complete answer by incorporating context
        context = item['context']
        
        # Try to find a sentence containing the answer
        sentences = context.split('.')
        answer_sentence = None
        for sent in sentences:
            if answer.lower() in sent.lower():
                answer_sentence = sent.strip() + '.'
                break
        
        if answer_sentence and len(answer_sentence) > len(answer):
            qa_pairs.append((question, answer_sentence))
        else:
            qa_pairs.append((question, answer))
        
        if limit and len(qa_pairs) >= limit:
            break
    
    print(f"   ✓ Extracted {len(qa_pairs)} Q&A pairs")
    return qa_pairs


def download_eli5(limit: int = None) -> List[Tuple[str, str]]:
    """
    Download ELI5 dataset and extract Q&A pairs.
    
    Returns list of (question, answer) tuples.
    """
    print("📥 Downloading ELI5 dataset...")
    dataset = load_dataset("eli5_category", split="train", trust_remote_code=True)
    
    qa_pairs = []
    
    for item in dataset:
        question = item['title'].strip()
        
        # ELI5 has multiple answers, use the top-scored one
        answers = item['answers']['text']
        if not answers:
            continue
        
        # Get first (usually best) answer
        answer = answers[0].strip()
        
        # ELI5 answers can be very long, truncate to reasonable length
        if len(answer) > 500:
            # Try to cut at sentence boundary
            sentences = answer.split('.')
            truncated = ""
            for sent in sentences:
                if len(truncated) + len(sent) < 450:
                    truncated += sent + "."
                else:
                    break
            answer = truncated.strip() if truncated else answer[:450] + "..."
        
        # Skip very short answers
        if len(answer) < 50:
            continue
        
        qa_pairs.append((question, answer))
        
        if limit and len(qa_pairs) >= limit:
            break
    
    print(f"   ✓ Extracted {len(qa_pairs)} Q&A pairs")
    return qa_pairs


def download_trivia(limit: int = None) -> List[Tuple[str, str]]:
    """
    Download TriviaQA dataset and extract Q&A pairs.
    
    Returns list of (question, answer) tuples.
    """
    print("📥 Downloading TriviaQA dataset...")
    dataset = load_dataset("trivia_qa", "rc", split="train")
    
    qa_pairs = []
    seen_questions = set()
    
    for item in dataset:
        question = item['question'].strip()
        
        # Skip duplicates
        if question.lower() in seen_questions:
            continue
        seen_questions.add(question.lower())
        
        # Get answer
        answer_data = item['answer']
        if not answer_data or not answer_data.get('value'):
            continue
        
        answer = answer_data['value'].strip()
        
        # TriviaQA has short answers, try to make them more complete
        # by forming a sentence
        if not question.endswith('?'):
            question += '?'
        
        # Create a simple answer sentence
        if len(answer) < 50:
            # Short factual answer - keep as is for now
            # Could enhance with "The answer is X" pattern
            pass
        
        qa_pairs.append((question, answer))
        
        if limit and len(qa_pairs) >= limit:
            break
    
    print(f"   ✓ Extracted {len(qa_pairs)} Q&A pairs")
    return qa_pairs


def download_quora_pairs(limit: int = None) -> List[Tuple[str, str]]:
    """
    Download Quora Question Pairs for paraphrase data.
    
    Returns list of (question1, question2) tuples that are paraphrases.
    """
    print("📥 Downloading Quora Question Pairs...")
    dataset = load_dataset("quora", split="train")
    
    paraphrase_pairs = []
    
    for item in dataset:
        # Only take duplicate pairs (is_duplicate == 1)
        if not item['is_duplicate']:
            continue
        
        q1 = item['questions']['text'][0].strip()
        q2 = item['questions']['text'][1].strip()
        
        # Skip very short questions
        if len(q1) < 15 or len(q2) < 15:
            continue
        
        paraphrase_pairs.append((q1, q2))
        
        if limit and len(paraphrase_pairs) >= limit:
            break
    
    print(f"   ✓ Extracted {len(paraphrase_pairs)} paraphrase pairs")
    return paraphrase_pairs


def download_commonsense(limit: int = None) -> List[Tuple[str, str]]:
    """
    Download CommonsenseQA for general knowledge.
    
    Returns list of (question, answer) tuples.
    """
    print("📥 Downloading CommonsenseQA dataset...")
    dataset = load_dataset("commonsense_qa", split="train")
    
    qa_pairs = []
    
    for item in dataset:
        question = item['question'].strip()
        
        # Get the correct answer
        answer_key = item['answerKey']
        choices = item['choices']
        
        # Find the correct choice
        answer = None
        for i, label in enumerate(choices['label']):
            if label == answer_key:
                answer = choices['text'][i]
                break
        
        if not answer:
            continue
        
        # Form a complete answer
        full_answer = f"The answer is: {answer}."
        
        qa_pairs.append((question, full_answer))
        
        if limit and len(qa_pairs) >= limit:
            break
    
    print(f"   ✓ Extracted {len(qa_pairs)} Q&A pairs")
    return qa_pairs


def save_qa_format(qa_pairs: List[Tuple[str, str]], output_path: Path):
    """Save Q&A pairs in Lilith's bootstrap format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Auto-generated training data\n")
        f.write(f"# Source: Open datasets\n")
        f.write(f"# Pairs: {len(qa_pairs)}\n\n")
        
        for question, answer in qa_pairs:
            # Clean up whitespace
            question = ' '.join(question.split())
            answer = ' '.join(answer.split())
            
            f.write(f"Q: {question}\n")
            f.write(f"A: {answer}\n\n")
    
    print(f"💾 Saved to {output_path}")


def save_json_format(qa_pairs: List[Tuple[str, str]], output_path: Path):
    """Save Q&A pairs in JSON format."""
    data = [{"user": q, "bot": a} for q, a in qa_pairs]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Saved to {output_path}")


def save_paraphrase_format(pairs: List[Tuple[str, str]], output_path: Path):
    """Save paraphrase pairs for training."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Paraphrase pairs for training\n")
        f.write(f"# Pairs: {len(pairs)}\n\n")
        
        for q1, q2 in pairs:
            q1 = ' '.join(q1.split())
            q2 = ' '.join(q2.split())
            
            f.write(f"Q1: {q1}\n")
            f.write(f"Q2: {q2}\n\n")
    
    print(f"💾 Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download open datasets for Lilith training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'dataset',
        choices=['squad', 'eli5', 'trivia', 'quora', 'commonsense', 'all'],
        help="Dataset to download"
    )
    parser.add_argument(
        '--output', '-o',
        help="Output file path (for single dataset)"
    )
    parser.add_argument(
        '--output-dir',
        default="data/datasets",
        help="Output directory (for 'all' option)"
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=1000,
        help="Maximum number of items to extract (default: 1000)"
    )
    parser.add_argument(
        '--format', '-f',
        choices=['txt', 'json'],
        default='txt',
        help="Output format (default: txt for bootstrap)"
    )
    
    args = parser.parse_args()
    
    check_datasets_library()
    
    print("🎓 Lilith Dataset Downloader")
    print("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_to_download = []
    
    if args.dataset == 'all':
        datasets_to_download = ['squad', 'eli5', 'trivia', 'commonsense', 'quora']
    else:
        datasets_to_download = [args.dataset]
    
    for dataset_name in datasets_to_download:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print('='*60)
        
        # Determine output path
        if args.output and len(datasets_to_download) == 1:
            output_path = Path(args.output)
        else:
            ext = 'json' if args.format == 'json' else 'txt'
            output_path = output_dir / f"{dataset_name}_training.{ext}"
        
        try:
            if dataset_name == 'squad':
                qa_pairs = download_squad(args.limit)
                if args.format == 'json':
                    save_json_format(qa_pairs, output_path)
                else:
                    save_qa_format(qa_pairs, output_path)
                    
            elif dataset_name == 'eli5':
                qa_pairs = download_eli5(args.limit)
                if args.format == 'json':
                    save_json_format(qa_pairs, output_path)
                else:
                    save_qa_format(qa_pairs, output_path)
                    
            elif dataset_name == 'trivia':
                qa_pairs = download_trivia(args.limit)
                if args.format == 'json':
                    save_json_format(qa_pairs, output_path)
                else:
                    save_qa_format(qa_pairs, output_path)
                    
            elif dataset_name == 'commonsense':
                qa_pairs = download_commonsense(args.limit)
                if args.format == 'json':
                    save_json_format(qa_pairs, output_path)
                else:
                    save_qa_format(qa_pairs, output_path)
                    
            elif dataset_name == 'quora':
                pairs = download_quora_pairs(args.limit)
                save_paraphrase_format(pairs, output_path)
                
        except Exception as e:
            print(f"   ❌ Error downloading {dataset_name}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("✅ Download complete!")
    print(f"   Output directory: {output_dir}")
    print("\nNext steps:")
    print("   1. Review the downloaded data")
    print("   2. Run: python scripts/bootstrap_qa.py --data-dir data")
    print("   Or use the JSON files with train_from_conversations.py")


if __name__ == "__main__":
    main()
