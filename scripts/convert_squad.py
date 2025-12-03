#!/usr/bin/env python3
"""
Convert raw SQuAD JSON to Q&A format for bootstrap_qa.py.
No dependencies on pyarrow or HuggingFace datasets.
"""

import json
import argparse
import random
from pathlib import Path


def convert_squad(input_path: str, output_path: str, limit: int = None, shuffle: bool = True):
    """Convert SQuAD JSON to Q&A training format."""
    
    print(f"Loading SQuAD data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    qa_pairs = []
    
    # SQuAD format: data -> articles -> paragraphs -> qas
    for article in data.get('data', []):
        for paragraph in article.get('paragraphs', []):
            context = paragraph.get('context', '')
            
            for qa in paragraph.get('qas', []):
                question = qa.get('question', '').strip()
                
                # Get the answer (SQuAD 2.0 may have unanswerable questions)
                answers = qa.get('answers', [])
                
                if answers:
                    # Use first answer
                    answer = answers[0].get('text', '').strip()
                    
                    if question and answer:
                        qa_pairs.append({
                            'question': question,
                            'answer': answer,
                            'context': context[:200]  # Store snippet for reference
                        })
    
    print(f"Found {len(qa_pairs)} Q&A pairs")
    
    if shuffle:
        random.shuffle(qa_pairs)
    
    if limit:
        qa_pairs = qa_pairs[:limit]
        print(f"Limited to {len(qa_pairs)} pairs")
    
    # Write in bootstrap_qa.py format
    print(f"Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# SQuAD Training Data\n")
        f.write("# Format: Q: question / A: answer\n")
        f.write(f"# Generated from {Path(input_path).name}\n")
        f.write(f"# Total pairs: {len(qa_pairs)}\n\n")
        
        for pair in qa_pairs:
            # Clean up the question and answer
            q = pair['question'].replace('\n', ' ').strip()
            a = pair['answer'].replace('\n', ' ').strip()
            
            # Skip very short or very long answers (they tend to be noise)
            if len(a) < 2 or len(a) > 300:
                continue
                
            f.write(f"Q: {q}\n")
            f.write(f"A: {a}\n\n")
    
    print(f"Done! Created {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert SQuAD JSON to training format')
    parser.add_argument('input', help='Path to SQuAD JSON file')
    parser.add_argument('-o', '--output', default='data/generated/squad_training.txt',
                        help='Output file path')
    parser.add_argument('-l', '--limit', type=int, default=1000,
                        help='Maximum number of Q&A pairs (default: 1000)')
    parser.add_argument('--no-shuffle', action='store_true',
                        help='Do not shuffle the data')
    
    args = parser.parse_args()
    
    convert_squad(
        args.input,
        args.output,
        limit=args.limit,
        shuffle=not args.no_shuffle
    )


if __name__ == '__main__':
    main()
