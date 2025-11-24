#!/usr/bin/env python3
"""
Conversational Training Pipeline - Learn from Dialogue Datasets

Ingests conversational data and learns:
1. Response patterns (semantic associations)
2. Grammatical structures (syntax patterns)
3. Successful turn-taking behaviors
4. Topic tracking and transitions

Supports multiple formats:
- JSON: [{"user": "...", "bot": "..."}, ...]
- CSV: user,bot columns
- Plain text: Alternating lines (user/bot/user/bot)
"""

import sys
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

from conversation_loop import ConversationLoop


@dataclass
class DialogueTurn:
    """A single turn in a dialogue."""
    user_text: str
    bot_text: str
    context: Dict = None  # Optional metadata
    

class ConversationalDataset:
    """Handles loading and parsing dialogue datasets."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.turns: List[DialogueTurn] = []
        
    def load(self) -> List[DialogueTurn]:
        """Load dataset based on file extension."""
        if self.file_path.suffix == '.json':
            return self._load_json()
        elif self.file_path.suffix == '.csv':
            return self._load_csv()
        elif self.file_path.suffix == '.txt':
            return self._load_text()
        else:
            raise ValueError(f"Unsupported format: {self.file_path.suffix}")
    
    def _load_json(self) -> List[DialogueTurn]:
        """
        Load JSON format:
        [
          {"user": "Hello!", "bot": "Hi there!"},
          {"user": "How are you?", "bot": "I'm doing well!"}
        ]
        """
        with open(self.file_path) as f:
            data = json.load(f)
        
        turns = []
        for item in data:
            if 'user' in item and 'bot' in item:
                turns.append(DialogueTurn(
                    user_text=item['user'],
                    bot_text=item['bot'],
                    context=item.get('context')
                ))
        
        print(f"✓ Loaded {len(turns)} turns from JSON")
        return turns
    
    def _load_csv(self) -> List[DialogueTurn]:
        """
        Load CSV format with columns: user, bot
        """
        turns = []
        with open(self.file_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'user' in row and 'bot' in row:
                    turns.append(DialogueTurn(
                        user_text=row['user'],
                        bot_text=row['bot']
                    ))
        
        print(f"✓ Loaded {len(turns)} turns from CSV")
        return turns
    
    def _load_text(self) -> List[DialogueTurn]:
        """
        Load plain text format (alternating lines):
        User: Hello!
        Bot: Hi there!
        User: How are you?
        Bot: I'm doing well!
        """
        with open(self.file_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        
        turns = []
        i = 0
        while i < len(lines) - 1:
            user_line = lines[i]
            bot_line = lines[i + 1]
            
            # Remove "User:" and "Bot:" prefixes if present
            user_text = user_line.replace('User:', '').replace('user:', '').strip()
            bot_text = bot_line.replace('Bot:', '').replace('bot:', '').strip()
            
            turns.append(DialogueTurn(
                user_text=user_text,
                bot_text=bot_text
            ))
            i += 2
        
        print(f"✓ Loaded {len(turns)} turns from text")
        return turns


class ConversationalTrainer:
    """Trains the neuro-symbolic system on conversational data."""
    
    def __init__(
        self,
        loop: ConversationLoop,
        use_grammar: bool = True,
        verbose: bool = True
    ):
        self.loop = loop
        self.use_grammar = use_grammar
        self.verbose = verbose
        
        # Training statistics
        self.patterns_learned = 0
        self.syntax_patterns_learned = 0
        self.turns_processed = 0
        
    def train_on_dataset(
        self,
        dataset: ConversationalDataset,
        learn_from_bot: bool = True,
        learn_from_user: bool = False
    ):
        """
        Train on a dialogue dataset.
        
        Args:
            dataset: Dataset to train on
            learn_from_bot: Learn response patterns from bot utterances
            learn_from_user: Learn from user patterns too (for diverse language)
        """
        turns = dataset.load()
        
        print("\n" + "="*70)
        print(f"🎓 TRAINING ON CONVERSATIONAL DATA")
        print("="*70)
        print(f"  Dataset: {dataset.file_path.name}")
        print(f"  Turns: {len(turns)}")
        print(f"  Learn from bot: {learn_from_bot}")
        print(f"  Learn from user: {learn_from_user}")
        print(f"  Grammar learning: {self.use_grammar}")
        print()
        
        initial_patterns = len(self.loop.fragment_store.patterns)
        initial_syntax = (len(self.loop.composer.syntax_stage.patterns) 
                         if hasattr(self.loop.composer, 'syntax_stage') 
                         and self.loop.composer.syntax_stage else 0)
        
        for i, turn in enumerate(turns, 1):
            self._process_turn(turn, learn_from_bot, learn_from_user)
            
            if self.verbose and i % 10 == 0:
                print(f"  Processed {i}/{len(turns)} turns...")
        
        # Final statistics
        final_patterns = len(self.loop.fragment_store.patterns)
        final_syntax = (len(self.loop.composer.syntax_stage.patterns) 
                       if hasattr(self.loop.composer, 'syntax_stage') 
                       and self.loop.composer.syntax_stage else 0)
        
        self.patterns_learned = final_patterns - initial_patterns
        self.syntax_patterns_learned = final_syntax - initial_syntax
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE")
        print("="*70)
        print(f"  Turns processed: {len(turns)}")
        print(f"  Response patterns learned: {self.patterns_learned}")
        print(f"  Syntax patterns learned: {self.syntax_patterns_learned}")
        print(f"  Total response patterns: {final_patterns}")
        print(f"  Total syntax patterns: {final_syntax}")
        print("="*70)
        print()
    
    def _process_turn(
        self,
        turn: DialogueTurn,
        learn_from_bot: bool,
        learn_from_user: bool
    ):
        """Process a single dialogue turn."""
        # Process user input through pipeline (updates working memory)
        self.loop.process_user_input(turn.user_text)
        
        # Learn from user patterns if enabled
        if learn_from_user:
            # Get recent context for trigger
            recent = self.loop.history.get_recent_turns(n=3)
            context = " ".join([t.user_input for t in recent[-3:]])
            
            self._learn_pattern(
                trigger=context,
                response=turn.user_text,
                intent="user_style"
            )
        
        # Learn from bot response
        if learn_from_bot:
            # Get current context from recent turns
            recent = self.loop.history.get_recent_turns(n=3)
            if recent:
                context = " ".join([t.user_input for t in recent])
            else:
                context = turn.user_text
            
            # Learn this as a successful response pattern
            self._learn_pattern(
                trigger=context,
                response=turn.bot_text,
                intent=self._classify_intent(turn.bot_text)
            )
            
            # Learn syntax pattern if grammar enabled
            if self.use_grammar and hasattr(self.loop.composer, 'syntax_stage'):
                syntax_stage = self.loop.composer.syntax_stage
                if syntax_stage:
                    tokens = turn.bot_text.split()
                    # Learn with positive feedback (it's from training data)
                    # Note: pos_tags will be auto-extracted by learn_pattern
                    syntax_stage.learn_pattern(
                        tokens=tokens,
                        pos_tags=syntax_stage._extract_pos_tags(tokens),  # Extract POS first
                        success_feedback=0.3  # Moderate confidence
                    )
        
        # Update conversation history
        self.loop.history.add_turn(
            user_input=turn.user_text,
            bot_response=turn.bot_text
        )
        
        self.turns_processed += 1
    
    def _learn_pattern(self, trigger: str, response: str, intent: str):
        """Learn a response pattern."""
        # Add to fragment store (it handles embeddings internally)
        fragment_id = self.loop.fragment_store.add_pattern(
            trigger_context=trigger,
            response_text=response,
            success_score=0.6,  # Training data gets good initial score
            intent=intent
        )
    
    def _classify_intent(self, text: str) -> str:
        """Simple intent classification."""
        text_lower = text.lower()
        
        if '?' in text:
            return "question"
        elif any(word in text_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return "greeting"
        elif any(word in text_lower for word in ['bye', 'goodbye', 'see you']):
            return "farewell"
        elif any(word in text_lower for word in ['yes', 'correct', 'exactly', 'right']):
            return "agreement"
        elif any(word in text_lower for word in ['interesting', 'fascinating', 'cool']):
            return "interest"
        else:
            return "statement"
    
    def save_learned_patterns(self, output_path: str | None = None):
        """Save learned patterns to disk."""
        if output_path:
            self.loop.fragment_store.storage_path = Path(output_path)
        
        # Patterns are auto-saved by add_pattern(), but force save here
        self.loop.fragment_store._save_patterns()
        print(f"💾 Saved {len(self.loop.fragment_store.patterns)} patterns to {self.loop.fragment_store.storage_path}")
        
        if hasattr(self.loop.composer, 'syntax_stage') and self.loop.composer.syntax_stage:
            syntax_stage = self.loop.composer.syntax_stage
            # Save syntax patterns
            syntax_stage._save_patterns()
            print(f"💾 Saved {len(syntax_stage.patterns)} syntax patterns to {syntax_stage.storage_path}")


def main():
    """Example training run."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train conversation system on dialogue data")
    parser.add_argument('dataset', help='Path to dataset (JSON/CSV/TXT)')
    parser.add_argument('--no-grammar', action='store_true', help='Disable grammar learning')
    parser.add_argument('--learn-user', action='store_true', help='Learn from user patterns too')
    parser.add_argument('--output', help='Save patterns to this path', default='conversation_patterns_trained.json')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # Initialize system
    print("\n🧠 Initializing Pure Neuro-Symbolic System...")
    loop = ConversationLoop(
        history_window=10,
        composition_mode="weighted_blend",
        use_grammar=not args.no_grammar
    )
    
    # Load dataset
    dataset = ConversationalDataset(args.dataset)
    
    # Train
    trainer = ConversationalTrainer(
        loop=loop,
        use_grammar=not args.no_grammar,
        verbose=not args.quiet
    )
    
    trainer.train_on_dataset(
        dataset=dataset,
        learn_from_bot=True,
        learn_from_user=args.learn_user
    )
    
    # Save learned patterns
    trainer.save_learned_patterns(args.output)
    
    print("\n🎉 Training complete! System ready for conversation.")
    print(f"\nTo test learned patterns, run:")
    print(f"  python minimal_conversation_demo.py")


if __name__ == "__main__":
    main()
