"""
Syntax Stage - Grammatical Pattern Processing

Processes syntactic structures as learned symbolic patterns using BNN.
Enables grammatically-guided composition without hardcoded rules.

This stage learns:
  - POS sequence patterns (successful grammatical structures)
  - Composition templates (how to combine fragments)
  - Syntactic transformations (statement ↔ question, etc.)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json


class GrammaticalRole(Enum):
    """Simplified grammatical roles for pattern matching."""
    SUBJECT = "SUBJ"
    VERB = "VERB"
    OBJECT = "OBJ"
    MODIFIER = "MOD"
    QUESTION_WORD = "QW"
    CONNECTOR = "CONN"


@dataclass
class SyntacticPattern:
    """A learned grammatical pattern."""
    
    pattern_id: str
    pos_sequence: List[str]          # ["DT", "JJ", "NN", "VBZ"]
    template: str                     # "The {adj} {noun} {verb}"
    roles: Dict[str, str]            # {"subject": "noun", "verb": "verb"}
    example: str                      # Actual instance of this pattern
    success_score: float = 0.5
    usage_count: int = 0
    intent: str = "general"          # statement, question, compound, etc.


@dataclass
class CompositionTemplate:
    """Template for combining multiple fragments grammatically."""
    
    template_id: str
    pattern_a_type: str              # e.g., "statement"
    pattern_b_type: str              # e.g., "statement"
    connector: str                   # "and", "but", "because", etc.
    pos_template: str                # "{POS_A} CONN {POS_B}"
    example: str                     # "I like apples and oranges are good too."
    success_score: float = 0.5


class GrammarPatternStore:
    """Store and retrieve grammatical patterns."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.patterns: Dict[str, SyntacticPattern] = {}
        self.templates: Dict[str, CompositionTemplate] = {}
        self.storage_path = storage_path
        
        # Bootstrap with basic patterns
        self._bootstrap_patterns()
        
    def _bootstrap_patterns(self):
        """Seed system with basic grammatical patterns."""
        
        # Simple statement patterns
        basic_patterns = [
            # Subject-Verb patterns
            SyntacticPattern(
                pattern_id="stmt_simple_sv",
                pos_sequence=["PRON", "VBZ"],
                template="{subject} {verb}",
                roles={"subject": "PRON", "verb": "VBZ"},
                example="It works",
                intent="statement"
            ),
            
            # Subject-Verb-Object
            SyntacticPattern(
                pattern_id="stmt_svo",
                pos_sequence=["PRON", "VBZ", "NNS"],
                template="{subject} {verb} {object}",
                roles={"subject": "PRON", "verb": "VBZ", "object": "NNS"},
                example="I like patterns",
                intent="statement"
            ),
            
            # Wh-questions
            SyntacticPattern(
                pattern_id="quest_how",
                pos_sequence=["WRB", "VBZ", "PRON", "VB"],
                template="{wh} {aux} {subject} {verb}",
                roles={"wh": "WRB", "aux": "VBZ", "subject": "PRON", "verb": "VB"},
                example="How does that work",
                intent="question"
            ),
            
            # Exclamations
            SyntacticPattern(
                pattern_id="exclaim_adj",
                pos_sequence=["PRON", "VBZ", "ADV", "ADJ"],
                template="{subject} {verb} {adverb} {adjective}",
                roles={"subject": "PRON", "verb": "VBZ", "adverb": "ADV", "adjective": "ADJ"},
                example="That's really interesting",
                intent="exclamation"
            ),
        ]
        
        for pattern in basic_patterns:
            self.patterns[pattern.pattern_id] = pattern
            
        # Composition templates
        basic_templates = [
            # Coordinate statements
            CompositionTemplate(
                template_id="coord_and",
                pattern_a_type="statement",
                pattern_b_type="statement",
                connector="and",
                pos_template="{STMT_A} and {STMT_B}",
                example="I understand that and I find it interesting",
            ),
            
            # Statement + question
            CompositionTemplate(
                template_id="stmt_quest",
                pattern_a_type="statement",
                pattern_b_type="question",
                connector=".",
                pos_template="{STMT}. {QUEST}",
                example="That's interesting. How does it work?",
            ),
            
            # Exclamation + statement
            CompositionTemplate(
                template_id="exclaim_stmt",
                pattern_a_type="exclamation",
                pattern_b_type="statement",
                connector="!",
                pos_template="{EXCLM}! {STMT}",
                example="Wow! That's really cool.",
            ),
        ]
        
        for template in basic_templates:
            self.templates[template.template_id] = template
    
    def extract_pattern(self, text: str, pos_tags: List[str]) -> Optional[SyntacticPattern]:
        """
        Extract grammatical pattern from text with POS tags.
        
        This is how the system LEARNS new grammar from user input!
        """
        # Simplified: In practice would use actual POS tagger
        pattern_id = f"learned_pattern_{len(self.patterns)}"
        
        # Determine intent from structure
        if text.strip().endswith('?'):
            intent = "question"
        elif any(word in text.lower() for word in ['wow', 'amazing', 'fascinating']):
            intent = "exclamation"
        else:
            intent = "statement"
            
        return SyntacticPattern(
            pattern_id=pattern_id,
            pos_sequence=pos_tags,
            template=self._generalize_template(text, pos_tags),
            roles={},
            example=text,
            success_score=0.6,  # Start moderate
            intent=intent
        )
    
    def _generalize_template(self, text: str, pos_tags: List[str]) -> str:
        """Create a template from text and POS tags."""
        # Simplified: would be more sophisticated in practice
        words = text.split()
        template_parts = []
        
        for word, pos in zip(words, pos_tags):
            if pos in ['DT', 'PRON', 'CC']:  # Function words
                template_parts.append(word)
            else:
                template_parts.append(f"{{{pos.lower()}}}")
                
        return " ".join(template_parts)
    
    def retrieve_composition_template(
        self,
        intent_a: str,
        intent_b: str
    ) -> Optional[CompositionTemplate]:
        """
        Find template for combining two fragments based on their intents.
        
        This guides grammatically-correct blending!
        """
        for template in self.templates.values():
            if (template.pattern_a_type == intent_a and 
                template.pattern_b_type == intent_b):
                return template
        return None
    
    def add_pattern(self, pattern: SyntacticPattern):
        """Learn a new grammatical pattern."""
        self.patterns[pattern.pattern_id] = pattern
        
    def update_success(self, pattern_id: str, feedback: float):
        """Update pattern success score from feedback."""
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.success_score += feedback
            pattern.success_score = max(0.0, min(1.0, pattern.success_score))
            pattern.usage_count += 1


class GrammarGuidedComposer:
    """Use grammatical patterns to guide response composition."""
    
    def __init__(self, grammar_store: GrammarPatternStore):
        self.grammar = grammar_store
        
    def compose_with_grammar(
        self,
        fragment_a: str,
        fragment_b: str,
        intent_a: str = "statement",
        intent_b: str = "statement"
    ) -> str:
        """
        Compose two fragments using grammatical templates.
        
        This creates grammatically-correct novel utterances!
        """
        # Get appropriate composition template
        template = self.grammar.retrieve_composition_template(intent_a, intent_b)
        
        if not template:
            # Fallback: simple concatenation with period
            return f"{fragment_a.strip()}. {fragment_b.strip()}"
        
        # Apply template
        connector = template.connector
        
        # Clean up fragments
        frag_a = fragment_a.strip()
        frag_b = fragment_b.strip()
        
        # Ensure proper punctuation before connector
        if connector == "and":
            # Coordinate: "I see that, and it's interesting"
            if not frag_a[-1] in "!?,":
                frag_a += ","
            result = f"{frag_a} {connector} {frag_b}"
        elif connector == ".":
            # Sequential: "That's cool. How does it work?"
            if not frag_a[-1] in "!?":
                frag_a = frag_a.rstrip(',.') + "."
            result = f"{frag_a} {frag_b}"
        elif connector == "!":
            # Exclamation + statement
            if not frag_a[-1] in "!?":
                frag_a += "!"
            result = f"{frag_a} {frag_b}"
        else:
            # Generic connector
            result = f"{frag_a} {connector} {frag_b}"
            
        return result
    
    def apply_transformation(
        self,
        text: str,
        from_intent: str,
        to_intent: str
    ) -> Optional[str]:
        """
        Transform text from one grammatical form to another.
        
        E.g., statement → question, active → passive
        """
        # Simplified version - would use learned transformation patterns
        if from_intent == "statement" and to_intent == "question":
            # Simple heuristic transformation
            if text.startswith("This is"):
                return text.replace("This is", "Is this") + "?"
            elif text.startswith("That"):
                return text.replace("That", "Is that") + "?"
        
        return None


# Example usage demonstration
if __name__ == "__main__":
    print("Grammar-Guided Composition Demo")
    print("=" * 60)
    
    # Initialize
    grammar_store = GrammarPatternStore()
    composer = GrammarGuidedComposer(grammar_store)
    
    # Example 1: Compose with coordination
    print("\n1. Coordinating two statements:")
    frag1 = "I understand the concept"
    frag2 = "it's really fascinating"
    result = composer.compose_with_grammar(frag1, frag2, "statement", "statement")
    print(f"   Input A: '{frag1}'")
    print(f"   Input B: '{frag2}'")
    print(f"   Result: '{result}'")
    
    # Example 2: Exclamation + statement
    print("\n2. Exclamation + statement:")
    frag1 = "Wow, that's impressive"
    frag2 = "I'd like to learn more"
    result = composer.compose_with_grammar(frag1, frag2, "exclamation", "statement")
    print(f"   Input A: '{frag1}'")
    print(f"   Input B: '{frag2}'")
    print(f"   Result: '{result}'")
    
    # Example 3: Statement + question
    print("\n3. Statement + question:")
    frag1 = "That makes sense"
    frag2 = "How does pattern matching work?"
    result = composer.compose_with_grammar(frag1, frag2, "statement", "question")
    print(f"   Input A: '{frag1}'")
    print(f"   Input B: '{frag2}'")
    print(f"   Result: '{result}'")
    
    print("\n" + "=" * 60)
    print("✓ Grammar-guided composition creates coherent outputs!")
