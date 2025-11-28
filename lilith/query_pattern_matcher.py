"""
Query Pattern Matcher - Parse incoming queries for structure and intent

Extracts semantic structure from user queries using pattern templates.
Enables slot-based retrieval and intent-guided response composition.

Example:
    Query: "what is rust"
    Match: "[QUERY_WORD] is [SUBJECT]" 
    Result: {intent: "definition", slots: {"SUBJECT": "rust"}}

Phase 2: Pattern-Based Query Understanding
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import re


@dataclass
class QueryMatch:
    """Result of matching a query to a pattern"""
    intent: str                      # "definition", "how_query", "explanation", etc.
    confidence: float                # How well the pattern matched
    slots: Dict[str, str]            # Extracted slot values
    pattern_template: str            # The matched pattern template
    original_query: str              # Original query text


class QueryPatternMatcher:
    """
    Matches user queries against patterns to extract intent and structure.
    
    Unlike PatternExtractor (which learns patterns from Wikipedia),
    this uses query-specific patterns to understand user questions.
    """
    
    def __init__(self):
        """Initialize with query-specific pattern templates."""
        self.query_patterns = self._bootstrap_query_patterns()
    
    def _bootstrap_query_patterns(self) -> List[Dict]:
        """
        Bootstrap with common query patterns.
        
        These are question patterns, not statement patterns.
        Focused on extracting what the user wants to know.
        """
        return [
            # Identity queries (about the AI itself) - MUST come before general "what is" pattern
            {
                "template": "what is your name",
                "regex": r"^(?:what\s+is|what's|whats)\s+your\s+name(?:\?|$)",
                "intent": "identity",
                "slots": [],
                "examples": [
                    "what is your name",
                    "what's your name",
                    "whats your name"
                ]
            },
            {
                "template": "who are you",
                "regex": r"^who\s+are\s+you(?:\?|$)",
                "intent": "identity",
                "slots": [],
                "examples": [
                    "who are you",
                    "who are you?"
                ]
            },
            
            # Definition queries
            {
                "template": "what is [SUBJECT]",
                "regex": r"^(?:what\s+is|what's|whats)\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\?|$)",
                "intent": "definition",
                "slots": ["SUBJECT"],
                "examples": [
                    "what is rust",
                    "what's machine learning",
                    "what is a blockchain"
                ]
            },
            {
                "template": "what are [SUBJECT]",
                "regex": r"^(?:what\s+are|what're)\s+(.+?)(?:\?|$)",
                "intent": "definition",
                "slots": ["SUBJECT"],
                "examples": [
                    "what are neural networks",
                    "what're microservices"
                ]
            },
            {
                "template": "define [SUBJECT]",
                "regex": r"^define\s+(.+?)(?:\?|$)",
                "intent": "definition",
                "slots": ["SUBJECT"],
                "examples": [
                    "define recursion",
                    "define machine learning"
                ]
            },
            # Knowledge-check / learned-knowledge queries
            {
                "template": "do you know what [SUBJECT] is",
                "regex": r"^do\s+you\s+know\s+what\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+is(?:\?|$)",
                "intent": "learned_knowledge",
                "slots": ["SUBJECT"],
                "examples": [
                    "do you know what a banana is",
                    "do you know what an apple is",
                    "do you know what photosynthesis is"
                ]
            },
            
            # Mechanism/How queries
            {
                "template": "how does [SUBJECT] work",
                "regex": r"^how\s+does\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+work(?:\?|$)",
                "intent": "how_query",
                "slots": ["SUBJECT"],
                "examples": [
                    "how does rust work",
                    "how does a blockchain work"
                ]
            },
            {
                "template": "how do [SUBJECT] work",
                "regex": r"^how\s+do\s+(.+?)\s+work(?:\?|$)",
                "intent": "how_query",
                "slots": ["SUBJECT"],
                "examples": [
                    "how do neural networks work"
                ]
            },
            {
                "template": "how to [ACTION]",
                "regex": r"^how\s+to\s+(.+?)(?:\?|$)",
                "intent": "how_to",
                "slots": ["ACTION"],
                "examples": [
                    "how to learn python",
                    "how to use docker"
                ]
            },
            {
                "template": "explain [SUBJECT]",
                "regex": r"^explain\s+(.+?)(?:\?|$)",
                "intent": "explanation",
                "slots": ["SUBJECT"],
                "examples": [
                    "explain blockchain",
                    "explain how rust works"
                ]
            },
            
            # Comparison queries
            {
                "template": "difference between [A] and [B]",
                "regex": r"(?:what(?:'s|\s+is)\s+the\s+)?difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
                "intent": "comparison",
                "slots": ["A", "B"],
                "examples": [
                    "difference between rust and c++",
                    "what's the difference between ML and AI"
                ]
            },
            {
                "template": "[A] vs [B]",
                "regex": r"^(.+?)\s+(?:vs|versus)\s+(.+?)(?:\?|$)",
                "intent": "comparison",
                "slots": ["A", "B"],
                "examples": [
                    "rust vs c++",
                    "docker vs kubernetes"
                ]
            },
            
            # Feature/Property queries
            {
                "template": "what does [SUBJECT] do",
                "regex": r"^what\s+does\s+(.+?)\s+do(?:\?|$)",
                "intent": "capability",
                "slots": ["SUBJECT"],
                "examples": [
                    "what does rust do",
                    "what does a compiler do"
                ]
            },
            {
                "template": "why [SUBJECT]",
                "regex": r"^why\s+(?:is\s+|does\s+)?(.+?)(?:\?|$)",
                "intent": "reason",
                "slots": ["SUBJECT"],
                "examples": [
                    "why rust",
                    "why is python popular"
                ]
            },
            
            # Example queries
            {
                "template": "example of [SUBJECT]",
                "regex": r"(?:give\s+me\s+an?\s+|show\s+me\s+an?\s+|what(?:'s|\s+is)\s+an?\s+)?example\s+of\s+(.+?)(?:\?|$)",
                "intent": "example",
                "slots": ["SUBJECT"],
                "examples": [
                    "example of recursion",
                    "give me an example of a design pattern"
                ]
            },
            
            # List queries
            {
                "template": "list [SUBJECT]",
                "regex": r"^(?:list|what\s+are)\s+(?:some\s+|the\s+)?(.+?)(?:\?|$)",
                "intent": "list",
                "slots": ["SUBJECT"],
                "examples": [
                    "list programming languages",
                    "what are design patterns"
                ]
            },
            
            # When/Where/Who queries
            {
                "template": "when was [SUBJECT] created",
                "regex": r"^when\s+(?:was|were)\s+(.+?)\s+(?:created|developed|invented|built|made)(?:\?|$)",
                "intent": "history",
                "slots": ["SUBJECT"],
                "examples": [
                    "when was rust created",
                    "when was python invented"
                ]
            },
            {
                "template": "who created [SUBJECT]",
                "regex": r"^who\s+(?:created|developed|invented|built|made)\s+(.+?)(?:\?|$)",
                "intent": "creator",
                "slots": ["SUBJECT"],
                "examples": [
                    "who created rust",
                    "who invented python"
                ]
            },
            
            # Meta queries (about the system itself)
            {
                "template": "what can you do",
                "regex": r"(?:what\s+)?(?:can|could|do)\s+you\s+do(?:\?|$)",  # Removed ^ anchor
                "intent": "capability",  # Match base pattern intent
                "slots": [],
                "examples": [
                    "what can you do",
                    "what do you do",
                    "what could you do",
                    "tell me what you can do"
                ]
            },
            {
                "template": "tell me about yourself",
                "regex": r"(?:tell\s+me\s+)?(?:about\s+)?(?:yourself|you|your\s+capabilities)(?:\?|$)",  # Removed ^ anchor
                "intent": "capability",  # Match base pattern intent
                "slots": [],
                "examples": [
                    "tell me about yourself",
                    "about you",
                    "your capabilities"
                ]
            },
            {
                "template": "what are you",
                "regex": r"(?:what\s+)?(?:are|is)\s+you(?:\?|$)",  # Removed ^ anchor
                "intent": "identity",  # Match base pattern intent
                "slots": [],
                "examples": [
                    "what are you",
                    "what is you"
                ]
            },
            {
                "template": "help",
                "regex": r"(?:^|\s)(?:help|how\s+do\s+i\s+use\s+(?:this|you))(?:\?|$)",  # Match at start or after whitespace
                "intent": "capability",  # Match base pattern intent
                "slots": [],
                "examples": [
                    "help",
                    "how do i use this",
                    "how do i use you"
                ]
            }
        ]
    
    def match_query(self, query: str) -> Optional[QueryMatch]:
        """
        Match query against patterns to extract intent and slots.
        
        Args:
            query: User query text
            
        Returns:
            QueryMatch if a pattern matched, None otherwise
        """
        query_clean = query.strip().lower()
        
        # Try each pattern in order
        for pattern in self.query_patterns:
            regex = re.compile(pattern["regex"], re.IGNORECASE)
            match = regex.match(query_clean)
            
            if match:
                # Extract slot values
                slots = {}
                for i, slot_name in enumerate(pattern["slots"]):
                    # Group indices are 1-based
                    slot_value = match.group(i + 1).strip()
                    slots[slot_name] = slot_value
                
                # Calculate confidence based on match quality
                confidence = self._calculate_confidence(match, query_clean)
                
                return QueryMatch(
                    intent=pattern["intent"],
                    confidence=confidence,
                    slots=slots,
                    pattern_template=pattern["template"],
                    original_query=query
                )
        
        # No pattern matched
        return None
    
    def _calculate_confidence(self, match: re.Match, query: str) -> float:
        """
        Calculate confidence score for a pattern match.
        
        Higher confidence for:
        - Longer matches (more specific)
        - Complete matches (entire query captured)
        - Clean extractions (no extra words)
        """
        matched_span = match.span()
        match_length = matched_span[1] - matched_span[0]
        query_length = len(query)
        
        # Percentage of query matched
        coverage = match_length / query_length if query_length > 0 else 0.0
        
        # Boost for complete matches
        if coverage > 0.95:
            confidence = 0.95
        elif coverage > 0.80:
            confidence = 0.85
        elif coverage > 0.60:
            confidence = 0.75
        else:
            confidence = 0.65
        
        return confidence
    
    def get_intent_description(self, intent: str) -> str:
        """Get human-readable description of intent."""
        descriptions = {
            "definition": "Asking for a definition or explanation of what something is",
            "how_query": "Asking how something works or operates",
            "how_to": "Asking how to do or use something",
            "explanation": "Requesting detailed explanation",
            "comparison": "Comparing two things",
            "capability": "Asking what something can do",
            "reason": "Asking why something is the case",
            "example": "Requesting examples",
            "list": "Requesting a list of items",
            "history": "Asking about when something was created",
            "creator": "Asking who created something"
        }
        return descriptions.get(intent, "General query")
    
    def extract_main_concept(self, query_match: QueryMatch) -> Optional[str]:
        """
        Extract the main concept from query match.
        
        Useful for focusing retrieval on the key topic.
        """
        # Primary slot names by intent
        primary_slots = {
            "definition": "SUBJECT",
            "how_query": "SUBJECT",
            "how_to": "ACTION",
            "explanation": "SUBJECT",
            "capability": "SUBJECT",
            "reason": "SUBJECT",
            "example": "SUBJECT",
            "list": "SUBJECT",
            "history": "SUBJECT",
            "creator": "SUBJECT",
            "comparison": "A"  # First item in comparison
        }
        
        primary_slot = primary_slots.get(query_match.intent)
        if primary_slot and primary_slot in query_match.slots:
            return query_match.slots[primary_slot]
        
        # Fallback: return first slot value
        if query_match.slots:
            return next(iter(query_match.slots.values()))
        
        return None


def demo():
    """Demo query pattern matcher."""
    print("=" * 80)
    print("Query Pattern Matcher Demo")
    print("=" * 80)
    print()
    
    matcher = QueryPatternMatcher()
    
    # Test queries
    test_queries = [
        "what is rust",
        "what's machine learning",
        "how does blockchain work",
        "how to learn python",
        "explain neural networks",
        "difference between rust and c++",
        "docker vs kubernetes",
        "why is python popular",
        "example of recursion",
        "when was rust created",
        "who created python",
        "what does a compiler do",
        "just a random statement",  # Should not match
    ]
    
    print("Testing query pattern matching:")
    print("-" * 80)
    print()
    
    for query in test_queries:
        match = matcher.match_query(query)
        
        if match:
            main_concept = matcher.extract_main_concept(match)
            print(f"Query: '{query}'")
            print(f"  ✓ Matched: {match.pattern_template}")
            print(f"  Intent: {match.intent} (confidence: {match.confidence:.2f})")
            print(f"  Slots: {match.slots}")
            print(f"  Main concept: {main_concept}")
            print()
        else:
            print(f"Query: '{query}'")
            print(f"  ✗ No pattern match")
            print()


if __name__ == "__main__":
    demo()
