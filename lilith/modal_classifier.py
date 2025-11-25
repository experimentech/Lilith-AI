"""
Modal Classifier - Route queries to appropriate backends

Prevents cross-modal contamination by identifying query modality
and routing to specialized backends.
"""

from enum import Enum
from typing import Optional, Dict, Tuple
import re


class Modality(Enum):
    """Query modality types"""
    LINGUISTIC = "linguistic"  # Natural language (default)
    MATH = "math"              # Mathematical computation
    CODE = "code"              # Programming/code
    VISUAL = "visual"          # Image-based (future)
    AUDIO = "audio"            # Audio-based (future)


class ModalClassifier:
    """
    Classify input modality using heuristics.
    
    Prevents modal contamination:
    - Math queries → MathBackend (not linguistic database)
    - Code queries → CodeBackend (not linguistic database)
    - Linguistic queries → ResponseComposer (existing)
    """
    
    def __init__(self):
        """Initialize modal classifier"""
        pass
    
    def classify(
        self, 
        query: str, 
        context: Optional[Dict] = None
    ) -> Tuple[Modality, float]:
        """
        Classify query modality.
        
        Args:
            query: User input query
            context: Optional context (for multi-modal inputs like images)
            
        Returns:
            (modality, confidence)
        """
        query_lower = query.lower()
        
        # 1. Math modality (highest priority - most specific)
        is_math, math_conf = self._is_math(query_lower, query)
        if is_math:
            return Modality.MATH, math_conf
        
        # 2. Code modality
        is_code, code_conf = self._is_code(query_lower)
        if is_code:
            return Modality.CODE, code_conf
        
        # 3. Visual modality (requires image input)
        if context and 'image' in context:
            return Modality.VISUAL, 1.0
        
        # 4. Audio modality (requires audio input)
        if context and 'audio' in context:
            return Modality.AUDIO, 1.0
        
        # 5. Default: Linguistic
        return Modality.LINGUISTIC, 1.0
    
    def _is_math(self, query_lower: str, query_original: str) -> Tuple[bool, float]:
        """
        Detect mathematical queries.
        
        Returns:
            (is_math, confidence)
        """
        # Strong indicators: operators with numbers
        arithmetic_pattern = r'\d+\s*[+\-*/^×÷]\s*\d+'
        if re.search(arithmetic_pattern, query_original):
            return True, 0.95
        
        # Math functions
        math_functions = [
            r'(sin|cos|tan|sec|csc|cot)\s*\(',
            r'(log|ln|exp|sqrt)\s*\(',
            r'(abs|floor|ceil|round)\s*\(',
        ]
        
        for pattern in math_functions:
            if re.search(pattern, query_lower):
                return True, 0.95
        
        # Equations (has equals sign with numbers/variables)
        if '=' in query_original:
            if re.search(r'[a-z]\s*=|=\s*\d', query_lower):
                return True, 0.90
        
        # Math keywords (strong)
        strong_math_keywords = [
            'derivative', 'integral', 'differentiate', 'integrate',
            'factor', 'expand', 'simplify', 'solve for',
            'square root', 'logarithm', 'cosine', 'sine', 'tangent'
        ]
        
        for keyword in strong_math_keywords:
            if keyword in query_lower:
                return True, 0.85
        
        # Math keywords (medium) - need additional evidence
        medium_math_keywords = ['calculate', 'compute', 'evaluate', 'solve']
        
        keyword_count = sum(1 for kw in medium_math_keywords if kw in query_lower)
        if keyword_count > 0:
            # Check for numbers or variables
            if re.search(r'\d+|[a-z]\s*[+\-*/^=]', query_lower):
                return True, 0.75
        
        # Operators present (weak indicator - need numbers)
        operators = ['+', '-', '*', '/', '^', '×', '÷']
        has_operator = any(op in query_original for op in operators)
        has_numbers = bool(re.search(r'\d', query_original))
        
        if has_operator and has_numbers:
            # Exclude false positives
            # "C++" should not be math
            if query_lower.count('++') > 0 or query_lower.count('c++') > 0:
                return False, 0.0
            
            return True, 0.70
        
        return False, 0.0
    
    def _is_code(self, query_lower: str) -> Tuple[bool, float]:
        """
        Detect code-related queries.
        
        Returns:
            (is_code, confidence)
        """
        # Strong indicators: code syntax
        code_syntax_patterns = [
            r'def\s+\w+\s*\(',  # Python function definition
            r'function\s+\w+\s*\(',  # JavaScript function
            r'class\s+\w+',  # Class definition
            r'import\s+\w+',  # Import statement
            r'from\s+\w+\s+import',  # Python import
            r'return\s+',  # Return statement
            r'=>',  # Arrow function
            r'\{\s*\w+\s*:\s*\w+\s*\}',  # Object/dict literal
        ]
        
        for pattern in code_syntax_patterns:
            if re.search(pattern, query_lower):
                return True, 0.95
        
        # Intent keywords with language hints
        code_intents = ['write a', 'create a', 'implement', 'code for', 'program to']
        lang_hints = ['python', 'javascript', 'java', 'c++', 'ruby', 'go', 'rust']
        
        has_intent = any(intent in query_lower for intent in code_intents)
        has_lang = any(lang in query_lower for lang in lang_hints)
        
        if has_intent and has_lang:
            return True, 0.90
        
        # Programming keywords
        programming_keywords = [
            'function', 'variable', 'loop', 'array', 'list', 'dict',
            'class', 'object', 'method', 'recursive', 'iterate',
            'algorithm', 'data structure'
        ]
        
        keyword_count = sum(1 for kw in programming_keywords if kw in query_lower)
        
        if keyword_count >= 2:
            return True, 0.80
        elif keyword_count == 1:
            # Check for language hint
            if has_lang:
                return True, 0.75
        
        # Common code patterns (weak)
        if 'code' in query_lower or 'script' in query_lower or 'program' in query_lower:
            if keyword_count > 0 or has_lang:
                return True, 0.65
        
        return False, 0.0
    
    def get_modality_name(self, modality: Modality) -> str:
        """Get human-readable name for modality"""
        names = {
            Modality.LINGUISTIC: "Natural Language",
            Modality.MATH: "Mathematics",
            Modality.CODE: "Programming",
            Modality.VISUAL: "Visual",
            Modality.AUDIO: "Audio"
        }
        return names.get(modality, "Unknown")
