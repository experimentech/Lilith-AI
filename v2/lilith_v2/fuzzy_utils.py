try:
    from rapidfuzz import fuzz, process, utils
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    import difflib
    RAPIDFUZZ_AVAILABLE = False

import re
from typing import List, Tuple, Optional, Dict

class FuzzyUtils:
    """
    Utility wrapper for fuzzy string matching, prioritizing RapidFuzz
    for C++ speed, falling back to difflib if necessary.
    """

    # Keywords critical for our semantic extraction patterns
    CRITICAL_KEYWORDS = {
        "type", "kind", "is", "are", "refers", "means", "part", "contains",
        "opposite", "similar", "synonym", "antonym", "definition", "called",
        "known", "as"
    }

    @staticmethod
    def ratio(s1: str, s2: str) -> float:
        """Standard Levenshtein similarity ratio (0-100)."""
        if RAPIDFUZZ_AVAILABLE:
            return fuzz.ratio(s1, s2)
        return difflib.SequenceMatcher(None, s1, s2).ratio() * 100

    @staticmethod
    def partial_ratio(s1: str, s2: str) -> float:
        """Partial ratio (substring matching)."""
        if RAPIDFUZZ_AVAILABLE:
            return fuzz.partial_ratio(s1, s2)
        # Difflib doesn't have a direct partial ratio equivalent easily accessible
        # implementing a naive one or just defaulting to ratio
        return difflib.SequenceMatcher(None, s1, s2).ratio() * 100

    @staticmethod
    def correct_keywords(text: str, threshold: float = 80.0) -> str:
        """
        Scan text for words that look like critical keywords and correct them.
        Example: "tyype" -> "type"
        """
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Skip short words to avoid false positives (e.g. 'is', 'as')
            # unless we are very sure
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if not clean_word:
                corrected_words.append(word)
                continue

            if clean_word in FuzzyUtils.CRITICAL_KEYWORDS:
                corrected_words.append(word)
                continue
            
            # Find best match in keywords
            if RAPIDFUZZ_AVAILABLE:
                match = process.extractOne(
                    clean_word, 
                    FuzzyUtils.CRITICAL_KEYWORDS, 
                    scorer=fuzz.ratio,
                    score_cutoff=threshold
                )
                if match:
                    # match is (match_str, score, index)
                    best_word, score, _ = match
                    # Preserve original case if possible? tricky. just use lower for keywords
                    corrected_words.append(best_word)
                else:
                    corrected_words.append(word)
            else:
                # Naive fallback
                matches = difflib.get_close_matches(clean_word, FuzzyUtils.CRITICAL_KEYWORDS, n=1, cutoff=threshold/100)
                if matches:
                    corrected_words.append(matches[0])
                else:
                    corrected_words.append(word)
                    
        return " ".join(corrected_words)
