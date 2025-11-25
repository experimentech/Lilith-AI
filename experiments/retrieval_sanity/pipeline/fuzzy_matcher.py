"""
Fuzzy Matching Module

Provides typo-tolerant string matching for pattern retrieval.
Makes the system robust to spelling errors, typos, and input variations.

Key techniques:
- Edit distance (Levenshtein) for character-level similarity
- Token overlap for word-level similarity
- Phonetic matching for sound-alike words
- Common pattern variations
"""

from typing import List, Tuple, Set
import re
from difflib import SequenceMatcher


class FuzzyMatcher:
    """
    Typo-tolerant string matching.
    
    Combines multiple similarity metrics to handle real-world messy input:
    - Character-level: Edit distance (typos, spelling errors)
    - Word-level: Token overlap (word order variations)
    - Phonetic: Sound-alike matching (homophones)
    """
    
    def __init__(
        self,
        edit_distance_threshold: float = 0.75,
        token_overlap_threshold: float = 0.6,
        enable_phonetic: bool = False
    ):
        """
        Initialize fuzzy matcher.
        
        Args:
            edit_distance_threshold: Minimum similarity for edit distance (0-1)
            token_overlap_threshold: Minimum token overlap ratio (0-1)
            enable_phonetic: Enable phonetic matching (computationally expensive)
        """
        self.edit_threshold = edit_distance_threshold
        self.token_threshold = token_overlap_threshold
        self.enable_phonetic = enable_phonetic
        
        # Common misspellings and variations
        self.common_variations = {
            'machine learning': {'machien learning', 'machin learning', 'machine learing'},
            'neural network': {'nueral network', 'neural netwrok', 'neurall network'},
            'quantum': {'quantom', 'qantum', 'kuantum'},
            'python': {'pyton', 'phyton', 'pythn'},
            'algorithm': {'algoritm', 'algorythm', 'algorithim'},
        }
        
        # Build reverse index for fast lookup
        self.variation_to_canonical = {}
        for canonical, variations in self.common_variations.items():
            for variant in variations:
                self.variation_to_canonical[variant.lower()] = canonical
    
    def fuzzy_match(
        self, 
        query: str, 
        candidate: str,
        min_similarity: float = 0.75
    ) -> Tuple[bool, float]:
        """
        Check if query fuzzy matches candidate.
        
        Args:
            query: User input (potentially has typos)
            candidate: Pattern trigger to match against
            min_similarity: Minimum similarity to consider a match
            
        Returns:
            (is_match, similarity_score)
        """
        # Normalize inputs
        query_norm = self._normalize(query)
        candidate_norm = self._normalize(candidate)
        
        # Exact match (fastest path)
        if query_norm == candidate_norm:
            return (True, 1.0)
        
        # Check common variations
        canonical_query = self.variation_to_canonical.get(query_norm)
        if canonical_query and canonical_query == candidate_norm:
            return (True, 0.95)  # High confidence for known variations
        
        # Calculate multiple similarity metrics
        similarities = []
        
        # 1. Edit distance similarity (character-level)
        edit_sim = self._edit_distance_similarity(query_norm, candidate_norm)
        similarities.append(('edit', edit_sim))
        
        # 2. Token overlap similarity (word-level)
        token_sim = self._token_overlap_similarity(query_norm, candidate_norm)
        similarities.append(('token', token_sim))
        
        # 3. Sequence matching (Python's built-in)
        seq_sim = SequenceMatcher(None, query_norm, candidate_norm).ratio()
        similarities.append(('sequence', seq_sim))
        
        # 4. Phonetic similarity (if enabled)
        if self.enable_phonetic:
            phon_sim = self._phonetic_similarity(query_norm, candidate_norm)
            similarities.append(('phonetic', phon_sim))
        
        # Combined score (weighted average)
        # Prioritize edit distance for short strings, token overlap for long strings
        query_words = len(query_norm.split())
        if query_words <= 2:
            # Short queries: prioritize character-level matching
            weights = {'edit': 0.5, 'token': 0.2, 'sequence': 0.3, 'phonetic': 0.0}
        else:
            # Long queries: prioritize word-level matching
            weights = {'edit': 0.2, 'token': 0.5, 'sequence': 0.2, 'phonetic': 0.1}
        
        combined_score = sum(sim * weights.get(name, 0.0) for name, sim in similarities)
        
        # Match if any strong signal OR combined score above threshold
        strong_match = any(sim >= 0.9 for _, sim in similarities)
        weak_match = combined_score >= min_similarity
        
        if strong_match or weak_match:
            return (True, combined_score)
        else:
            return (False, combined_score)
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase, strip, normalize whitespace
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _edit_distance_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate similarity based on Levenshtein edit distance.
        
        Returns value between 0 (completely different) and 1 (identical).
        """
        if not s1 or not s2:
            return 0.0
        
        # Levenshtein distance
        distance = self._levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        
        # Convert to similarity (1 - normalized distance)
        similarity = 1.0 - (distance / max_len)
        return max(0.0, similarity)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein edit distance between two strings.
        
        Number of single-character edits (insertions, deletions, substitutions).
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        # Use dynamic programming
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _token_overlap_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate similarity based on word/token overlap.
        
        Good for handling word order variations.
        """
        tokens1 = set(s1.split())
        tokens2 = set(s2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Jaccard similarity
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _phonetic_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate phonetic similarity (sound-alike matching).
        
        Uses simple phonetic rules (simplified Soundex-like approach).
        For full phonetic matching, would use metaphone or soundex libraries.
        """
        # Simplified phonetic encoding
        def phonetic_encode(text: str) -> str:
            # Remove vowels except first letter
            encoded = text[0] if text else ''
            for c in text[1:]:
                if c not in 'aeiou':
                    encoded += c
            return encoded
        
        # Encode both strings
        phon1 = phonetic_encode(s1)
        phon2 = phonetic_encode(s2)
        
        # Compare encodings
        return self._edit_distance_similarity(phon1, phon2)
    
    def find_best_matches(
        self,
        query: str,
        candidates: List[str],
        topk: int = 5,
        min_similarity: float = 0.65
    ) -> List[Tuple[str, float]]:
        """
        Find best matching candidates for a query.
        
        Args:
            query: User input (potentially has typos)
            candidates: List of pattern triggers to match against
            topk: Return top K matches
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (candidate, similarity_score) tuples, sorted by score
        """
        matches = []
        
        for candidate in candidates:
            is_match, score = self.fuzzy_match(query, candidate, min_similarity)
            if is_match:
                matches.append((candidate, score))
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:topk]


class TypoCorrector:
    """
    Simple typo correction for common mistakes.
    
    Can be extended with more sophisticated spell-checking libraries
    (e.g., pyspellchecker, autocorrect) if needed.
    """
    
    def __init__(self):
        # Common typo patterns (character swaps, duplications, omissions)
        self.common_typos = {
            'teh': 'the',
            'adn': 'and',
            'taht': 'that',
            'waht': 'what',
            'hwo': 'how',
            'wiht': 'with',
            'thsi': 'this',
            'tehre': 'there',
            'abuot': 'about',
            'machien': 'machine',
            'learing': 'learning',
            'netwrok': 'network',
            'nueral': 'neural',
            'algoritm': 'algorithm',
            'pyton': 'python',
        }
    
    def correct(self, text: str) -> Tuple[str, bool]:
        """
        Attempt to correct typos in text.
        
        Returns:
            (corrected_text, was_corrected)
        """
        words = text.split()
        corrected_words = []
        was_corrected = False
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.common_typos:
                corrected_words.append(self.common_typos[word_lower])
                was_corrected = True
            else:
                corrected_words.append(word)
        
        corrected_text = ' '.join(corrected_words)
        return (corrected_text, was_corrected)
    
    def suggest_corrections(self, word: str) -> List[str]:
        """
        Suggest possible corrections for a word.
        
        Returns list of suggestions (empty if none found).
        """
        word_lower = word.lower()
        
        # Check exact match in typo dictionary
        if word_lower in self.common_typos:
            return [self.common_typos[word_lower]]
        
        # Could extend with edit distance suggestions from dictionary
        # For now, just return empty
        return []


# Convenience function for external use
def fuzzy_match_score(query: str, candidate: str) -> float:
    """
    Quick fuzzy matching score between two strings.
    
    Returns similarity score between 0 and 1.
    """
    matcher = FuzzyMatcher()
    _, score = matcher.fuzzy_match(query, candidate, min_similarity=0.0)
    return score
