"""
Linguistic Processor for Lilith V2.

Provides multi-stage BioNN/Database processing for input text:
1. Intake: Noise normalization, typo correction  
2. Tokenization: Break into words with POS tags
3. Syntax: PMFlow encoding of grammatical structure
4. Symbolic Frame: Extract (actor, action, target, modifiers)

This bridges V1's staged pipeline with V2's relational graph storage.
Each stage uses a BioNN (PMFlow encoder) paired with graph storage.

Key improvements over V1:
- Uses V2's RelationalGraphStore instead of flat JSON
- Proper multi-tenant support
- Integrated with concept grounding pipeline
- Stores linguistic artifacts as graph nodes for reasoning
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import hashlib
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Token:
    """Minimal token representation."""
    text: str
    position: int
    pos: str = "UNK"  # Part of speech
    lemma: Optional[str] = None

@dataclass
class ParsedSentence:
    """Syntactic structure extracted from text."""
    tokens: List[Token]
    subject_index: Optional[int] = None
    verb_index: Optional[int] = None
    object_index: Optional[int] = None  
    modifiers: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    intent: Optional[str] = None
    negation_indices: List[int] = field(default_factory=list)
    pos_sequence: List[str] = field(default_factory=list)  # For syntax BioNN

@dataclass
class SymbolicFrame:
    """Language-agnostic action representation."""
    actor: Optional[str]
    action: Optional[str]
    target: Optional[str]
    modifiers: Dict[str, str]
    attributes: Dict[str, str]
    confidence: float
    raw_text: str
    
    def as_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor,
            "action": self.action,
            "target": self.target,
            "modifiers": self.modifiers,
            "attributes": self.attributes,
            "confidence": self.confidence,
        }

@dataclass
class LinguisticArtifact:
    """Complete processing result for an utterance."""
    original_text: str
    normalized_text: str
    parsed: ParsedSentence
    frame: SymbolicFrame
    syntax_embedding: Optional[torch.Tensor] = None
    semantic_embedding: Optional[torch.Tensor] = None
    concepts: List[str] = field(default_factory=list)  # Grounded concept IDs


# =============================================================================
# Intake Layer - Noise Normalization
# =============================================================================

class IntakeProcessor:
    """Input normalization layer."""
    
    def __init__(self):
        self.typo_corrections = {
            "teh": "the",
            "dont": "don't",
            "cant": "can't",
            "wont": "won't",
            "recieve": "receive",
            "definately": "definitely",
            "seperate": "separate",
            "occurence": "occurrence",
            "beleive": "believe",
            "wierd": "weird",
            "untill": "until",
            "begining": "beginning",
            "occured": "occurred",
        }
        
        self.filler_phrases = [
            "i mean", "you know", "like", "um", "uh", "well",
            "actually", "basically", "literally", "honestly",
            "okay so", "alright so", "anyway", "so basically"
        ]
    
    def normalize(self, text: str) -> str:
        """Apply intake normalization."""
        # 1. Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 2. Apply typo corrections
        words = text.split()
        fixed = [self.typo_corrections.get(w.lower(), w) for w in words]
        text = ' '.join(fixed)
        
        return text
    
    def strip_fillers(self, text: str) -> str:
        """Remove discourse fillers from start of text."""
        cleaned = text.lower().strip()
        
        for filler in sorted(self.filler_phrases, key=len, reverse=True):
            if cleaned.startswith(filler):
                cleaned = cleaned[len(filler):].lstrip(", ")
                # Restore original casing by finding the remainder
                text = text[len(text) - len(cleaned) - (len(text.strip()) - len(cleaned)):].strip()
                
        return text if text else cleaned


# =============================================================================
# Parser Layer - Tokenization and POS Tagging
# =============================================================================

class LexicalParser:
    """Heuristic tokenization and POS tagging."""
    
    # Word category sets (expanded from V1)
    PRONOUNS = frozenset([
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves",
        "this", "that", "these", "those", "who", "whom", "which", "what",
        "mine", "yours", "his", "hers", "ours", "theirs",
    ])
    
    DETERMINERS = frozenset([
        "the", "a", "an", "this", "that", "these", "those", 
        "my", "your", "his", "her", "its", "our", "their",
        "some", "any", "no", "every", "each", "either", "neither",
    ])
    
    QUESTION_WORDS = frozenset([
        "who", "what", "where", "when", "why", "how", "which", "whose", "whom"
    ])
    
    PREPOSITIONS = frozenset([
        "in", "on", "at", "to", "for", "with", "by", "from", "of", "about",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "under", "over", "among", "across", "behind", "beyond",
    ])
    
    AUXILIARIES = frozenset([
        "is", "are", "am", "was", "were", "be", "been", "being",
        "have", "has", "had", "having",
        "do", "does", "did", "doing",
        "will", "would", "shall", "should", 
        "can", "could", "may", "might", "must",
        "don't", "doesn't", "didn't", "won't", "wouldn't",
        "can't", "couldn't", "shouldn't", "mustn't",
        "isn't", "aren't", "wasn't", "weren't",
    ])
    
    COMMON_VERBS = frozenset([
        "go", "goes", "went", "gone", "going",
        "come", "comes", "came", "coming",
        "get", "gets", "got", "getting",
        "make", "makes", "made", "making",
        "know", "knows", "knew", "known", "knowing",
        "think", "thinks", "thought", "thinking",
        "take", "takes", "took", "taken", "taking",
        "see", "sees", "saw", "seen", "seeing",
        "want", "wants", "wanted", "wanting",
        "use", "uses", "used", "using",
        "find", "finds", "found", "finding",
        "give", "gives", "gave", "given", "giving",
        "tell", "tells", "told", "telling",
        "ask", "asks", "asked", "asking",
        "need", "needs", "needed", "needing",
        "feel", "feels", "felt", "feeling",
        "try", "tries", "tried", "trying",
        "like", "likes", "liked", "liking",
        "help", "helps", "helped", "helping",
        "talk", "talks", "talked", "talking",
        "say", "says", "said", "saying",
        "learn", "learns", "learned", "learning",
        "work", "works", "worked", "working",
    ])
    
    COMMON_ADJECTIVES = frozenset([
        "good", "bad", "new", "old", "great", "little", "big", "small",
        "high", "low", "long", "short", "first", "last", "next", 
        "right", "wrong", "true", "false", "real", "important",
        "best", "better", "nice", "cool", "interesting", "amazing",
    ])
    
    COMMON_ADVERBS = frozenset([
        "not", "just", "also", "very", "often", "too", "really",
        "never", "always", "sometimes", "usually", "probably",
        "here", "there", "now", "then", "today", "tomorrow", "yesterday",
        "well", "still", "already", "even", "only", "quite",
    ])
    
    NEGATIONS = frozenset(["not", "never", "no", "none", "nothing", "neither", "nobody"])
    
    INTERJECTIONS = frozenset([
        "oh", "ah", "wow", "hey", "hi", "hello", "bye", "goodbye",
        "yes", "no", "yeah", "nah", "ok", "okay", "please", "thanks",
    ])
    
    TOKEN_SPLIT_RE = re.compile(r"[\w']+", re.UNICODE)
    
    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens."""
        return self.TOKEN_SPLIT_RE.findall(text)
    
    def assign_pos(self, token: str, prev_pos: Optional[str] = None) -> str:
        """Assign part-of-speech tag using context-aware heuristics."""
        t = token.lower()
        
        # Punctuation
        if token in '.,!?;:"\'-()[]{}':
            return "PUNCT"
        
        # Question words
        if t in self.QUESTION_WORDS:
            return "WRB"
        
        # Interjections
        if t in self.INTERJECTIONS:
            return "UH"
        
        # Determiners
        if t in self.DETERMINERS:
            return "DT"
        
        # Pronouns (but not after determiners)
        if t in self.PRONOUNS and prev_pos != "DT":
            return "PRON"
        
        # Prepositions
        if t in self.PREPOSITIONS:
            return "IN"
        
        # Auxiliaries
        if t in self.AUXILIARIES:
            return "AUX"
        
        # Common adverbs
        if t in self.COMMON_ADVERBS:
            return "RB"
        
        # Common verbs (check morphology for tense)
        if t in self.COMMON_VERBS:
            if t.endswith("ing"):
                return "VBG"
            elif t.endswith("ed"):
                return "VBD"
            elif t.endswith("s") and not t.endswith("ss"):
                return "VBZ"
            return "VB"
        
        # Common adjectives
        if t in self.COMMON_ADJECTIVES:
            return "JJ"
        
        # Morphological rules for unknown words
        if t.endswith("ing"):
            return "VBG"
        if t.endswith("ed"):
            return "VBD"
        if t.endswith("ly"):
            return "RB"
        if t.endswith(("tion", "sion", "ment", "ness", "ity")):
            return "NN"
        if t.endswith("ful") or t.endswith("ous") or t.endswith("ive"):
            return "JJ"
        
        # Numbers
        if t.isdigit():
            return "NUM"
        
        # Context-aware disambiguation
        if prev_pos in ("DT", "JJ", "PRON"):
            # After determiner or adjective, likely noun
            return "NN"
        if prev_pos in ("MD", "TO", "AUX"):
            # After modal or "to", likely verb
            return "VB"
        
        return "UNK"
    
    def lemmatize(self, token: str, pos: str) -> str:
        """Basic lemmatization."""
        t = token.lower()
        
        if pos.startswith("VB"):
            # Verb lemmatization
            for suffix in ("ing", "ed", "es", "s"):
                if t.endswith(suffix) and len(t) > len(suffix) + 2:
                    return t[:-len(suffix)]
            # Irregular verbs
            irregulars = {
                "went": "go", "gone": "go",
                "saw": "see", "seen": "see", 
                "knew": "know", "known": "know",
                "thought": "think",
                "made": "make",
                "took": "take", "taken": "take",
                "came": "come",
                "got": "get",
                "said": "say",
                "told": "tell",
                "felt": "feel",
            }
            return irregulars.get(t, t)
        
        if pos == "NN" or pos == "NNS":
            # Noun lemmatization
            if t.endswith("ies") and len(t) > 4:
                return t[:-3] + "y"
            for suffix in ("es", "s"):
                if t.endswith(suffix) and len(t) > len(suffix) + 1:
                    return t[:-len(suffix)]
        
        return t
    
    def parse(self, text: str) -> ParsedSentence:
        """Full parse of text into structured representation."""
        raw_tokens = self.tokenize(text)
        tokens: List[Token] = []
        pos_sequence: List[str] = []
        prev_pos = None
        
        # First pass: tokenize and tag
        for idx, tok in enumerate(raw_tokens):
            pos = self.assign_pos(tok, prev_pos)
            lemma = self.lemmatize(tok, pos)
            tokens.append(Token(text=tok.lower(), position=idx, pos=pos, lemma=lemma))
            pos_sequence.append(pos)
            prev_pos = pos
        
        # Calculate recognition confidence
        unknown_count = sum(1 for t in tokens if t.pos == "UNK")
        confidence = 1.0 - (unknown_count / max(len(tokens), 1))
        
        # Second pass: identify roles
        subject_idx = self._find_subject(tokens)
        verb_idx = self._find_verb(tokens)
        object_idx = self._find_object(tokens, verb_idx)
        
        # Detect modifiers
        modifiers: Dict[str, str] = {}
        negation_indices: List[int] = []
        
        for i, t in enumerate(tokens):
            if t.text.lower() in self.NEGATIONS:
                negation_indices.append(i)
                modifiers["negated"] = "true"
        
        # Detect intent
        intent = None
        if text.strip().endswith("?"):
            intent = "question"
        if tokens and tokens[0].pos == "WRB":
            intent = tokens[0].text  # "what", "how", etc.
        
        return ParsedSentence(
            tokens=tokens,
            subject_index=subject_idx,
            verb_index=verb_idx,
            object_index=object_idx,
            modifiers=modifiers,
            confidence=confidence,
            intent=intent,
            negation_indices=negation_indices,
            pos_sequence=pos_sequence,
        )
    
    def _find_subject(self, tokens: List[Token]) -> Optional[int]:
        """Find subject (first pronoun or first noun before verb)."""
        # First try pronouns
        for t in tokens:
            if t.pos == "PRON":
                return t.position
        # Then try nouns
        for t in tokens:
            if t.pos == "NN":
                return t.position
        return None
    
    def _find_verb(self, tokens: List[Token]) -> Optional[int]:
        """Find main verb (skip auxiliaries)."""
        for t in tokens:
            if t.pos.startswith("VB") and t.text not in self.AUXILIARIES:
                return t.position
        # Fallback to any verb including aux
        for t in tokens:
            if t.pos.startswith("VB") or t.pos == "AUX":
                return t.position
        return None
    
    def _find_object(self, tokens: List[Token], verb_idx: Optional[int]) -> Optional[int]:
        """Find object (first noun or pronoun after verb)."""
        if verb_idx is None:
            return None
        for t in tokens[verb_idx + 1:]:
            if t.pos in ("NN", "PRON"):
                return t.position
        return None


# =============================================================================
# Symbolic Frame Builder
# =============================================================================

def build_frame(text: str, parsed: ParsedSentence) -> SymbolicFrame:
    """Build symbolic representation from parsed sentence."""
    
    def get_token(idx: Optional[int]) -> Optional[str]:
        if idx is not None and 0 <= idx < len(parsed.tokens):
            return parsed.tokens[idx].text
        return None
    
    actor = get_token(parsed.subject_index)
    action = get_token(parsed.verb_index)
    target = get_token(parsed.object_index)
    
    modifiers = dict(parsed.modifiers)
    
    attributes = {
        "token_count": str(len(parsed.tokens)),
        "subject_present": str(parsed.subject_index is not None),
        "verb_present": str(parsed.verb_index is not None),
        "object_present": str(parsed.object_index is not None),
        "pos_pattern": "_".join(parsed.pos_sequence[:5]),  # First 5 POS tags
    }
    
    if parsed.intent:
        attributes["intent"] = parsed.intent
    if parsed.negation_indices:
        attributes["negated"] = "true"
    
    return SymbolicFrame(
        actor=actor,
        action=action,
        target=target,
        modifiers=modifiers,
        attributes=attributes,
        confidence=parsed.confidence,
        raw_text=text,
    )


# =============================================================================
# Main Linguistic Processor - Combines All Stages
# =============================================================================

class LinguisticProcessor:
    """
    Multi-stage linguistic processing with BioNN/Database pairing.
    
    Each stage:
    - Intake: Normalize text
    - Parse: Tokenize + POS tag
    - Syntax: PMFlow encode POS pattern (stores in syntax_patterns table)
    - Frame: Build symbolic representation (stores in frames table)
    - Ground: Map to existing concepts (uses concept grounding)
    """
    
    def __init__(
        self,
        graph_store = None,  # V2 RelationalGraphStore or MultiTenantGraphManager
        encoder = None,      # PMFlow encoder for syntax embeddings
    ):
        self.intake = IntakeProcessor()
        self.parser = LexicalParser()
        self.graph = graph_store
        self.encoder = encoder
        
        logger.info("LinguisticProcessor initialized")
    
    def process(self, text: str, tenant_id: str = None) -> LinguisticArtifact:
        """
        Full linguistic processing pipeline.
        
        Returns LinguisticArtifact with all intermediate results.
        """
        # 1. Intake: Normalize
        normalized = self.intake.normalize(text)
        
        # 2. Parse: Tokenize + POS
        parsed = self.parser.parse(normalized)
        
        # 3. Frame: Build symbolic representation
        frame = build_frame(normalized, parsed)
        
        # 4. Syntax Embedding (if encoder available)
        syntax_emb = None
        if self.encoder and parsed.pos_sequence:
            pos_string = " ".join(parsed.pos_sequence)
            try:
                syntax_emb = self.encoder.encode(pos_string)
            except Exception as e:
                logger.warning(f"Syntax encoding failed: {e}")
        
        # 5. Store artifacts in graph (if available)
        if self.graph:
            self._store_artifacts(text, normalized, parsed, frame, tenant_id)
        
        return LinguisticArtifact(
            original_text=text,
            normalized_text=normalized,
            parsed=parsed,
            frame=frame,
            syntax_embedding=syntax_emb,
        )
    
    def _store_artifacts(
        self, 
        original: str,
        normalized: str,
        parsed: ParsedSentence,
        frame: SymbolicFrame,
        tenant_id: str = None
    ):
        """Store linguistic artifacts in the relational graph."""
        try:
            # Generate stable IDs
            text_hash = hashlib.md5(original.lower().encode()).hexdigest()[:12]
            utterance_id = f"utt_{text_hash}"
            frame_id = f"frame_{text_hash}"
            
            # Store utterance node
            self.graph.add_node(
                node_id=utterance_id,
                node_type="utterance_parsed",
                term=normalized,
                confidence=parsed.confidence,
                data={
                    "original": original,
                    "tokens": [t.text for t in parsed.tokens],
                    "pos_sequence": parsed.pos_sequence,
                    "intent": parsed.intent,
                },
                tenant_id=tenant_id
            )
            
            # Store symbolic frame node
            self.graph.add_node(
                node_id=frame_id,
                node_type="symbolic_frame",
                term=f"{frame.actor or '_'} {frame.action or '_'} {frame.target or '_'}",
                confidence=frame.confidence,
                data=frame.as_dict(),
                tenant_id=tenant_id
            )
            
            # Link utterance to frame
            self.graph.add_edge(
                source=utterance_id,
                target=frame_id,
                edge_type="has_frame",
                confidence=1.0,
                tenant_id=tenant_id
            )
            
            # Store individual tokens with their POS (for syntax pattern learning)
            for token in parsed.tokens:
                if token.pos != "UNK" and token.pos != "PUNCT":
                    token_id = f"tok_{token.text}_{token.pos}"
                    try:
                        self.graph.add_node(
                            node_id=token_id,
                            node_type="lexical_token",
                            term=token.text,
                            confidence=1.0,
                            data={"pos": token.pos, "lemma": token.lemma},
                            tenant_id=tenant_id
                        )
                        # Link token to utterance
                        self.graph.add_edge(
                            source=utterance_id,
                            target=token_id,
                            edge_type="contains_token",
                            confidence=1.0,
                            tenant_id=tenant_id
                        )
                    except Exception:
                        pass  # Token may already exist, that's fine
            
            # Store POS pattern for syntax learning
            if parsed.pos_sequence:
                pos_pattern = "_".join(parsed.pos_sequence[:8])  # Truncate long patterns
                pattern_id = f"pos_{hashlib.md5(pos_pattern.encode()).hexdigest()[:8]}"
                
                self.graph.add_node(
                    node_id=pattern_id,
                    node_type="pos_pattern",
                    term=pos_pattern,
                    confidence=parsed.confidence,
                    data={"length": len(parsed.pos_sequence), "intent": parsed.intent},
                    tenant_id=tenant_id
                )
                self.graph.add_edge(
                    source=utterance_id,
                    target=pattern_id,
                    edge_type="has_pos_pattern",
                    confidence=1.0,
                    tenant_id=tenant_id
                )
                
        except Exception as e:
            logger.warning(f"Failed to store linguistic artifacts: {e}")
    
    def get_tokens_for_grounding(self, artifact: LinguisticArtifact) -> List[str]:
        """
        Extract terms suitable for concept grounding.
        
        Returns lemmatized nouns, verbs, and adjectives (content words).
        """
        content_pos = {"NN", "VB", "VBD", "VBG", "VBZ", "JJ"}
        terms = []
        
        for token in artifact.parsed.tokens:
            if token.pos in content_pos or token.pos == "UNK":
                term = token.lemma or token.text
                if len(term) >= 3:  # Skip very short words
                    terms.append(term)
        
        return terms
    
    def extract_key_phrases(self, artifact: LinguisticArtifact) -> List[str]:
        """
        Extract meaningful phrases (noun phrases, verb phrases).
        """
        phrases = []
        tokens = artifact.parsed.tokens
        
        # Simple noun phrase extraction (DT? JJ* NN+)
        i = 0
        while i < len(tokens):
            phrase_tokens = []
            
            # Optional determiner
            if tokens[i].pos == "DT":
                phrase_tokens.append(tokens[i].text)
                i += 1
                if i >= len(tokens):
                    break
            
            # Optional adjectives
            while i < len(tokens) and tokens[i].pos == "JJ":
                phrase_tokens.append(tokens[i].text)
                i += 1
            
            # Required noun
            if i < len(tokens) and tokens[i].pos == "NN":
                phrase_tokens.append(tokens[i].text)
                i += 1
                # Additional nouns (compound nouns)
                while i < len(tokens) and tokens[i].pos == "NN":
                    phrase_tokens.append(tokens[i].text)
                    i += 1
                    
                if len(phrase_tokens) >= 2:  # Only multi-word phrases
                    phrases.append(" ".join(phrase_tokens))
            else:
                i += 1
        
        return phrases


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "Token",
    "ParsedSentence",
    "SymbolicFrame",
    "LinguisticArtifact",
    "IntakeProcessor",
    "LexicalParser",
    "LinguisticProcessor",
    "build_frame",
]
