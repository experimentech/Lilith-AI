"""
Syntax Stage - BioNN-based Grammatical Pattern Processing

Full cognitive stage implementation using PMFlow BioNN for syntactic structures.
Learns grammatical patterns through reinforcement, just like semantic stage.

Stage flow:
  Input (tokens + POS tags) → PMFlow Encoder → Syntax Embedding
  Syntax Embedding → Retrieve similar patterns → Composition templates
  Feedback → Plasticity updates → Better grammar over time

Plasticity features (PMFlow 0.3.0):
  - vectorized_pm_plasticity: Hebbian-style reinforcement learning
  - contrastive_plasticity: Pull similar grammar patterns, push dissimilar
  - MultiScale support: Coarse (sentence type) + Fine (specific structure)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import json

import torch
import numpy as np

from .embedding import PMFlowEmbeddingEncoder
from .stage_coordinator import StageType, StageConfig, StageArtifact

# Import PMFlow plasticity functions (0.3.1 features)
try:
    from pmflow import vectorized_pm_plasticity, contrastive_plasticity, batch_plasticity_update
    PMFLOW_PLASTICITY_AVAILABLE = True
except ImportError:
    vectorized_pm_plasticity = None
    contrastive_plasticity = None
    batch_plasticity_update = None
    PMFLOW_PLASTICITY_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class SyntaxPlasticityReport:
    """Summary of a syntax plasticity update."""
    
    feedback: float
    threshold: float
    delta_centers: float
    delta_mus: float
    pattern_id: str
    plasticity_type: str = "reinforcement"  # reinforcement, contrastive
    
    def as_dict(self) -> Dict[str, Any]:
        return {
            "feedback": float(self.feedback),
            "threshold": float(self.threshold),
            "delta_centers": float(self.delta_centers),
            "delta_mus": float(self.delta_mus),
            "pattern_id": self.pattern_id,
            "plasticity_type": self.plasticity_type,
        }


@dataclass
class SyntacticPattern:
    """A learned grammatical pattern stored in syntax_memory."""
    
    pattern_id: str
    pos_sequence: List[str]          # ["DT", "JJ", "NN", "VBZ"]
    embedding: torch.Tensor          # PMFlow encoding of POS sequence
    template: str                     # "The {adj} {noun} {verb}"
    example: str                      # Actual instance
    success_score: float = 0.5       # Reinforcement score
    usage_count: int = 0
    intent: str = "general"          # statement, question, compound


@dataclass
class CompositionTemplate:
    """Template for grammatically combining fragments."""
    
    template_id: str
    pattern_a_type: str              # e.g., "statement"
    pattern_b_type: str              # e.g., "question"
    connector: str                   # "and", ".", "because"
    embedding: torch.Tensor          # PMFlow encoding of template structure
    example: str
    success_score: float = 0.5


class SyntaxStage:
    """
    Cognitive stage for grammatical pattern processing using BioNN.
    
    This is a FULL stage implementation, parallel to INTAKE/SEMANTIC:
    - Dedicated PMFlow encoder for POS sequences
    - Separate syntax_memory database namespace  
    - Independent plasticity learning
    - Reinforcement-based grammar improvement
    """
    
    def __init__(
        self,
        config: Optional[StageConfig] = None,
        encoder: Optional[PMFlowEmbeddingEncoder] = None,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize syntax stage.
        
        Args:
            config: Stage configuration (uses defaults if None)
            encoder: PMFlow encoder for POS sequences (creates new if None)
            storage_path: Where to store learned patterns
        """
        # Default config
        if config is None:
            config = StageConfig(
                stage_type=StageType.REASONING,  # Use REASONING slot for syntax
                encoder_config={
                    "latent_dim": 32,  # Smaller than semantic (simpler patterns)
                    "num_centers": 64,  # Enough for common POS sequences
                },
                db_namespace="syntax_memory",
                plasticity_enabled=True,
                plasticity_lr=1e-3  # Faster learning for discrete patterns
            )
        
        self.config = config
        self.stage_type = config.stage_type
        
        # Initialize PMFlow encoder for POS sequences
        if encoder is None:
            logger.info("Initializing PMFlow encoder for syntax stage...")
            self.encoder = PMFlowEmbeddingEncoder(
                latent_dim=config.encoder_config.get("latent_dim", 32),
                seed=config.encoder_config.get("seed", 42),
            )
        else:
            self.encoder = encoder
            
        # Storage
        self.storage_path = storage_path or Path("syntax_patterns.json")
        self.pmflow_state_path = self.storage_path.with_suffix('.pt')  # PMFlow state
        
        # Pattern stores
        self.patterns: Dict[str, SyntacticPattern] = {}
        self.templates: Dict[str, CompositionTemplate] = {}
        
        # Plasticity tracking
        self.plasticity_reports: List[SyntaxPlasticityReport] = []
        self.total_plasticity_updates = 0
        
        # Load or bootstrap
        if self.storage_path.exists():
            self._load_patterns()
        else:
            self._bootstrap_patterns()
        
        # Load PMFlow state if available (plasticity-learned weights)
        if self.pmflow_state_path.exists():
            self._load_pmflow_state()
            
        logger.info(
            f"Syntax stage initialized: {len(self.patterns)} patterns, "
            f"{len(self.templates)} templates, "
            f"plasticity={'enabled' if PMFLOW_PLASTICITY_AVAILABLE else 'disabled'}"
        )
    
    def process(
        self,
        tokens: List[str],
        pos_tags: Optional[List[str]] = None
    ) -> StageArtifact:
        """
        Process tokens through syntax stage using BioNN.
        
        Args:
            tokens: Input tokens
            pos_tags: Part-of-speech tags (extracts if None)
            
        Returns:
            StageArtifact with syntax embedding and matched patterns
        """
        # Extract POS if not provided
        if pos_tags is None:
            pos_tags = self._extract_pos_tags(tokens)
        
        # Encode POS sequence using PMFlow BioNN
        pos_string = " ".join(pos_tags)
        embedding, latent, activations = self.encoder.encode_with_components(pos_string)
        
        # Calculate confidence from activation energy
        confidence = self._compute_confidence(activations)
        
        # Retrieve similar grammatical patterns
        matched_patterns = self._retrieve_patterns(embedding, topk=5)
        
        # Determine syntactic intent
        intent = self._classify_intent(pos_tags, tokens)
        
        return StageArtifact(
            stage=self.stage_type,
            embedding=embedding,
            confidence=confidence,
            tokens=tokens,
            activations=activations,
            metadata={
                "pos_sequence": pos_tags,
                "pos_string": pos_string,
                "intent": intent,
                "matched_patterns": [
                    {"id": p.pattern_id, "score": score, "template": p.template}
                    for p, score in matched_patterns
                ],
                "activation_energy": float(torch.norm(activations).item()),
            }
        )
    
    # ─────────────────────────────────────────────────────────────
    # POS Tagging - Improved heuristics (no external dependencies)
    # ─────────────────────────────────────────────────────────────
    
    # Word lists for POS tagging
    _QUESTION_WORDS = frozenset(['who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom'])
    _PRONOUNS = frozenset([
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves',
        'this', 'that', 'these', 'those', 'who', 'whom', 'which', 'what',
        'mine', 'yours', 'his', 'hers', 'ours', 'theirs',
        'anyone', 'everyone', 'someone', 'nobody', 'anybody', 'everybody', 'somebody',
        'anything', 'everything', 'something', 'nothing'
    ])
    _DETERMINERS = frozenset(['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'some', 'any', 'no', 'every', 'each', 'either', 'neither', 'both', 'all', 'few', 'many', 'much', 'several'])
    _COORDINATORS = frozenset(['and', 'or', 'but', 'nor', 'yet', 'so', 'for'])
    _SUBORDINATORS = frozenset(['if', 'because', 'although', 'though', 'while', 'when', 'where', 'after', 'before', 'since', 'until', 'unless', 'whereas', 'whether', 'once', 'as'])
    _PREPOSITIONS = frozenset(['in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'over', 'among', 'along', 'across', 'behind', 'beyond', 'near', 'within', 'without', 'against', 'toward', 'towards', 'upon', 'throughout', 'beside', 'besides', 'around', 'outside', 'inside'])
    _AUXILIARIES = frozenset(['is', 'are', 'am', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must', "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't", "shouldn't", "mustn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't"])
    _MODALS = frozenset(['can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would', 'ought'])
    _COMMON_VERBS = frozenset(['go', 'goes', 'went', 'gone', 'going', 'come', 'comes', 'came', 'coming', 'get', 'gets', 'got', 'getting', 'make', 'makes', 'made', 'making', 'know', 'knows', 'knew', 'known', 'knowing', 'think', 'thinks', 'thought', 'thinking', 'take', 'takes', 'took', 'taken', 'taking', 'see', 'sees', 'saw', 'seen', 'seeing', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'give', 'gives', 'gave', 'given', 'giving', 'tell', 'tells', 'told', 'telling', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'put', 'puts', 'putting', 'mean', 'means', 'meant', 'meaning', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'begin', 'begins', 'began', 'begun', 'beginning', 'seem', 'seems', 'seemed', 'seeming', 'help', 'helps', 'helped', 'helping', 'show', 'shows', 'showed', 'shown', 'showing', 'hear', 'hears', 'heard', 'hearing', 'play', 'plays', 'played', 'playing', 'run', 'runs', 'ran', 'running', 'move', 'moves', 'moved', 'moving', 'like', 'likes', 'liked', 'liking', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'learn', 'learns', 'learned', 'learning', 'change', 'changes', 'changed', 'changing', 'lead', 'leads', 'led', 'leading', 'understand', 'understands', 'understood', 'understanding', 'watch', 'watches', 'watched', 'watching', 'follow', 'follows', 'followed', 'following', 'stop', 'stops', 'stopped', 'stopping', 'create', 'creates', 'created', 'creating', 'speak', 'speaks', 'spoke', 'spoken', 'speaking', 'read', 'reads', 'reading', 'spend', 'spends', 'spent', 'spending', 'grow', 'grows', 'grew', 'grown', 'growing', 'open', 'opens', 'opened', 'opening', 'walk', 'walks', 'walked', 'walking', 'win', 'wins', 'won', 'winning', 'teach', 'teaches', 'taught', 'teaching', 'offer', 'offers', 'offered', 'offering', 'remember', 'remembers', 'remembered', 'remembering', 'love', 'loves', 'loved', 'loving', 'consider', 'considers', 'considered', 'considering', 'appear', 'appears', 'appeared', 'appearing', 'buy', 'buys', 'bought', 'buying', 'wait', 'waits', 'waited', 'waiting', 'serve', 'serves', 'served', 'serving', 'die', 'dies', 'died', 'dying', 'send', 'sends', 'sent', 'sending', 'expect', 'expects', 'expected', 'expecting', 'build', 'builds', 'built', 'building', 'stay', 'stays', 'stayed', 'staying', 'fall', 'falls', 'fell', 'fallen', 'falling', 'cut', 'cuts', 'cutting', 'reach', 'reaches', 'reached', 'reaching', 'kill', 'kills', 'killed', 'killing', 'remain', 'remains', 'remained', 'remaining', 'suggest', 'suggests', 'suggested', 'suggesting', 'raise', 'raises', 'raised', 'raising', 'pass', 'passes', 'passed', 'passing', 'sell', 'sells', 'sold', 'selling', 'require', 'requires', 'required', 'requiring', 'report', 'reports', 'reported', 'reporting', 'decide', 'decides', 'decided', 'deciding', 'pull', 'pulls', 'pulled', 'pulling'])
    _COMMON_ADJECTIVES = frozenset(['good', 'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other', 'old', 'right', 'big', 'high', 'different', 'small', 'large', 'next', 'early', 'young', 'important', 'few', 'public', 'bad', 'same', 'able', 'free', 'sure', 'clear', 'full', 'special', 'easy', 'strong', 'true', 'whole', 'real', 'best', 'better', 'nice', 'cool', 'interesting', 'amazing', 'awesome', 'beautiful', 'wonderful', 'terrible', 'horrible', 'fantastic', 'excellent', 'perfect', 'simple', 'hard', 'soft', 'fast', 'slow', 'hot', 'cold', 'warm', 'dark', 'light', 'open', 'close', 'short', 'happy', 'sad', 'angry', 'afraid', 'alone', 'certain', 'likely', 'possible', 'impossible', 'necessary', 'ready', 'sorry', 'wrong', 'late', 'recent', 'main', 'major', 'general', 'specific', 'particular', 'available', 'popular', 'common', 'natural', 'physical', 'final', 'local', 'international', 'national', 'political', 'economic', 'social', 'human', 'personal', 'private', 'military', 'legal', 'medical', 'single', 'various', 'similar', 'dead', 'central', 'current', 'foreign', 'federal', 'normal', 'serious', 'financial', 'basic', 'pretty'])
    _COMMON_ADVERBS = frozenset(['not', 'just', 'also', 'very', 'often', 'however', 'too', 'usually', 'really', 'early', 'never', 'always', 'sometimes', 'together', 'likely', 'simply', 'generally', 'instead', 'actually', 'already', 'again', 'rather', 'almost', 'especially', 'ever', 'quickly', 'probably', 'certainly', 'perhaps', 'maybe', 'finally', 'recently', 'still', 'well', 'here', 'there', 'now', 'then', 'today', 'tomorrow', 'yesterday', 'soon', 'ago', 'yet', 'later', 'even', 'only', 'quite', 'enough', 'thus', 'therefore', 'hence', 'extremely', 'absolutely', 'completely', 'totally', 'entirely', 'nearly', 'barely', 'hardly', 'merely', 'mostly', 'partly', 'slightly', 'somewhat', 'fairly', 'rather'])
    _INTERJECTIONS = frozenset(['oh', 'ah', 'wow', 'hey', 'hi', 'hello', 'bye', 'goodbye', 'yes', 'no', 'yeah', 'nah', 'ok', 'okay', 'please', 'thanks', 'thank', 'sorry', 'excuse', 'well', 'um', 'uh', 'hmm', 'huh', 'oops', 'ouch', 'yay', 'hooray', 'alas', 'whoa', 'gee', 'gosh', 'damn', 'darn', 'shoot', 'yikes', 'phew', 'ugh', 'meh', 'aw', 'aww'])
    
    def _extract_pos_tags(self, tokens: List[str]) -> List[str]:
        """
        Extract POS tags from tokens using improved heuristics.
        
        BioNN-native approach: No external NLP dependencies.
        Uses expanded word lists, suffix rules, and context hints.
        
        POS Tag Key:
            WRB  - Wh-adverb (question words)
            PRON - Pronoun
            DT   - Determiner
            CC   - Coordinating conjunction
            IN   - Preposition/subordinating conjunction
            MD   - Modal
            VB   - Verb (base form)
            VBZ  - Verb (3rd person singular present)
            VBD  - Verb (past tense)
            VBG  - Verb (gerund/present participle)
            VBN  - Verb (past participle)
            NN   - Noun (singular)
            NNS  - Noun (plural)
            JJ   - Adjective
            RB   - Adverb
            UH   - Interjection
            PUNCT- Punctuation
        """
        pos_tags = []
        prev_tag = None
        
        for i, token in enumerate(tokens):
            token_lower = token.lower().strip()
            
            # Handle punctuation
            if token in '.,!?;:"\'-()[]{}':
                pos_tags.append('PUNCT')
                prev_tag = 'PUNCT'
                continue
            
            # Question words
            if token_lower in self._QUESTION_WORDS:
                pos_tags.append('WRB')
                prev_tag = 'WRB'
                continue
            
            # Interjections (often at start)
            if token_lower in self._INTERJECTIONS:
                pos_tags.append('UH')
                prev_tag = 'UH'
                continue
            
            # Determiners
            if token_lower in self._DETERMINERS:
                pos_tags.append('DT')
                prev_tag = 'DT'
                continue
            
            # Pronouns (but not when used as determiners)
            if token_lower in self._PRONOUNS and prev_tag not in ('DT',):
                pos_tags.append('PRON')
                prev_tag = 'PRON'
                continue
            
            # Coordinating conjunctions
            if token_lower in self._COORDINATORS:
                pos_tags.append('CC')
                prev_tag = 'CC'
                continue
            
            # Prepositions and subordinating conjunctions
            if token_lower in self._PREPOSITIONS or token_lower in self._SUBORDINATORS:
                pos_tags.append('IN')
                prev_tag = 'IN'
                continue
            
            # Modals
            if token_lower in self._MODALS:
                pos_tags.append('MD')
                prev_tag = 'MD'
                continue
            
            # Auxiliaries (as verbs)
            if token_lower in self._AUXILIARIES:
                pos_tags.append('VBZ')
                prev_tag = 'VBZ'
                continue
            
            # Common verbs
            if token_lower in self._COMMON_VERBS:
                # Try to determine verb form from suffix
                if token_lower.endswith('ing'):
                    pos_tags.append('VBG')
                elif token_lower.endswith('ed'):
                    pos_tags.append('VBD')
                elif token_lower.endswith('s') and not token_lower.endswith('ss'):
                    pos_tags.append('VBZ')
                else:
                    pos_tags.append('VB')
                prev_tag = pos_tags[-1]
                continue
            
            # Common adjectives
            if token_lower in self._COMMON_ADJECTIVES:
                pos_tags.append('JJ')
                prev_tag = 'JJ'
                continue
            
            # Common adverbs
            if token_lower in self._COMMON_ADVERBS:
                pos_tags.append('RB')
                prev_tag = 'RB'
                continue
            
            # ─────────────────────────────────────────────────────
            # Suffix-based rules for unknown words
            # ─────────────────────────────────────────────────────
            
            # Adverbs: -ly suffix
            if token_lower.endswith('ly') and len(token_lower) > 3:
                pos_tags.append('RB')
                prev_tag = 'RB'
                continue
            
            # Gerunds/present participles: -ing
            if token_lower.endswith('ing') and len(token_lower) > 4:
                # Could be VBG or NN (e.g., "building")
                # After determiner → noun, otherwise → verb
                if prev_tag in ('DT', 'JJ', 'PRON'):
                    pos_tags.append('NN')
                else:
                    pos_tags.append('VBG')
                prev_tag = pos_tags[-1]
                continue
            
            # Past tense/participle: -ed
            if token_lower.endswith('ed') and len(token_lower) > 3:
                # After "have/has/had" → VBN, otherwise → VBD
                if prev_tag == 'VBZ':  # Approximation for auxiliary
                    pos_tags.append('VBN')
                else:
                    pos_tags.append('VBD')
                prev_tag = pos_tags[-1]
                continue
            
            # Adjectives: common suffixes
            if any(token_lower.endswith(suf) for suf in ['ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ish', 'ic', 'ical']):
                pos_tags.append('JJ')
                prev_tag = 'JJ'
                continue
            
            # Nouns: common suffixes
            if any(token_lower.endswith(suf) for suf in ['tion', 'sion', 'ment', 'ness', 'ity', 'ance', 'ence', 'er', 'or', 'ist', 'ism', 'dom', 'ship', 'hood']):
                pos_tags.append('NN')
                prev_tag = 'NN'
                continue
            
            # Plural nouns: -s/-es (but not verbs ending in -ss)
            if (token_lower.endswith('s') or token_lower.endswith('es')) and len(token_lower) > 2:
                if not token_lower.endswith('ss') and prev_tag in ('DT', 'JJ', 'NN', None):
                    pos_tags.append('NNS')
                    prev_tag = 'NNS'
                    continue
                elif prev_tag in ('PRON', 'NN', 'NNS'):
                    # After subject, likely a verb
                    pos_tags.append('VBZ')
                    prev_tag = 'VBZ'
                    continue
            
            # ─────────────────────────────────────────────────────
            # Context-based fallbacks
            # ─────────────────────────────────────────────────────
            
            # After determiner → likely noun or adjective
            if prev_tag == 'DT':
                # If next word exists and looks like noun, this is adjective
                if i + 1 < len(tokens) and not tokens[i + 1].lower() in self._AUXILIARIES:
                    pos_tags.append('JJ')  # Assume adjective before noun
                else:
                    pos_tags.append('NN')
                prev_tag = pos_tags[-1]
                continue
            
            # After adjective → likely noun
            if prev_tag == 'JJ':
                pos_tags.append('NN')
                prev_tag = 'NN'
                continue
            
            # After modal/auxiliary → likely verb
            if prev_tag in ('MD', 'VBZ'):
                pos_tags.append('VB')
                prev_tag = 'VB'
                continue
            
            # After preposition → likely noun
            if prev_tag == 'IN':
                pos_tags.append('NN')
                prev_tag = 'NN'
                continue
            
            # Default: noun
            pos_tags.append('NN')
            prev_tag = 'NN'
                
        return pos_tags
    
    def _compute_confidence(self, activations: torch.Tensor) -> float:
        """Compute confidence from PMFlow activation strength."""
        energy = float(torch.norm(activations, p=2).item())
        # Normalize to [0, 1]
        return min(1.0, energy / 8.0)
    
    def _retrieve_patterns(
        self,
        query_embedding: torch.Tensor,
        topk: int = 5
    ) -> List[Tuple[SyntacticPattern, float]]:
        """
        Retrieve similar grammatical patterns using BioNN similarity.
        
        This is the key: grammar patterns are retrieved via learned embeddings!
        """
        if not self.patterns:
            return []
        
        query_np = query_embedding.detach().cpu().numpy().flatten()
        query_norm = query_np / (np.linalg.norm(query_np) + 1e-8)
        
        scored_patterns = []
        for pattern in self.patterns.values():
            pattern_np = pattern.embedding.detach().cpu().numpy().flatten()
            pattern_norm = pattern_np / (np.linalg.norm(pattern_np) + 1e-8)
            
            # Cosine similarity
            similarity = float(np.dot(query_norm, pattern_norm))
            
            # Weight by success score
            weighted_score = similarity * pattern.success_score
            
            scored_patterns.append((pattern, weighted_score))
        
        # Sort by score
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return scored_patterns[:topk]
    
    def _classify_intent(self, pos_tags: List[str], tokens: List[str]) -> str:
        """Classify syntactic intent from POS sequence."""
        # Question
        if pos_tags[0] == 'WRB' or any(t.strip().endswith('?') for t in tokens):
            return "question"
        # Exclamation  
        elif any(t.strip().endswith('!') for t in tokens):
            return "exclamation"
        # Coordination (compound)
        elif 'CC' in pos_tags:
            return "compound"
        # Default statement
        else:
            return "statement"
    
    def learn_pattern(
        self,
        tokens: List[str],
        pos_tags: List[str],
        success_feedback: float
    ) -> str:
        """
        Learn a new grammatical pattern from observation.
        
        This is reinforcement learning for grammar!
        
        Args:
            tokens: Example tokens
            pos_tags: POS sequence
            success_feedback: How well this pattern worked (-1.0 to 1.0)
            
        Returns:
            pattern_id of learned pattern
        """
        # Encode POS sequence
        pos_string = " ".join(pos_tags)
        embedding = self.encoder.encode(pos_string)
        
        # Create pattern
        pattern_id = f"syntax_learned_{len(self.patterns)}"
        intent = self._classify_intent(pos_tags, tokens)
        template = self._generalize_template(tokens, pos_tags)
        
        pattern = SyntacticPattern(
            pattern_id=pattern_id,
            pos_sequence=pos_tags,
            embedding=embedding,
            template=template,
            example=" ".join(tokens),
            success_score=0.5 + (success_feedback * 0.3),  # Start near success
            usage_count=0,
            intent=intent
        )
        
        self.patterns[pattern_id] = pattern
        self._save_patterns()
        
        logger.info(f"Learned syntax pattern: {template} (intent: {intent})")
        
        return pattern_id
    
    def update_pattern_success(
        self,
        pattern_id: str,
        feedback: float,
        learning_rate: float = 0.1
    ) -> Optional[SyntaxPlasticityReport]:
        """
        Update pattern success score via reinforcement.
        
        This is how grammar improves over time!
        
        Returns:
            SyntaxPlasticityReport if plasticity was applied, None otherwise
        """
        if pattern_id not in self.patterns:
            return None
            
        pattern = self.patterns[pattern_id]
        pattern.success_score += feedback * learning_rate
        pattern.success_score = np.clip(pattern.success_score, 0.0, 1.0)
        pattern.usage_count += 1
        
        report = None
        
        # Apply plasticity to BioNN if enabled
        if self.config.plasticity_enabled:
            report = self._apply_plasticity(pattern, feedback)
            
            if report is not None:
                self.plasticity_reports.append(report)
                self.total_plasticity_updates += 1
                
                # Auto-save PMFlow state every 10 plasticity updates
                if self.total_plasticity_updates % 10 == 0:
                    self._save_pmflow_state()
        
        return report
    
    def _apply_plasticity(
        self, 
        pattern: SyntacticPattern, 
        feedback: float,
        contrastive_pairs: Optional[List[Tuple[str, str, bool]]] = None
    ) -> Optional[SyntaxPlasticityReport]:
        """
        Apply BioNN plasticity update for this pattern using PMFlow 0.3.0 features.
        
        This implements real neuroplastic learning:
        1. Reinforcement: vectorized_pm_plasticity adjusts centers/mus based on feedback
        2. Contrastive: contrastive_plasticity separates grammatically different patterns
        
        Args:
            pattern: The syntactic pattern to reinforce
            feedback: Feedback signal (-1.0 to 1.0)
            contrastive_pairs: Optional list of (pattern_id, other_pattern_id, is_similar)
                              for contrastive learning
        
        Returns:
            SyntaxPlasticityReport with delta metrics, or None if plasticity unavailable
        """
        if not PMFLOW_PLASTICITY_AVAILABLE or vectorized_pm_plasticity is None:
            logger.debug("Plasticity not available, skipping BioNN update")
            return None
        
        # Get the PMFlow field from encoder
        pm_field = self.encoder.pm_field
        
        # Handle MultiScalePMField (has fine_field and coarse_field)
        if hasattr(pm_field, 'fine_field'):
            target_field = pm_field.fine_field  # Use fine field for syntax
        else:
            target_field = pm_field
        
        device = target_field.centers.device
        
        # Re-encode the pattern to get latent and refined embeddings
        pos_string = " ".join(pattern.pos_sequence)
        _, latent_cpu, refined_cpu = self.encoder.encode_with_components(pos_string)
        
        latent = latent_cpu.to(device)
        refined = refined_cpu.to(device)
        
        # Ensure batch dimension
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        if refined.dim() == 1:
            refined = refined.unsqueeze(0)
        
        # Record state before update
        before_centers = target_field.centers.detach().clone()
        before_mus = target_field.mus.detach().clone()
        
        # Determine learning rates based on feedback sign and magnitude
        # Positive feedback → larger LR to reinforce
        # Negative feedback → smaller LR to gently correct
        base_mu_lr = self.config.plasticity_lr
        base_c_lr = self.config.plasticity_lr
        
        if feedback > 0:
            # Reinforce good patterns more strongly
            mu_lr = base_mu_lr * (1.0 + feedback)
            c_lr = base_c_lr * (1.0 + feedback)
        else:
            # Gentle correction for bad patterns
            mu_lr = base_mu_lr * 0.5
            c_lr = base_c_lr * 0.3
        
        # Apply vectorized plasticity (Hebbian-style update)
        # This adjusts centers toward the pattern and modifies mus based on activity
        vectorized_pm_plasticity(
            target_field, 
            latent, 
            refined, 
            mu_lr=mu_lr, 
            c_lr=c_lr
        )
        
        plasticity_type = "reinforcement"
        
        # Apply contrastive plasticity if available and pairs provided
        if contrastive_plasticity is not None and contrastive_pairs:
            similar_pairs = []
            dissimilar_pairs = []
            
            for pattern_id_a, pattern_id_b, is_similar in contrastive_pairs:
                if pattern_id_a not in self.patterns or pattern_id_b not in self.patterns:
                    continue
                
                pattern_a = self.patterns[pattern_id_a]
                pattern_b = self.patterns[pattern_id_b]
                
                # Use latent embeddings for contrastive plasticity (PMFlow operates in latent space)
                pos_a = " ".join(pattern_a.pos_sequence)
                pos_b = " ".join(pattern_b.pos_sequence)
                _, latent_a, _ = self.encoder.encode_with_components(pos_a)
                _, latent_b, _ = self.encoder.encode_with_components(pos_b)
                
                latent_a = latent_a.to(device)
                latent_b = latent_b.to(device)
                
                if is_similar:
                    similar_pairs.append((latent_a, latent_b))
                else:
                    dissimilar_pairs.append((latent_a, latent_b))
            
            if similar_pairs or dissimilar_pairs:
                contrastive_plasticity(
                    target_field,
                    similar_pairs=similar_pairs,
                    dissimilar_pairs=dissimilar_pairs,
                    mu_lr=mu_lr * 0.5,  # Gentler for contrastive
                    c_lr=c_lr * 0.5,
                    margin=1.0
                )
                plasticity_type = "reinforcement+contrastive"
        
        # Compute deltas
        delta_centers = torch.norm(target_field.centers - before_centers, p=2).item()
        delta_mus = torch.norm(target_field.mus - before_mus, p=2).item()
        
        report = SyntaxPlasticityReport(
            feedback=feedback,
            threshold=0.5,  # Not used for syntax (always apply)
            delta_centers=delta_centers,
            delta_mus=delta_mus,
            pattern_id=pattern.pattern_id,
            plasticity_type=plasticity_type,
        )
        
        logger.debug(
            "Applied syntax plasticity: pattern=%s feedback=%.2f Δcenters=%.4f Δmus=%.4f type=%s",
            pattern.pattern_id,
            feedback,
            delta_centers,
            delta_mus,
            plasticity_type,
        )
        
        return report
    
    def apply_contrastive_learning(
        self,
        similar_intents: Optional[List[Tuple[str, str]]] = None,
        dissimilar_intents: Optional[List[Tuple[str, str]]] = None,
    ) -> Optional[SyntaxPlasticityReport]:
        """
        Apply contrastive plasticity to improve grammatical structure separation.
        
        This teaches the BioNN that:
        - Questions ("WRB VBZ PRON") should be similar to other questions
        - Questions should be dissimilar from statements ("PRON VBZ ADJ")
        
        Args:
            similar_intents: List of (intent_a, intent_b) that should be close
            dissimilar_intents: List of (intent_a, intent_b) that should be far
            
        Returns:
            SyntaxPlasticityReport or None
        """
        if contrastive_plasticity is None:
            logger.debug("Contrastive plasticity not available")
            return None
        
        # Default: questions similar to questions, statements similar to statements
        if similar_intents is None:
            similar_intents = [("question", "question"), ("statement", "statement")]
        if dissimilar_intents is None:
            dissimilar_intents = [("question", "statement")]
        
        # Group patterns by intent
        by_intent: Dict[str, List[SyntacticPattern]] = {}
        for pattern in self.patterns.values():
            if pattern.intent not in by_intent:
                by_intent[pattern.intent] = []
            by_intent[pattern.intent].append(pattern)
        
        # Build contrastive pairs using LATENT embeddings (not final embeddings)
        # Contrastive plasticity operates in PMFlow latent space, not output space
        similar_pairs = []
        dissimilar_pairs = []
        
        pm_field = self.encoder.pm_field
        if hasattr(pm_field, 'fine_field'):
            target_field = pm_field.fine_field
        else:
            target_field = pm_field
        device = target_field.centers.device
        
        # Helper to get latent embedding for a pattern
        def get_latent(pattern: SyntacticPattern) -> torch.Tensor:
            pos_string = " ".join(pattern.pos_sequence)
            _, latent, _ = self.encoder.encode_with_components(pos_string)
            return latent.to(device)
        
        # Similar pairs: patterns with same intent type
        for intent_a, intent_b in similar_intents:
            patterns_a = by_intent.get(intent_a, [])
            patterns_b = by_intent.get(intent_b, [])
            
            for p_a in patterns_a[:3]:  # Limit to avoid explosion
                for p_b in patterns_b[:3]:
                    if p_a.pattern_id != p_b.pattern_id:
                        latent_a = get_latent(p_a)
                        latent_b = get_latent(p_b)
                        similar_pairs.append((latent_a, latent_b))
        
        # Dissimilar pairs: patterns with different intent types
        for intent_a, intent_b in dissimilar_intents:
            patterns_a = by_intent.get(intent_a, [])
            patterns_b = by_intent.get(intent_b, [])
            
            for p_a in patterns_a[:3]:
                for p_b in patterns_b[:3]:
                    latent_a = get_latent(p_a)
                    latent_b = get_latent(p_b)
                    dissimilar_pairs.append((latent_a, latent_b))
        
        if not similar_pairs and not dissimilar_pairs:
            logger.debug("No contrastive pairs found")
            return None
        
        before_centers = target_field.centers.detach().clone()
        before_mus = target_field.mus.detach().clone()
        
        contrastive_plasticity(
            target_field,
            similar_pairs=similar_pairs,
            dissimilar_pairs=dissimilar_pairs,
            mu_lr=self.config.plasticity_lr * 0.5,
            c_lr=self.config.plasticity_lr * 0.5,
            margin=1.0
        )
        
        delta_centers = torch.norm(target_field.centers - before_centers, p=2).item()
        delta_mus = torch.norm(target_field.mus - before_mus, p=2).item()
        
        report = SyntaxPlasticityReport(
            feedback=0.0,
            threshold=0.0,
            delta_centers=delta_centers,
            delta_mus=delta_mus,
            pattern_id="contrastive_batch",
            plasticity_type="contrastive",
        )
        
        logger.info(
            "Applied contrastive syntax plasticity: %d similar, %d dissimilar pairs, "
            "Δcenters=%.4f Δmus=%.4f",
            len(similar_pairs),
            len(dissimilar_pairs),
            delta_centers,
            delta_mus,
        )
        
        return report

    def check_and_correct(self, text: str) -> str:
        """
        Grammar refinement pass - check adapted text for grammatical issues.
        
        This is used AFTER pattern adaptation to fix grammatical errors
        while preserving the contextual meaning.
        
        Args:
            text: Adapted response text to refine
            
        Returns:
            Grammatically refined text
        """
        # Simple grammar fixes for common errors
        refined = text
        
        # Fix common word order issues
        # "discuss think" → "think about", "discuss you?" → "discuss that?"
        refined = refined.replace("discuss think", "think about")
        refined = refined.replace("discuss what", "discuss that")
        refined = refined.replace("discuss you?", "discuss that?")
        refined = refined.replace("discuss it?", "discuss that?")
        
        # Fix interrogative + pronoun combinations
        refined = refined.replace(" you? with", " that with")
        refined = refined.replace(" it? with", " that with")
        
        # Fix verb agreement
        # "I is" → "I am", "they is" → "they are"
        refined = refined.replace(" I is ", " I am ")
        refined = refined.replace(" they is ", " they are ")
        refined = refined.replace(" we is ", " we are ")
        refined = refined.replace(" you is ", " you are ")
        
        # Fix double words
        import re
        refined = re.sub(r'\b(\w+)\s+\1\b', r'\1', refined)
        
        # Fix punctuation in middle of sentence (e.g., "movies?" → "movies")
        refined = re.sub(r'([a-z])\?(?=\s+with|about)', r'\1', refined)
        refined = re.sub(r'([a-z])!(?=\s+with|about)', r'\1', refined)
        
        # Fix spacing before punctuation
        refined = re.sub(r'\s+([?.!,;:])', r'\1', refined)
        refined = re.sub(r'([?.!])\s*([?.!])', r'\1', refined)  # Remove double punctuation
        
        # Capitalize first letter
        if refined and refined[0].islower():
            refined = refined[0].upper() + refined[1:]
        
        # Future: Use BioNN to learn common corrections
        # For now, use rule-based fixes for demonstrable improvement
        
        return refined
    
    def learn_correction(self, incorrect: str, correct: str):
        """
        Learn from grammar corrections for future refinement.
        
        When a correction is made, this can be used to improve
        the check_and_correct() method over time.
        
        Args:
            incorrect: The incorrect text
            correct: The corrected version
        """
        # Future: Store correction patterns and learn via BioNN
        # For now, this is a placeholder for the learning mechanism
        pass
    
    def _generalize_template(self, tokens: List[str], pos_tags: List[str]) -> str:
        """Create generalized template from example."""
        parts = []
        for token, pos in zip(tokens, pos_tags):
            if pos in ['DT', 'PRON', 'CC', 'VBZ']:  # Function words
                parts.append(token.lower())
            else:
                parts.append(f"{{{pos.lower()}}}")
        return " ".join(parts)
    
    def _bootstrap_patterns(self):
        """Bootstrap with basic syntactic patterns using BioNN encoding."""
        basic_patterns = [
            (["PRON", "VBZ"], "it works", "statement"),
            (["PRON", "VBZ", "ADJ"], "that's interesting", "statement"),
            (["WRB", "VBZ", "PRON", "VB"], "how does that work", "question"),
            (["DT", "ADJ", "NN"], "the good point", "phrase"),
        ]
        
        for pos_seq, example, intent in basic_patterns:
            pos_string = " ".join(pos_seq)
            embedding = self.encoder.encode(pos_string)
            
            pattern_id = f"syntax_seed_{len(self.patterns)}"
            template = self._generalize_template(example.split(), pos_seq)
            
            pattern = SyntacticPattern(
                pattern_id=pattern_id,
                pos_sequence=pos_seq,
                embedding=embedding,
                template=template,
                example=example,
                intent=intent
            )
            
            self.patterns[pattern_id] = pattern
    
    def _save_patterns(self):
        """Save learned patterns to storage."""
        # Convert to serializable format
        data = {
            "patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "pos_sequence": p.pos_sequence,
                    "embedding": p.embedding.tolist(),
                    "template": p.template,
                    "example": p.example,
                    "success_score": p.success_score,
                    "usage_count": p.usage_count,
                    "intent": p.intent,
                }
                for p in self.patterns.values()
            ]
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_patterns(self):
        """Load patterns from storage."""
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
        
        for p_data in data.get("patterns", []):
            pattern = SyntacticPattern(
                pattern_id=p_data["pattern_id"],
                pos_sequence=p_data["pos_sequence"],
                embedding=torch.tensor(p_data["embedding"]),
                template=p_data["template"],
                example=p_data["example"],
                success_score=p_data["success_score"],
                usage_count=p_data["usage_count"],
                intent=p_data["intent"],
            )
            self.patterns[pattern.pattern_id] = pattern
    
    def _save_pmflow_state(self):
        """Save PMFlow field state (learned plasticity weights)."""
        pm_field = self.encoder.pm_field
        
        # Handle MultiScalePMField (has fine_field and coarse_field)
        if hasattr(pm_field, 'fine_field') and hasattr(pm_field, 'coarse_field'):
            payload = {
                "type": "multiscale",
                "fine_centers": pm_field.fine_field.centers.detach().cpu(),
                "fine_mus": pm_field.fine_field.mus.detach().cpu(),
                "coarse_centers": pm_field.coarse_field.centers.detach().cpu(),
                "coarse_mus": pm_field.coarse_field.mus.detach().cpu(),
                "coarse_projection": pm_field.coarse_projection.weight.detach().cpu(),
                "total_plasticity_updates": self.total_plasticity_updates,
            }
        else:
            # Standard PMField
            payload = {
                "type": "standard",
                "centers": pm_field.centers.detach().cpu(),
                "mus": pm_field.mus.detach().cpu(),
                "total_plasticity_updates": self.total_plasticity_updates,
            }
        
        self.pmflow_state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self.pmflow_state_path)
        logger.debug(f"Saved syntax PMFlow state to {self.pmflow_state_path}")
    
    def _load_pmflow_state(self):
        """Load PMFlow field state (learned plasticity weights)."""
        if not self.pmflow_state_path.exists():
            return
        
        pm_field = self.encoder.pm_field
        device = pm_field.fine_field.centers.device if hasattr(pm_field, 'fine_field') else pm_field.centers.device
        
        try:
            payload = torch.load(self.pmflow_state_path, map_location=device)
            
            with torch.no_grad():
                # Handle MultiScalePMField
                if payload.get("type") == "multiscale":
                    if hasattr(pm_field, 'fine_field') and hasattr(pm_field, 'coarse_field'):
                        pm_field.fine_field.centers.copy_(payload["fine_centers"].to(device))
                        pm_field.fine_field.mus.copy_(payload["fine_mus"].to(device))
                        pm_field.coarse_field.centers.copy_(payload["coarse_centers"].to(device))
                        pm_field.coarse_field.mus.copy_(payload["coarse_mus"].to(device))
                        pm_field.coarse_projection.weight.copy_(payload["coarse_projection"].to(device))
                # Handle standard PMField (backward compatibility)
                elif "centers" in payload:
                    if hasattr(pm_field, 'centers'):
                        pm_field.centers.copy_(payload["centers"].to(device))
                    if "mus" in payload and hasattr(pm_field, 'mus'):
                        pm_field.mus.copy_(payload["mus"].to(device))
            
            self.total_plasticity_updates = payload.get("total_plasticity_updates", 0)
            logger.info(
                f"Loaded syntax PMFlow state: {self.total_plasticity_updates} prior plasticity updates"
            )
        except Exception as e:
            logger.warning(f"Failed to load PMFlow state: {e}")
    
    def save_state(self):
        """Save all state (patterns and PMFlow weights)."""
        self._save_patterns()
        if PMFLOW_PLASTICITY_AVAILABLE:
            self._save_pmflow_state()
    
    def get_plasticity_stats(self) -> Dict[str, Any]:
        """Get statistics about plasticity updates."""
        if not self.plasticity_reports:
            return {
                "total_updates": self.total_plasticity_updates,
                "recent_updates": 0,
                "avg_delta_centers": 0.0,
                "avg_delta_mus": 0.0,
                "plasticity_available": PMFLOW_PLASTICITY_AVAILABLE,
            }
        
        recent = self.plasticity_reports[-100:]  # Last 100 updates
        return {
            "total_updates": self.total_plasticity_updates,
            "recent_updates": len(recent),
            "avg_delta_centers": np.mean([r.delta_centers for r in recent]),
            "avg_delta_mus": np.mean([r.delta_mus for r in recent]),
            "by_type": {
                t: len([r for r in recent if r.plasticity_type == t])
                for t in set(r.plasticity_type for r in recent)
            },
            "plasticity_available": PMFLOW_PLASTICITY_AVAILABLE,
        }

