"""
Learned Vocabulary Encoder - Trainable Semantic Word Embeddings

The HashedEmbeddingEncoder destroys semantic information because synonyms
hash to completely different indices. This encoder fixes that by:

1. Maintaining trainable word embeddings for known vocabulary
2. Learning semantic relationships through contrastive training
3. Falling back to hashing for truly unknown words
4. Growing vocabulary from experience

This is the missing piece - without trainable base embeddings, 
ContrastiveLearner can only reorganize the PMFlow physics landscape,
it can't give synonyms similar initial vectors.

Design Philosophy:
- Learn from experience: Start with empty/minimal vocab, grow through use
- Gradients flow: Unlike hashing, embeddings are differentiable
- Graceful fallback: Unknown words get hash vectors until learned
- Persistence: Vocabulary and learned embeddings survive restarts
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedVocabularyEncoder(nn.Module):
    """
    Trainable vocabulary encoder with semantic learning capability.
    
    Unlike HashedEmbeddingEncoder which uses non-differentiable hashing,
    this encoder maintains trainable word embeddings that can be optimized
    through contrastive learning to capture semantic relationships.
    
    Key Features:
    - Trainable nn.Embedding for known vocabulary
    - Automatic vocabulary expansion from experience
    - Hash fallback for OOV words
    - Persistent vocabulary and embeddings
    - Compatible with ContrastiveLearner and PMFlow pipeline
    
    Example:
        encoder = LearnedVocabularyEncoder(dimension=64)
        encoder.add_words(["cat", "dog", "animal"])  # Register vocabulary
        emb = encoder.encode("the cat is an animal")  # Get embedding
        
        # Train with contrastive learning
        learner = ContrastiveLearner(pmflow_encoder)  # Uses this as base
        learner.add_pair("cat", "dog", "positive")
        learner.train()  # Gradients flow to word embeddings!
    """
    
    def __init__(
        self,
        *,
        dimension: int = 64,
        max_vocab_size: int = 50000,
        device: Optional[torch.device] = None,
        pad_idx: int = 0,
        unk_idx: int = 1,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.max_vocab_size = max_vocab_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        
        # Vocabulary mapping: word -> index
        self.word_to_idx: Dict[str, int] = {
            "<PAD>": pad_idx,
            "<UNK>": unk_idx,
        }
        self.idx_to_word: Dict[int, str] = {
            pad_idx: "<PAD>",
            unk_idx: "<UNK>",
        }
        self.next_idx = 2  # Next available index
        
        # Trainable embedding layer - will be resized as vocab grows
        # Start with small initial size
        initial_size = 1000
        self.embedding = nn.Embedding(
            num_embeddings=initial_size,
            embedding_dim=dimension,
            padding_idx=pad_idx,
        )
        self._embedding_size = initial_size
        
        # Initialize with small random values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        # Keep PAD as zeros
        with torch.no_grad():
            self.embedding.weight[pad_idx].zero_()
        
        # Track word frequency for importance weighting
        self.word_counts: Dict[str, int] = {}
        
        # For hash fallback: cache computed hash vectors
        self._hash_cache: Dict[str, torch.Tensor] = {}
        
        self.to(self.device)
    
    def _expand_embedding(self, new_size: int) -> None:
        """Expand embedding layer to accommodate more vocabulary."""
        if new_size <= self._embedding_size:
            return
        
        # Double size to avoid frequent resizing
        new_size = max(new_size, self._embedding_size * 2)
        new_size = min(new_size, self.max_vocab_size)
        
        # Create new larger embedding
        old_weight = self.embedding.weight.data.clone()
        old_size = self._embedding_size
        
        self.embedding = nn.Embedding(
            num_embeddings=new_size,
            embedding_dim=self.dimension,
            padding_idx=self.pad_idx,
        ).to(self.device)
        
        # Copy old weights
        with torch.no_grad():
            self.embedding.weight[:old_size].copy_(old_weight)
            # Initialize new weights
            nn.init.normal_(self.embedding.weight[old_size:], mean=0.0, std=0.1)
        
        self._embedding_size = new_size
    
    def add_word(self, word: str) -> int:
        """
        Add a word to vocabulary and return its index.
        
        If word already exists, returns existing index.
        If vocabulary is full, returns UNK index.
        """
        word = word.lower().strip()
        
        if word in self.word_to_idx:
            return self.word_to_idx[word]
        
        if self.next_idx >= self.max_vocab_size:
            return self.unk_idx
        
        # Expand if needed
        if self.next_idx >= self._embedding_size:
            self._expand_embedding(self.next_idx + 1)
        
        idx = self.next_idx
        self.word_to_idx[word] = idx
        self.idx_to_word[idx] = word
        self.next_idx += 1
        
        return idx
    
    def add_words(self, words: Iterable[str]) -> List[int]:
        """Add multiple words to vocabulary."""
        return [self.add_word(w) for w in words]
    
    def has_word(self, word: str) -> bool:
        """Check if word is in learned vocabulary."""
        return word.lower().strip() in self.word_to_idx
    
    def vocab_size(self) -> int:
        """Current vocabulary size (excluding special tokens)."""
        return self.next_idx - 2  # Exclude PAD and UNK
    
    def _hash_word(self, word: str) -> torch.Tensor:
        """
        Hash fallback for unknown words.
        
        Creates a deterministic pseudo-embedding from the word's hash.
        This is similar to HashedEmbeddingEncoder but creates a dense
        vector rather than sparse one-hot buckets.
        """
        if word in self._hash_cache:
            return self._hash_cache[word].to(self.device)
        
        # Use SHA256 for deterministic bytes
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        
        # Convert to float vector
        # Use enough bytes for dimension (4 bytes per float)
        num_bytes = min(len(digest), self.dimension * 4)
        values = []
        for i in range(0, num_bytes, 4):
            # Convert 4 bytes to float in [-1, 1] range
            val = int.from_bytes(digest[i:i+4], "big")
            val = (val / (2**32 - 1)) * 2 - 1  # Normalize to [-1, 1]
            values.append(val)
        
        # Pad or truncate to dimension
        while len(values) < self.dimension:
            values.append(0.0)
        values = values[:self.dimension]
        
        # Normalize
        vec = torch.tensor(values, dtype=torch.float32)
        vec = F.normalize(vec, dim=0)
        
        self._hash_cache[word] = vec
        return vec.to(self.device)
    
    def get_word_embedding(self, word: str) -> torch.Tensor:
        """
        Get embedding for a single word.
        
        Uses learned embedding if word is in vocabulary,
        otherwise falls back to hash embedding.
        """
        word = word.lower().strip()
        
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            return self.embedding.weight[idx]
        else:
            # Hash fallback
            return self._hash_word(word)
    
    def encode(self, tokens: Iterable[str] | str) -> torch.Tensor:
        """
        Encode text or tokens to a single embedding vector.
        
        For multi-word input, averages word embeddings (bag of words).
        This is compatible with the HashedEmbeddingEncoder interface.
        
        Args:
            tokens: Text string or iterable of tokens
            
        Returns:
            Tensor of shape (1, dimension) - normalized embedding
        """
        if isinstance(tokens, str):
            tokens = tokens.lower().split()
        else:
            tokens = [t.lower().strip() for t in tokens]
        
        if not tokens:
            # Empty input -> zero vector
            return torch.zeros(1, self.dimension, device=self.device)
        
        # Track word usage
        for token in tokens:
            self.word_counts[token] = self.word_counts.get(token, 0) + 1
        
        # Get embeddings for each token
        embeddings = []
        for token in tokens:
            emb = self.get_word_embedding(token)
            embeddings.append(emb)
        
        # Average pool (bag of words)
        stacked = torch.stack(embeddings)
        pooled = stacked.mean(dim=0, keepdim=True)
        
        # Normalize
        normalized = F.normalize(pooled, p=2, dim=1)
        
        return normalized
    
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode multiple texts, returning (batch_size, dimension)."""
        embeddings = [self.encode(text) for text in texts]
        return torch.cat(embeddings, dim=0)
    
    def auto_add_from_text(self, text: str, min_freq: int = 2) -> Set[str]:
        """
        Automatically add words from text to vocabulary.
        
        Words must appear at least min_freq times before being added.
        This prevents one-time typos from polluting vocabulary.
        
        Args:
            text: Text to process
            min_freq: Minimum frequency before adding to vocab
            
        Returns:
            Set of newly added words
        """
        tokens = text.lower().split()
        added = set()
        
        for token in tokens:
            # Skip very short tokens
            if len(token) < 2:
                continue
            
            # Track frequency
            self.word_counts[token] = self.word_counts.get(token, 0) + 1
            
            # Add if frequent enough and not already in vocab
            if (self.word_counts[token] >= min_freq 
                and token not in self.word_to_idx):
                self.add_word(token)
                added.add(token)
        
        return added
    
    def get_similar_words(
        self, 
        word: str, 
        k: int = 5,
        threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Find words with similar embeddings.
        
        Args:
            word: Query word
            k: Maximum number of results
            threshold: Minimum similarity score
            
        Returns:
            List of (word, similarity) tuples, sorted by similarity
        """
        if not self.has_word(word):
            return []
        
        query_emb = self.get_word_embedding(word).unsqueeze(0)
        
        # Get all learned embeddings
        all_embeddings = self.embedding.weight[:self.next_idx]
        
        # Compute cosine similarity
        similarities = F.cosine_similarity(
            query_emb.expand(self.next_idx, -1),
            all_embeddings,
            dim=1
        )
        
        # Get top-k (excluding the word itself and special tokens)
        results = []
        for idx in range(2, self.next_idx):  # Skip PAD and UNK
            word_i = self.idx_to_word[idx]
            if word_i == word:
                continue
            sim = similarities[idx].item()
            if sim >= threshold:
                results.append((word_i, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    # ─────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────
    
    def save(self, path: Path) -> None:
        """Save vocabulary and embeddings to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary as JSON
        vocab_data = {
            "word_to_idx": self.word_to_idx,
            "idx_to_word": {str(k): v for k, v in self.idx_to_word.items()},
            "next_idx": self.next_idx,
            "dimension": self.dimension,
            "word_counts": self.word_counts,
        }
        with open(path.with_suffix(".vocab.json"), "w") as f:
            json.dump(vocab_data, f, indent=2)
        
        # Save embeddings as torch state
        torch.save({
            "embedding_weight": self.embedding.weight.detach().cpu(),
            "embedding_size": self._embedding_size,
        }, path.with_suffix(".embeddings.pt"))
    
    def load(self, path: Path) -> None:
        """Load vocabulary and embeddings from disk."""
        path = Path(path)
        
        # Load vocabulary
        vocab_path = path.with_suffix(".vocab.json")
        if vocab_path.exists():
            with open(vocab_path) as f:
                vocab_data = json.load(f)
            
            self.word_to_idx = vocab_data["word_to_idx"]
            self.idx_to_word = {int(k): v for k, v in vocab_data["idx_to_word"].items()}
            self.next_idx = vocab_data["next_idx"]
            self.word_counts = vocab_data.get("word_counts", {})
            
            # Verify dimension matches
            if vocab_data.get("dimension") != self.dimension:
                raise ValueError(
                    f"Saved dimension {vocab_data.get('dimension')} "
                    f"doesn't match {self.dimension}"
                )
        
        # Load embeddings
        emb_path = path.with_suffix(".embeddings.pt")
        if emb_path.exists():
            state = torch.load(emb_path, map_location=self.device)
            saved_weight = state["embedding_weight"]
            saved_size = state["embedding_size"]
            
            # Resize if needed
            if saved_size != self._embedding_size:
                self._expand_embedding(saved_size)
            
            with torch.no_grad():
                self.embedding.weight.copy_(saved_weight.to(self.device))
    
    # ─────────────────────────────────────────────────────────────
    # Integration with ContrastiveLearner
    # ─────────────────────────────────────────────────────────────
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get parameters for optimizer (used by ContrastiveLearner)."""
        return [self.embedding.weight]
    
    def compute_word_pair_loss(
        self,
        word_a: str,
        word_b: str,
        should_be_similar: bool,
        margin: float = 0.3,
    ) -> torch.Tensor:
        """
        Compute contrastive loss for a word pair.
        
        This can be used for direct word-level training,
        separate from the full ContrastiveLearner pipeline.
        
        Args:
            word_a: First word
            word_b: Second word
            should_be_similar: True if positive pair, False if negative
            margin: Margin for hinge loss on negative pairs
            
        Returns:
            Scalar loss tensor
        """
        # Get embeddings
        emb_a = self.get_word_embedding(word_a).unsqueeze(0)
        emb_b = self.get_word_embedding(word_b).unsqueeze(0)
        
        # Cosine similarity
        sim = F.cosine_similarity(emb_a, emb_b)
        
        if should_be_similar:
            # Pull together: minimize (1 - similarity)
            loss = 1.0 - sim
        else:
            # Push apart: penalize if similarity > -margin
            loss = F.relu(sim + margin)
        
        return loss.mean()
    
    def bootstrap_core_semantics(
        self,
        learning_rate: float = 0.01,
        epochs: int = 10,
    ) -> Dict[str, float]:
        """
        Bootstrap core semantic relationships through direct training.
        
        This teaches basic word relationships before any user interaction.
        Can be called during initialization for a semantic head start.
        
        Returns:
            Dict with training metrics
        """
        # Core semantic pairs
        positive_pairs = [
            # Synonyms
            ("big", "large"), ("small", "tiny"), ("fast", "quick"),
            ("happy", "joyful"), ("sad", "unhappy"), ("angry", "mad"),
            ("smart", "intelligent"), ("dumb", "stupid"),
            ("beautiful", "pretty"), ("ugly", "hideous"),
            
            # Related concepts
            ("cat", "dog"), ("cat", "kitten"), ("dog", "puppy"),
            ("car", "vehicle"), ("bike", "bicycle"),
            ("computer", "laptop"), ("phone", "smartphone"),
            ("eat", "food"), ("drink", "water"),
            ("run", "walk"), ("jump", "leap"),
            
            # Actions with similar purposes
            ("search", "find"), ("look", "search"), ("seek", "find"),
            ("ask", "question"), ("answer", "reply"), ("say", "speak"),
            ("start", "begin"), ("stop", "end"), ("finish", "complete"),
            ("make", "create"), ("build", "construct"),
            ("help", "assist"), ("fix", "repair"),
        ]
        
        negative_pairs = [
            # Antonyms
            ("hot", "cold"), ("big", "small"), ("fast", "slow"),
            ("happy", "sad"), ("good", "bad"), ("light", "dark"),
            ("up", "down"), ("left", "right"), ("yes", "no"),
            ("true", "false"), ("old", "young"), ("new", "old"),
            ("love", "hate"), ("win", "lose"),
            
            # Unrelated
            ("cat", "computer"), ("dog", "algorithm"),
            ("happy", "keyboard"), ("search", "banana"),
        ]
        
        # Add all words to vocabulary
        all_words = set()
        for a, b in positive_pairs + negative_pairs:
            all_words.add(a)
            all_words.add(b)
        self.add_words(all_words)
        
        # Training
        optimizer = torch.optim.AdamW(
            [self.embedding.weight],
            lr=learning_rate,
        )
        
        total_loss = 0.0
        num_updates = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Train on positive pairs
            for a, b in positive_pairs:
                optimizer.zero_grad()
                loss = self.compute_word_pair_loss(a, b, should_be_similar=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_updates += 1
            
            # Train on negative pairs
            for a, b in negative_pairs:
                optimizer.zero_grad()
                loss = self.compute_word_pair_loss(a, b, should_be_similar=False)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_updates += 1
            
            total_loss += epoch_loss
        
        # Measure learned semantics
        avg_pos_sim = 0.0
        for a, b in positive_pairs[:10]:
            emb_a = self.get_word_embedding(a)
            emb_b = self.get_word_embedding(b)
            avg_pos_sim += F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
        avg_pos_sim /= 10
        
        avg_neg_sim = 0.0
        for a, b in negative_pairs[:10]:
            emb_a = self.get_word_embedding(a)
            emb_b = self.get_word_embedding(b)
            avg_neg_sim += F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
        avg_neg_sim /= 10
        
        return {
            "avg_loss": total_loss / num_updates if num_updates > 0 else 0.0,
            "vocab_size": self.vocab_size(),
            "avg_positive_similarity": avg_pos_sim,
            "avg_negative_similarity": avg_neg_sim,
            "margin": avg_pos_sim - avg_neg_sim,
        }
    
    def __repr__(self) -> str:
        return (
            f"LearnedVocabularyEncoder("
            f"dim={self.dimension}, "
            f"vocab={self.vocab_size()}, "
            f"device={self.device})"
        )


class SemanticPMFlowEncoder:
    """
    PMFlow encoder with learned vocabulary base.
    
    This combines:
    1. LearnedVocabularyEncoder: Trainable word embeddings (semantic)
    2. PMFlow physics: Trajectory tracing and intent injection
    
    Unlike the standard PMFlowEmbeddingEncoder which uses hashing,
    this encoder's base embeddings are trainable, enabling true
    semantic learning through contrastive training.
    
    This is a drop-in replacement for PMFlowEmbeddingEncoder with
    the same interface, for use with ActionPlanner and CognitiveStage.
    """
    
    def __init__(
        self,
        *,
        dimension: int = 96,
        latent_dim: int = 48,
        seed: int = 13,
        combine_mode: str = "concat",
        device: Optional[torch.device] = None,
        enable_flow: bool = True,
        vocab_path: Optional[Path] = None,
        bootstrap_semantics: bool = True,
    ) -> None:
        if combine_mode not in {"concat", "pm-only"}:
            raise ValueError("combine_mode must be 'concat' or 'pm-only'")
        
        self.combine_mode = combine_mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.enable_flow = enable_flow
        
        # Use learned vocabulary instead of hashing
        self.base_encoder = LearnedVocabularyEncoder(
            dimension=dimension,
            device=self.device,
        )
        self.dimension = self.base_encoder.dimension
        
        # Bootstrap core semantics
        if bootstrap_semantics:
            metrics = self.base_encoder.bootstrap_core_semantics(epochs=10)
            print(f"Bootstrapped semantics: vocab={metrics['vocab_size']}, "
                  f"margin={metrics['margin']:.3f}")
        
        # Load saved vocabulary if provided
        if vocab_path and Path(vocab_path).with_suffix(".vocab.json").exists():
            self.base_encoder.load(vocab_path)
        
        # Projection matrix (trainable)
        self._projection = self._build_projection_matrix(
            self.dimension, latent_dim, seed
        ).to(self.device)
        self._projection.requires_grad_(True)
        
        # PMFlow field (same as PMFlowEmbeddingEncoder)
        self.pm_field = self._init_pm_field(latent_dim, seed, enable_flow)
        self.pm_field.to(self.device)
        self.pm_field.eval()
        
        self._state_path: Optional[Path] = None
    
    @staticmethod
    def _build_projection_matrix(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        matrix = rng.standard_normal((input_dim, output_dim), dtype=np.float32)
        return torch.from_numpy(matrix)
    
    @staticmethod
    def _init_pm_field(latent_dim: int, seed: int, enable_flow: bool = False):
        """Create a deterministic PMFlow field."""
        from pmflow.core.pmflow import MultiScalePMField, ParallelPMField

        try:
            field = MultiScalePMField(
                d_latent=latent_dim,
                n_centers_fine=128,
                n_centers_coarse=32,
                steps_fine=5,
                steps_coarse=3,
                dt=0.15,
                beta=1.2,
                clamp=3.0,
                enable_flow=enable_flow,
            )
            
            generator = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                # Initialize fine field
                centres_fine = torch.randn(
                    field.fine_field.centers.shape,
                    generator=generator,
                    device=field.fine_field.centers.device,
                ) * 0.5
                mus_fine = torch.full(
                    field.fine_field.mus.shape,
                    0.35,
                    device=field.fine_field.mus.device,
                )
                field.fine_field.centers.copy_(centres_fine)
                field.fine_field.mus.copy_(mus_fine)
                
                if hasattr(field.fine_field, 'omegas'):
                    omegas_fine = torch.randn(
                        field.fine_field.omegas.shape,
                        generator=generator,
                        device=field.fine_field.omegas.device,
                    ) * 0.01
                    field.fine_field.omegas.copy_(omegas_fine)
                
                # Initialize coarse field
                centres_coarse = torch.randn(
                    field.coarse_field.centers.shape,
                    generator=generator,
                    device=field.coarse_field.centers.device,
                ) * 0.5
                mus_coarse = torch.full(
                    field.coarse_field.mus.shape,
                    0.35,
                    device=field.coarse_field.mus.device,
                )
                field.coarse_field.centers.copy_(centres_coarse)
                field.coarse_field.mus.copy_(mus_coarse)
                
                if hasattr(field.coarse_field, 'omegas'):
                    omegas_coarse = torch.randn(
                        field.coarse_field.omegas.shape,
                        generator=generator,
                        device=field.coarse_field.omegas.device,
                    ) * 0.01
                    field.coarse_field.omegas.copy_(omegas_coarse)
            
            return field
            
        except Exception:
            # Fallback to parallel field
            field = ParallelPMField(
                d_latent=latent_dim, 
                steps=5, 
                dt=0.08, 
                beta=0.9, 
                clamp=2.5,
                enable_flow=enable_flow,
            )
            generator = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                centres = torch.randn(
                    field.centers.shape,
                    generator=generator,
                    device=field.centers.device,
                ) * 0.5
                mus = torch.full(field.mus.shape, 0.35, device=field.mus.device)
                field.centers.copy_(centres)
                field.mus.copy_(mus)
                
                if hasattr(field, 'omegas'):
                    omegas = torch.randn(
                        field.omegas.shape,
                        generator=generator,
                        device=field.omegas.device,
                    ) * 0.01
                    field.omegas.copy_(omegas)
            return field
    
    def encode(self, tokens: Iterable[str] | str) -> torch.Tensor:
        """Encode text to embedding (compatible with PMFlowEmbeddingEncoder)."""
        combined, _, _ = self._encode_internal(tokens)
        return combined
    
    def encode_with_components(
        self, 
        tokens: Iterable[str] | str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return embedding with PMFlow latent and raw activations."""
        return self._encode_internal(tokens)
    
    def _encode_internal(
        self, 
        tokens: Iterable[str] | str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Internal encoding with all components."""
        # Get base embedding from learned vocabulary
        base = self.base_encoder.encode(tokens)
        
        # Project to latent dim
        latent = base @ self._projection
        
        # Pass through PMFlow
        pm_output = self.pm_field(latent)
        if isinstance(pm_output, tuple) and len(pm_output) == 3:
            raw_refined = pm_output[2]
        else:
            raw_refined = pm_output
        
        refined = F.normalize(raw_refined, p=2, dim=1)
        
        if self.combine_mode == "concat":
            hashed = F.normalize(base, p=2, dim=1)
            combined = torch.cat([hashed, refined], dim=1)
        else:
            combined = refined
        
        return combined.cpu(), latent.detach().cpu(), raw_refined.detach().cpu()
    
    # ─────────────────────────────────────────────────────────────
    # Agentic Physics API (pass-through to pm_field if available)
    # ─────────────────────────────────────────────────────────────
    
    def inject_intent(self, tokens: Iterable[str] | str, strength: float = 0.5) -> None:
        """Inject goal intent for trajectory tracing."""
        if not hasattr(self.pm_field, 'inject_omega_intent'):
            return
        
        base = self.base_encoder.encode(tokens)
        latent = base @ self._projection
        
        self.pm_field.inject_omega_intent(latent.squeeze(0), strength)
    
    def clear_intent(self) -> None:
        """Clear injected intent."""
        if hasattr(self.pm_field, 'clear_omega_intent'):
            self.pm_field.clear_omega_intent()
    
    def trace_trajectory(
        self,
        tokens: Iterable[str] | str,
        steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Trace trajectory through concept space."""
        base = self.base_encoder.encode(tokens)
        latent = base @ self._projection
        
        if hasattr(self.pm_field, 'forward_with_trajectory'):
            _, trajectory = self.pm_field.forward_with_trajectory(latent, steps=steps)
            
            # Compute metrics
            if trajectory.shape[1] > 1:
                diffs = trajectory[:, 1:] - trajectory[:, :-1]
                path_length = torch.norm(diffs, dim=-1).sum().item()
                displacement = torch.norm(trajectory[:, -1] - trajectory[:, 0]).item()
                efficiency = displacement / max(path_length, 1e-6)
            else:
                path_length = 0.0
                displacement = 0.0
                efficiency = 1.0
            
            return trajectory, {
                'path_length': path_length,
                'displacement': displacement,
                'efficiency': efficiency,
            }
        else:
            # Fallback: just return the embedding as a single-point trajectory
            output = self.pm_field(latent)
            if isinstance(output, tuple):
                output = output[2]
            return output.unsqueeze(1), {'path_length': 0.0, 'displacement': 0.0, 'efficiency': 1.0}
    
    # ─────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────
    
    def attach_state_path(self, path: Optional[Path]) -> None:
        """Set path for automatic state persistence."""
        self._state_path = path
        if path and Path(path).with_suffix(".vocab.json").exists():
            self.load_state(path)
    
    def save_state(self, path: Optional[Path] = None) -> None:
        """Save all encoder state."""
        path = path or self._state_path
        if path is None:
            return
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary encoder
        self.base_encoder.save(path)
        
        # Save projection matrix
        torch.save({
            "projection": self._projection.detach().cpu(),
        }, path.with_suffix(".projection.pt"))
        
        # Save PMFlow field
        if hasattr(self.pm_field, 'fine_field'):
            payload = {
                "type": "multiscale",
                "fine_centers": self.pm_field.fine_field.centers.detach().cpu(),
                "fine_mus": self.pm_field.fine_field.mus.detach().cpu(),
                "coarse_centers": self.pm_field.coarse_field.centers.detach().cpu(),
                "coarse_mus": self.pm_field.coarse_field.mus.detach().cpu(),
                "coarse_projection": self.pm_field.coarse_projection.weight.detach().cpu(),
            }
            if hasattr(self.pm_field.fine_field, 'omegas'):
                payload["fine_omegas"] = self.pm_field.fine_field.omegas.detach().cpu()
            if hasattr(self.pm_field.coarse_field, 'omegas'):
                payload["coarse_omegas"] = self.pm_field.coarse_field.omegas.detach().cpu()
        else:
            payload = {
                "type": "standard",
                "centers": self.pm_field.centers.detach().cpu(),
                "mus": self.pm_field.mus.detach().cpu(),
            }
            if hasattr(self.pm_field, 'omegas'):
                payload["omegas"] = self.pm_field.omegas.detach().cpu()
        
        torch.save(payload, path.with_suffix(".pmflow.pt"))
    
    def load_state(self, path: Optional[Path] = None) -> None:
        """Load all encoder state."""
        path = path or self._state_path
        if path is None:
            return
        
        path = Path(path)
        
        # Load vocabulary encoder
        if path.with_suffix(".vocab.json").exists():
            self.base_encoder.load(path)
        
        # Load projection matrix
        proj_path = path.with_suffix(".projection.pt")
        if proj_path.exists():
            state = torch.load(proj_path, map_location=self.device)
            with torch.no_grad():
                self._projection.copy_(state["projection"].to(self.device))
        
        # Load PMFlow field
        pmflow_path = path.with_suffix(".pmflow.pt")
        if pmflow_path.exists():
            payload = torch.load(pmflow_path, map_location=self.device)
            
            with torch.no_grad():
                if payload.get("type") == "multiscale":
                    if hasattr(self.pm_field, 'fine_field'):
                        self.pm_field.fine_field.centers.copy_(
                            payload["fine_centers"].to(self.device))
                        self.pm_field.fine_field.mus.copy_(
                            payload["fine_mus"].to(self.device))
                        self.pm_field.coarse_field.centers.copy_(
                            payload["coarse_centers"].to(self.device))
                        self.pm_field.coarse_field.mus.copy_(
                            payload["coarse_mus"].to(self.device))
                        self.pm_field.coarse_projection.weight.copy_(
                            payload["coarse_projection"].to(self.device))
                        
                        if "fine_omegas" in payload and hasattr(self.pm_field.fine_field, 'omegas'):
                            self.pm_field.fine_field.omegas.copy_(
                                payload["fine_omegas"].to(self.device))
                        if "coarse_omegas" in payload and hasattr(self.pm_field.coarse_field, 'omegas'):
                            self.pm_field.coarse_field.omegas.copy_(
                                payload["coarse_omegas"].to(self.device))
                else:
                    if hasattr(self.pm_field, 'centers'):
                        self.pm_field.centers.copy_(payload["centers"].to(self.device))
                    if hasattr(self.pm_field, 'mus'):
                        self.pm_field.mus.copy_(payload["mus"].to(self.device))
                    if "omegas" in payload and hasattr(self.pm_field, 'omegas'):
                        self.pm_field.omegas.copy_(payload["omegas"].to(self.device))
    
    # ─────────────────────────────────────────────────────────────
    # ContrastiveLearner integration
    # ─────────────────────────────────────────────────────────────
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get all trainable parameters for ContrastiveLearner.
        
        Includes:
        - Word embeddings (base_encoder)
        - Projection matrix
        - PMFlow field parameters
        """
        params = list(self.base_encoder.get_trainable_parameters())
        params.append(self._projection)
        
        # PMFlow parameters
        if hasattr(self.pm_field, 'fine_field'):
            params.extend([
                self.pm_field.fine_field.centers,
                self.pm_field.fine_field.mus,
                self.pm_field.coarse_field.centers,
                self.pm_field.coarse_field.mus,
            ])
            if hasattr(self.pm_field, 'coarse_projection'):
                params.extend(self.pm_field.coarse_projection.parameters())
        else:
            params.extend([
                self.pm_field.centers,
                self.pm_field.mus,
            ])
        
        if hasattr(self.pm_field, 'fine_field') and hasattr(self.pm_field.fine_field, 'omegas'):
            params.append(self.pm_field.fine_field.omegas)
        if hasattr(self.pm_field, 'omegas'):
            params.append(self.pm_field.omegas)
        
        return params
    
    def add_word(self, word: str) -> int:
        """Add word to vocabulary (pass-through to base_encoder)."""
        return self.base_encoder.add_word(word)
    
    def add_words(self, words) -> list:
        """Add multiple words to vocabulary."""
        return self.base_encoder.add_words(words)
    
    def has_word(self, word: str) -> bool:
        """Check if word is in vocabulary."""
        return self.base_encoder.has_word(word)
    
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.base_encoder.vocab_size()
    
    def auto_add_from_text(self, text: str, min_freq: int = 2) -> set:
        """Auto-add words from text to vocabulary."""
        return self.base_encoder.auto_add_from_text(text, min_freq)
    
    def __repr__(self) -> str:
        return (
            f"SemanticPMFlowEncoder("
            f"dim={self.dimension}, "
            f"latent={self.latent_dim}, "
            f"vocab={self.base_encoder.vocab_size()}, "
            f"flow={self.enable_flow})"
        )
