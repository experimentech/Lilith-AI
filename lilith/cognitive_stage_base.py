"""
Cognitive Stage Base Library

Common patterns and functionality for all BNN-based cognitive stages.
Each stage has domain-specific latent space but shares these patterns:

- Pattern storage with embeddings AND latent representations
- Latent-space retrieval (not full embeddings)
- Contrastive learning integration
- Plasticity/reinforcement updates
- Success score tracking
- SQLite-backed persistence

Stages using this library:
- Intake: Character/token normalization (latent_dim=16)
- Syntax: Grammatical patterns (latent_dim=32)
- Semantic: Concept relationships (latent_dim=96)
- World: Grounded representations (latent_dim=64)
- Pragmatic: Response composition (latent_dim=48)

Key Design:
- Latent space (32-96 dim) is where learning/comparison happens
- Full embeddings (144 dim) are final output with PMFlow refinement
- Contrastive learning shapes latent space geometry
- Each stage has domain-appropriate latent_dim
"""

from __future__ import annotations

import logging
import sqlite3
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Callable
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Core Data Structures
# ──────────────────────────────────────────────────────────────────

@dataclass
class CognitivePattern:
    """
    A learned pattern stored in a cognitive stage.
    
    Stores BOTH full embedding (144-dim) and latent (stage-specific dim).
    Retrieval uses latent space, output uses full embedding.
    """
    pattern_id: str
    content: str                      # The pattern content (text, POS sequence, etc.)
    embedding: torch.Tensor           # Full embedding (144-dim PMFlow output)
    latent: torch.Tensor              # Latent representation (stage-specific dim)
    success_score: float = 0.5        # Reinforcement score (0-1)
    usage_count: int = 0
    intent: str = "general"           # Pattern category/type
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "pattern_id": self.pattern_id,
            "content": self.content,
            "embedding": self.embedding.cpu().tolist() if isinstance(self.embedding, torch.Tensor) else self.embedding,
            "latent": self.latent.cpu().tolist() if isinstance(self.latent, torch.Tensor) else self.latent,
            "success_score": float(self.success_score),
            "usage_count": int(self.usage_count),
            "intent": self.intent,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CognitivePattern:
        """Create from dictionary."""
        return cls(
            pattern_id=data["pattern_id"],
            content=data["content"],
            embedding=torch.tensor(data["embedding"]) if isinstance(data["embedding"], list) else data["embedding"],
            latent=torch.tensor(data["latent"]) if isinstance(data["latent"], list) else data["latent"],
            success_score=data["success_score"],
            usage_count=data["usage_count"],
            intent=data["intent"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class PlasticityReport:
    """Summary of a plasticity update."""
    feedback: float
    pattern_id: str
    delta_centers: float
    delta_mus: float
    plasticity_type: str = "reinforcement"  # reinforcement, contrastive, hybrid
    
    def as_dict(self) -> Dict[str, Any]:
        return {
            "feedback": float(self.feedback),
            "pattern_id": self.pattern_id,
            "delta_centers": float(self.delta_centers),
            "delta_mus": float(self.delta_mus),
            "plasticity_type": self.plasticity_type,
        }


@dataclass
class RetrievalResult:
    """Result from pattern retrieval."""
    pattern: CognitivePattern
    similarity: float                 # Cosine similarity in latent space
    confidence: float                 # Adjusted by success_score and usage


# ──────────────────────────────────────────────────────────────────
# Abstract Base Class
# ──────────────────────────────────────────────────────────────────

class CognitiveStageBase(ABC):
    """
    Abstract base for all BNN-based cognitive stages.
    
    Provides common functionality:
    - Pattern storage (SQLite or JSON)
    - Latent-space retrieval
    - Success score updates
    - Contrastive learning
    - Plasticity updates
    
    Subclasses implement:
    - Domain-specific encoding
    - Pattern bootstrapping
    - Specialized processing
    """
    
    def __init__(
        self,
        encoder,
        stage_name: str,
        latent_dim: int,
        storage_path: Optional[Path] = None,
        use_sqlite: bool = True,
        plasticity_enabled: bool = True,
        plasticity_lr: float = 1e-3,
    ):
        """
        Initialize cognitive stage.
        
        Args:
            encoder: PMFlowEmbeddingEncoder with latent space
            stage_name: Name of this stage (e.g., "syntax", "semantic")
            latent_dim: Dimension of latent space for this stage
            storage_path: Where to store patterns (uses default if None)
            use_sqlite: Use SQLite instead of JSON
            plasticity_enabled: Enable PMFlow plasticity updates
            plasticity_lr: Learning rate for plasticity
        """
        self.encoder = encoder
        self.stage_name = stage_name
        self.latent_dim = latent_dim
        self.plasticity_enabled = plasticity_enabled
        self.plasticity_lr = plasticity_lr
        
        # Storage paths
        if storage_path is None:
            storage_path = Path(f"data/{stage_name}_patterns")
        self.storage_path = storage_path
        self.use_sqlite = use_sqlite
        
        if use_sqlite:
            self.db_path = storage_path.with_suffix('.db')
        else:
            self.json_path = storage_path.with_suffix('.json')
        
        self.pmflow_state_path = storage_path.with_suffix('.pt')
        
        # In-memory pattern cache (for non-SQLite)
        self.patterns: Dict[str, CognitivePattern] = {}
        
        # Plasticity tracking
        self.plasticity_reports: List[PlasticityReport] = []
        self.total_plasticity_updates = 0
        
        # Initialize storage
        if use_sqlite:
            self._init_sqlite()
        else:
            self._load_json()
        
        # Load PMFlow state if available
        if self.pmflow_state_path.exists():
            self._load_pmflow_state()
        
        logger.info(
            f"{stage_name} stage initialized: latent_dim={latent_dim}, "
            f"storage={'SQLite' if use_sqlite else 'JSON'}, "
            f"plasticity={plasticity_enabled}"
        )
    
    # ──────────────────────────────────────────────────────────────
    # Abstract Methods - Implement in Subclasses
    # ──────────────────────────────────────────────────────────────
    
    @abstractmethod
    def _bootstrap_patterns(self) -> List[CognitivePattern]:
        """
        Create seed patterns for this stage.
        
        Returns:
            List of initial patterns
        """
        pass
    
    @abstractmethod
    def _encode_content(self, content: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode content to full embedding and latent.
        
        Args:
            content: Stage-specific content (text, POS sequence, etc.)
            
        Returns:
            (full_embedding, latent_representation)
        """
        pass
    
    # ──────────────────────────────────────────────────────────────
    # Core Pattern Operations
    # ──────────────────────────────────────────────────────────────
    
    def encode_with_latent(self, content: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode content returning BOTH full embedding and latent.
        
        This is the standard encoding interface. Subclasses can override
        _encode_content for domain-specific processing.
        
        Args:
            content: Content to encode
            
        Returns:
            (full_embedding, latent_representation)
        """
        return self._encode_content(content)
    
    def add_pattern(
        self,
        content: str,
        intent: str = "general",
        success_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CognitivePattern:
        """
        Add a new pattern to this stage.
        
        Args:
            content: Pattern content
            intent: Pattern category
            success_score: Initial success rating
            metadata: Additional metadata
            
        Returns:
            Created CognitivePattern
        """
        # Generate unique ID
        pattern_id = f"{self.stage_name}_{intent}_{len(self.patterns)}"
        
        # Encode to get both full embedding and latent
        embedding, latent = self.encode_with_latent(content)
        
        # Create pattern
        pattern = CognitivePattern(
            pattern_id=pattern_id,
            content=content,
            embedding=embedding,
            latent=latent,
            success_score=success_score,
            usage_count=0,
            intent=intent,
            metadata=metadata or {},
        )
        
        # Store
        if self.use_sqlite:
            self._insert_pattern_sqlite(pattern)
        else:
            self.patterns[pattern_id] = pattern
            self._save_json()
        
        return pattern
    
    def retrieve_similar(
        self,
        query_content: str,
        topk: int = 5,
        min_similarity: float = 0.0,
        intent_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve patterns similar to query using LATENT space.
        
        CRITICAL: Comparison happens in latent space, not full embeddings.
        This is where BNN learning has shaped the geometry.
        
        Args:
            query_content: Query to match
            topk: Number of results to return
            min_similarity: Minimum similarity threshold
            intent_filter: Optional intent category filter
            
        Returns:
            List of RetrievalResult objects sorted by confidence
        """
        # Encode query to latent space
        _, query_latent = self.encode_with_latent(query_content)
        
        # Normalize for cosine similarity
        query_latent = F.normalize(query_latent, p=2, dim=-1)
        
        results = []
        
        if self.use_sqlite:
            patterns = self._fetch_patterns_sqlite(intent_filter=intent_filter)
        else:
            patterns = [
                p for p in self.patterns.values()
                if intent_filter is None or p.intent == intent_filter
            ]
        
        for pattern in patterns:
            # Compare in LATENT space (not full embeddings!)
            pattern_latent = F.normalize(pattern.latent, p=2, dim=-1)
            similarity = float(F.cosine_similarity(query_latent, pattern_latent, dim=-1).item())
            
            if similarity < min_similarity:
                continue
            
            # Adjust confidence by success score and usage
            usage_bonus = min(0.1, pattern.usage_count * 0.01)
            confidence = similarity * pattern.success_score + usage_bonus
            
            results.append(RetrievalResult(
                pattern=pattern,
                similarity=similarity,
                confidence=confidence,
            ))
        
        # Sort by confidence descending
        results.sort(key=lambda r: r.confidence, reverse=True)
        
        return results[:topk]
    
    def update_success(
        self,
        pattern_id: str,
        feedback: float,
        learning_rate: float = 0.1,
        apply_plasticity: bool = True,
    ) -> Optional[PlasticityReport]:
        """
        Update pattern success score via reinforcement.
        
        Args:
            pattern_id: Pattern to update
            feedback: Feedback signal (-1.0 to 1.0)
            learning_rate: How fast to update success score
            apply_plasticity: Whether to apply PMFlow plasticity
            
        Returns:
            PlasticityReport if plasticity was applied, None otherwise
        """
        # Get pattern
        if self.use_sqlite:
            pattern = self._fetch_pattern_sqlite(pattern_id)
        else:
            pattern = self.patterns.get(pattern_id)
        
        if pattern is None:
            logger.warning(f"Pattern {pattern_id} not found")
            return None
        
        # Update success score
        new_score = pattern.success_score + feedback * learning_rate
        pattern.success_score = float(np.clip(new_score, 0.0, 1.0))
        pattern.usage_count += 1
        
        # Save updated pattern
        if self.use_sqlite:
            self._update_pattern_sqlite(pattern)
        else:
            self.patterns[pattern_id] = pattern
            self._save_json()
        
        # Apply PMFlow plasticity if enabled
        report = None
        if apply_plasticity and self.plasticity_enabled:
            report = self._apply_plasticity(pattern, feedback)
            
            if report is not None:
                self.plasticity_reports.append(report)
                self.total_plasticity_updates += 1
                
                # Auto-save PMFlow state periodically
                if self.total_plasticity_updates % 10 == 0:
                    self._save_pmflow_state()
        
        return report
    
    def add_contrastive_pairs(
        self,
        similar_pairs: Optional[List[Tuple[str, str]]] = None,
        dissimilar_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Optional[PlasticityReport]:
        """
        Apply batch contrastive learning to shape latent space.
        
        This is critical for proper BNN learning - teaches the system
        which patterns should be close vs far in latent space.
        
        Args:
            similar_pairs: List of (content_a, content_b) that should be close
            dissimilar_pairs: List of (content_a, content_b) that should be far
            
        Returns:
            PlasticityReport if update was applied
        """
        try:
            from pmflow import contrastive_plasticity
        except ImportError:
            logger.warning("PMFlow contrastive_plasticity not available")
            return None
        
        if not self.plasticity_enabled:
            return None
        
        # Get PMFlow field
        pm_field = self.encoder.pm_field
        if hasattr(pm_field, 'fine_field'):
            target_field = pm_field.fine_field
        else:
            target_field = pm_field
        
        device = target_field.centers.device
        
        # Encode pairs to latent space
        similar_latents = []
        if similar_pairs:
            for content_a, content_b in similar_pairs:
                _, latent_a = self.encode_with_latent(content_a)
                _, latent_b = self.encode_with_latent(content_b)
                similar_latents.append((latent_a.to(device), latent_b.to(device)))
        
        dissimilar_latents = []
        if dissimilar_pairs:
            for content_a, content_b in dissimilar_pairs:
                _, latent_a = self.encode_with_latent(content_a)
                _, latent_b = self.encode_with_latent(content_b)
                dissimilar_latents.append((latent_a.to(device), latent_b.to(device)))
        
        if not similar_latents and not dissimilar_latents:
            return None
        
        # Snapshot before update
        before_centers = target_field.centers.detach().clone()
        before_mus = target_field.mus.detach().clone()
        
        # Apply contrastive plasticity in latent space
        contrastive_plasticity(
            target_field,
            similar_pairs=similar_latents,
            dissimilar_pairs=dissimilar_latents,
            mu_lr=self.plasticity_lr * 0.5,
            c_lr=self.plasticity_lr * 0.5,
            margin=1.0,
        )
        
        # Measure changes
        delta_centers = torch.norm(target_field.centers - before_centers, p=2).item()
        delta_mus = torch.norm(target_field.mus - before_mus, p=2).item()
        
        report = PlasticityReport(
            feedback=0.0,
            pattern_id="contrastive_batch",
            delta_centers=delta_centers,
            delta_mus=delta_mus,
            plasticity_type="contrastive",
        )
        
        self.plasticity_reports.append(report)
        self.total_plasticity_updates += 1
        
        logger.info(
            f"{self.stage_name}: Applied contrastive learning - "
            f"{len(similar_latents)} similar, {len(dissimilar_latents)} dissimilar pairs, "
            f"Δcenters={delta_centers:.4f}, Δmus={delta_mus:.4f}"
        )
        
        return report
    
    # ──────────────────────────────────────────────────────────────
    # PMFlow Plasticity
    # ──────────────────────────────────────────────────────────────
    
    def _apply_plasticity(
        self,
        pattern: CognitivePattern,
        feedback: float,
    ) -> Optional[PlasticityReport]:
        """
        Apply PMFlow plasticity update for this pattern.
        
        Args:
            pattern: Pattern that received feedback
            feedback: Feedback signal (-1.0 to 1.0)
            
        Returns:
            PlasticityReport or None
        """
        try:
            from pmflow import vectorized_pm_plasticity
        except ImportError:
            logger.debug("PMFlow plasticity not available")
            return None
        
        # Get PMFlow field
        pm_field = self.encoder.pm_field
        if hasattr(pm_field, 'fine_field'):
            target_field = pm_field.fine_field
        else:
            target_field = pm_field
        
        device = target_field.centers.device
        
        # Snapshot before update
        before_centers = target_field.centers.detach().clone()
        before_mus = target_field.mus.detach().clone()
        
        # Apply vectorized plasticity using latent representation
        # Note: PMFlow expects z_batch (latent) and h_batch (refined/full embedding)
        pattern_latent = pattern.latent.to(device)
        pattern_embedding = pattern.embedding.to(device)
        
        if pattern_latent.dim() == 1:
            pattern_latent = pattern_latent.unsqueeze(0)
        if pattern_embedding.dim() == 1:
            pattern_embedding = pattern_embedding.unsqueeze(0)
        
        # Scale learning rates by feedback signal (positive = strengthen, negative = weaken)
        effective_mu_lr = self.plasticity_lr * abs(feedback)
        effective_c_lr = self.plasticity_lr * abs(feedback)
        
        vectorized_pm_plasticity(
            target_field,
            pattern_latent,      # z_batch: latent representation
            pattern_embedding,   # h_batch: refined/full embedding
            mu_lr=effective_mu_lr,
            c_lr=effective_c_lr,
        )
        
        # Measure changes
        delta_centers = torch.norm(target_field.centers - before_centers, p=2).item()
        delta_mus = torch.norm(target_field.mus - before_mus, p=2).item()
        
        report = PlasticityReport(
            feedback=feedback,
            pattern_id=pattern.pattern_id,
            delta_centers=delta_centers,
            delta_mus=delta_mus,
            plasticity_type="reinforcement",
        )
        
        logger.debug(
            f"{self.stage_name}: Plasticity update - pattern={pattern.pattern_id}, "
            f"feedback={feedback:.2f}, Δcenters={delta_centers:.4f}, Δmus={delta_mus:.4f}"
        )
        
        return report
    
    def _save_pmflow_state(self):
        """Save PMFlow field state (learned plasticity weights)."""
        pm_field = self.encoder.pm_field
        
        # Handle MultiScalePMField
        if hasattr(pm_field, 'fine_field') and hasattr(pm_field, 'coarse_field'):
            payload = {
                "type": "multiscale",
                "stage_name": self.stage_name,
                "latent_dim": self.latent_dim,
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
                "stage_name": self.stage_name,
                "latent_dim": self.latent_dim,
                "centers": pm_field.centers.detach().cpu(),
                "mus": pm_field.mus.detach().cpu(),
                "total_plasticity_updates": self.total_plasticity_updates,
            }
        
        self.pmflow_state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self.pmflow_state_path)
        logger.debug(f"{self.stage_name}: Saved PMFlow state to {self.pmflow_state_path}")
    
    def _load_pmflow_state(self):
        """Load PMFlow field state (learned plasticity weights)."""
        if not self.pmflow_state_path.exists():
            return
        
        pm_field = self.encoder.pm_field
        device = pm_field.fine_field.centers.device if hasattr(pm_field, 'fine_field') else pm_field.centers.device
        
        try:
            payload = torch.load(self.pmflow_state_path, map_location=device)
            
            # Verify this is the right stage
            if payload.get("stage_name") != self.stage_name:
                logger.warning(
                    f"PMFlow state is for stage '{payload.get('stage_name')}', "
                    f"not '{self.stage_name}' - skipping load"
                )
                return
            
            # Load based on type
            if payload["type"] == "multiscale":
                pm_field.fine_field.centers.data = payload["fine_centers"].to(device)
                pm_field.fine_field.mus.data = payload["fine_mus"].to(device)
                pm_field.coarse_field.centers.data = payload["coarse_centers"].to(device)
                pm_field.coarse_field.mus.data = payload["coarse_mus"].to(device)
                pm_field.coarse_projection.weight.data = payload["coarse_projection"].to(device)
            else:
                pm_field.centers.data = payload["centers"].to(device)
                pm_field.mus.data = payload["mus"].to(device)
            
            self.total_plasticity_updates = payload.get("total_plasticity_updates", 0)
            
            logger.info(
                f"{self.stage_name}: Loaded PMFlow state - "
                f"{self.total_plasticity_updates} previous updates"
            )
        except Exception as e:
            logger.error(f"{self.stage_name}: Failed to load PMFlow state: {e}")
    
    # ──────────────────────────────────────────────────────────────
    # SQLite Storage
    # ──────────────────────────────────────────────────────────────
    
    def _init_sqlite(self):
        """Initialize SQLite database with schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        
        # Enable WAL mode for better concurrent access
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        
        # Create schema
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.stage_name}_patterns (
                pattern_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                latent BLOB NOT NULL,
                success_score REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 0,
                intent TEXT DEFAULT 'general',
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Indexes for efficient querying
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.stage_name}_intent
            ON {self.stage_name}_patterns(intent)
        """)
        
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.stage_name}_success
            ON {self.stage_name}_patterns(success_score DESC)
        """)
        
        conn.commit()
        conn.close()
        
        # Bootstrap if empty
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute(f"SELECT COUNT(*) FROM {self.stage_name}_patterns")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count == 0:
            seed_patterns = self._bootstrap_patterns()
            for pattern in seed_patterns:
                self._insert_pattern_sqlite(pattern)
    
    def _insert_pattern_sqlite(self, pattern: CognitivePattern):
        """Insert pattern into SQLite."""
        conn = sqlite3.connect(str(self.db_path))
        
        # Convert tensors to bytes
        embedding_bytes = pattern.embedding.cpu().numpy().tobytes()
        latent_bytes = pattern.latent.cpu().numpy().tobytes()
        metadata_json = json.dumps(pattern.metadata)
        
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {self.stage_name}_patterns
            (pattern_id, content, embedding, latent, success_score, usage_count, intent, metadata, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                pattern.pattern_id,
                pattern.content,
                embedding_bytes,
                latent_bytes,
                pattern.success_score,
                pattern.usage_count,
                pattern.intent,
                metadata_json,
            ),
        )
        
        conn.commit()
        conn.close()
    
    def _fetch_pattern_sqlite(self, pattern_id: str) -> Optional[CognitivePattern]:
        """Fetch single pattern from SQLite."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute(
            f"SELECT * FROM {self.stage_name}_patterns WHERE pattern_id = ?",
            (pattern_id,),
        )
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return self._row_to_pattern(row)
    
    def _fetch_patterns_sqlite(
        self,
        intent_filter: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[CognitivePattern]:
        """Fetch multiple patterns from SQLite."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        
        if intent_filter:
            query = f"SELECT * FROM {self.stage_name}_patterns WHERE intent = ?"
            params = (intent_filter,)
        else:
            query = f"SELECT * FROM {self.stage_name}_patterns"
            params = ()
        
        if limit:
            query += " LIMIT ?"
            params = params + (limit,)
        
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_pattern(row) for row in rows]
    
    def _update_pattern_sqlite(self, pattern: CognitivePattern):
        """Update existing pattern in SQLite."""
        self._insert_pattern_sqlite(pattern)  # INSERT OR REPLACE handles update
    
    def _row_to_pattern(self, row: sqlite3.Row) -> CognitivePattern:
        """Convert SQLite row to CognitivePattern."""
        # Reconstruct tensors from bytes
        embedding_array = np.frombuffer(row["embedding"], dtype=np.float32)
        embedding = torch.from_numpy(embedding_array)
        
        latent_array = np.frombuffer(row["latent"], dtype=np.float32)
        latent = torch.from_numpy(latent_array)
        
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        
        return CognitivePattern(
            pattern_id=row["pattern_id"],
            content=row["content"],
            embedding=embedding,
            latent=latent,
            success_score=row["success_score"],
            usage_count=row["usage_count"],
            intent=row["intent"],
            metadata=metadata,
        )
    
    # ──────────────────────────────────────────────────────────────
    # JSON Storage (Fallback)
    # ──────────────────────────────────────────────────────────────
    
    def _save_json(self):
        """Save patterns to JSON file."""
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "stage_name": self.stage_name,
            "latent_dim": self.latent_dim,
            "patterns": [p.to_dict() for p in self.patterns.values()],
        }
        
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_json(self):
        """Load patterns from JSON file."""
        if not self.json_path.exists():
            # Bootstrap with seed patterns
            seed_patterns = self._bootstrap_patterns()
            for pattern in seed_patterns:
                self.patterns[pattern.pattern_id] = pattern
            self._save_json()
            return
        
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            
            for pattern_dict in data.get("patterns", []):
                pattern = CognitivePattern.from_dict(pattern_dict)
                self.patterns[pattern.pattern_id] = pattern
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"{self.stage_name}: Failed to load JSON, starting fresh")
    
    # ──────────────────────────────────────────────────────────────
    # Utility Methods
    # ──────────────────────────────────────────────────────────────
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this stage."""
        if self.use_sqlite:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute(f"SELECT COUNT(*), AVG(success_score) FROM {self.stage_name}_patterns")
            count, avg_success = cursor.fetchone()
            conn.close()
            pattern_count = count or 0
            avg_score = avg_success or 0.0
        else:
            pattern_count = len(self.patterns)
            avg_score = np.mean([p.success_score for p in self.patterns.values()]) if self.patterns else 0.0
        
        return {
            "stage_name": self.stage_name,
            "latent_dim": self.latent_dim,
            "total_patterns": pattern_count,
            "average_success": float(avg_score),
            "plasticity_updates": self.total_plasticity_updates,
            "storage": "SQLite" if self.use_sqlite else "JSON",
        }
