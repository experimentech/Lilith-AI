# PMFlow Enhanced v0.3.0 - Quick Integration Guide

## Installation

```bash
# From Lilith project root
cd /home/tmumford/Coding/LLM/lilith
python setup_pmflow_enhanced.py develop
```

Or for editable install:
```bash
pip install -e .
```

## Integration Steps

### Step 1: Update SemanticStage to use MultiScalePMField

**File**: `experiments/retrieval_sanity/pipeline/stage_coordinator.py`

Add at top:
```python
from pmflow_bnn_enhanced import (
    MultiScalePMField,
    AttentionGatedPMField,
    contrastive_plasticity,
    batch_plasticity_update,
)
```

Modify SemanticStage.__init__:
```python
class SemanticStage(CognitiveStage):
    def __init__(self, config: StageConfig):
        # Don't call super().__init__ yet - we need custom encoder
        
        # Create multi-scale PMField encoder
        from pmflow_bnn_enhanced import MultiScalePMField
        
        latent_dim = config.encoder_config.get('latent_dim', 32)
        dimension = config.encoder_config.get('dimension', 64)
        
        # Build encoder with multi-scale field
        self.ms_field = MultiScalePMField(
            d_latent=latent_dim,
            n_centers_fine=dimension * 2,  # 128 for dimension=64
            n_centers_coarse=dimension // 2,  # 32 for dimension=64
            steps_fine=5,
            steps_coarse=3
        )
        
        # Initialize concept taxonomy
        from .concept_taxonomy import ConceptTaxonomy, CompositionalQuery
        self.taxonomy = ConceptTaxonomy()
        self.compositor = CompositionalQuery(self.taxonomy)
        
        # Rest of initialization
        self.config = config
        # ... (keep existing code)
```

### Step 2: Add Contrastive Learning to Training Loop

In `stage_coordinator.py`, add to CognitiveStage:

```python
class CognitiveStage:
    def __init__(self, config: StageConfig):
        # ... existing code ...
        
        # Contrastive learning buffers
        self.recent_embeddings = []
        self.max_buffer_size = 100
        
    def apply_plasticity(self, artifact, reward):
        """Enhanced plasticity with contrastive learning."""
        
        # Standard task-based plasticity
        if reward < 0.5:  # Poor performance
            batch_plasticity_update(
                self.encoder.ms_field.fine_field,  # or pm_field
                [artifact.embedding],
                mu_lr=self.config.plasticity_lr,
                c_lr=self.config.plasticity_lr
            )
        
        # Collect for contrastive learning
        self.recent_embeddings.append(artifact.embedding)
        if len(self.recent_embeddings) > self.max_buffer_size:
            self.recent_embeddings.pop(0)
        
        # Periodic contrastive update
        if len(self.recent_embeddings) >= 20:
            self._contrastive_update()
    
    def _contrastive_update(self):
        """Apply contrastive learning from buffer."""
        # Generate pairs using concept taxonomy
        similar_pairs = []
        dissimilar_pairs = []
        
        # Use concepts from artifacts to determine similarity
        # (implementation depends on your metadata tracking)
        
        # Apply contrastive plasticity
        contrastive_plasticity(
            self.encoder.ms_field.fine_field,
            similar_pairs,
            dissimilar_pairs,
            mu_lr=self.config.plasticity_lr * 0.5,
            c_lr=self.config.plasticity_lr * 0.5,
            margin=1.0
        )
```

### Step 3: Use Hybrid Similarity for Retrieval

In your retrieval code:

```python
from pmflow_bnn_enhanced import hybrid_similarity, EnergyBasedPMField

# Wrap your PMField with energy capability
energy_field = EnergyBasedPMField(d_latent=latent_dim, n_centers=n_centers)
energy_field.centers = your_pmfield.centers  # Copy parameters
energy_field.mus = your_pmfield.mus

# Use hybrid similarity for retrieval
def retrieve_top_k(query_emb, corpus_embeddings, k=5):
    scores = []
    for doc_emb in corpus_embeddings:
        score = hybrid_similarity(
            query_emb,
            doc_emb,
            energy_field,
            cosine_weight=0.7,
            energy_weight=0.3
        )
        scores.append(score)
    
    top_k = torch.topk(torch.tensor(scores), k=k)
    return top_k.indices, top_k.values
```

## Testing Integration

After integration, run:

```bash
# Test that nothing broke
pytest tests/ -q

# Run multistage demo
python experiments/retrieval_sanity/demo_multistage.py

# Visualize improvements
python experiments/retrieval_sanity/visualize_stages.py
```

## Expected Results

After integration with MultiScalePMField + Contrastive Learning:

**Before** (current):
- Silhouette: 0.211
- "outdoor location" P@3: 67%

**After** (predicted):
- Silhouette: **0.30-0.35** (multiscale only)
- Silhouette: **0.35-0.45** (+ contrastive)
- "outdoor location" P@3: **80-90%**
- Computational cost: +30-40%

## Troubleshooting

### Import Error
```python
ModuleNotFoundError: No module named 'pmflow_bnn_enhanced'
```
**Fix**: Run `python setup_pmflow_enhanced.py develop` from Lilith root

### Dimension Mismatch
```python
RuntimeError: inconsistent tensor size
```
**Fix**: Ensure `d_latent` matches between fine/coarse fields and downstream usage

### Poor Performance
If clustering doesn't improve:
1. Check contrastive pairs are being generated correctly
2. Verify learning rates aren't too large (start with 5e-4)
3. Ensure enough training examples (>100 per category)

## Next Steps

1. **Immediate**: Integrate MultiScalePMField into SemanticStage
2. **Short-term**: Add contrastive learning to plasticity mechanism  
3. **Medium-term**: Use hybrid_similarity for retrieval
4. **Long-term**: Add AttentionGatedPMField for better context flow

Each step is independent - integrate incrementally and measure improvements!
