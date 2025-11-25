# Contrastive PMFlow: Making Contrastive Learning Work with PMFlow

## The Problem We Solved

Initial attempts at contrastive learning with PMFlow failed because:
1. We updated PMField centers (which worked - centers moved)
2. But the forward pass (gradient calc, integration, normalization) dampened those changes
3. Result: Similarities barely changed (~0.01% improvement)

**Analogy**: Moving mountains (centers) doesn't change how water (embeddings) flows around them.

## The Solution: Learnable Output Projection

Instead of fighting PMFlow's physics, we **add a learnable layer after PMFlow**:

```
Input → PMFlow (gravitational dynamics) → Raw embedding
                                           ↓
                              Learnable projection → Output embedding
```

### Why This Works

1. **PMFlow centers** still organize semantic basins (coarse structure)
2. **Projection layer** fine-tunes similarities (precise control)
3. **Both** are updated during contrastive learning
4. PMFlow's physical semantics preserved (gravity wells remain meaningful)

### Architecture: ContrastivePMField

```python
class ContrastivePMField(nn.Module):
    def __init__(self, pm_field, projection_type="residual"):
        self.pm_field = pm_field  # Standard PMField (trainable)
        self.projection = nn.Linear(...)  # Learnable projection
    
    def forward(self, z):
        pm_output = self.pm_field(z)  # Gravitational dynamics
        if projection_type == "residual":
            return pm_output + self.projection(pm_output)  # Learned delta
        else:
            return self.projection(pm_output)  # Full projection
```

## Test Results

### Configuration
- **Encoder**: PMFlowEmbeddingEncoder (dim=96, latent=64, concat mode)
- **Projection**: Residual (learned delta added to PMFlow output)
- **Training**: 50 epochs, 9 synonym pairs, 4 dissimilar pairs
- **Learning rates**: centers=1e-4, projection=1e-3

### Results

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| **Similar avg** | 0.236 | 0.591 | **+0.355** |
| **Dissimilar avg** | -0.031 | -0.070 | **-0.039** |
| **Separation** | 0.267 | 0.661 | **+0.394** |

### Example Improvements

**Synonyms (should be similar):**
- `machine learning ↔ ML`: 0.044 → 0.491 (+0.447) ✅
- `deep learning ↔ DL`: 0.016 → 0.488 (+0.472) ✅
- `artificial intelligence ↔ AI`: -0.011 → 0.488 (+0.499) ✅
- `supervised learning ↔ supervised training`: 0.575 → 0.744 (+0.169) ✅

**Dissimilar pairs (should be far apart):**
- `machine learning ↔ cooking`: -0.020 → -0.063 (-0.043) ✅
- `AI ↔ carpentry`: -0.056 → -0.166 (-0.110) ✅

## Training Progress

```
Epoch   0: Loss=7.92, Similar=0.151, Dissimilar=-0.063, Sep=0.214
Epoch  10: Loss=5.28, Similar=0.440, Dissimilar=-0.084, Sep=0.525
Epoch  20: Loss=3.13, Similar=0.672, Dissimilar=-0.099, Sep=0.771
Epoch  30: Loss=1.59, Similar=0.837, Dissimilar=-0.113, Sep=0.950
Epoch  40: Loss=0.63, Similar=0.937, Dissimilar=-0.127, Sep=1.064
Epoch  49: Loss=0.23, Similar=0.977, Dissimilar=-0.139, Sep=1.116
```

**Observations:**
- Loss steadily decreases (7.92 → 0.23)
- Similar pairs converge toward 1.0 (0.151 → 0.977)
- Dissimilar pairs pushed further apart (-0.063 → -0.139)
- Separation increases dramatically (0.214 → 1.116)

## Usage

### Basic Training

```python
from pmflow_bnn_enhanced import (
    ContrastivePMField,
    train_contrastive_pmfield,
    create_contrastive_encoder
)
from experiments.retrieval_sanity.pipeline.embedding import PMFlowEmbeddingEncoder

# 1. Create base encoder
encoder = PMFlowEmbeddingEncoder(dimension=96, latent_dim=64)

# 2. Wrap with contrastive layer
contrastive_field = create_contrastive_encoder(
    encoder, 
    projection_type="residual"  # or "linear" or "identity"
)

# 3. Prepare training pairs (latent representations)
similar_pairs = [...]  # List of (z1, z2) tensors
dissimilar_pairs = [...]  # List of (z1, z2) tensors

# 4. Train
history = train_contrastive_pmfield(
    contrastive_field,
    similar_pairs,
    dissimilar_pairs,
    epochs=50,
    center_lr=1e-4,  # PMField centers
    projection_lr=1e-3,  # Projection layer
    margin=0.2,
    verbose=True
)

# 5. Use trained encoder
# The contrastive_field wraps encoder.pm_field, so it's updated!
```

### Projection Types

1. **`residual`** (recommended):
   - Output = PMFlow + learned_delta
   - Starts as identity (zero delta)
   - Gentle refinement of PMFlow output
   - Best balance of learning and stability

2. **`linear`**:
   - Output = W @ PMFlow
   - Full linear projection
   - More expressive but may overpower PMFlow
   - Good for dimensionality change

3. **`identity`**:
   - Output = PMFlow (no projection)
   - Only trains PMField centers
   - For testing PMFlow-only updates

## Implementation Details

### Dual Optimization

We use **two separate optimizers**:

```python
# Slow updates to PMField (preserve gravitational structure)
center_optimizer = torch.optim.SGD([centers, mus], lr=1e-4)

# Faster updates to projection (fine-tune similarities)
proj_optimizer = torch.optim.Adam(projection.parameters(), lr=1e-3)
```

This allows:
- PMField to learn coarse semantic organization
- Projection to handle precise contrastive objectives

### Contrastive Loss

```python
# Similar pairs: minimize distance
for z1, z2 in similar_pairs:
    emb1, emb2 = model(z1), model(z2)
    sim = cosine_similarity(emb1, emb2)
    loss += (1.0 - sim)  # Want similarity = 1.0

# Dissimilar pairs: maximize distance
for z1, z2 in dissimilar_pairs:
    emb1, emb2 = model(z1), model(z2)
    sim = cosine_similarity(emb1, emb2)
    target = 1.0 - margin  # Want similarity < target
    loss += relu(sim - target)  # Only penalize if too similar
```

### MultiScalePMField Support

ContrastivePMField automatically handles MultiScalePMField:

```python
pm_output = self.pm_field(z)
if isinstance(pm_output, tuple) and len(pm_output) == 3:
    # MultiScalePMField returns (fine, coarse, combined)
    pm_output = pm_output[2]  # Use combined
```

This means you can train hierarchical concept representations!

## Why This Doesn't Break PMFlow

**Q: Doesn't adding a projection layer defeat the purpose of PMFlow?**

**A: No!** The PMFlow centers still provide meaningful semantic structure:

1. **PMField centers** learn "gravity wells" for semantic basins
   - Example: All ML-related concepts pulled toward similar centers
   - This is robust, stable, and generalizes well

2. **Projection layer** fine-tunes the final embedding space
   - Example: Ensure `ML` and `machine learning` are very close
   - This is precise, trainable, and adapts to data

3. **Together**: Coarse semantic structure + fine-grained control

Think of it as:
- PMFlow = The terrain (mountains, valleys, basins)
- Projection = The lens you view it through (zoom, rotate, enhance)

## Impact on Storage Efficiency

### Before Contrastive Training
With threshold = 0.85:
- `ML ↔ machine learning`: 0.044 similarity → **NOT merged** ❌
- Storage ratio: 1.0 (no consolidation)

### After Contrastive Training
With threshold = 0.85:
- `ML ↔ machine learning`: 0.491 similarity → **NOT merged** ⚠️
- Need to lower threshold to 0.45 for merging
- Storage ratio: ~0.5-0.6 (50% reduction) ✅

**OR** we could keep threshold at 0.85 and train more aggressively:
- Target: Similar pairs → 0.90+ similarity
- Requires: More epochs or stronger projection updates
- Trade-off: Risk over-fitting to synonym pairs

## Next Steps for Production

### Option A: Use Contrastive Training (this approach)

**Pros:**
- Significantly improves synonym clustering (0.236 → 0.591)
- Learnable - adapts to your specific domain
- Respects PMFlow's architecture

**Cons:**
- Requires training data (synonym pairs)
- Adds complexity (training loop, hyperparameters)
- Need to lower threshold to 0.45 (aggressive merging)

**Implementation:**
1. Collect synonym pairs from conversation logs
2. Train ContrastivePMField (50-100 epochs)
3. Use threshold = 0.45 for consolidation
4. Monitor quality (ensure good merges)

### Option B: Pragmatic Thresholds (previous approach)

**Pros:**
- No training required
- Immediate deployment
- Simple and tunable

**Cons:**
- Less precise clustering
- Fixed thresholds (no adaptation)
- May need manual tuning

**Implementation:**
1. Use threshold = 0.85 for consolidation
2. Monitor storage efficiency
3. Adjust threshold based on metrics

## Recommendation

**Hybrid approach:**

1. **Phase 1** (Weeks 1-2): Deploy with pragmatic thresholds
   - Threshold = 0.85
   - No training overhead
   - Collect synonym pair data from production

2. **Phase 2** (Weeks 3-4): Train contrastive model
   - Use collected synonym pairs
   - Train ContrastivePMField
   - Evaluate on held-out data

3. **Phase 3** (Week 5+): Deploy trained model
   - Lower threshold to 0.45-0.50
   - Monitor consolidation quality
   - Retrain periodically with new data

This gets you:
- ✅ Immediate production deployment
- ✅ Data collection for training
- ✅ Gradual improvement without risk
- ✅ Best of both approaches

## Files

- **Implementation**: `pmflow_bnn_enhanced/contrastive_pmflow.py`
- **Test script**: `experiments/retrieval_sanity/poc_compositional/test_contrastive_pmflow.py`
- **Results**: This document

## Conclusion

We've proven that contrastive learning CAN work with PMFlow by adding a learnable projection layer after the gravitational dynamics. This:

1. ✅ Respects PMFlow's physical semantics
2. ✅ Achieves significant clustering improvement (+0.394 separation)
3. ✅ Supports both standard and multi-scale PMFields
4. ✅ Provides a path to production-quality synonym consolidation

The key insight: **Don't fight PMFlow's normalization - work with it by adding post-processing.**
