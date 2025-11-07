# Non-CLS Transformer Architecture Proposal

**Date:** 2025-11-05
**Status:** DRAFT - Awaiting Approval
**Purpose:** Design non-CLS variants of FT-Transformer and CSN-Transformer for comparison

---

## Table of Contents
1. [Background & Motivation](#background--motivation)
2. [Non-CLS Aggregation Strategies](#non-cls-aggregation-strategies)
3. [FT-Transformer Non-CLS Options](#ft-transformer-non-cls-options)
4. [CSN-Transformer Non-CLS Options](#csn-transformer-non-cls-options)
5. [Parameter & Memory Analysis](#parameter--memory-analysis)
6. [Recommended Architectures](#recommended-architectures)
7. [Implementation Plan](#implementation-plan)
8. [Expected Performance Characteristics](#expected-performance-characteristics)
9. [Questions for Decision](#questions-for-decision)

---

## Background & Motivation

### Current CLS-Based Architecture

Both FT-Transformer and CSN-Transformer currently use **CLS (classification) tokens** for sequence aggregation:

**FT-Transformer CLS:**
```
Input: [batch, seq_len, features]
↓
Tokenization: [batch, num_tokens, d_model]
↓
Prepend CLS: [batch, 1 + num_tokens, d_model]
↓
Transformer: All tokens attend to each other
↓
Extract CLS: cls_output = output[:, 0, :]  # [batch, d_model]
↓
Prediction: Linear(cls_output) → [batch, output_dim]
```

**CSN-Transformer CLS:**
```
Categorical Path: [batch, num_cat, d_model] → CLS₁ → [batch, d_model]
Numerical Path: [batch, seq_len, d_model] → CLS₂ → [batch, d_model]
Fusion: cat([CLS₁, CLS₂]) → [batch, 2*d_model]
Prediction: Linear(fused) → [batch, output_dim]
```

### Why Explore Non-CLS Variants?

1. **CLS token is learned but may not be optimal** - The CLS token learns to aggregate through self-attention, but explicit pooling strategies might be more effective for time series

2. **Parameter efficiency** - CLS token adds parameters and one extra position in attention computation

3. **Interpretability** - Explicit pooling (mean, max, attention) is more interpretable than learned CLS aggregation

4. **Domain-specific optimization** - Time series forecasting might benefit from temporal-aware aggregation rather than generic CLS learning

5. **Empirical validation** - Compare whether CLS provides actual benefits or if simpler pooling works as well

---

## Non-CLS Aggregation Strategies

### Option 1: Mean Pooling

**Description:** Average all token representations across the sequence dimension

```python
# Instead of extracting CLS token
aggregated = torch.mean(transformer_output, dim=1)  # [batch, d_model]
```

**Pros:**
- ✅ Zero additional parameters
- ✅ Simple and interpretable
- ✅ Equal weight to all tokens
- ✅ Smooth gradients (all tokens contribute equally)
- ✅ No extra attention computation

**Cons:**
- ❌ Cannot learn importance weighting
- ❌ All tokens treated equally (may not be optimal)
- ❌ Sensitive to outlier token representations

**Parameters Added:** 0

---

### Option 2: Max Pooling

**Description:** Take maximum value across sequence dimension for each feature

```python
# Instead of extracting CLS token
aggregated, _ = torch.max(transformer_output, dim=1)  # [batch, d_model]
```

**Pros:**
- ✅ Zero additional parameters
- ✅ Focuses on most salient features
- ✅ Invariant to sequence length
- ✅ Good for sparse, important signals

**Cons:**
- ❌ Loses information from non-maximum tokens
- ❌ Non-smooth gradients (only max token gets gradient)
- ❌ May ignore important but not maximal features
- ❌ Less effective for dense, continuous signals like time series

**Parameters Added:** 0

---

### Option 3: Attention Pooling (Single Query)

**Description:** Learnable query vector that attends to all tokens

```python
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attention = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)

    def forward(self, tokens):
        # tokens: [batch, seq_len, d_model]
        batch_size = tokens.size(0)
        query = self.query.expand(batch_size, -1, -1)  # [batch, 1, d_model]

        # Query attends to all tokens
        output, attn_weights = self.attention(query, tokens, tokens)
        return output.squeeze(1)  # [batch, d_model]
```

**Pros:**
- ✅ Learnable importance weighting
- ✅ Interpretable attention weights
- ✅ Minimal parameters (just query vector + attention)
- ✅ Can adapt to which tokens are important
- ✅ Similar to CLS but more explicit

**Cons:**
- ❌ Adds attention computation
- ❌ More parameters than mean/max pooling
- ❌ Single attention head may be limiting

**Parameters Added:**
- Query vector: `d_model` = 128
- Attention weights: `3 * d_model * d_model` = 49,152
- **Total: ~49,280 parameters**

---

### Option 4: Multi-Head Attention Pooling

**Description:** Multiple learnable queries with multi-head attention

```python
class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, tokens):
        # tokens: [batch, seq_len, d_model]
        batch_size = tokens.size(0)
        query = self.query.expand(batch_size, -1, -1)  # [batch, 1, d_model]

        # Multi-head attention: query attends to all tokens
        output, attn_weights = self.attention(query, tokens, tokens)
        return output.squeeze(1)  # [batch, d_model]
```

**Pros:**
- ✅ Rich learnable aggregation (multi-head)
- ✅ Can capture different aggregation patterns per head
- ✅ More expressive than single-head
- ✅ Interpretable attention weights

**Cons:**
- ❌ More parameters than single-head
- ❌ Extra attention computation
- ❌ May overfit on small datasets

**Parameters Added:**
- Query vector: `d_model` = 128
- Multi-head attention: `3 * d_model * d_model + d_model * d_model` = 65,536
- **Total: ~65,664 parameters**

---

### Option 5: Weighted Average (Learnable)

**Description:** Learn a weight for each position, then weighted average

```python
class WeightedAveragePooling(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(max_seq_len, 1))

    def forward(self, tokens):
        # tokens: [batch, seq_len, d_model]
        seq_len = tokens.size(1)
        weights = F.softmax(self.weights[:seq_len], dim=0)  # [seq_len, 1]
        aggregated = (tokens * weights).sum(dim=1)  # [batch, d_model]
        return aggregated
```

**Pros:**
- ✅ Very few parameters
- ✅ Position-aware weighting
- ✅ Smooth gradients
- ✅ Interpretable weights

**Cons:**
- ❌ Context-independent (same weights for all samples)
- ❌ Requires fixed max_seq_len
- ❌ Less flexible than attention pooling

**Parameters Added:** `max_seq_len` (e.g., 100 for safety) = **100 parameters**

---

### Option 6: Temporal Attention Pooling (Time-Series Specific)

**Description:** Attention pooling with temporal decay bias

```python
class TemporalAttentionPooling(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        # Learnable temporal decay parameter
        self.temporal_decay = nn.Parameter(torch.tensor(0.1))

    def forward(self, tokens):
        # tokens: [batch, seq_len, d_model]
        batch_size, seq_len, _ = tokens.shape
        query = self.query.expand(batch_size, -1, -1)

        # Create temporal bias (more recent timesteps weighted higher)
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.float)
        temporal_bias = torch.exp(-self.temporal_decay * (seq_len - 1 - positions))
        temporal_bias = temporal_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]

        # Attention with temporal bias
        output, attn_weights = self.attention(query, tokens, tokens)

        # Apply temporal bias to attention
        # (This is a simplified version; full implementation would modify attention scores)
        return output.squeeze(1)
```

**Pros:**
- ✅ Time-series aware (recency bias)
- ✅ Learnable decay parameter
- ✅ More suitable for forecasting
- ✅ Multi-head expressiveness

**Cons:**
- ❌ More complex implementation
- ❌ Adds temporal decay parameter
- ❌ May introduce inductive bias that's not always beneficial

**Parameters Added:** ~65,664 + 1 = **~65,665 parameters**

---

## FT-Transformer Non-CLS Options

### Current FT-Transformer CLS Architecture

```python
# Tokenization
tokens = [CLS, num_token_0, ..., num_token_79, cat_token_0, cat_token_1]
# Shape: [batch, 83, 128]  # 1 CLS + 80 numerical + 2 categorical

# Transformer
transformer_output = transformer(tokens)  # [batch, 83, 128]

# Extraction
cls_output = transformer_output[:, 0, :]  # [batch, 128]

# Prediction
predictions = head(cls_output)  # [batch, output_dim]
```

**Current Parameters (CLS-related):**
- CLS token: 128 parameters
- CLS position in attention: increases attention matrix from [82, 82] to [83, 83]

---

### Proposed FT-Transformer Non-CLS Variants

#### **FT-MeanPool: Mean Pooling Aggregation**

```python
# Tokenization (NO CLS token)
tokens = [num_token_0, ..., num_token_79, cat_token_0, cat_token_1]
# Shape: [batch, 82, 128]  # 80 numerical + 2 categorical

# Transformer
transformer_output = transformer(tokens)  # [batch, 82, 128]

# Aggregation: Mean pooling
aggregated = torch.mean(transformer_output, dim=1)  # [batch, 128]

# Prediction
predictions = head(aggregated)  # [batch, output_dim]
```

**Model Name:** `FT_TRANSFORMER_MEANPOOL`

**Parameters:**
- Remove: CLS token (-128)
- Remove: CLS position in transformer (slight reduction in attention)
- **Net change: -128 parameters, faster attention (82² vs 83²)**

---

#### **FT-AttentionPool: Attention Pooling Aggregation**

```python
# Tokenization (NO CLS token)
tokens = [num_token_0, ..., num_token_79, cat_token_0, cat_token_1]
# Shape: [batch, 82, 128]

# Transformer
transformer_output = transformer(tokens)  # [batch, 82, 128]

# Aggregation: Multi-head attention pooling
aggregated = attention_pooling(transformer_output)  # [batch, 128]

# Prediction
predictions = head(aggregated)  # [batch, output_dim]
```

**Model Name:** `FT_TRANSFORMER_ATTNPOOL`

**Parameters:**
- Remove: CLS token (-128)
- Add: Attention pooling (+65,664)
- **Net change: +65,536 parameters**

---

#### **FT-TemporalPool: Temporal Attention Pooling (Time-Series Optimized)**

```python
# Tokenization (NO CLS token)
tokens = [num_token_0, ..., num_token_79, cat_token_0, cat_token_1]
# Shape: [batch, 82, 128]

# Transformer
transformer_output = transformer(tokens)  # [batch, 82, 128]

# Aggregation: Temporal attention pooling with recency bias
aggregated = temporal_attention_pooling(transformer_output)  # [batch, 128]

# Prediction
predictions = head(aggregated)  # [batch, output_dim]
```

**Model Name:** `FT_TRANSFORMER_TEMPORALPOOL`

**Parameters:**
- Remove: CLS token (-128)
- Add: Temporal attention pooling (+65,665)
- **Net change: +65,537 parameters**

---

## CSN-Transformer Non-CLS Options

### Current CSN-Transformer CLS Architecture

```python
# PATH 1: Categorical
cat_tokens = [CLS₁, cat_token_0, cat_token_1]  # [batch, 3, 128]
cat_output = cat_transformer(cat_tokens)
cls1 = cat_output[:, 0, :]  # [batch, 128]

# PATH 2: Numerical
num_tokens = [CLS₂, num_proj_0, ..., num_proj_9]  # [batch, 11, 128]
num_output = num_transformer(num_tokens)
cls2 = num_output[:, 0, :]  # [batch, 128]

# Fusion
fused = cat([cls1, cls2], dim=1)  # [batch, 256]

# Prediction
predictions = head(fused)  # [batch, output_dim]
```

**Current Parameters (CLS-related):**
- CLS₁ token: 128 parameters
- CLS₂ token: 128 parameters
- **Total: 256 parameters**

---

### Proposed CSN-Transformer Non-CLS Variants

#### **CSN-MeanPool: Dual Mean Pooling**

```python
# PATH 1: Categorical (NO CLS₁)
cat_tokens = [cat_token_0, cat_token_1]  # [batch, 2, 128]
cat_output = cat_transformer(cat_tokens)
cat_aggregated = torch.mean(cat_output, dim=1)  # [batch, 128]

# PATH 2: Numerical (NO CLS₂)
num_tokens = [num_proj_0, ..., num_proj_9]  # [batch, 10, 128]
num_output = num_transformer(num_tokens)
num_aggregated = torch.mean(num_output, dim=1)  # [batch, 128]

# Fusion
fused = cat([cat_aggregated, num_aggregated], dim=1)  # [batch, 256]

# Prediction
predictions = head(fused)  # [batch, output_dim]
```

**Model Name:** `CSN_TRANSFORMER_MEANPOOL`

**Parameters:**
- Remove: CLS₁ token (-128)
- Remove: CLS₂ token (-128)
- **Net change: -256 parameters**

---

#### **CSN-AttentionPool: Dual Attention Pooling**

```python
# PATH 1: Categorical (NO CLS₁)
cat_tokens = [cat_token_0, cat_token_1]  # [batch, 2, 128]
cat_output = cat_transformer(cat_tokens)
cat_aggregated = cat_attention_pooling(cat_output)  # [batch, 128]

# PATH 2: Numerical (NO CLS₂)
num_tokens = [num_proj_0, ..., num_proj_9]  # [batch, 10, 128]
num_output = num_transformer(num_tokens)
num_aggregated = num_attention_pooling(num_output)  # [batch, 128]

# Fusion
fused = cat([cat_aggregated, num_aggregated], dim=1)  # [batch, 256]

# Prediction
predictions = head(fused)  # [batch, output_dim]
```

**Model Name:** `CSN_TRANSFORMER_ATTNPOOL`

**Parameters:**
- Remove: CLS₁ token (-128)
- Remove: CLS₂ token (-128)
- Add: Categorical attention pooling (+65,664)
- Add: Numerical attention pooling (+65,664)
- **Net change: +130,816 parameters**

---

#### **CSN-TemporalPool: Categorical Mean + Numerical Temporal Attention**

**Rationale:** Categorical features don't have temporal order, so mean pooling is sufficient. Numerical sequences benefit from temporal-aware pooling.

```python
# PATH 1: Categorical (Mean pooling - no temporal order)
cat_tokens = [cat_token_0, cat_token_1]  # [batch, 2, 128]
cat_output = cat_transformer(cat_tokens)
cat_aggregated = torch.mean(cat_output, dim=1)  # [batch, 128]

# PATH 2: Numerical (Temporal attention pooling - has temporal order)
num_tokens = [num_proj_0, ..., num_proj_9]  # [batch, 10, 128]
num_output = num_transformer(num_tokens)
num_aggregated = temporal_attention_pooling(num_output)  # [batch, 128]

# Fusion
fused = cat([cat_aggregated, num_aggregated], dim=1)  # [batch, 256]

# Prediction
predictions = head(fused)  # [batch, output_dim]
```

**Model Name:** `CSN_TRANSFORMER_HYBRIDPOOL`

**Parameters:**
- Remove: CLS₁ token (-128)
- Remove: CLS₂ token (-128)
- Add: Numerical temporal attention pooling (+65,665)
- **Net change: +65,409 parameters**

---

## Parameter & Memory Analysis

### FT-Transformer Variants Comparison

| Variant | Total Parameters | vs CLS | Attention Matrix | Memory (Forward) |
|---------|-----------------|--------|------------------|------------------|
| **FT-CLS** (current) | 613,205 | baseline | 32 × 8 × 83² ≈ 3.5 MB | ~11.86 MB |
| **FT-MeanPool** | 613,077 | **-128** | 32 × 8 × 82² ≈ 3.4 MB | ~11.5 MB |
| **FT-AttentionPool** | 678,741 | **+65,536** | 32 × 8 × 82² + pooling | ~12.8 MB |
| **FT-TemporalPool** | 678,742 | **+65,537** | 32 × 8 × 82² + pooling | ~12.8 MB |

### CSN-Transformer Variants Comparison

| Variant | Total Parameters | vs CLS | Attention Complexity | Memory (Forward) |
|---------|------------------|--------|---------------------|------------------|
| **CSN-CLS** (current) | 1,205,845 | baseline | (3² + 11²) × layers | ~628 KB |
| **CSN-MeanPool** | 1,205,589 | **-256** | (2² + 10²) × layers | ~580 KB |
| **CSN-AttentionPool** | 1,336,405 | **+130,560** | (2² + 10²) × layers + pooling | ~750 KB |
| **CSN-HybridPool** | 1,271,254 | **+65,409** | (2² + 10²) × layers + num pooling | ~690 KB |

---

## Recommended Architectures

Based on the analysis, I recommend implementing **3 new model types** (2 for FT, 1 for CSN):

### 1. FT-MeanPool (Simplest Baseline)

**Name:** `FT_TRANSFORMER_MEANPOOL`

**Why:**
- Zero parameter overhead (actually saves 128 params)
- Fastest training/inference (smaller attention matrix)
- Good baseline to validate if CLS is necessary
- Simplest to implement and debug

**Use Case:**
- When parameter efficiency matters
- When CLS token learning may be overkill
- As baseline for comparison

---

### 2. FT-TemporalPool (Time-Series Optimized)

**Name:** `FT_TRANSFORMER_TEMPORALPOOL`

**Why:**
- Time-series specific design (recency bias)
- Learnable temporal weighting
- May outperform CLS for forecasting tasks
- Interpretable attention weights

**Use Case:**
- When temporal patterns are crucial
- When you want interpretable aggregation
- When willing to trade parameters for performance

---

### 3. CSN-HybridPool (Domain-Optimized)

**Name:** `CSN_TRANSFORMER_HYBRIDPOOL`

**Why:**
- Best of both worlds: mean for categorical, temporal for numerical
- Domain-aware design (categorical has no temporal order)
- Moderate parameter increase (+65K vs +130K for dual attention)
- Should outperform mean-only and be more efficient than dual attention

**Use Case:**
- When categorical and numerical have different aggregation needs
- When parameter efficiency matters (vs dual attention)
- Best suited for stock forecasting with static categorical + temporal numerical

---

## Implementation Plan

### Phase 1: Create Pooling Modules (New File)

**File:** `tf_predictor/core/base/pooling.py`

```python
"""
Pooling strategies for transformer sequence aggregation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class MeanPooling(nn.Module):
    """Simple mean pooling across sequence dimension."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=1)

class MultiHeadAttentionPooling(nn.Module):
    """Learnable multi-head attention pooling."""
    def __init__(self, d_model: int, num_heads: int = 8):
        # Implementation details
        pass

class TemporalAttentionPooling(nn.Module):
    """Temporal attention pooling with recency bias."""
    def __init__(self, d_model: int, num_heads: int = 8):
        # Implementation details
        pass
```

### Phase 2: Create Non-CLS Model Classes

**File 1:** `tf_predictor/core/ft_model.py` (modify existing)

Add new classes:
- `FTTransformerMeanPoolModel`
- `FTTransformerTemporalPoolModel`

**File 2:** `tf_predictor/core/csn_model.py` (modify existing)

Add new class:
- `CSNTransformerHybridPoolModel`

### Phase 3: Update Model Registry

**File:** `tf_predictor/core/predictor.py`

```python
# Add to model registry
MODEL_REGISTRY = {
    # Existing
    'ft_transformer_cls': FTTransformerCLSModel,
    'csn_transformer_cls': CSNTransformerCLSModel,

    # New non-CLS variants
    'ft_transformer_meanpool': FTTransformerMeanPoolModel,
    'ft_transformer_temporalpool': FTTransformerTemporalPoolModel,
    'csn_transformer_hybridpool': CSNTransformerHybridPoolModel,
}
```

### Phase 4: Testing

Create comprehensive tests:
- Input/output shape verification
- Parameter counting validation
- Forward pass correctness
- Gradient flow verification
- Comparison with CLS variants

### Phase 5: Documentation

- Update README with new model types
- Add architecture diagrams
- Document when to use each variant
- Add performance comparison section (after empirical results)

---

## Expected Performance Characteristics

### Computational Efficiency

**Training Speed (relative to CLS):**
- **MeanPool:** ~5% faster (smaller attention, no CLS learning)
- **AttentionPool:** ~3% slower (extra pooling attention)
- **TemporalPool:** ~3% slower (extra pooling + temporal computation)

**Inference Speed (relative to CLS):**
- **MeanPool:** ~5% faster
- **AttentionPool:** ~2% slower
- **TemporalPool:** ~2% slower

### Convergence

**Expected Epochs to Convergence:**
- **MeanPool:** May converge slightly slower (less learnable aggregation)
- **TemporalPool:** May converge faster (temporal inductive bias)
- **AttentionPool:** Similar to CLS

### Generalization

**Expected Test Performance:**
- **MeanPool:** Good baseline, may struggle with complex aggregation needs
- **TemporalPool:** Potentially best for time series (temporal bias matches domain)
- **AttentionPool:** Similar to CLS, more interpretable

### Memory Usage

**Peak GPU Memory (training):**
- **MeanPool:** Lowest (no extra parameters, smaller attention)
- **AttentionPool:** Slightly higher than CLS
- **TemporalPool:** Slightly higher than CLS

---

## Questions for Decision

Before proceeding with implementation, please confirm:

### 1. Architecture Selection

**Question:** Do you agree with the recommended 3 variants, or would you like to add/remove any?

Recommended:
- ✅ FT-MeanPool (simplest baseline)
- ✅ FT-TemporalPool (time-series optimized)
- ✅ CSN-HybridPool (domain-optimized)

Alternative options to consider:
- FT-AttentionPool (standard attention pooling)
- CSN-MeanPool (dual mean pooling)
- CSN-AttentionPool (dual attention pooling)

### 2. Temporal Pooling Design

**Question:** For TemporalAttentionPooling, should we:

**Option A:** Learnable temporal decay (exponential recency bias)
```python
temporal_bias = exp(-learnable_decay * (T - t))
```

**Option B:** Fixed recent-heavy weighting
```python
temporal_bias = fixed_weights[t]  # e.g., [0.1, 0.1, ..., 0.5, 1.0]
```

**Option C:** Full learnable per-position weights
```python
temporal_bias = learnable_weights[t]
```

**Recommendation:** Option A (learnable decay) - flexible but interpretable

### 3. Numerical Feature Aggregation in CSN-HybridPool

**Question:** For CSN numerical path, should we use:

**Option A:** TemporalAttentionPooling (time-aware, +65K params)
**Option B:** Standard attention pooling (no temporal bias, +65K params)
**Option C:** Weighted mean with learnable position weights (minimal params)

**Recommendation:** Option A (temporal) - best for time series

### 4. Testing Strategy

**Question:** What metrics should we use for comparison?

Proposed metrics:
- ✅ MAE/MSE on validation set
- ✅ Per-group performance (for grouped forecasting)
- ✅ Training time per epoch
- ✅ Inference time
- ✅ Peak memory usage
- ✅ Parameter count
- ✅ Convergence speed (epochs to reach threshold)

Any additional metrics?

### 5. Backward Compatibility

**Question:** Should we support loading old CLS models with new code?

**Recommendation:** Yes - keep CLS models unchanged, add new variants as separate classes

### 6. Default Model Type

**Question:** After testing, should we change the default model type if non-CLS performs better?

**Current default:** `ft_transformer_cls` / `csn_transformer_cls`

**Recommendation:** Keep CLS as default until empirical results prove non-CLS is superior

---

## Next Steps

Once you approve the architecture and answer the questions above, I will:

1. ✅ Implement pooling modules in `tf_predictor/core/base/pooling.py`
2. ✅ Create non-CLS model classes
3. ✅ Update model registry
4. ✅ Write comprehensive tests
5. ✅ Update documentation
6. ✅ Create comparison benchmarks

**Estimated Implementation Time:** 2-3 hours
**Estimated Testing Time:** 1 hour
**Estimated Documentation Time:** 30 minutes

---

## Appendix: Detailed Parameter Calculations

### FT-MeanPool Parameter Breakdown
```
Numerical Tokenizer:     2,048
Categorical Embeddings:  16,340
Temporal Pos Encoding:   1,280
Transformer (3 layers):  593,280
Prediction Head:         129
─────────────────────────────
TOTAL:                   613,077 (-128 vs CLS)
```

### FT-TemporalPool Parameter Breakdown
```
Numerical Tokenizer:          2,048
Categorical Embeddings:       16,340
Temporal Pos Encoding:        1,280
Transformer (3 layers):       593,280
Temporal Attention Pooling:   65,665
Prediction Head:              129
─────────────────────────────────
TOTAL:                        678,742 (+65,537 vs CLS)
```

### CSN-HybridPool Parameter Breakdown
```
Categorical Path:
  - Embeddings:              16,340
  - Transformer (3 layers):  593,280
  - Subtotal:                609,620

Numerical Path:
  - Projection:              1,152
  - Pos Encoding:            1,280
  - Transformer (3 layers):  593,280
  - Temporal Pooling:        65,665
  - Subtotal:                661,377

Prediction Head:             257
─────────────────────────────────
TOTAL:                       1,271,254 (+65,409 vs CLS)
```

---

**END OF PROPOSAL**

---

## Approval Checklist

Please review and approve/modify:

- [ ] Recommended architectures (FT-MeanPool, FT-TemporalPool, CSN-HybridPool)
- [ ] Temporal pooling design (learnable decay vs fixed vs full learnable)
- [ ] Numerical aggregation strategy for CSN
- [ ] Testing metrics
- [ ] Implementation plan
- [ ] Any other concerns or modifications

Once approved, I'll proceed with implementation!
