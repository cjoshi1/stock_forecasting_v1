# Pooling Strategies Implementation Plan

**Date:** 2025-11-06
**Status:** Planning Phase
**Goal:** Implement 4 pooling strategies for both FT-Transformer and CSN-Transformer architectures

---

## Table of Contents
1. [Overview](#overview)
2. [Model Nomenclature](#model-nomenclature)
3. [Architecture Plan](#architecture-plan)
4. [Implementation Strategy](#implementation-strategy)
5. [File Organization](#file-organization)
6. [API Design](#api-design)
7. [Testing Plan](#testing-plan)

---

## Overview

### Current State
- ✅ **FTTransformerCLSModel**: FT-Transformer with CLS token aggregation
- ✅ **CSNTransformerCLSModel**: CSN-Transformer with CLS token aggregation (dual-path)

### Target State
Add 4 additional pooling strategies to both architectures:
1. **Attention Pooling** (single-head)
2. **Multi-Head Attention Pooling**
3. **Weighted Average** (learnable position weights)
4. **Temporal Attention Pooling** (multi-head + recency bias)

**Total:** 5 pooling types × 2 architectures = 10 model variants

---

## Model Nomenclature

### Proposed Naming Convention

We have three options:

#### **Option A: Separate Classes (Explicit)**
```python
# FT-Transformer variants
FTTransformerCLS                    # Current implementation
FTTransformerAttentionPooling       # Single-head attention pooling
FTTransformerMultiHeadPooling       # Multi-head attention pooling
FTTransformerWeightedAvg            # Learnable weighted average
FTTransformerTemporalPooling        # Temporal attention pooling

# CSN-Transformer variants
CSNTransformerCLS                   # Current implementation
CSNTransformerAttentionPooling      # Single-head attention pooling
CSNTransformerMultiHeadPooling      # Multi-head attention pooling
CSNTransformerWeightedAvg           # Learnable weighted average
CSNTransformerTemporalPooling       # Temporal attention pooling
```

**Pros:**
- ✅ Explicit and clear naming
- ✅ Easy to understand what each model does
- ✅ Type-safe (each class is distinct)

**Cons:**
- ❌ Code duplication across classes
- ❌ 10 separate classes to maintain
- ❌ Hard to add new pooling strategies

---

#### **Option B: Single Class with Parameter (Dynamic)**
```python
# FT-Transformer with pooling parameter
FTTransformer(pooling_type='cls')
FTTransformer(pooling_type='attention')
FTTransformer(pooling_type='multihead')
FTTransformer(pooling_type='weighted_avg')
FTTransformer(pooling_type='temporal')

# CSN-Transformer with pooling parameter
CSNTransformer(pooling_type='cls')
CSNTransformer(pooling_type='attention')
CSNTransformer(pooling_type='multihead')
CSNTransformer(pooling_type='weighted_avg')
CSNTransformer(pooling_type='temporal')
```

**Pros:**
- ✅ Minimal code duplication
- ✅ Easy to add new pooling strategies
- ✅ Single point of configuration
- ✅ Flexible and extensible

**Cons:**
- ❌ Runtime parameter validation needed
- ❌ Less explicit in code
- ❌ Type hints less specific

---

#### **Option C: Hybrid (Composition Pattern)**
```python
# Create pooling modules separately
pooling_cls = CLSTokenPooling(d_token=128)
pooling_attention = AttentionPooling(d_token=128)
pooling_multihead = MultiHeadPooling(d_token=128, n_heads=8)
pooling_weighted = WeightedAveragePooling(d_token=128)
pooling_temporal = TemporalAttentionPooling(d_token=128, n_heads=8)

# Inject into base models
model = FTTransformer(pooling_module=pooling_attention, ...)
model = CSNTransformer(pooling_module=pooling_multihead, ...)
```

**Pros:**
- ✅ Maximum flexibility
- ✅ Easy to test pooling modules independently
- ✅ Clean separation of concerns
- ✅ Easy to add custom pooling strategies

**Cons:**
- ❌ More complex initialization
- ❌ Users need to understand composition

---

### **Recommended Approach: Option B (with Option C internally)**

**External API:** Simple string parameter
```python
model = FTTransformer(pooling_type='attention', d_token=128, n_heads=8)
model = CSNTransformer(pooling_type='multihead', d_token=128, n_heads=8)
```

**Internal Implementation:** Composition pattern with pooling modules
```python
# Inside FTTransformer.__init__
self.pooling = self._create_pooling_module(pooling_type)
```

This gives us:
- ✅ Simple user-facing API (Option B)
- ✅ Clean modular code (Option C)
- ✅ Easy to extend with new strategies
- ✅ Easy to test

---

## Architecture Plan

### FT-Transformer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FT-TRANSFORMER PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Features: [batch, seq_len, n_features]                   │
│         ↓                                                        │
│  Feature Tokenization (Project to d_token)                      │
│         ↓                                                        │
│  [batch, seq_len, d_token]                                      │
│         ↓                                                        │
│  ╔═══════════════════════════════════════════════════════╗      │
│  ║          TRANSFORMER ENCODER LAYERS                   ║      │
│  ║  • Self-Attention                                     ║      │
│  ║  • Feed-Forward                                       ║      │
│  ║  • Layer Normalization                               ║      │
│  ║  (Repeated n_layers times)                           ║      │
│  ╚═══════════════════════════════════════════════════════╝      │
│         ↓                                                        │
│  [batch, seq_len, d_token]                                      │
│         ↓                                                        │
│  ┌─────────────────────────────────────────────────────┐       │
│  │         POOLING STRATEGY (Configurable)             │       │
│  ├─────────────────────────────────────────────────────┤       │
│  │  Option 1: CLS Token (extract position 0)          │       │
│  │  Option 2: Attention Pooling (learnable query)     │       │
│  │  Option 3: Multi-Head Pooling (n_heads queries)    │       │
│  │  Option 4: Weighted Avg (learnable weights)        │       │
│  │  Option 5: Temporal Pooling (time-aware attention) │       │
│  └─────────────────────────────────────────────────────┘       │
│         ↓                                                        │
│  [batch, d_token]                                               │
│         ↓                                                        │
│  Prediction Head (Multi-Horizon)                                │
│         ↓                                                        │
│  [batch, prediction_horizon] or [batch, n_targets × horizons]  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### CSN-Transformer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   CSN-TRANSFORMER PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Categorical Features          Numerical Features               │
│  [batch, n_categorical]        [batch, seq_len, n_numerical]    │
│         ↓                               ↓                        │
│  Embed to d_token                Project to d_token             │
│         ↓                               ↓                        │
│  ╔═══════════════════╗        ╔════════════════════╗           │
│  ║ CATEGORICAL PATH  ║        ║   NUMERICAL PATH   ║           │
│  ║ Transformer       ║        ║   Transformer      ║           │
│  ║ (n_layers)        ║        ║   (n_layers)       ║           │
│  ╚═══════════════════╝        ╚════════════════════╝           │
│         ↓                               ↓                        │
│  [batch, n_cat, d_token]       [batch, seq_len, d_token]       │
│         ↓                               ↓                        │
│  ┌─────────────────┐            ┌──────────────────┐           │
│  │ POOLING STRATEGY│            │ POOLING STRATEGY │           │
│  │ (Same as above) │            │ (Same as above)  │           │
│  └─────────────────┘            └──────────────────┘           │
│         ↓                               ↓                        │
│  [batch, d_token]              [batch, d_token]                │
│         └───────────────┬───────────────┘                       │
│                         ↓                                        │
│                   Concatenate                                    │
│                         ↓                                        │
│                [batch, 2 × d_token]                             │
│                         ↓                                        │
│              Prediction Head (Multi-Horizon)                    │
│                         ↓                                        │
│  [batch, prediction_horizon] or [batch, n_targets × horizons]  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Point:** CSN applies the SAME pooling strategy to BOTH pathways independently, then concatenates.

---

## Implementation Strategy

### Phase 1: Create Pooling Modules (New File)

**File:** `tf_predictor/core/base/pooling.py`

Create base class and 5 implementations:

```python
class PoolingModule(nn.Module):
    """Base class for all pooling strategies."""

    def __init__(self, d_token: int):
        super().__init__()
        self.d_token = d_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_token]
        Returns:
            output: [batch, d_token]
        """
        raise NotImplementedError


class CLSTokenPooling(PoolingModule):
    """Extract CLS token at position 0."""
    def forward(self, x):
        return x[:, 0, :]  # [batch, d_token]


class AttentionPooling(PoolingModule):
    """Single-head attention pooling with learnable query."""
    def __init__(self, d_token: int, dropout: float = 0.1):
        super().__init__(d_token)
        self.query = nn.Parameter(torch.randn(1, 1, d_token))
        self.attention = nn.MultiheadAttention(d_token, num_heads=1, dropout=dropout, batch_first=True)

    def forward(self, x):
        # Expand query for batch: [1, 1, d_token] → [batch, 1, d_token]
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)

        # Attention: query attends to all tokens
        output, _ = self.attention(query, x, x)  # [batch, 1, d_token]
        return output.squeeze(1)  # [batch, d_token]


class MultiHeadPooling(PoolingModule):
    """Multi-head attention pooling."""
    def __init__(self, d_token: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__(d_token)
        self.query = nn.Parameter(torch.randn(1, 1, d_token))
        self.attention = nn.MultiheadAttention(d_token, num_heads=n_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        output, _ = self.attention(query, x, x)
        return output.squeeze(1)


class WeightedAveragePooling(PoolingModule):
    """Learnable weighted average over sequence positions."""
    def __init__(self, d_token: int, max_seq_len: int = 100):
        super().__init__(d_token)
        # Learnable weights for each position
        self.weights = nn.Parameter(torch.randn(1, max_seq_len, 1))

    def forward(self, x):
        # x: [batch, seq_len, d_token]
        seq_len = x.size(1)

        # Get weights for current sequence length
        w = self.weights[:, :seq_len, :]  # [1, seq_len, 1]
        w = torch.softmax(w, dim=1)       # Normalize

        # Weighted sum: [batch, seq_len, d_token] * [1, seq_len, 1]
        output = (x * w).sum(dim=1)  # [batch, d_token]
        return output


class TemporalAttentionPooling(PoolingModule):
    """Multi-head attention with temporal/recency bias."""
    def __init__(self, d_token: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__(d_token)
        self.query = nn.Parameter(torch.randn(1, 1, d_token))
        self.attention = nn.MultiheadAttention(d_token, num_heads=n_heads, dropout=dropout, batch_first=True)

        # Learnable temporal bias (recent timesteps weighted higher)
        self.temporal_bias = nn.Parameter(torch.randn(1, 100, 1))  # Max 100 timesteps

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Add temporal bias to keys
        bias = self.temporal_bias[:, :seq_len, :]  # [1, seq_len, 1]
        x_biased = x + bias  # Broadcast: [batch, seq_len, d_token]

        # Attention with biased keys
        query = self.query.expand(batch_size, -1, -1)
        output, _ = self.attention(query, x_biased, x)  # Query biased keys, get original values
        return output.squeeze(1)
```

---

### Phase 2: Update FT-Transformer

**File:** `tf_predictor/core/ft_model.py`

```python
class FTTransformer(TransformerBasedModel):
    """FT-Transformer with configurable pooling strategy."""

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        d_token: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        pooling_type: str = 'cls',  # NEW PARAMETER
        dropout: float = 0.1,
        activation: str = 'relu',
        prediction_horizons: int = 1
    ):
        super().__init__(d_token=d_token, n_heads=n_heads, n_layers=n_layers)

        # Feature tokenization
        self.feature_projection = nn.Linear(num_features, d_token)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, sequence_length, d_token) * 0.02)

        # Add CLS token only if using CLS pooling
        if pooling_type == 'cls':
            from .base.embeddings import CLSToken
            self.cls_token = CLSToken(d_token)
            transformer_seq_len = sequence_length + 1
        else:
            self.cls_token = None
            transformer_seq_len = sequence_length

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=4 * d_token,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Create pooling module
        from .base.pooling import create_pooling_module
        self.pooling = create_pooling_module(
            pooling_type=pooling_type,
            d_token=d_token,
            n_heads=n_heads,
            max_seq_len=transformer_seq_len,
            dropout=dropout
        )

        # Prediction head
        self.prediction_head = MultiHorizonHead(d_token, prediction_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, num_features]

        # Project features to tokens
        x = self.feature_projection(x)  # [batch, seq_len, d_token]

        # Add positional encoding
        x = x + self.pos_encoding

        # Optionally prepend CLS token
        if self.cls_token is not None:
            x = self.cls_token(x)  # [batch, seq_len+1, d_token]

        # Transformer
        x = self.transformer(x)  # [batch, seq_len(+1), d_token]

        # Pooling to aggregate sequence
        x = self.pooling(x)  # [batch, d_token]

        # Prediction
        output = self.prediction_head(x)  # [batch, prediction_horizons]

        return output
```

---

### Phase 3: Update CSN-Transformer

**File:** `tf_predictor/core/csn_model.py`

Similar pattern - add `pooling_type` parameter and apply pooling to both pathways:

```python
class CSNTransformer(nn.Module):
    """CSN-Transformer with configurable pooling strategy."""

    def __init__(
        self,
        categorical_features: Dict[str, int],
        num_numerical_features: int,
        sequence_length: int,
        d_token: int = 128,
        n_layers: int = 3,
        n_heads: int = 8,
        pooling_type: str = 'cls',  # NEW PARAMETER
        dropout: float = 0.1,
        output_dim: int = 1,
        prediction_horizons: int = 1
    ):
        super().__init__()

        # ... (existing setup code)

        # Create pooling modules for BOTH pathways
        from .base.pooling import create_pooling_module

        # Categorical pathway pooling
        self.categorical_pooling = create_pooling_module(
            pooling_type=pooling_type,
            d_token=d_token,
            n_heads=n_heads,
            max_seq_len=len(categorical_features) + (1 if pooling_type == 'cls' else 0),
            dropout=dropout
        )

        # Numerical pathway pooling
        self.numerical_pooling = create_pooling_module(
            pooling_type=pooling_type,
            d_token=d_token,
            n_heads=n_heads,
            max_seq_len=sequence_length + (1 if pooling_type == 'cls' else 0),
            dropout=dropout
        )

        # ... (rest of init)

    def forward(self, categorical_inputs, numerical_inputs):
        # Process categorical pathway
        cat_tokens = self.categorical_processor(categorical_inputs)  # [batch, n_cat(+1), d_token]
        cat_pooled = self.categorical_pooling(cat_tokens)            # [batch, d_token]

        # Process numerical pathway
        num_tokens = self.numerical_processor(numerical_inputs)      # [batch, seq_len(+1), d_token]
        num_pooled = self.numerical_pooling(num_tokens)              # [batch, d_token]

        # Concatenate both pathways
        fused = torch.cat([cat_pooled, num_pooled], dim=-1)         # [batch, 2*d_token]

        # Prediction
        output = self.prediction_head(fused)

        return output
```

---

### Phase 4: Update TimeSeriesPredictor

**File:** `tf_predictor/core/predictor.py`

Add `pooling_type` parameter:

```python
class TimeSeriesPredictor:
    def __init__(
        self,
        target_column: Union[str, List[str]],
        model_type: str = 'ft_transformer_cls',
        pooling_type: str = 'cls',  # NEW PARAMETER
        # ... other parameters
    ):
        self.pooling_type = pooling_type
        # ...

    def _create_model(self):
        if 'ft_transformer' in self.model_type:
            return FTTransformer(
                pooling_type=self.pooling_type,  # Pass to model
                # ... other parameters
            )
        elif 'csn_transformer' in self.model_type:
            return CSNTransformer(
                pooling_type=self.pooling_type,  # Pass to model
                # ... other parameters
            )
```

---

## File Organization

```
tf_predictor/
├── core/
│   ├── base/
│   │   ├── __init__.py
│   │   ├── model_interface.py       # Base classes (existing)
│   │   ├── embeddings.py            # CLSToken, etc. (existing)
│   │   ├── pooling.py               # NEW: All pooling modules
│   │   └── prediction_heads.py      # Prediction heads (existing)
│   ├── ft_model.py                  # MODIFIED: Add pooling_type
│   ├── csn_model.py                 # MODIFIED: Add pooling_type
│   └── predictor.py                 # MODIFIED: Add pooling_type
├── tests/
│   └── test_pooling.py              # NEW: Unit tests for pooling
└── docs/
    └── POOLING_IMPLEMENTATION_PLAN.md  # This file
```

---

## API Design

### User-Facing API (Simple)

```python
from tf_predictor.core.predictor import TimeSeriesPredictor

# Option 1: CLS token (current default)
predictor = TimeSeriesPredictor(
    target_column='close',
    model_type='ft_transformer',
    pooling_type='cls',  # Default
    d_token=128,
    n_heads=8,
    n_layers=3
)

# Option 2: Attention pooling
predictor = TimeSeriesPredictor(
    target_column='close',
    model_type='ft_transformer',
    pooling_type='attention',
    d_token=128,
    n_heads=8,
    n_layers=3
)

# Option 3: Multi-head pooling
predictor = TimeSeriesPredictor(
    target_column='close',
    model_type='csn_transformer',
    pooling_type='multihead',
    d_token=128,
    n_heads=8,  # Used for pooling
    n_layers=3
)

# Option 4: Weighted average
predictor = TimeSeriesPredictor(
    target_column='close',
    model_type='ft_transformer',
    pooling_type='weighted_avg',
    d_token=128,
    n_heads=8,
    n_layers=3
)

# Option 5: Temporal attention
predictor = TimeSeriesPredictor(
    target_column='close',
    model_type='ft_transformer',
    pooling_type='temporal',
    d_token=128,
    n_heads=8,
    n_layers=3
)
```

### Valid Values

```python
VALID_POOLING_TYPES = [
    'cls',           # CLS token (extract position 0)
    'attention',     # Single-head attention pooling
    'multihead',     # Multi-head attention pooling
    'weighted_avg',  # Learnable weighted average
    'temporal'       # Temporal attention pooling with recency bias
]
```

---

## Testing Plan

### Unit Tests (`test_pooling.py`)

```python
def test_cls_token_pooling():
    """Test CLS token extraction."""
    pooling = CLSTokenPooling(d_token=128)
    x = torch.randn(32, 10, 128)  # [batch, seq_len, d_token]
    output = pooling(x)
    assert output.shape == (32, 128)
    assert torch.allclose(output, x[:, 0, :])  # Should match first token


def test_attention_pooling():
    """Test single-head attention pooling."""
    pooling = AttentionPooling(d_token=128)
    x = torch.randn(32, 10, 128)
    output = pooling(x)
    assert output.shape == (32, 128)


def test_multihead_pooling():
    """Test multi-head attention pooling."""
    pooling = MultiHeadPooling(d_token=128, n_heads=8)
    x = torch.randn(32, 10, 128)
    output = pooling(x)
    assert output.shape == (32, 128)


def test_weighted_average_pooling():
    """Test weighted average pooling."""
    pooling = WeightedAveragePooling(d_token=128, max_seq_len=100)
    x = torch.randn(32, 10, 128)
    output = pooling(x)
    assert output.shape == (32, 128)


def test_temporal_pooling():
    """Test temporal attention pooling."""
    pooling = TemporalAttentionPooling(d_token=128, n_heads=8)
    x = torch.randn(32, 10, 128)
    output = pooling(x)
    assert output.shape == (32, 128)


def test_ft_transformer_with_pooling():
    """Test FT-Transformer with different pooling strategies."""
    for pooling_type in ['cls', 'attention', 'multihead', 'weighted_avg', 'temporal']:
        model = FTTransformer(
            num_features=10,
            sequence_length=20,
            d_token=128,
            n_heads=8,
            n_layers=2,
            pooling_type=pooling_type,
            prediction_horizons=1
        )

        x = torch.randn(32, 20, 10)
        output = model(x)
        assert output.shape == (32, 1)


def test_csn_transformer_with_pooling():
    """Test CSN-Transformer with different pooling strategies."""
    categorical_features = {'symbol': 10, 'sector': 5}

    for pooling_type in ['cls', 'attention', 'multihead', 'weighted_avg', 'temporal']:
        model = CSNTransformer(
            categorical_features=categorical_features,
            num_numerical_features=8,
            sequence_length=20,
            d_token=128,
            n_heads=8,
            n_layers=2,
            pooling_type=pooling_type,
            prediction_horizons=1
        )

        cat_inputs = {'symbol': torch.randint(0, 10, (32,)), 'sector': torch.randint(0, 5, (32,))}
        num_inputs = torch.randn(32, 20, 8)

        output = model(cat_inputs, num_inputs)
        assert output.shape == (32, 1)
```

### Integration Tests

```python
def test_end_to_end_with_pooling():
    """Test full pipeline with different pooling types."""
    df = create_test_data()

    for pooling_type in ['cls', 'attention', 'multihead', 'weighted_avg', 'temporal']:
        predictor = TimeSeriesPredictor(
            target_column='close',
            model_type='ft_transformer',
            pooling_type=pooling_type,
            sequence_length=10,
            d_token=32,
            n_heads=4,
            n_layers=2
        )

        # Train
        predictor.fit(df, epochs=2)

        # Predict
        predictions = predictor.predict(df)
        assert predictions.shape[0] > 0

        # Evaluate
        metrics = predictor.evaluate(df)
        assert 'mae' in metrics
```

---

## Summary

### Proposed Plan

1. **Nomenclature:** Use `pooling_type='cls'|'attention'|'multihead'|'weighted_avg'|'temporal'` parameter
2. **Implementation:** Create separate pooling modules in `pooling.py`, compose them in models
3. **API:** Simple string parameter exposed through `TimeSeriesPredictor`
4. **Testing:** Comprehensive unit and integration tests

### Next Steps

1. ✅ Review and approve this plan
2. ⏳ Implement Phase 1: Create `pooling.py` with all modules
3. ⏳ Implement Phase 2: Update `FTTransformer`
4. ⏳ Implement Phase 3: Update `CSNTransformer`
5. ⏳ Implement Phase 4: Update `TimeSeriesPredictor`
6. ⏳ Write unit tests
7. ⏳ Write integration tests
8. ⏳ Update documentation

---

**Questions for Review:**

1. Do you agree with the **nomenclature** (Option B: `pooling_type` parameter)?
2. Should we keep CLS token implementation or refactor it to use the pooling module?
3. Any changes to the pooling module designs?
4. Should `n_heads` parameter be used for pooling or separate `pooling_n_heads`?
5. For CSN-Transformer, should both pathways use the same pooling or allow different pooling per pathway?

Please review and let me know if this plan looks good to proceed with implementation!
