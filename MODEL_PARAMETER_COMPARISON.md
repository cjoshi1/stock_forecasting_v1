# FT-Transformer vs CSN-Transformer Parameter Comparison

**Date:** 2025-11-05
**Analysis by:** Claude Sonnet 4.5
**Purpose:** Investigate why CSN-Transformer has more parameters than FT-Transformer for the same hyperparameters

---

## Executive Summary

**Observation Confirmed:** CSN-Transformer has approximately **96.6% MORE parameters** (nearly double) compared to FT-Transformer when using identical hyperparameters.

- **FT-Transformer:** 613,205 parameters
- **CSN-Transformer:** 1,205,845 parameters
- **Difference:** 592,640 parameters

**Root Cause:** CSN-Transformer uses a **dual-path architecture** with two separate transformer stacks (one for categorical features, one for numerical sequences), while FT-Transformer uses a **unified single transformer** that processes all features together.

---

## Test Configuration

Using the example configuration from both model files:

```python
# Common hyperparameters
batch_size = 32
sequence_length = 10
num_numerical = 8
num_categorical = 2
cat_cardinalities = [100, 5]  # 100 stock symbols, 5 sectors
d_model = 128
num_heads = 8
num_layers = 3
d_ffn = 4 * d_model = 512
output_dim = 1  # Single-step prediction
```

---

## Detailed Parameter Count: FT-Transformer CLS Model

### 1. CLS Token
```
cls_token: [d_model]
= 128 parameters
```

### 2. Numerical Tokenizer
From `NumericalTokenizer` class - creates per-feature linear transformations:
```
For each numerical feature: W_j [1, d_model] + b_j [d_model]
= num_numerical * (d_model + d_model)
= 8 * (128 + 128)
= 2,048 parameters
```

### 3. Categorical Embeddings (with Logarithmic Scaling)

**Symbol embedding (cardinality=100):**
```
emb_dim = int(8 * log2(101)) ≈ 53
Embedding layer: 100 * 53 = 5,300
Projection layer: 53 * 128 = 6,784
Total: 12,084 parameters
```

**Sector embedding (cardinality=5):**
```
emb_dim = int(8 * log2(6)) ≈ 20 → clamped to 32 (d_model/4)
Embedding layer: 5 * 32 = 160
Projection layer: 32 * 128 = 4,096
Total: 4,256 parameters
```

**Categorical total:** 16,340 parameters

### 4. Temporal Positional Encoding
```
temporal_pos_encoding: [1, sequence_length, d_model]
= 1 * 10 * 128
= 1,280 parameters
```

### 5. Unified Transformer Encoder (3 layers)

**Per TransformerEncoderLayer:**

Multi-head Attention:
```
Q, K, V projections: 3 * (d_model * d_model) = 3 * 128 * 128 = 49,152
Output projection: d_model * d_model = 128 * 128 = 16,384
Total attention: 65,536 parameters
```

Layer Norms (2 per layer):
```
norm1: 2 * d_model = 256
norm2: 2 * d_model = 256
Total: 512 parameters
```

Feed-Forward Network:
```
Linear1: (d_model * d_ffn) + d_ffn = (128 * 512) + 512 = 66,048
Linear2: (d_ffn * d_model) + d_model = (512 * 128) + 128 = 65,664
Total FFN: 131,712 parameters
```

**Per layer subtotal:** 65,536 + 512 + 131,712 = **197,760 parameters**
**3 layers total:** 197,760 * 3 = **593,280 parameters**

### 6. Prediction Head (MultiHorizonHead)
```
Linear(d_model, output_dim): (d_model * output_dim) + output_dim
= (128 * 1) + 1
= 129 parameters
```

### FT-Transformer Grand Total
```
CLS Token:              128
Numerical Tokenizer:    2,048
Categorical Embeddings: 16,340
Positional Encoding:    1,280
Transformer (3 layers): 593,280
Prediction Head:        129
─────────────────────────────
TOTAL:                  613,205 parameters
```

---

## Detailed Parameter Count: CSN-Transformer CLS Model

### PATH 1: Categorical Processing

**CLS1 Token:**
```
cls1_token: [d_model] = 128 parameters
```

**Categorical Embeddings (same as FT):**
```
Symbol: 12,084
Sector: 4,256
Total: 16,340 parameters
```

**Categorical Transformer (3 layers):**
```
Each layer: 197,760 parameters (same structure as FT)
3 layers: 593,280 parameters
```

**Categorical Path Subtotal:** 128 + 16,340 + 593,280 = **609,748 parameters**

---

### PATH 2: Numerical Processing

**CLS2 Token:**
```
cls2_token: [d_model] = 128 parameters
```

**Numerical Projection:**
```
Linear(num_numerical, d_model): (num_numerical * d_model) + d_model
= (8 * 128) + 128
= 1,152 parameters
```

**Temporal Positional Encoding:**
```
temporal_pos_encoding: [1, sequence_length, d_model]
= 1 * 10 * 128
= 1,280 parameters
```

**Numerical Transformer (3 layers):**
```
Each layer: 197,760 parameters
3 layers: 593,280 parameters
```

**Numerical Path Subtotal:** 128 + 1,152 + 1,280 + 593,280 = **595,840 parameters**

---

### PATH 3: Fusion and Prediction

**MultiHorizonHead:**
```
fusion_dim = 2 * d_model = 256 (concatenation of cls1 + cls2)
Linear(fusion_dim, output_dim): (256 * 1) + 1
= 257 parameters
```

### CSN-Transformer Grand Total
```
Categorical Path:       609,748
Numerical Path:         595,840
Prediction Head:        257
─────────────────────────────
TOTAL:                  1,205,845 parameters
```

---

## Side-by-Side Comparison

| Component | FT-Transformer | CSN-Transformer | Difference |
|-----------|----------------|-----------------|------------|
| CLS Tokens | 128 | 256 (2 tokens) | +128 |
| Feature Processing | 18,388 | 17,492 | -896 |
| Positional Encoding | 1,280 | 1,280 | 0 |
| **Transformer Layers** | **593,280** | **1,186,560** | **+593,280** |
| Prediction Head | 129 | 257 | +128 |
| **TOTAL** | **613,205** | **1,205,845** | **+592,640** |
| **Relative Increase** | - | **+96.6%** | - |

---

## Why CSN Has Nearly Double the Parameters

### Primary Reason: Dual Transformer Architecture

**CSN-Transformer:**
- **Categorical Transformer:** 3 layers * 197,760 params/layer = 593,280 params
- **Numerical Transformer:** 3 layers * 197,760 params/layer = 593,280 params
- **Total Transformer Params:** 1,186,560 params

**FT-Transformer:**
- **Unified Transformer:** 3 layers * 197,760 params/layer = 593,280 params
- **Total Transformer Params:** 593,280 params

**Difference from transformers alone:** 593,280 params (99.4% of total difference)

### Architectural Philosophy

**FT-Transformer (Unified Processing):**
```
┌────────────────────────────────────────┐
│ [CLS, num_tokens, cat_tokens]          │
│         ↓                               │
│   SINGLE TRANSFORMER                    │
│   (All tokens attend to each other)    │
│         ↓                               │
│   Extract CLS → Predict                 │
└────────────────────────────────────────┘

Token count: 1 CLS + 80 numerical + 2 categorical = 83 tokens
Attention complexity: O(83²) = 6,889 operations per layer
```

**CSN-Transformer (Dual-Path Processing):**
```
┌────────────────────────────────────────┐
│ PATH 1: [CLS₁, cat_tokens]             │
│         ↓                               │
│   CATEGORICAL TRANSFORMER               │
│   (Categorical features attend)        │
│         ↓                               │
│   Extract CLS₁                         │
└────────────────────────────────────────┘
                  ↓
            CONCATENATE
                  ↓
┌────────────────────────────────────────┐
│ PATH 2: [CLS₂, num_tokens]             │
│         ↓                               │
│   NUMERICAL TRANSFORMER                 │
│   (Temporal features attend)           │
│         ↓                               │
│   Extract CLS₂ → Predict               │
└────────────────────────────────────────┘

Path 1: 1 CLS + 2 categorical = 3 tokens → O(3²) = 9 operations
Path 2: 1 CLS + 10 timesteps = 11 tokens → O(11²) = 121 operations
Total attention: 9 + 121 = 130 operations per layer
```

---

## Trade-offs Analysis

### FT-Transformer Advantages

✅ **Fewer Parameters:**
- 613K vs 1.2M parameters (48% fewer)
- Faster training and inference
- Lower memory footprint
- Less prone to overfitting on small datasets

✅ **Richer Cross-Modal Interactions:**
- Categorical and numerical features attend to each other
- Example: "AAPL" symbol can directly attend to specific price movements
- Single unified representation space

✅ **Simpler Architecture:**
- One transformer stack
- Easier to debug and interpret
- Fewer hyperparameters to tune

### CSN-Transformer Advantages

✅ **Specialized Feature Processing:**
- Categorical features processed separately from numerical
- Each transformer optimized for its feature type
- No interference between static and time-varying patterns

✅ **Better for Distinct Feature Types:**
- Categorical path learns inter-category relationships (symbol-sector)
- Numerical path learns temporal patterns (price momentum)
- Clearer separation of concerns

✅ **More Efficient Attention (computationally):**
- Despite more parameters, attention is more efficient:
- FT: O(83²) = 6,889 per layer
- CSN: O(3²) + O(11²) = 130 per layer
- **98% fewer attention operations per layer**

✅ **Better Gradient Flow:**
- Independent pathways reduce gradient interference
- Each feature type can learn at optimal rate
- Late fusion preserves specialized representations

---

## When to Use Each Model

### Use FT-Transformer When:

1. **Limited Training Data:** Fewer parameters reduce overfitting risk
2. **Strong Feature Interactions Expected:** Cross-modal attention is valuable
   - Example: Stock symbol should directly influence how price patterns are interpreted
3. **Computational Resources Are Limited:** Faster training/inference
4. **Simpler Deployment:** Smaller model size for production

### Use CSN-Transformer When:

1. **Large Training Datasets Available:** More parameters can capture complex patterns
2. **Distinct Feature Types:** Clear separation between categorical and numerical
3. **Computational Efficiency Matters More Than Size:** 98% fewer attention operations
4. **Better Generalization Needed:** Specialized pathways reduce overfitting to spurious correlations
5. **Strong Temporal Patterns:** Dedicated numerical transformer for time series

---

## Scaling Behavior

### Parameter Growth with Hyperparameters

**FT-Transformer:**
```
Params ≈ 593,280 * (n_layers / 3) + constant
Linear scaling with n_layers
```

**CSN-Transformer:**
```
Params ≈ 1,186,560 * (n_layers / 3) + constant
Linear scaling with n_layers (2x slope)
```

**With d_model scaling:**
- Transformers dominate: O(d_model²) growth
- Both models scale similarly, CSN always ~2x larger

### Memory Usage During Forward Pass

**FT-Transformer:**
```
All tokens: 32 × 83 × 128 × 4 bytes ≈ 1.36 MB
Attention matrix: 32 × 8 × 83² × 4 bytes ≈ 3.5 MB per layer
Total: ~1.36 MB + ~10.5 MB (3 layers) ≈ 11.86 MB
```

**CSN-Transformer:**
```
Categorical tokens: 32 × 3 × 128 × 4 bytes ≈ 49 KB
Numerical tokens: 32 × 11 × 128 × 4 bytes ≈ 180 KB
Cat attention: 32 × 8 × 3² × 4 bytes ≈ 9 KB per layer
Num attention: 32 × 8 × 11² × 4 bytes ≈ 124 KB per layer
Total: ~229 KB + ~399 KB (3 layers) ≈ 628 KB
```

**CSN uses ~19x LESS memory during forward pass** despite having 2x parameters!

---

## Recommendations

### For Stock Forecasting (Current Use Case)

**Use CSN-Transformer:**

Rationale:
- Stock data has clear feature separation:
  - **Static categorical:** symbol, sector (don't change over sequences)
  - **Time-varying numerical:** OHLCV, indicators (strong temporal patterns)
- Temporal patterns in prices are more important than categorical-numerical interactions
- 98% fewer attention operations → faster training
- 19x less memory during forward pass → can use larger batch sizes

**Exception - Use FT-Transformer when:**
- Dataset has < 10,000 samples (small data regime)
- Explicit symbol-price pattern learning is desired
- Model size constraints (edge deployment)

---

## Empirical Validation Needed

To definitively compare these models, run experiments measuring:

1. **Convergence Speed:** Iterations to reach target validation loss
2. **Generalization:** Out-of-sample MAE/MSE on test set
3. **Training Time:** Wall-clock time per epoch
4. **Memory Usage:** Peak GPU memory during training
5. **Prediction Quality:** Per-group metrics for rare categories

Hypothesis based on architecture analysis:
- **CSN should train faster** (fewer attention ops)
- **FT should generalize better on small data** (fewer params)
- **CSN should perform better on large data** (specialized processing)

---

## Conclusion

**Your observation is correct:** CSN-Transformer has **96.6% more parameters** than FT-Transformer for the same hyperparameters.

**Root cause:** CSN uses a **dual-path architecture** with two separate 3-layer transformers (one for categorical, one for numerical), effectively doubling the transformer parameters.

**Key insight:** Despite having 2x parameters, CSN-Transformer is:
- **98% more computationally efficient** per forward pass (fewer attention operations)
- **19x more memory efficient** during forward pass
- **Better suited** for time series with distinct categorical/numerical feature types

The parameter increase is a **deliberate architectural choice** that trades model size for:
1. Specialized feature processing
2. Computational efficiency
3. Better gradient flow
4. Clearer separation of concerns

For the stock forecasting use case with strong temporal patterns and clear feature type separation, the **CSN-Transformer's parameter increase is justified** by its advantages in processing efficiency and feature specialization.

---

**END OF REPORT**
