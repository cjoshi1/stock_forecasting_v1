# Pooling Strategies Explained: A Comprehensive Comparison

**Date:** 2025-11-06
**Purpose:** Detailed explanation of different pooling strategies for transformer sequence aggregation

---

## Table of Contents
1. [Overview](#overview)
2. [Strategy 1: Attention Pooling (Single-Head)](#strategy-1-attention-pooling-single-head)
3. [Strategy 2: Multi-Head Attention Pooling](#strategy-2-multi-head-attention-pooling)
4. [Strategy 3: Weighted Average (Learnable)](#strategy-3-weighted-average-learnable)
5. [Strategy 4: Temporal Attention Pooling](#strategy-4-temporal-attention-pooling)
6. [Side-by-Side Comparison](#side-by-side-comparison)
7. [When to Use Each Strategy](#when-to-use-each-strategy)

---

## Overview

All pooling strategies solve the same problem: **How to aggregate a sequence of token representations into a single vector?**

```
Input:  [batch, seq_len, d_token]  (e.g., [32, 10, 128])
Goal:   [batch, d_token]            (e.g., [32, 128])
```

The difference is **HOW** they decide which tokens are important and how to combine them.

---

## Strategy 1: Attention Pooling (Single-Head)

### 🎯 Core Idea
Use a **single learnable query vector** that "asks" the sequence: "Which tokens are most relevant for the final prediction?"

### 📐 Mathematical Formulation

```python
# Learnable parameters
Q = learnable query vector        # Shape: [1, d_token] = [1, 128]
W_q, W_k, W_v = attention weights # Each: [d_token, d_token]

# Forward pass
tokens: [batch, seq_len, d_token] = [32, 10, 128]

# Step 1: Project tokens to Keys and Values
K = tokens @ W_k  # [32, 10, 128] - Keys
V = tokens @ W_v  # [32, 10, 128] - Values

# Step 2: Expand query for batch
Q_batch = Q.expand(batch, 1, d_token)  # [32, 1, 128]

# Step 3: Compute attention scores
scores = Q_batch @ K.transpose(-2, -1)  # [32, 1, 10]
scores = scores / sqrt(d_token)         # Scale by sqrt(128)

# Step 4: Softmax to get attention weights
attention_weights = softmax(scores, dim=-1)  # [32, 1, 10]
# attention_weights[i] tells us how much to weight each of the 10 tokens

# Step 5: Weighted sum of values
output = attention_weights @ V  # [32, 1, 128]
output = output.squeeze(1)      # [32, 128]
```

### 🎨 Visual Representation

```
┌─────────────────────────────────────────────────────────────────┐
│                  SINGLE-HEAD ATTENTION POOLING                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Sequence:                                                │
│  ┌────────┬────────┬────────┬─────┬────────┐                  │
│  │ Token₀ │ Token₁ │ Token₂ │ ... │ Token₉ │                  │
│  │ [128]  │ [128]  │ [128]  │     │ [128]  │                  │
│  └────────┴────────┴────────┴─────┴────────┘                  │
│       ↓        ↓        ↓             ↓                         │
│  Project to Keys and Values (W_k, W_v)                         │
│       ↓        ↓        ↓             ↓                         │
│  ┌────────┬────────┬────────┬─────┬────────┐                  │
│  │   K₀   │   K₁   │   K₂   │ ... │   K₉   │  Keys           │
│  └────────┴────────┴────────┴─────┴────────┘                  │
│  ┌────────┬────────┬────────┬─────┬────────┐                  │
│  │   V₀   │   V₁   │   V₂   │ ... │   V₉   │  Values         │
│  └────────┴────────┴────────┴─────┴────────┘                  │
│                                                                 │
│  Learnable Query:                                               │
│  ┌──────────────────┐                                          │
│  │    Q [128]       │  "What should I pay attention to?"       │
│  └──────────────────┘                                          │
│          ↓                                                      │
│  Compute Attention Scores: Q · K^T                             │
│  ┌─────┬─────┬─────┬─────┬─────┐                             │
│  │ 0.1 │ 0.05│ 0.2 │ ... │ 0.15│  ← Raw scores                │
│  └─────┴─────┴─────┴─────┴─────┘                             │
│          ↓                                                      │
│  Softmax (normalize to sum to 1):                              │
│  ┌─────┬─────┬─────┬─────┬─────┐                             │
│  │ 0.12│ 0.08│ 0.25│ ... │ 0.18│  ← Attention weights         │
│  └─────┴─────┴─────┴─────┴─────┘                             │
│          ↓                                                      │
│  Weighted Sum: Σ(attention_weights[i] × V[i])                  │
│          ↓                                                      │
│  ┌──────────────────┐                                          │
│  │  Output [128]    │  ← Aggregated representation            │
│  └──────────────────┘                                          │
│                                                                 │
│  Example: If attention_weights = [0.12, 0.08, 0.25, ..., 0.18]│
│           Output = 0.12×V₀ + 0.08×V₁ + 0.25×V₂ + ... + 0.18×V₉│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 📊 Parameters

```python
Query vector (Q):        d_token = 128
W_q (query projection):  d_token × d_token = 128 × 128 = 16,384
W_k (key projection):    d_token × d_token = 128 × 128 = 16,384
W_v (value projection):  d_token × d_token = 128 × 128 = 16,384
Total:                   ~49,280 parameters
```

### ✅ Pros
- Learnable importance weighting (model decides which tokens matter)
- Interpretable attention weights (can visualize which timesteps model focuses on)
- Context-dependent (different samples get different weights)
- Single attention head is computationally efficient

### ❌ Cons
- Single attention head may be limiting (can only learn one attention pattern)
- More parameters than simple mean pooling
- Slightly slower than parameter-free methods

---

## Strategy 2: Multi-Head Attention Pooling

### 🎯 Core Idea
Use **multiple learnable queries** (one per head) that can learn **different attention patterns** simultaneously.

### 📐 Mathematical Formulation

```python
# Learnable parameters
Q = learnable query vector        # Shape: [1, d_token] = [1, 128]
W_q, W_k, W_v, W_o = weights     # Multi-head projections

# Forward pass
tokens: [batch, seq_len, d_token] = [32, 10, 128]
num_heads = 8
head_dim = d_token / num_heads = 128 / 8 = 16

# Step 1: Project tokens for multi-head
Q_batch = Q.expand(batch, 1, d_token)  # [32, 1, 128]
K = tokens @ W_k  # [32, 10, 128]
V = tokens @ W_v  # [32, 10, 128]

# Step 2: Split into multiple heads
# Reshape to [batch, seq_len, num_heads, head_dim]
Q_heads = Q_batch.view(batch, 1, num_heads, head_dim)      # [32, 1, 8, 16]
K_heads = K.view(batch, seq_len, num_heads, head_dim)      # [32, 10, 8, 16]
V_heads = V.view(batch, seq_len, num_heads, head_dim)      # [32, 10, 8, 16]

# Transpose to [batch, num_heads, seq_len, head_dim]
Q_heads = Q_heads.transpose(1, 2)  # [32, 8, 1, 16]
K_heads = K_heads.transpose(1, 2)  # [32, 8, 10, 16]
V_heads = V_heads.transpose(1, 2)  # [32, 8, 10, 16]

# Step 3: Compute attention for each head independently
scores = Q_heads @ K_heads.transpose(-2, -1)  # [32, 8, 1, 10]
scores = scores / sqrt(head_dim)               # Scale by sqrt(16)
attention_weights = softmax(scores, dim=-1)    # [32, 8, 1, 10]

# Step 4: Apply attention to values (per head)
attended = attention_weights @ V_heads  # [32, 8, 1, 16]

# Step 5: Concatenate heads and project
attended = attended.transpose(1, 2)     # [32, 1, 8, 16]
attended = attended.reshape(batch, 1, d_token)  # [32, 1, 128]

output = attended @ W_o  # Output projection: [32, 1, 128]
output = output.squeeze(1)  # [32, 128]
```

### 🎨 Visual Representation

```
┌─────────────────────────────────────────────────────────────────┐
│                 MULTI-HEAD ATTENTION POOLING                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Sequence:                                                │
│  ┌────────┬────────┬────────┬─────┬────────┐                  │
│  │ Token₀ │ Token₁ │ Token₂ │ ... │ Token₉ │  [32, 10, 128]   │
│  └────────┴────────┴────────┴─────┴────────┘                  │
│                        ↓                                        │
│              Split into 8 heads (each 16-dim)                   │
│                        ↓                                        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ HEAD 1:  [Token₀] [Token₁] ... [Token₉]  (dim 0-15)      │ │
│  │           ↓ Attention Pattern 1                            │ │
│  │         [0.2, 0.1, 0.05, ..., 0.15]  ← Focuses on recent  │ │
│  ├───────────────────────────────────────────────────────────┤ │
│  │ HEAD 2:  [Token₀] [Token₁] ... [Token₉]  (dim 16-31)     │ │
│  │           ↓ Attention Pattern 2                            │ │
│  │         [0.1, 0.1, 0.1, ..., 0.1]  ← Uniform weighting    │ │
│  ├───────────────────────────────────────────────────────────┤ │
│  │ HEAD 3:  [Token₀] [Token₁] ... [Token₉]  (dim 32-47)     │ │
│  │           ↓ Attention Pattern 3                            │ │
│  │         [0.05, 0.1, 0.3, ..., 0.05]  ← Focuses on middle  │ │
│  ├───────────────────────────────────────────────────────────┤ │
│  │  ... (5 more heads)                                        │ │
│  └───────────────────────────────────────────────────────────┘ │
│                        ↓                                        │
│           Each head produces: [32, 1, 16]                      │
│                        ↓                                        │
│              Concatenate all heads                              │
│         [32, 1, 16] × 8 heads → [32, 1, 128]                  │
│                        ↓                                        │
│              Output projection (W_o)                            │
│                        ↓                                        │
│  ┌──────────────────┐                                          │
│  │  Output [128]    │  ← Aggregated with multiple patterns    │
│  └──────────────────┘                                          │
│                                                                 │
│  KEY INSIGHT: Each head learns a different attention pattern!  │
│  - Head 1 might focus on recent timesteps                      │
│  - Head 2 might focus on early timesteps                       │
│  - Head 3 might focus on peaks/important events                │
│  - Head 4 might average everything uniformly                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 📊 Parameters

```python
Query vector (Q):        d_token = 128
W_q (query projection):  d_token × d_token = 128 × 128 = 16,384
W_k (key projection):    d_token × d_token = 128 × 128 = 16,384
W_v (value projection):  d_token × d_token = 128 × 128 = 16,384
W_o (output projection): d_token × d_token = 128 × 128 = 16,384
Total:                   ~65,664 parameters
```

### ✅ Pros
- **Multiple attention patterns**: Each head can learn different aggregation strategies
- More expressive than single-head
- Can capture diverse relationships (e.g., Head 1 focuses on recent, Head 2 on peaks)
- Interpretable per-head attention weights

### ❌ Cons
- More parameters than single-head (~65K vs ~49K)
- Slightly more computation
- May overfit on small datasets
- 8 heads might be overkill for simple sequences

---

## Strategy 3: Weighted Average (Learnable)

### 🎯 Core Idea
Learn a **fixed weight for each position** in the sequence, then compute weighted average.

### 📐 Mathematical Formulation

```python
# Learnable parameters
position_weights = learnable vector  # Shape: [max_seq_len] = [100] for safety

# Forward pass
tokens: [batch, seq_len, d_token] = [32, 10, 128]

# Step 1: Get weights for current sequence length
weights = position_weights[:seq_len]  # [10]

# Step 2: Apply softmax to normalize (sum to 1)
weights = softmax(weights, dim=0)  # [10]

# Step 3: Expand for broadcasting
weights = weights.unsqueeze(0).unsqueeze(-1)  # [1, 10, 1]

# Step 4: Weighted sum
output = (tokens * weights).sum(dim=1)  # [32, 128]

# Example:
# If weights = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.13, 0.15, 0.15]
# Output[i] = 0.05×Token₀[i] + 0.06×Token₁[i] + ... + 0.15×Token₉[i]
```

### 🎨 Visual Representation

```
┌─────────────────────────────────────────────────────────────────┐
│                   WEIGHTED AVERAGE POOLING                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Learnable Position Weights (trained):                         │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐│
│  │ w₀  │ w₁  │ w₂  │ w₃  │ w₄  │ w₅  │ w₆  │ w₇  │ w₈  │ w₉  ││
│  │0.05 │0.06 │0.07 │0.08 │0.09 │0.10 │0.12 │0.13 │0.15 │0.15 ││
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘│
│    ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓    │
│                                                                 │
│  Input Sequence:                                                │
│  ┌────────┬────────┬────────┬────────┬─────┬────────┐         │
│  │ Token₀ │ Token₁ │ Token₂ │ Token₃ │ ... │ Token₉ │         │
│  │ [128]  │ [128]  │ [128]  │ [128]  │     │ [128]  │         │
│  └────────┴────────┴────────┴────────┴─────┴────────┘         │
│     ×        ×        ×        ×              ×                 │
│   0.05     0.06     0.07     0.08           0.15               │
│                                                                 │
│  Weighted Sum:                                                  │
│  Output = 0.05×Token₀ + 0.06×Token₁ + ... + 0.15×Token₉       │
│                        ↓                                        │
│  ┌──────────────────┐                                          │
│  │  Output [128]    │  ← Position-weighted average             │
│  └──────────────────┘                                          │
│                                                                 │
│  KEY CHARACTERISTICS:                                           │
│  ✓ Same weights for ALL samples (position-based only)          │
│  ✓ Weights are learned during training                         │
│  ✓ After training, weights might show pattern like:            │
│    - Increasing (recent timesteps more important)               │
│    - Decreasing (early timesteps more important)                │
│    - Peaked (middle timesteps more important)                   │
│                                                                 │
│  Example trained weights for time series:                       │
│  t=0  t=1  t=2  t=3  t=4  t=5  t=6  t=7  t=8  t=9              │
│  ──────────────────────────────────────────────────             │
│  0.05 0.06 0.07 0.08 0.09 0.10 0.12 0.13 0.15 0.15  (recency)  │
│                                                   ↑↑  More      │
│                                                   weight        │
└─────────────────────────────────────────────────────────────────┘
```

### 📊 Parameters

```python
Position weights: max_seq_len = 100 (for flexibility)
Total:            100 parameters
```

### ✅ Pros
- **Very few parameters** (only 100!)
- Position-aware weighting (can learn recency bias)
- Simple and interpretable
- Smooth gradients (all positions contribute)
- Fast computation (no attention mechanism)

### ❌ Cons
- **Context-independent**: Same weights for ALL samples (not adaptive)
- Requires fixed `max_seq_len` (though can be large for safety)
- Cannot adapt based on content (e.g., can't focus on specific events)
- Less flexible than attention-based methods

### 🔍 Example After Training

```
For stock price forecasting, trained weights might show:

Position:  t=0   t=1   t=2   t=3   t=4   t=5   t=6   t=7   t=8   t=9
Weight:   0.02  0.03  0.04  0.05  0.07  0.09  0.11  0.13  0.20  0.26
          └────── Past (less weight) ──────┘  └──── Recent (more) ────┘

Interpretation: Model learned that recent timesteps are more important
for prediction, which makes sense for time series forecasting!
```

---

## Strategy 4: Temporal Attention Pooling

### 🎯 Core Idea
Multi-head attention pooling **with an additional temporal bias** that gives preference to recent timesteps.

### 📐 Mathematical Formulation

```python
# Learnable parameters
Q = learnable query vector             # Shape: [1, d_token] = [1, 128]
W_q, W_k, W_v, W_o = attention weights # Multi-head projections
temporal_decay = learnable scalar      # Shape: [1], e.g., 0.1

# Forward pass
tokens: [batch, seq_len, d_token] = [32, 10, 128]
num_heads = 8

# Step 1: Standard multi-head attention (same as Strategy 2)
Q_batch = Q.expand(batch, 1, d_token)
K = tokens @ W_k
V = tokens @ W_v
# ... (split into heads, as in Strategy 2)

# Step 2: Compute attention scores
scores = Q_heads @ K_heads.transpose(-2, -1)  # [32, 8, 1, 10]
scores = scores / sqrt(head_dim)

# Step 3: ADD TEMPORAL BIAS (this is the key difference!)
positions = torch.arange(seq_len)  # [0, 1, 2, ..., 9]
# More recent positions (higher index) get higher bias
temporal_bias = exp(-temporal_decay × (seq_len - 1 - positions))
# Example with temporal_decay=0.1, seq_len=10:
#   pos=0: exp(-0.1 × 9) = exp(-0.9) ≈ 0.41
#   pos=5: exp(-0.1 × 4) = exp(-0.4) ≈ 0.67
#   pos=9: exp(-0.1 × 0) = exp(0)   = 1.00
# Temporal bias: [0.41, 0.45, 0.50, 0.55, 0.61, 0.67, 0.74, 0.82, 0.90, 1.00]

# Step 4: Apply temporal bias to scores
temporal_bias = temporal_bias.view(1, 1, 1, seq_len)  # [1, 1, 1, 10]
scores = scores + torch.log(temporal_bias)  # Add in log-space (multiply in prob space)

# Step 5: Continue with standard attention
attention_weights = softmax(scores, dim=-1)  # [32, 8, 1, 10]
attended = attention_weights @ V_heads       # [32, 8, 1, 16]
# ... (concatenate heads and project output)
```

### 🎨 Visual Representation

```
┌─────────────────────────────────────────────────────────────────┐
│                TEMPORAL ATTENTION POOLING                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Sequence (time series):                                 │
│  ┌────────┬────────┬────────┬────────┬─────┬────────┐         │
│  │ t=0    │ t=1    │ t=2    │ t=3    │ ... │ t=9    │         │
│  │ (old)  │        │        │        │     │ (recent)│         │
│  └────────┴────────┴────────┴────────┴─────┴────────┘         │
│                        ↓                                        │
│         Step 1: Multi-Head Attention (8 heads)                 │
│                        ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Compute attention scores (before temporal bias)        │  │
│  │  HEAD 1 raw scores: [0.8, 0.6, 0.9, 0.7, ..., 0.5]     │  │
│  └─────────────────────────────────────────────────────────┘  │
│                        ↓                                        │
│         Step 2: Add Temporal Bias                              │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Temporal bias (exponential decay):                     │  │
│  │                                                          │  │
│  │  Position:  t=0  t=1  t=2  t=3  t=4  t=5  t=6  t=7  t=8  t=9│
│  │  Bias:      0.41 0.45 0.50 0.55 0.61 0.67 0.74 0.82 0.90 1.0│
│  │             └─────── Past (penalized) ──────┘  └─ Recent ──┘│
│  │                                                          │  │
│  │  Visual:                                                 │  │
│  │  │                                              ████████ │  │
│  │  │                                        ██████████████ │  │
│  │  │                                  ████████████████████ │  │
│  │  │                            ██████████████████████████ │  │
│  │  │                      ████████████████████████████████ │  │
│  │  │                ████████████████████████████████████   │  │
│  │  │          ████████████████████████████████████         │  │
│  │  │    ████████████████████████████████████               │  │
│  │  │████████████████████████████████                       │  │
│  │  └───────────────────────────────────────────────────   │  │
│  │     t=0  t=1  t=2  t=3  t=4  t=5  t=6  t=7  t=8  t=9    │  │
│  └─────────────────────────────────────────────────────────┘  │
│                        ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Modified attention scores (after adding bias):         │  │
│  │  HEAD 1 scores: score[i] = raw_score[i] + log(bias[i]) │  │
│  │                                                          │  │
│  │  Result: Recent timesteps get BOOSTED!                  │  │
│  │  - t=0 (old):    score = 0.8 + log(0.41) = 0.8 - 0.89  │  │
│  │  - t=9 (recent): score = 0.5 + log(1.00) = 0.5 + 0.0   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                        ↓                                        │
│         Step 3: Softmax (normalize)                            │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Final attention weights (after softmax):               │  │
│  │  [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18]│
│  │   └─── Lower weights ────┘  └──── Higher weights ─────┘ │  │
│  │                                                          │  │
│  │  NOTICE: Even though raw scores were similar,           │  │
│  │  recent timesteps get more weight after temporal bias!  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                        ↓                                        │
│         Step 4: Apply attention to values                      │
│                        ↓                                        │
│  ┌──────────────────┐                                          │
│  │  Output [128]    │  ← Aggregated with recency bias         │
│  └──────────────────┘                                          │
│                                                                 │
│  KEY INSIGHT: Combines benefits of both approaches:            │
│  ✓ Content-based attention (learns what's important)           │
│  ✓ Temporal bias (automatically weighs recent more)            │
│  ✓ Learnable decay parameter (model decides how strong)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 📊 Parameters

```python
Query vector (Q):        d_token = 128
W_q, W_k, W_v, W_o:      4 × (d_token × d_token) = 65,536
Temporal decay:          1 scalar
Total:                   ~65,665 parameters
```

### ✅ Pros
- **Time-series optimized**: Built-in recency bias
- **Adaptive**: Attention can still focus on important events (not purely position-based)
- **Learnable decay**: Model learns optimal temporal weighting
- **Multi-head**: Can learn diverse patterns per head
- **Interpretable**: Can visualize both attention weights AND temporal bias

### ❌ Cons
- Most parameters (~65K)
- Most computational cost
- Temporal bias might not always be beneficial (sometimes old data is important)
- More complex to implement

### 🔍 Example Behavior

```
Scenario 1: Normal timesteps
────────────────────────────
t=0  t=1  t=2  t=3  t=4  t=5  t=6  t=7  t=8  t=9
│    │    │    │    │    │    │    │    │    │
Content attention:  uniform weighting (nothing special)
Temporal bias:      0.41 0.45 0.50 0.55 0.61 0.67 0.74 0.82 0.90 1.0
Final weights:      0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.18
                    └─────── Recent gets more weight ────────────┘


Scenario 2: Important spike at t=3
───────────────────────────────────
t=0  t=1  t=2  t=3  t=4  t=5  t=6  t=7  t=8  t=9
│    │    │    🔥   │    │    │    │    │    │
                 ↑ Important event!

Content attention:  focuses on t=3 (high raw score)
Temporal bias:      0.41 0.45 0.50 0.55 0.61 0.67 0.74 0.82 0.90 1.0
Final weights:      0.08 0.09 0.10 0.35 0.10 0.11 0.12 0.13 0.14 0.15
                              ↑
                    Attention wins! Still focuses on important event
                    even though it's in the past


BENEFIT: Gets best of both worlds!
- Normal case: Recent timesteps weighted more (temporal bias dominates)
- Special events: Can still focus on important past events (attention wins)
```

---

## Side-by-Side Comparison

### Quick Reference Table

| Feature | Attention Pooling | Multi-Head Attention | Weighted Average | Temporal Attention |
|---------|------------------|---------------------|------------------|-------------------|
| **Parameters** | ~49K | ~66K | ~100 | ~66K |
| **Computation** | Medium | High | Low | High |
| **Context-Aware** | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| **Position-Aware** | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| **Time-Series Optimized** | ❌ No | ❌ No | Partial | ✅ Yes |
| **Interpretability** | Good | Good | Excellent | Excellent |
| **Flexibility** | Medium | High | Low | High |
| **Multiple Patterns** | ❌ Single | ✅ Multi | ❌ Single | ✅ Multi |

### Computational Cost Comparison

```
Input: [32, 10, 128]  (batch=32, seq_len=10, d_token=128)

Method                  Operations                          FLOPs (approx)
─────────────────────────────────────────────────────────────────────────
Mean Pooling           sum + divide                         ~41K
Weighted Average       multiply + sum                       ~82K
Attention Pooling      Q·K^T + softmax + weighted sum      ~1.3M
Multi-Head Attention   8 heads × (Q·K^T + attention)       ~2.1M
Temporal Attention     Multi-head + exp + log               ~2.2M

Speed ranking (fastest to slowest):
1. Mean Pooling (not in this comparison, but baseline)
2. Weighted Average        ←─ Fastest learnable method
3. Single-Head Attention
4. Multi-Head Attention
5. Temporal Attention      ←─ Slowest but most sophisticated
```

### Memory Footprint

```
Additional Memory During Forward Pass:

Weighted Average:
  - Position weights: [10] = 40 bytes
  - Total: ~40 bytes

Attention Pooling (Single-Head):
  - Query: [32, 1, 128] = 16 KB
  - Attention scores: [32, 1, 10] = 1.3 KB
  - Total: ~17 KB

Multi-Head Attention:
  - Query: [32, 1, 128] = 16 KB
  - Attention scores: [32, 8, 1, 10] = 10 KB
  - Total: ~26 KB

Temporal Attention:
  - Same as multi-head + temporal bias: [10] = 40 bytes
  - Total: ~26 KB
```

---

## When to Use Each Strategy

### Use **Attention Pooling (Single-Head)** When:

✅ You want learnable, context-dependent aggregation
✅ Dataset is medium-sized (not huge, not tiny)
✅ Interpretability matters (can visualize attention weights)
✅ You need moderate parameter count
✅ Single attention pattern is sufficient

❌ Avoid if:
- Dataset is very small (may overfit with 49K extra params)
- Need to capture multiple diverse patterns
- Computational budget is extremely tight

**Example Use Cases:**
- Aggregating document representations for classification
- Pooling image patches in vision transformers
- General-purpose sequence aggregation

---

### Use **Multi-Head Attention Pooling** When:

✅ You need to capture **multiple aggregation strategies**
✅ Large dataset available (can support 66K extra params)
✅ Different heads can learn complementary patterns
✅ Maximum expressiveness is needed
✅ Slightly higher compute cost is acceptable

❌ Avoid if:
- Small dataset (risk of overfitting)
- Sequence is very short (8 heads may be overkill for 3 tokens)
- Tight parameter budget
- Need fast inference

**Example Use Cases:**
- Complex sequences where different aspects matter (e.g., price patterns, volume patterns, sentiment)
- Large-scale models with abundant data
- When single-head attention doesn't capture enough nuance

---

### Use **Weighted Average (Learnable)** When:

✅ **Minimal parameters** is critical (~100 params)
✅ Position matters more than content
✅ All samples should use same position weights
✅ Simple, interpretable solution needed
✅ Fast computation is important

❌ Avoid if:
- Need context-dependent weighting (different samples need different weights)
- Content of tokens matters more than position
- Sequence length varies widely (requires large max_seq_len)

**Example Use Cases:**
- Time series where recency bias is consistent across all samples
- Fixed-format sequences (e.g., always 10 timesteps)
- When you know position-based weighting is appropriate
- Resource-constrained environments

**Real-World Example:**
```
Stock price forecasting with 10 timesteps:
  - Training learns: [0.02, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.18, 0.22]
  - Interpretation: Recent prices matter more (increasing weights)
  - All stocks use same position weights (appropriate for this domain)
```

---

### Use **Temporal Attention Pooling** When:

✅ **Time series forecasting** is the task
✅ Recency bias is important BUT exceptions matter
✅ Want both content-based AND position-based weighting
✅ Large dataset supports 66K params
✅ Need maximum performance for temporal data
✅ Interpretability of temporal patterns matters

❌ Avoid if:
- Not working with time series (temporal bias is wasted)
- Small dataset (complex model may overfit)
- Sequences don't have temporal ordering (e.g., categorical features)
- Need fastest possible inference

**Example Use Cases:**
- **Stock price forecasting** (perfect fit!)
- Weather prediction
- Sensor data analysis
- Any time series where "recent usually matters more, but not always"

**Real-World Example:**
```
Stock price prediction:
  - Normal days: Temporal bias ensures recent prices weighted more
  - Earnings announcement at t=3: Attention focuses on that spike
  - Market crash at t=1: Attention can still focus on this important event

Result: Best of both worlds!
```

---

## Recommendations for Your Stock Forecasting Use Case

Based on the NON_CLS_ARCHITECTURE_PROPOSAL.md and your domain:

### For **FT-Transformer**:
**Recommended:** Temporal Attention Pooling

**Why:**
- Stock prices are time series (temporal bias makes sense)
- Recent prices usually matter more, but important events (earnings, crashes) should still get attention
- Large enough dataset to support 66K params
- Multi-head allows learning diverse patterns (momentum, volatility, trends)

### For **CSN-Transformer**:
**Recommended:** Hybrid approach (as proposed in document)

**Categorical Path:** Weighted Average or Mean Pooling
- Categorical features (symbol, sector) have no temporal order
- Same symbols/sectors across all samples
- Simple aggregation sufficient

**Numerical Path:** Temporal Attention Pooling
- Price/volume sequences are temporal
- Recency matters but important events should be captured
- Benefits from sophisticated aggregation

---

## Implementation Priority

If implementing these, I recommend this order:

1. **Start with Weighted Average** (simplest, 100 params)
   - Baseline to see if position-based weighting helps
   - Fast to implement and test

2. **Add Attention Pooling (Single-Head)** (medium, 49K params)
   - Test if content-based weighting improves over position-only

3. **Implement Temporal Attention** (sophisticated, 66K params)
   - Combine best of both: content + position
   - Expected to be best for time series

4. **Try Multi-Head Attention** (high expressiveness, 66K params)
   - If temporal doesn't help, try pure multi-head
   - Good for comparison

---

**END OF DOCUMENT**
