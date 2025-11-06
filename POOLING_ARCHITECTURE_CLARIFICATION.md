# Where Does Pooling Happen? Architecture Clarification

**Date:** 2025-11-06
**Purpose:** Clarify the exact position of pooling in the transformer architecture

---

## The Critical Question

**Q: Does pooling happen on the output of transformer layers?**

**A: YES! Pooling happens AFTER all transformer layers have processed the tokens.**

---

## Complete Architecture Flow

### CLS-Based Architecture (Current)

```
┌─────────────────────────────────────────────────────────────────────┐
│                       COMPLETE DATA FLOW                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  STAGE 1: Tokenization                                             │
│  ──────────────────────                                            │
│  Raw Input: [batch, seq_len, features]                             │
│           ↓                                                         │
│  Numerical Tokenizer + Categorical Embeddings                      │
│           ↓                                                         │
│  Tokens: [batch, num_tokens, d_token]                              │
│  Example: [32, 82, 128]  (80 numerical + 2 categorical)            │
│           ↓                                                         │
│  Add CLS Token                                                      │
│           ↓                                                         │
│  [batch, 83, 128]  (1 CLS + 82 feature tokens)                    │
│                                                                     │
│  ═══════════════════════════════════════════════════════════        │
│                                                                     │
│  STAGE 2: Transformer Processing (3 Layers)                        │
│  ─────────────────────────────────────────                         │
│  Input: [32, 83, 128]                                              │
│           ↓                                                         │
│  ┌────────────────────────────────────────────┐                   │
│  │  Transformer Layer 1                       │                   │
│  │  - Self-Attention (all tokens attend)      │                   │
│  │  - Feed-Forward Network                    │                   │
│  │  Output: [32, 83, 128]                     │                   │
│  └────────────────────────────────────────────┘                   │
│           ↓                                                         │
│  ┌────────────────────────────────────────────┐                   │
│  │  Transformer Layer 2                       │                   │
│  │  - Self-Attention                          │                   │
│  │  - Feed-Forward Network                    │                   │
│  │  Output: [32, 83, 128]                     │                   │
│  └────────────────────────────────────────────┘                   │
│           ↓                                                         │
│  ┌────────────────────────────────────────────┐                   │
│  │  Transformer Layer 3                       │                   │
│  │  - Self-Attention                          │                   │
│  │  - Feed-Forward Network                    │                   │
│  │  Output: [32, 83, 128] ◄──────────────────┼─ Transformer output│
│  └────────────────────────────────────────────┘    (contextualized)│
│                                                                     │
│  At this point, ALL 83 tokens have been processed and              │
│  contextualized through attention. Each token has "seen"           │
│  all other tokens and incorporated their information.              │
│                                                                     │
│  ═══════════════════════════════════════════════════════════        │
│                                                                     │
│  STAGE 3: Aggregation (POOLING HAPPENS HERE!)                     │
│  ────────────────────────────────────────────                      │
│  Transformer output: [32, 83, 128]                                 │
│           ↓                                                         │
│  ┌────────────────────────────────────────────┐                   │
│  │  CLS Token Extraction                      │                   │
│  │  aggregated = output[:, 0, :]              │                   │
│  │                                             │                   │
│  │  Select FIRST token (the CLS token)        │                   │
│  └────────────────────────────────────────────┘                   │
│           ↓                                                         │
│  Aggregated: [32, 128]  ◄───────────────────── Single vector      │
│                                                  per sample         │
│  ═══════════════════════════════════════════════════════════        │
│                                                                     │
│  STAGE 4: Prediction                                               │
│  ──────────────────                                                │
│  Input: [32, 128]                                                  │
│           ↓                                                         │
│  Linear(128, output_dim)                                           │
│           ↓                                                         │
│  Predictions: [32, output_dim]                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Non-CLS Architecture (Proposed)

```
┌─────────────────────────────────────────────────────────────────────┐
│                  NON-CLS ARCHITECTURE FLOW                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  STAGE 1: Tokenization (NO CLS TOKEN)                              │
│  ────────────────────────────────────                              │
│  Raw Input: [batch, seq_len, features]                             │
│           ↓                                                         │
│  Numerical Tokenizer + Categorical Embeddings                      │
│           ↓                                                         │
│  Tokens: [batch, num_tokens, d_token]                              │
│  Example: [32, 82, 128]  (80 numerical + 2 categorical)            │
│                                                                     │
│  NO CLS token prepended!                                            │
│                                                                     │
│  ═══════════════════════════════════════════════════════════        │
│                                                                     │
│  STAGE 2: Transformer Processing (3 Layers)                        │
│  ─────────────────────────────────────────                         │
│  Input: [32, 82, 128]                                              │
│           ↓                                                         │
│  ┌────────────────────────────────────────────┐                   │
│  │  Transformer Layer 1                       │                   │
│  │  Output: [32, 82, 128]                     │                   │
│  └────────────────────────────────────────────┘                   │
│           ↓                                                         │
│  ┌────────────────────────────────────────────┐                   │
│  │  Transformer Layer 2                       │                   │
│  │  Output: [32, 82, 128]                     │                   │
│  └────────────────────────────────────────────┘                   │
│           ↓                                                         │
│  ┌────────────────────────────────────────────┐                   │
│  │  Transformer Layer 3                       │                   │
│  │  Output: [32, 82, 128] ◄──────────────────┼─ Transformer output│
│  └────────────────────────────────────────────┘    (contextualized)│
│                                                                     │
│  All 82 feature tokens have been contextualized through attention. │
│                                                                     │
│  ═══════════════════════════════════════════════════════════        │
│                                                                     │
│  STAGE 3: Aggregation (POOLING HAPPENS HERE!)                     │
│  ────────────────────────────────────────────                      │
│  Transformer output: [32, 82, 128]                                 │
│           ↓                                                         │
│  ┌────────────────────────────────────────────┐                   │
│  │  CHOOSE ONE POOLING STRATEGY:              │                   │
│  │                                             │                   │
│  │  Option A: Mean Pooling                    │                   │
│  │  aggregated = torch.mean(output, dim=1)    │                   │
│  │                                             │                   │
│  │  Option B: Attention Pooling               │                   │
│  │  aggregated = attention_pool(output)       │                   │
│  │                                             │                   │
│  │  Option C: Multi-Head Attention Pooling    │                   │
│  │  aggregated = multihead_pool(output)       │                   │
│  │                                             │                   │
│  │  Option D: Temporal Attention Pooling      │                   │
│  │  aggregated = temporal_pool(output)        │                   │
│  └────────────────────────────────────────────┘                   │
│           ↓                                                         │
│  Aggregated: [32, 128]  ◄───────────────────── Single vector      │
│                                                  per sample         │
│  ═══════════════════════════════════════════════════════════        │
│                                                                     │
│  STAGE 4: Prediction                                               │
│  ──────────────────                                                │
│  Input: [32, 128]                                                  │
│           ↓                                                         │
│  Linear(128, output_dim)                                           │
│           ↓                                                         │
│  Predictions: [32, output_dim]                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Insights

### 1. Pooling is AFTER Transformer, Not Before

```
❌ WRONG:
Raw features → Pooling → Transformer → Prediction

✅ CORRECT:
Raw features → Tokenization → Transformer → Pooling → Prediction
              [batch, N, 128]   ↓           [batch, N, 128]
                            3 layers of
                         contextualization
```

### 2. Why Pooling Comes After Transformer

**The transformer's job is to CONTEXTUALIZE each token:**

```
Before Transformer:
Token₀: [raw price at t=0]
Token₁: [raw price at t=1]
Token₂: [raw price at t=2]
...
Each token only knows about itself

After Transformer:
Token₀: [price at t=0, aware of all other timesteps, patterns, etc.]
Token₁: [price at t=1, aware of all other timesteps, patterns, etc.]
Token₂: [price at t=2, aware of all other timesteps, patterns, etc.]
...
Each token has incorporated information from ALL other tokens via attention
```

**Then pooling aggregates these contextualized tokens:**

```
Pooling takes contextualized tokens → combines them → single representation

This single representation contains:
- Information from all timesteps
- Patterns discovered by transformer
- Relationships between features
- Temporal dependencies
```

### 3. What Each Token Represents After Transformer

```
Input Token (before transformer):
  [feature value, position encoding]
  Example: "Price is $100 at t=0"

Output Token (after transformer):
  [contextualized representation]
  Example: "Price is $100 at t=0, AND I know:
    - Price trend is increasing
    - Volume is high at nearby timesteps
    - This is symbol AAPL (from categorical token)
    - Recent volatility pattern
    - Correlation with other features"

The transformer has enriched each token with CONTEXT!
```

---

## Detailed Example with Real Data

### Setup
```python
batch_size = 32
sequence_length = 10
num_numerical = 8
d_token = 128
n_layers = 3
```

### Step-by-Step Shape Transformations

```
┌──────────────────────────────────────────────────────────────────┐
│ Stage                    Shape              Description           │
├──────────────────────────────────────────────────────────────────┤
│ Raw Input                [32, 10, 8]        Price, volume, etc.  │
│                                             10 timesteps          │
│                                             8 features            │
├──────────────────────────────────────────────────────────────────┤
│ After Tokenization       [32, 82, 128]      80 num + 2 cat       │
│                                             Each: 128-dim vector  │
├──────────────────────────────────────────────────────────────────┤
│ After Transformer L1     [32, 82, 128]      Contextualized       │
│                                             by layer 1            │
├──────────────────────────────────────────────────────────────────┤
│ After Transformer L2     [32, 82, 128]      Further              │
│                                             contextualized        │
├──────────────────────────────────────────────────────────────────┤
│ After Transformer L3     [32, 82, 128] ◄─── THIS IS WHERE       │
│                                             POOLING OPERATES!     │
├──────────────────────────────────────────────────────────────────┤
│ After Pooling            [32, 128]          Single vector        │
│                                             per sample            │
├──────────────────────────────────────────────────────────────────┤
│ After Prediction Head    [32, 1]            Final predictions    │
└──────────────────────────────────────────────────────────────────┘
```

### What Pooling Sees

```python
# Transformer output (what pooling receives as input)
transformer_output = [
    [Token₀_contextualized, Token₁_contextualized, ..., Token₈₁_contextualized],  # Sample 0
    [Token₀_contextualized, Token₁_contextualized, ..., Token₈₁_contextualized],  # Sample 1
    ...
    [Token₀_contextualized, Token₁_contextualized, ..., Token₈₁_contextualized],  # Sample 31
]
# Shape: [32, 82, 128]

# Each token has been enriched by 3 layers of self-attention
# Token₀ now contains information about:
#   - Itself (original price at t=0)
#   - All other timesteps (via attention)
#   - Categorical features (symbol, sector)
#   - Patterns learned by transformer (momentum, volatility, etc.)

# Pooling aggregates these 82 contextualized tokens
if pooling_type == "mean":
    aggregated = torch.mean(transformer_output, dim=1)  # [32, 128]

elif pooling_type == "attention":
    # Learnable query asks: "Which of these 82 contextualized tokens matter most?"
    aggregated = attention_pooling(transformer_output)  # [32, 128]

elif pooling_type == "temporal":
    # Multi-head attention + temporal bias on contextualized tokens
    aggregated = temporal_attention_pooling(transformer_output)  # [32, 128]
```

---

## Why This Order Makes Sense

### Analogy: Team Meeting → Summary

```
Raw Data (individuals' thoughts):
  Person 0: "I saw prices rise"
  Person 1: "I saw high volume"
  Person 2: "Market sentiment positive"
  ...

After Transformer (everyone has discussed and shared):
  Person 0: "I saw prices rise, AND based on group discussion:
             - Volume correlates with my price observation
             - Sentiment aligns
             - Historical pattern matches
             - Everyone agrees there's a trend"
  Person 1: "I saw high volume, AND based on discussion:
             - This confirms Person 0's price rise
             - Timing matches Person 2's sentiment
             - Pattern is significant"
  ...

Pooling (create summary report):
  - CLS approach: Designated note-taker summarizes
  - Mean pooling: Average everyone's informed opinions
  - Attention pooling: Weight people based on expertise
  - Temporal pooling: Weight recent speakers more, but still consider important past points
```

### Technical Reasoning

1. **Transformer enriches each token with context**
   - Token₀ learns about Token₁, Token₂, ..., Token₈₁
   - All tokens exchange information via attention
   - Result: Rich, contextualized representations

2. **Pooling aggregates enriched tokens**
   - Operating on contextualized tokens (not raw features)
   - Each token already contains global information
   - Aggregation creates a single summary representation

3. **Prediction head makes final prediction**
   - Receives single 128-dim vector
   - This vector contains information from ALL tokens
   - Has been processed through 3 transformer layers
   - Has been aggregated by pooling strategy

---

## CSN-Transformer: Dual-Path Pooling

For CSN, pooling happens **independently** on each path's transformer output:

```
Categorical Path:
  Tokens: [32, 2, 128]
       ↓
  Cat Transformer (3 layers)
       ↓
  Output: [32, 2, 128] ◄──────────── Pooling operates HERE
       ↓
  Pooling (mean or attention)
       ↓
  Aggregated: [32, 128]

Numerical Path:
  Tokens: [32, 10, 128]
       ↓
  Num Transformer (3 layers)
       ↓
  Output: [32, 10, 128] ◄─────────── Pooling operates HERE
       ↓
  Pooling (temporal attention)
       ↓
  Aggregated: [32, 128]

Fusion:
  Cat: [32, 128] + Num: [32, 128]
       ↓
  Concatenate
       ↓
  Fused: [32, 256]
       ↓
  Prediction Head
       ↓
  Output: [32, 1]
```

---

## Summary

**Q: Does pooling happen on the output of transformer layers?**

**A: YES, absolutely!**

The complete flow is:
1. **Tokenization**: Convert raw features to tokens
2. **Transformer**: Process tokens through multiple attention layers (contextualization)
3. **Pooling** ← HAPPENS HERE: Aggregate contextualized tokens
4. **Prediction**: Make final prediction from aggregated representation

Pooling operates on **contextualized token representations** that have been enriched through multiple transformer layers, NOT on raw input features.

---

**END OF CLARIFICATION**
