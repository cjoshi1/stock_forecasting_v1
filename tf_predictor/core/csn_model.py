"""
CSNTransformer: Categorical-Seasonal-Numerical Transformer for Time Series Prediction.

🧠 CSN-TRANSFORMER CLS MODEL (DUAL-PATH ARCHITECTURE)
======================================================

The CSN-Transformer CLS Model is a dual-path architecture specifically designed for time
series forecasting with STATIC categorical features and TIME-VARYING numerical sequences.
Unlike unified approaches, it processes categorical and numerical features through separate
specialized transformers before late fusion.

📊 ARCHITECTURE OVERVIEW:
┌─────────────────────────────────────────────────────────────────────────────┐
│  STATIC Categorical Features → Categorical Transformer → CLS₁               │
│  [batch, num_categorical]    → [embeddings + attention]  → [batch, d_token] │
│                                                                    ↓         │
│                                                               CONCATENATE    │
│                                                                    ↓         │
│  TIME-VARYING Numerical Seq  → Numerical Transformer      → CLS₂  → Predict │
│  [batch, seq_len, num_num]   → [projection + attention]  → [batch, d_token] │
└─────────────────────────────────────────────────────────────────────────────┘

🔄 DATA FLOW WITH MATRIX DIMENSIONS:

Example Configuration:
- batch_size = 32
- sequence_length = 10
- num_numerical = 8 (price, volume, technical indicators, etc.)
- num_categorical = 2 (symbol, sector)
- cat_cardinalities = [100, 5] (100 stock symbols, 5 sectors)
- d_model = 128 (embedding dimension)
- num_heads = 8
- num_layers = 3
- output_dim = 1 (single-step forecast)

🏗️ DUAL-PATH PROCESSING ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────────────┐
│ PATH 1: CATEGORICAL PROCESSING (STATIC FEATURES)                            │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│ Step 1: Input Format                                                         │
│   x_cat: [32, 2]  # [batch_size, num_categorical]                          │
│   Values are integer indices (label encoded):                               │
│     - Column 0: symbol indices [0-99]                                       │
│     - Column 1: sector indices [0-4]                                        │
│                                                                              │
│ Step 2: Categorical Embedding with Logarithmic Scaling                      │
│   FORMULA: emb_dim = int(8 * log2(cardinality + 1))                        │
│            emb_dim = clamp(emb_dim, d_model/4, d_model)                    │
│                                                                              │
│   symbol_emb = Embedding(100, 53)(x_cat[:, 0])  # [32, 53]                │
│   sector_emb = Embedding(5, 32)(x_cat[:, 1])    # [32, 32]                │
│                                                                              │
│ Step 3: Project to d_model                                                  │
│   symbol_proj = Linear(53, 128)(symbol_emb)     # [32, 128]                │
│   sector_proj = Linear(32, 128)(sector_emb)     # [32, 128]                │
│                                                                              │
│ Step 4: Stack Categorical Tokens                                            │
│   cat_tokens = stack([symbol_proj, sector_proj], dim=1)                    │
│   →  [32, 2, 128]  # [batch, num_categorical, d_token]                     │
│                                                                              │
│ Step 5: Add CLS₁ Token                                                      │
│   cls1_token = CLSToken(d_model)                # [1, 1, 128]              │
│   cls1_expanded = cls1_token.expand(32, -1, -1) # [32, 1, 128]            │
│   tokens_with_cls = cat([cls1_expanded, cat_tokens], dim=1)               │
│   →  [32, 3, 128]  # [batch, 1 + num_categorical, d_model]                │
│                                                                              │
│ Step 6: Categorical Transformer (3 layers)                                  │
│   for layer in range(num_layers):                                          │
│     # Multi-Head Self-Attention (num_heads=8)                              │
│     Q, K, V = tokens @ W_q, W_k, W_v  # [32, 3, 128]                      │
│     d_head = 128 / 8 = 16                                                  │
│     attention = softmax(QK^T / √16) @ V                                    │
│     tokens = LayerNorm(tokens + attention)                                 │
│                                                                              │
│     # Feed-Forward Network                                                  │
│     ffn = Linear(ReLU(Linear(tokens, 512)), 128)                          │
│     tokens = LayerNorm(tokens + ffn)                                       │
│   →  [32, 3, 128]                                                          │
│                                                                              │
│ Step 7: Extract CLS₁                                                        │
│   cls1_output = tokens[:, 0, :]  # [32, 128]                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PATH 2: NUMERICAL SEQUENCE PROCESSING (TIME-VARYING FEATURES)               │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│ Step 1: Input Format                                                         │
│   x_num: [32, 10, 8]  # [batch_size, sequence_length, num_numerical]      │
│   Time-varying features across 10 timesteps:                                │
│     - open, high, low, close, volume                                        │
│     - technical indicators (RSI, MACD, etc.)                                │
│                                                                              │
│ Step 2: Project Numerical Features to d_model                               │
│   num_proj = Linear(8, 128)(x_num)                                         │
│   →  [32, 10, 128]  # [batch, sequence_length, d_model]                   │
│                                                                              │
│ Step 3: Add CLS₂ Token                                                      │
│   cls2_token = CLSToken(d_model)                # [1, 1, 128]              │
│   cls2_expanded = cls2_token.expand(32, -1, -1) # [32, 1, 128]            │
│   tokens_with_cls = cat([cls2_expanded, num_proj], dim=1)                 │
│   →  [32, 11, 128]  # [batch, 1 + sequence_length, d_model]               │
│                                                                              │
│ Step 4: Add Positional Encoding (Temporal)                                  │
│   positions = [0, 1, 2, ..., 10]                                           │
│   pos_encoding = PositionalEncoding(positions)  # [32, 11, 128]           │
│   tokens = tokens_with_cls + pos_encoding                                  │
│                                                                              │
│ Step 5: Numerical Transformer (3 layers)                                    │
│   for layer in range(num_layers):                                          │
│     # Multi-Head Self-Attention (num_heads=8)                              │
│     Q, K, V = tokens @ W_q, W_k, W_v  # [32, 11, 128]                     │
│     attention = softmax(QK^T / √16) @ V                                    │
│     tokens = LayerNorm(tokens + attention)                                 │
│                                                                              │
│     # Feed-Forward Network                                                  │
│     ffn = Linear(ReLU(Linear(tokens, 512)), 128)                          │
│     tokens = LayerNorm(tokens + ffn)                                       │
│   →  [32, 11, 128]                                                         │
│                                                                              │
│ Step 6: Extract CLS₂                                                        │
│   cls2_output = tokens[:, 0, :]  # [32, 128]                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PATH 3: LATE FUSION & PREDICTION                                            │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│ Step 1: Concatenate CLS Tokens                                              │
│   fused_representation = cat([cls1_output, cls2_output], dim=1)           │
│   →  [32, 256]  # [batch, 2 * d_model]                                    │
│                                                                              │
│ Step 2: Prediction Head (MultiHorizonHead)                                  │
│   predictions = Linear(256, output_dim)(fused_representation)              │
│   →  [32, 1]  # [batch, output_dim]                                       │
│                                                                              │
│   For multi-horizon (output_dim=3):                                         │
│     predictions = Linear(256, 3)(fused_representation)                     │
│     →  [32, 3]  # Forecasts for t+1, t+2, t+3                             │
└─────────────────────────────────────────────────────────────────────────────┘

🎯 KEY ARCHITECTURAL FEATURES:

1. Logarithmic Embedding Scaling for Categorical Features:
   - emb_dim = int(8 * log2(cardinality + 1))
   - Bounds: [d_model/4, d_model] = [32, 128]
   - Information-theoretic capacity matching

2. Dual Transformer Processing:
   - Separate attention mechanisms for categorical and numerical
   - Categorical: learns feature interactions (symbol-sector relationships)
   - Numerical: learns temporal patterns (price momentum, trends)

3. CLS Token Strategy:
   - CLS₁: aggregates categorical feature information
   - CLS₂: aggregates temporal sequence information
   - Late fusion preserves both representations

4. Positional Encoding:
   - Only applied to numerical path (temporal sequences)
   - Not needed for categorical (static features have no temporal order)

🧠 COMPUTATIONAL COMPLEXITY:

Memory Usage (per forward pass):
- Categorical tokens: 32 × 3 × 128 × 4 bytes ≈ 49 KB
- Numerical tokens: 32 × 11 × 128 × 4 bytes ≈ 180 KB
- Categorical attention: 32 × 8 × 3² × 4 bytes ≈ 9 KB (per layer)
- Numerical attention: 32 × 8 × 11² × 4 bytes ≈ 124 KB (per layer)
- Total: ~362 KB + ~399 KB (3 layers) ≈ 761 KB

Time Complexity:
- Categorical path: O(L_cat × (T_cat² × d + T_cat × d²))
  where T_cat = 1 + num_categorical = 3
- Numerical path: O(L_num × (T_num² × d + T_num × d²))
  where T_num = 1 + sequence_length = 11
- Dominated by numerical path due to longer sequence

🎨 ADVANTAGES OVER UNIFIED ARCHITECTURES:

1. Specialized Feature Processing:
   - Categorical: embeddings for discrete features
   - Numerical: projections for continuous sequences
   - No mixing of different feature types in early layers

2. Reduced Computational Cost:
   - Two smaller transformers instead of one large
   - Categorical attention: O(3²) vs unified O(14²)
   - Numerical attention: O(11²) vs unified O(14²)
   - Total: O(9 + 121) = 130 vs O(196) operations

3. Better Feature Learning:
   - Categorical features learn inter-feature relationships
   - Numerical features learn temporal dependencies
   - No interference between static and time-varying patterns

4. Flexibility for Missing Modalities:
   - Can handle numerical-only data (x_cat=None)
   - Can adapt to varying categorical feature counts
   - Graceful degradation when features are absent

⚡ USAGE NOTES:

1. When num_categorical = 0:
   - Only numerical path is used
   - cls1_output is omitted from fusion
   - Behaves like standard sequence transformer

2. Feature Preparation:
   - Categorical features: integer indices [0, cardinality-1]
   - Numerical features: normalized/scaled continuous values
   - Cardinalities must match actual unique values in data

3. Training Considerations:
   - Separate learning rates for categorical and numerical paths
   - Dropout applied independently to each path
   - Batch normalization can stabilize categorical embeddings

This dual-path architecture is optimal for financial time series forecasting where
static entity features (symbol, sector) combine with dynamic market data (OHLCV, indicators).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from .base.model_interface import TransformerBasedModel
from .base.embeddings import CLSToken
from .base.prediction_heads import MultiHorizonHead


class CSNTransformerBlock(nn.Module):
    """
    Standard transformer block used in both categorical and numerical processors.

    Architecture: MultiHeadAttention → Add&Norm → FeedForward → Add&Norm
    This is the core building block that enables feature interaction learning.
    """

    def __init__(self, d_token: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Multi-head self-attention mechanism
        # Allows each position to attend to all positions in the sequence
        self.attention = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)

        # Layer normalization for training stability
        self.norm1 = nn.LayerNorm(d_token)  # After attention
        self.norm2 = nn.LayerNorm(d_token)  # After feed-forward

        # Position-wise feed-forward network
        # Applies same transformation to each position independently
        self.feed_forward = nn.Sequential(
            nn.Linear(d_token, d_ff),       # Expand to hidden dimension
            nn.ReLU(),                      # Non-linear activation
            nn.Dropout(dropout),            # Regularization
            nn.Linear(d_ff, d_token),       # Project back to d_token
            nn.Dropout(dropout)             # Final regularization
        )

    def forward(self, x):
        """
        Args:
            x: Input tokens [batch_size, seq_len, d_token]
        Returns:
            Output tokens [batch_size, seq_len, d_token]
        """
        # Self-attention with residual connection and layer norm
        # Each token can attend to every other token in the sequence
        attn_out, _ = self.attention(x, x, x)  # Q=K=V=x for self-attention
        x = self.norm1(x + attn_out)           # Residual connection + normalization

        # Feed-forward with residual connection and layer norm
        # Apply position-wise transformation
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)             # Residual connection + normalization

        return x


class CategoricalProcessor(nn.Module):
    """
    Processes categorical features through embeddings and transformer blocks.

    This processor handles discrete features like:
    - Temporal features: year, quarter, month, day_of_week
    - Seasonal features: month_sin, month_cos, hour_sin, hour_cos (discretized)
    - Other categorical features: region, category, etc.

    Architecture: Embeddings → CLS+Features → Positional → Transformer → CLS Output
    """

    def __init__(self,
                 categorical_features: Dict[str, int],  # feature_name: num_unique_values
                 d_token: int,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.d_token = d_token
        self.feature_names = list(categorical_features.keys())

        # Create embedding table for each categorical feature
        # Each feature gets its own vocabulary and embedding space
        self.embeddings = nn.ModuleDict()
        for feature_name, vocab_size in categorical_features.items():
            # Add 1 for potential unknown/padding values (vocab_size + 1)
            # Maps discrete values [0, vocab_size-1] → dense vectors [d_token]
            self.embeddings[feature_name] = nn.Embedding(vocab_size + 1, d_token)

        # CLS token for categorical feature aggregation
        # This token will collect information from all categorical features
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        # Stack of transformer blocks for categorical feature interaction
        d_ff = d_token * 4  # Standard transformer feed-forward dimension
        self.transformer_blocks = nn.ModuleList([
            CSNTransformerBlock(d_token, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Positional embeddings to distinguish feature positions
        # Categorical features don't have inherent order, but position helps attention
        max_seq_len = len(categorical_features) + 1  # +1 for CLS token
        self.pos_embedding = nn.Embedding(max_seq_len, d_token)

    def forward(self, categorical_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process categorical features through embeddings and transformer.

        Args:
            categorical_inputs: Dict mapping feature names to index tensors
                               e.g., {'year': [32], 'quarter': [32], 'month_sin': [32]}
        Returns:
            cls_output: [batch_size, d_token] - Aggregated categorical representation
        """
        batch_size = next(iter(categorical_inputs.values())).size(0)
        device = next(iter(categorical_inputs.values())).device

        # Step 1: Convert categorical indices to dense embeddings
        embedded_features = []
        for feature_name in self.feature_names:
            if feature_name in categorical_inputs:
                # Lookup embedding: [batch_size] → [batch_size, d_token]
                embedded = self.embeddings[feature_name](categorical_inputs[feature_name])
                embedded_features.append(embedded)

        # Step 2: Stack all feature embeddings
        if embedded_features:
            # Stack: List of [batch, d_token] → [batch, num_features, d_token]
            feature_embeddings = torch.stack(embedded_features, dim=1)
        else:
            # Handle case with no categorical features
            feature_embeddings = torch.zeros(batch_size, 0, self.d_token, device=device)

        # Step 3: Add CLS token for aggregation
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, d_token]

        # Concatenate: [batch, 1, d_token] + [batch, num_features, d_token]
        # Result: [batch, 1 + num_features, d_token]
        x = torch.cat([cls_tokens, feature_embeddings], dim=1)

        # Step 4: Add positional embeddings
        # Help model distinguish between different categorical features
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)  # [batch, seq_len, d_token]
        x = x + pos_emb

        # Step 5: Apply transformer blocks for feature interaction
        # Each categorical feature can attend to every other categorical feature
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)  # [batch, 1 + num_features, d_token]

        # Step 6: Extract CLS token output
        # CLS token has aggregated information from all categorical features
        cls_output = x[:, 0, :]  # [batch_size, d_token]
        return cls_output


class NumericalProcessor(nn.Module):
    """
    Processes numerical sequences through projection and transformer blocks.

    This processor handles continuous features like:
    - Price sequences: [open, high, low, close] over time
    - Volume patterns: trading volume across time periods
    - Technical indicators: moving averages, volatility, momentum
    - Ratio features: price/volume ratios, percentage changes

    Architecture: Projection → CLS+Sequences → Positional → Transformer → CLS Output
    """

    def __init__(self,
                 num_numerical_features: int,
                 sequence_length: int,
                 d_token: int,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.num_features = num_numerical_features
        self.sequence_length = sequence_length
        self.d_token = d_token

        # Project each timestep's numerical features to d_token dimension
        # Maps [num_features] → [d_token] for each timestep
        self.feature_projection = nn.Linear(num_numerical_features, d_token)

        # CLS token for numerical sequence aggregation
        # This token will collect temporal patterns from numerical sequences
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        # Stack of transformer blocks for temporal pattern learning
        d_ff = d_token * 4  # Standard transformer feed-forward dimension
        self.transformer_blocks = nn.ModuleList([
            CSNTransformerBlock(d_token, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Positional embeddings for temporal order
        # Critical for preserving time series ordering in attention
        max_seq_len = sequence_length + 1  # +1 for CLS token
        self.pos_embedding = nn.Embedding(max_seq_len, d_token)

    def forward(self, numerical_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through numerical processor with detailed step-by-step computation.

        Args:
            numerical_inputs: [batch_size, sequence_length, num_features]
                            Time series data with multiple numerical features per timestep

        Returns:
            cls_output: [batch_size, d_token] - Aggregated temporal representation

        Processing Flow:
        1. Feature Projection: Map each timestep's features to embedding space
        2. CLS Token Addition: Add aggregation token for sequence representation
        3. Positional Encoding: Add temporal order information
        4. Transformer Processing: Learn temporal patterns via self-attention
        5. CLS Extraction: Extract aggregated sequence representation
        """
        # Input validation and shape extraction
        batch_size, seq_len, num_features = numerical_inputs.shape
        device = numerical_inputs.device

        # Step 1: Feature Projection - Convert continuous features to embeddings
        # Transform each timestep: [num_features] → [d_token]
        # This allows the transformer to work with consistent dimensionality
        x = self.feature_projection(numerical_inputs)  # [batch_size, seq_len, d_token]

        # Step 2: Add CLS Token for sequence aggregation
        # CLS token will learn to aggregate information from all timesteps
        # Shape transformation: [1, 1, d_token] → [batch_size, 1, d_token]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Concatenate CLS with projected sequences
        # [batch_size, 1, d_token] + [batch_size, seq_len, d_token]
        # Result: [batch_size, seq_len+1, d_token]
        x = torch.cat([cls_tokens, x], dim=1)

        # Step 3: Add Positional Embeddings for temporal order
        # Critical for time series: position 0=CLS, positions 1...seq_len=timesteps
        # Without this, the model can't distinguish between different timesteps
        total_seq_len = x.size(1)  # seq_len + 1 (for CLS)
        positions = torch.arange(total_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)  # [batch_size, seq_len+1, d_token]
        x = x + pos_emb  # Element-wise addition of positional information

        # Step 4: Apply Transformer Blocks for temporal pattern learning
        # Each block allows every position to attend to every other position
        # CLS token can attend to all timesteps, timesteps can attend to each other
        for transformer_block in self.transformer_blocks:
            # Self-attention + feed-forward with residual connections
            x = transformer_block(x)  # [batch_size, seq_len+1, d_token]

        # Step 5: Extract CLS Token Output
        # CLS token (position 0) has aggregated information from entire sequence
        # This becomes our fixed-size representation of the variable-length sequence
        cls_output = x[:, 0, :]  # [batch_size, d_token]

        # Return the aggregated temporal representation
        # This will be concatenated with categorical CLS for final prediction
        return cls_output


class CSNTransformer(nn.Module):
    """
    🧠 CSN-TRANSFORMER: MAIN ARCHITECTURE CLASS
    ===========================================

    The Categorical-Seasonal-Numerical Transformer implements dual-pathway processing
    for time series data with mixed feature types. This is the main orchestrator class
    that coordinates categorical and numerical processors for unified predictions.

    🏗️ DUAL-PATHWAY ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ CATEGORICAL PATH:                                                           │
    │ Features: {year, quarter, month_sin, month_cos, ...}                       │
    │ Pipeline: Discrete Values → Embeddings → Transformer → CLS₁                │
    │ Output: [batch_size, d_model] representation of categorical patterns       │
    └─────────────────────────────────────────────────────────────────────────────┘
                                         ↓
                                    FUSION LAYER
                                         ↓
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ NUMERICAL PATH:                                                             │
    │ Features: [price, volume, technical_indicators] sequences                  │
    │ Pipeline: Sequences → Projection → Transformer → CLS₂                      │
    │ Output: [batch_size, d_model] representation of temporal patterns          │
    └─────────────────────────────────────────────────────────────────────────────┘
                                         ↓
                              PREDICTION HEAD (Multi-Horizon)
                                         ↓
                            [batch_size, prediction_horizons, output_dim]

    🎯 MULTI-HORIZON PREDICTION CAPABILITY:

    Single-Horizon Mode (prediction_horizons=1):
    - Predicts only the next time step
    - Output shape: [batch_size, 1] or [batch_size] after squeeze
    - Example: Predict tomorrow's closing price

    Multi-Horizon Mode (prediction_horizons>1):
    - Predicts multiple future time steps simultaneously
    - Output shape: [batch_size, prediction_horizons]
    - Example: prediction_horizons=3 → predict steps 1, 2, 3 ahead
    - Enables better long-term forecasting with consistent predictions

    📊 MATHEMATICAL FORMULATION:

    Let:
    - C = categorical features {c₁, c₂, ..., cₖ}
    - N = numerical sequences [n₁, n₂, ..., nₜ] where each nᵢ ∈ ℝᵈ
    - E_c = categorical embedding function
    - E_n = numerical projection function
    - T_c = categorical transformer
    - T_n = numerical transformer

    Forward Pass:
    1. CLS₁ = T_c(E_c(C))                    # Categorical representation
    2. CLS₂ = T_n(E_n(N))                    # Numerical representation
    3. fused = Concat(CLS₁, CLS₂)            # Feature fusion
    4. output = MLP(fused)                   # Final prediction

    Multi-horizon extension:
    5. if prediction_horizons > 1:
         output = output.view(batch, horizons, features)

    🧠 ATTENTION MECHANISMS:

    Categorical Attention:
    - Features attend to other categorical features
    - Learns relationships like "Q4 + December → holiday season"
    - Seasonal patterns: sin/cos pairs attend to each other

    Numerical Attention:
    - Timesteps attend to other timesteps in sequence
    - Learns temporal patterns like "price drop → volume spike"
    - CLS token aggregates across all timesteps

    ⚡ PERFORMANCE CHARACTERISTICS:

    Memory Usage (batch_size=32, d_model=128):
    - Categorical pathway: ~80-120 KB per layer
    - Numerical pathway: ~100-150 KB per layer
    - Fusion layer: ~32 KB
    - Total: O(batch_size × d_model × (seq_len + num_categories))

    Computational Complexity:
    - Categorical: O(num_categories² × d_model) per layer
    - Numerical: O(sequence_length² × d_model) per layer
    - Independent scaling with different feature dimensions

    🎨 DESIGN BENEFITS:

    1. Feature Type Specialization:
       - Categorical features use discrete embeddings
       - Numerical features use continuous projections
       - Each pathway optimized for its data type

    2. Reduced Interference:
       - Categorical and numerical features don't interfere during learning
       - Late fusion preserves specialized representations
       - Better gradient flow for each feature type

    3. Seasonal Pattern Handling:
       - Sin/cos seasonal features treated as high-resolution categorical
       - Learnable embeddings capture complex seasonal relationships
       - Better than raw numerical seasonal encoding

    4. Scalability:
       - Adding categorical features only affects categorical pathway
       - Adding numerical features only affects numerical pathway
       - Memory usage scales independently

    🚀 USAGE PATTERNS:

    Time Series with Heavy Categorical Components:
    - Financial data: sector, market_cap_category, exchanges
    - Retail data: store_type, promotion_category, seasonal_events
    - Energy data: region, source_type, regulatory_zone

    Time Series with Strong Seasonal Patterns:
    - Daily patterns: hour_sin/hour_cos for intraday cycles
    - Weekly patterns: day_of_week, weekend indicators
    - Yearly patterns: month_sin/month_cos, quarter indicators

    Mixed Feature Types:
    - IoT sensor data: device_type + sensor readings
    - Economic indicators: region + time series metrics
    - Social media: platform + engagement time series
    """

    def __init__(self,
                 categorical_features: Dict[str, int],
                 num_numerical_features: int,
                 sequence_length: int,
                 d_token: int = 128,
                 n_layers: int = 3,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 output_dim: int = 1,
                 prediction_horizons: int = 1):
        super().__init__()

        self.d_token = d_token
        self.has_categorical = len(categorical_features) > 0
        self.has_numerical = num_numerical_features > 0
        self.prediction_horizons = prediction_horizons

        # Categorical processor
        if self.has_categorical:
            self.categorical_processor = CategoricalProcessor(
                categorical_features, d_token, n_layers, n_heads, dropout
            )

        # Numerical processor
        if self.has_numerical:
            self.numerical_processor = NumericalProcessor(
                num_numerical_features, sequence_length, d_token, n_layers, n_heads, dropout
            )

        # Determine fusion dimension
        fusion_dim = 0
        if self.has_categorical:
            fusion_dim += d_token
        if self.has_numerical:
            fusion_dim += d_token

        # Fully connected layers for final prediction - multi-horizon support
        final_output_dim = output_dim * prediction_horizons
        self.prediction_head = nn.Sequential(
            nn.Linear(fusion_dim, d_token),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, d_token // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token // 2, final_output_dim)
        )

        self.output_dim = output_dim

    def forward(self,
                categorical_inputs: Optional[Dict[str, torch.Tensor]] = None,
                numerical_inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the complete CSN-Transformer architecture.

        Args:
            categorical_inputs: Dict of categorical feature tensors
                              e.g., {'year': [32], 'quarter': [32], 'month_sin': [32]}
            numerical_inputs: [batch_size, sequence_length, num_features]
                            Time series sequences with numerical features

        Returns:
            predictions: [batch_size] for single-horizon OR
                        [batch_size, prediction_horizons] for multi-horizon

        Processing Flow:
        1. Dual Processing: Run categorical and numerical processors in parallel
        2. CLS Fusion: Concatenate the learned representations
        3. Prediction: Generate final predictions with multi-horizon support
        4. Reshaping: Format output based on prediction horizon configuration
        """
        cls_tokens = []  # Collect CLS outputs from both pathways

        # PATHWAY 1: CATEGORICAL FEATURE PROCESSING
        # Process discrete features (year, quarter, seasonal embeddings, etc.)
        if self.has_categorical and categorical_inputs is not None:
            # Categorical processor handles:
            # - Feature embedding lookup
            # - CLS token aggregation via attention
            # - Returns: [batch_size, d_model]
            cat_cls = self.categorical_processor(categorical_inputs)
            cls_tokens.append(cat_cls)

        # PATHWAY 2: NUMERICAL SEQUENCE PROCESSING
        # Process continuous time series features (prices, volumes, indicators, etc.)
        if self.has_numerical and numerical_inputs is not None:
            # Numerical processor handles:
            # - Feature projection to embedding space
            # - Temporal attention across sequence
            # - CLS token aggregation
            # - Returns: [batch_size, d_model]
            num_cls = self.numerical_processor(numerical_inputs)
            cls_tokens.append(num_cls)

        # VALIDATION: Ensure at least one input pathway is active
        if not cls_tokens:
            raise ValueError("At least one of categorical or numerical inputs must be provided")

        # FUSION: Concatenate CLS representations
        # Combine categorical and numerical representations into unified feature vector
        # Shape: [batch_size, d_model] + [batch_size, d_model] → [batch_size, 2*d_model]
        # If only one pathway: [batch_size, d_model]
        fused_cls = torch.cat(cls_tokens, dim=1)  # [batch_size, fusion_dim]

        # PREDICTION HEAD: Generate final predictions
        # Multi-layer perceptron transforms fused representation to predictions
        # For multi-horizon: outputs prediction_horizons * output_dim values
        output = self.prediction_head(fused_cls)

        # MULTI-HORIZON RESHAPING
        # Transform flat output to structured multi-horizon format
        if self.prediction_horizons > 1:
            batch_size = output.size(0)
            # Reshape: [batch, horizons*features] → [batch, horizons, features] → [batch, horizons]
            # The squeeze(-1) removes the last dimension if output_dim=1
            return output.view(batch_size, self.prediction_horizons, self.output_dim).squeeze(-1)

        # SINGLE-HORIZON OUTPUT
        # Return as-is for single prediction per sample
        return output


# For compatibility with existing FT-Transformer interface
class CSNTransformerPredictor(nn.Module):
    """
    🔗 CSN-TRANSFORMER PREDICTOR WRAPPER
    ===================================

    Compatibility wrapper that provides a unified interface matching the FTTransformerPredictor.
    This ensures seamless integration with the existing tf_predictor ecosystem while leveraging
    the specialized dual-pathway architecture of CSN-Transformer.

    Purpose:
    - Maintains API compatibility with FTTransformerPredictor
    - Enables drop-in replacement for FT-Transformer in existing workflows
    - Provides consistent interface for both single and multi-horizon prediction
    - Simplifies model selection in the TimeSeriesPredictor base class

    Usage Pattern:
    ```python
    # Direct instantiation (compatible with FT-Transformer interface)
    model = CSNTransformerPredictor(
        categorical_features={'year': 5, 'quarter': 4},
        num_numerical_features=8,
        sequence_length=10,
        prediction_horizons=3  # Multi-horizon support
    )

    # Forward pass (same interface as FT-Transformer)
    predictions = model(categorical_inputs, numerical_inputs)
    ```

    Interface Consistency:
    - Same constructor parameters as FTTransformerPredictor
    - Same forward() method signature
    - Same output shapes and behavior
    - Automatic output_dim=1 setting for regression tasks

    This wrapper enables the TimeSeriesPredictor base class to seamlessly switch
    between FT-Transformer and CSN-Transformer based on feature characteristics
    without changing the downstream prediction pipeline.
    """

    def __init__(self,
                 categorical_features: Dict[str, int],
                 num_numerical_features: int,
                 sequence_length: int,
                 d_token: int = 128,
                 n_layers: int = 3,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 prediction_horizons: int = 1):
        super().__init__()

        self.model = CSNTransformer(
            categorical_features=categorical_features,
            num_numerical_features=num_numerical_features,
            sequence_length=sequence_length,
            d_token=d_token,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            output_dim=1,
            prediction_horizons=prediction_horizons
        )

    def forward(self, categorical_inputs=None, numerical_inputs=None):
        return self.model(categorical_inputs, numerical_inputs)


class CSNTransformerCLSModel(TransformerBasedModel):
    """
    🧠 CSN-TRANSFORMER WITH CLS TOKENS FOR CATEGORICAL FEATURES (CSN_TRANSFORMER_CLS)
    ==================================================================================

    This model extends the CSN-Transformer architecture to handle STATIC categorical features
    alongside TIME-VARYING numerical features using a DUAL-PATH processing approach.

    📊 ARCHITECTURE OVERVIEW:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  PATH 1: Categorical Features → Categorical Transformer → CLS₁              │
    │                                                                ↓             │
    │                                                           FUSION             │
    │                                                                ↓             │
    │  PATH 2: Numerical Sequences → Numerical Transformer → CLS₂  → Prediction  │
    └─────────────────────────────────────────────────────────────────────────────┘

    🔄 DATA FLOW WITH MATRIX DIMENSIONS:

    Example Configuration:
    - batch_size = 32
    - sequence_length = 10
    - num_numerical = 8 (price, volume, seasonal features, etc.)
    - num_categorical = 2 (symbol, sector)
    - cat_cardinalities = [100, 5] (100 symbols, 5 sectors)
    - d_model = 128 (embedding dimension)

    Step 1: Input Format
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ x_num: [32, 10, 8]     # Numerical sequences (time-varying)                 │
    │                        # Shape: [batch, seq_len, num_numerical_features]    │
    │                                                                              │
    │ x_cat: [32, 2]         # Categorical features (static per sequence)         │
    │                        # Shape: [batch, num_categorical_features]           │
    │                        # Values are integer indices (label encoded)         │
    └─────────────────────────────────────────────────────────────────────────────┘

    Step 2: Categorical Embedding with Logarithmic Dimension Scaling
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ EMBEDDING DIMENSION FORMULA:                                                │
    │   emb_dim = int(8 * log2(cardinality + 1))                                 │
    │   emb_dim = clamp(emb_dim, d_model/4, d_model)                             │
    │                                                                              │
    │ RATIONALE:                                                                   │
    │   - Logarithmic scaling: Information-theoretic capacity                     │
    │   - Lower bound (d_model/4): Ensures minimum representational capacity     │
    │   - Upper bound (d_model): Prevents excessive dimensions                   │
    │                                                                              │
    │ EXAMPLES (d_model=128):                                                      │
    │   cardinality=10   → emb_dim = int(8*log2(11)) = 27 → clamp → 32          │
    │   cardinality=100  → emb_dim = int(8*log2(101)) = 53 → clamp → 53         │
    │   cardinality=1000 → emb_dim = int(8*log2(1001)) = 79 → clamp → 79        │
    │   cardinality=10000 → emb_dim = int(8*log2(10001)) = 106 → clamp → 106    │
    │                                                                              │
    │ CATEGORICAL EMBEDDING:                                                       │
    │   symbol_embedding: Embedding(100, 53)   # 100 symbols → 53 dims           │
    │   sector_embedding: Embedding(5, 32)     # 5 sectors → 32 dims             │
    │                                                                              │
    │   x_cat: [32, 2] (integer indices)                                          │
    │   → symbol_emb: [32, 53]                                                    │
    │   → sector_emb: [32, 32]                                                    │
    │                                                                              │
    │ PROJECT TO d_model:                                                          │
    │   symbol_proj: Linear(53, 128) → [32, 128]                                 │
    │   sector_proj: Linear(32, 128) → [32, 128]                                 │
    │   cat_tokens: [32, 2, 128]  # Stacked categorical tokens                   │
    └─────────────────────────────────────────────────────────────────────────────┘

    Step 3: PATH 1 - Categorical Processing
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ 1. Add CLS₁ Token: [32, 1, 128]                                             │
    │ 2. Categorical Tokens: [32, 2, 128]                                         │
    │                                                                              │
    │ Concatenate:                                                                 │
    │   cat_tokens_with_cls = cat([CLS₁, cat_tokens], dim=1)                     │
    │   → [32, 3, 128]  # 1 CLS + 2 categorical                                  │
    │                                                                              │
    │ Categorical Transformer:                                                     │
    │   Multi-Head Self-Attention (num_heads=8):                                  │
    │     Q, K, V = cat_tokens_with_cls @ W_q, W_k, W_v                          │
    │     Attention(Q,K,V) = softmax(QK^T/√16)V                                   │
    │     output: [32, 3, 128]                                                    │
    │                                                                              │
    │   Feed-Forward Network:                                                      │
    │     FFN(x) = Linear(GELU(Linear(x, 512)), 128)                             │
    │     output: [32, 3, 128]                                                    │
    │                                                                              │
    │   Repeat for num_layers (typically 2-4)                                     │
    │                                                                              │
    │ Extract CLS₁:                                                               │
    │   cls1_output = cat_transformer_output[:, 0, :]  # [32, 128]               │
    └─────────────────────────────────────────────────────────────────────────────┘

    Step 4: PATH 2 - Numerical Sequence Processing
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ 1. Project numerical features to d_model:                                   │
    │    x_num_proj = Linear(8, 128)(x_num)  # [32, 10, 8] → [32, 10, 128]      │
    │                                                                              │
    │ 2. Add CLS₂ Token: [32, 1, 128]                                             │
    │                                                                              │
    │ 3. Add positional encoding (learnable):                                     │
    │    pos_encoding: [1, 10, 128]                                               │
    │    x_num_proj = x_num_proj + pos_encoding                                  │
    │                                                                              │
    │ Concatenate:                                                                 │
    │   num_tokens_with_cls = cat([CLS₂, x_num_proj], dim=1)                     │
    │   → [32, 11, 128]  # 1 CLS + 10 timesteps                                  │
    │                                                                              │
    │ Numerical Transformer:                                                       │
    │   Multi-Head Self-Attention (num_heads=8):                                  │
    │     Q, K, V = num_tokens_with_cls @ W_q, W_k, W_v                          │
    │     Attention(Q,K,V) = softmax(QK^T/√16)V                                   │
    │     output: [32, 11, 128]                                                   │
    │                                                                              │
    │   Feed-Forward Network:                                                      │
    │     FFN(x) = Linear(GELU(Linear(x, 512)), 128)                             │
    │     output: [32, 11, 128]                                                   │
    │                                                                              │
    │   Repeat for num_layers (typically 2-4)                                     │
    │                                                                              │
    │ Extract CLS₂:                                                               │
    │   cls2_output = num_transformer_output[:, 0, :]  # [32, 128]               │
    └─────────────────────────────────────────────────────────────────────────────┘

    Step 5: Fusion and Prediction
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ Concatenate CLS tokens:                                                     │
    │   fused_cls = cat([cls1_output, cls2_output], dim=1)                       │
    │   → [32, 256]  # [batch, 2 * d_model]                                      │
    │                                                                              │
    │ Prediction Head:                                                             │
    │   predictions = Linear(fused_cls, output_dim)  # [32, 256] → [32, output_dim]│
    │                                                                              │
    │ For multi-horizon forecasting (prediction_horizon=3):                       │
    │   output_dim = 3                                                             │
    │   predictions: [32, 3]  # Predictions for horizons 1, 2, 3                 │
    └─────────────────────────────────────────────────────────────────────────────┘

    🎯 KEY DESIGN DECISIONS:

    1. **Dual-Path Processing**: Separate transformers for categorical and numerical
       - Categorical Transformer: Processes static context features
       - Numerical Transformer: Processes time-varying sequences
       - Each path has its own CLS token

    2. **Categorical Embedding Dimensions**: Logarithmic scaling with bounds
       - Formula: emb_dim = clamp(int(8 * log2(cardinality + 1)), d_model/4, d_model)
       - Balances capacity with efficiency
       - Information-theoretic foundation

    3. **Fusion Strategy**: Late fusion via concatenation
       - CLS₁ captures categorical context
       - CLS₂ captures temporal dynamics
       - Concatenated representation combines both perspectives

    4. **Output Dimension**: 2 * d_model after fusion
       - Richer representation than single CLS token
       - Allows model to maintain separate categorical and temporal information

    🚀 ADVANTAGES:

    1. **Specialized Processing**: Each feature type processed by dedicated pathway
    2. **Flexible Architecture**: Independent control over categorical and numerical paths
    3. **Rich Representations**: Two CLS tokens provide complementary information
    4. **Scalable**: Logarithmic embedding dimensions scale efficiently

    ⚠️ IMPORTANT NOTES:

    1. **Unseen Categories**: Will throw an error (by design)
       - Categorical features must be label encoded during training
       - New categories at inference time are not supported

    2. **Input Format**:
       - x_num: Numerical sequences [batch, seq_len, num_numerical]
       - x_cat: Categorical indices [batch, num_categorical] (dtype: long)

    3. **Model Capacity**: Fusion layer outputs 2 * d_model dimensions
       - More parameters than FT_Transformer_CLS
       - May require more training data

    4. **Categorical Columns**: Must be specified and encoded in predictor.py
       - Encoding: sklearn.preprocessing.LabelEncoder
       - Sequence creation: Extract from last timestep

    This implementation provides a robust dual-path foundation for time series forecasting
    with mixed feature types, maintaining separate processing streams until late fusion.
    """

    def __init__(
        self,
        sequence_length: int,
        num_numerical: int,
        num_categorical: int,
        cat_cardinalities: List[int],
        output_dim: int,
        d_token: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize CSN-Transformer with dual-path CLS tokens for categorical features.

        Args:
            sequence_length: Length of input sequences (lookback window)
            num_numerical: Number of numerical features per timestep
            num_categorical: Number of categorical features
            cat_cardinalities: List of cardinalities for each categorical feature
            output_dim: Output dimension (num_targets * prediction_horizon)
            d_token: Embedding dimension (token size)
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__(d_token=d_token, n_heads=n_heads, n_layers=n_layers)

        self.sequence_length = sequence_length
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.cat_cardinalities = cat_cardinalities
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.activation_name = activation

        # Validate inputs
        if num_categorical > 0 and len(cat_cardinalities) != num_categorical:
            raise ValueError(f"cat_cardinalities length ({len(cat_cardinalities)}) must match num_categorical ({num_categorical})")

        # PATH 1: Categorical Processing
        if num_categorical > 0:
            import math

            # CLS token for categorical path
            self.cls1_token = CLSToken(d_token)

            # Categorical embeddings with logarithmic dimension scaling
            self.cat_embeddings = nn.ModuleList()
            self.cat_projections = nn.ModuleList()

            for cardinality in cat_cardinalities:
                # Calculate embedding dimension using logarithmic scaling
                emb_dim = int(8 * math.log2(cardinality + 1))
                # Clamp to bounds [d_token/4, d_token]
                min_dim = d_token // 4
                max_dim = d_token
                emb_dim = max(min_dim, min(max_dim, emb_dim))

                # Create embedding layer
                embedding = nn.Embedding(cardinality, emb_dim)
                self.cat_embeddings.append(embedding)

                # Project to d_token if needed
                if emb_dim != d_token:
                    projection = nn.Linear(emb_dim, d_token)
                    self.cat_projections.append(projection)
                else:
                    self.cat_projections.append(nn.Identity())

            # Categorical transformer
            cat_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=n_heads,
                dim_feedforward=4 * d_token,
                dropout=dropout,
                activation=activation,
                batch_first=True
            )
            self.cat_transformer = nn.TransformerEncoder(cat_encoder_layer, num_layers=n_layers)

        # PATH 2: Numerical Processing
        if num_numerical > 0:
            # CLS token for numerical path
            self.cls2_token = CLSToken(d_token)

            # Project numerical features to d_token
            self.num_projection = nn.Linear(num_numerical, d_token)

            # Positional encoding for temporal sequences
            self.temporal_pos_encoding = nn.Parameter(torch.randn(1, sequence_length, d_token) * 0.02)

            # Numerical transformer
            num_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=n_heads,
                dim_feedforward=4 * d_token,
                dropout=dropout,
                activation=activation,
                batch_first=True
            )
            self.num_transformer = nn.TransformerEncoder(num_encoder_layer, num_layers=n_layers)

        # Fusion and Prediction Head
        # Input dimension is 2 * d_token (CLS₁ + CLS₂)
        fusion_dim = 2 * d_token if (num_categorical > 0 and num_numerical > 0) else d_token

        self.head = MultiHorizonHead(
            d_input=fusion_dim,
            prediction_horizons=output_dim,
            hidden_dim=None,
            dropout=dropout
        )

    def forward(self, x_num: torch.Tensor, x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with dual-path processing.

        Args:
            x_num: Numerical sequences [batch_size, sequence_length, num_numerical]
            x_cat: Categorical features [batch_size, num_categorical] (integer indices)

        Returns:
            predictions: [batch_size, output_dim]
        """
        batch_size = x_num.shape[0]
        cls_outputs = []

        # PATH 1: Categorical Processing
        if x_cat is not None and self.num_categorical > 0:
            # Step 1: Embed categorical features
            cat_tokens_list = []
            for i in range(self.num_categorical):
                cat_indices = x_cat[:, i]  # [batch]
                emb = self.cat_embeddings[i](cat_indices)  # [batch, emb_dim]
                projected = self.cat_projections[i](emb)  # [batch, d_token]
                cat_tokens_list.append(projected.unsqueeze(1))  # [batch, 1, d_token]

            cat_tokens = torch.cat(cat_tokens_list, dim=1)  # [batch, num_categorical, d_token]

            # Step 2: Add CLS₁ token
            cls1_tokens = self.cls1_token(batch_size)  # [batch, 1, d_token]
            cat_tokens_with_cls = torch.cat([cls1_tokens, cat_tokens], dim=1)  # [batch, 1+num_cat, d_token]

            # Step 3: Categorical transformer
            cat_output = self.cat_transformer(cat_tokens_with_cls)  # [batch, 1+num_cat, d_token]

            # Step 4: Extract CLS₁
            cls1_output = cat_output[:, 0, :]  # [batch, d_token]
            cls_outputs.append(cls1_output)

        # PATH 2: Numerical Processing
        if self.num_numerical > 0:
            # Step 1: Project numerical features
            x_num_proj = self.num_projection(x_num)  # [batch, seq_len, d_token]

            # Step 2: Add positional encoding
            x_num_proj = x_num_proj + self.temporal_pos_encoding  # [batch, seq_len, d_token]

            # Step 3: Add CLS₂ token
            cls2_tokens = self.cls2_token(batch_size)  # [batch, 1, d_token]
            num_tokens_with_cls = torch.cat([cls2_tokens, x_num_proj], dim=1)  # [batch, 1+seq_len, d_token]

            # Step 4: Numerical transformer
            num_output = self.num_transformer(num_tokens_with_cls)  # [batch, 1+seq_len, d_token]

            # Step 5: Extract CLS₂
            cls2_output = num_output[:, 0, :]  # [batch, d_token]
            cls_outputs.append(cls2_output)

        # FUSION: Concatenate CLS outputs
        if len(cls_outputs) == 2:
            fused_cls = torch.cat(cls_outputs, dim=1)  # [batch, 2*d_token]
        else:
            fused_cls = cls_outputs[0]  # Only one path available

        # Prediction
        predictions = self.head(fused_cls)  # [batch, output_dim]

        return predictions

    def get_model_config(self) -> Dict[str, Any]:
        """Get the current configuration of the model."""
        return {
            'model_type': 'csn_transformer_cls',
            'd_token': self.d_token,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'dropout': self.dropout_rate,
            'activation': self.activation_name,
            'sequence_length': self.sequence_length,
            'num_numerical': self.num_numerical,
            'num_categorical': self.num_categorical,
            'cat_cardinalities': self.cat_cardinalities,
            'output_dim': self.output_dim
        }

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get attention weights from the last forward pass.

        Returns:
            Attention weights tensor or None if not available.
        """
        return None  # Not implemented for CLS models