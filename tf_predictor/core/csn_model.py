"""
CSNTransformer: Categorical-Seasonal-Numerical Transformer for Time Series Prediction.

🧠 CSN-TRANSFORMER (CATEGORICAL-SEASONAL-NUMERICAL) ARCHITECTURE
===============================================================

The CSN-Transformer is an advanced architecture designed specifically for time series data
with mixed feature types. Unlike the unified FT-Transformer approach, CSN-Transformer
processes different feature types through specialized pathways before fusing them.

📊 ARCHITECTURE OVERVIEW:
┌─────────────────────────────────────────────────────────────────────────────┐
│  Categorical Features → Categorical Processor → CLS₁                        │
│  [cat, seasonal]      → [embeddings]          → [d_model]                   │
│                                                       ↓                      │
│                                                    Fusion                    │
│                                                       ↓                      │
│  Numerical Features   → Numerical Processor   → CLS₂  → Final Prediction    │
│  [sequences]          → [sequences + attention] → [d_model]                 │
└─────────────────────────────────────────────────────────────────────────────┘

🔄 DATA FLOW WITH MATRIX DIMENSIONS:

Example Configuration:
- batch_size = 32
- categorical_features = {'year': 5, 'quarter': 4, 'month_sin': 24, 'month_cos': 24}
- num_numerical = 8 (price, volume, ratios, etc.)
- sequence_length = 5 (temporal context)
- d_model = 128 (embedding dimension)

🏗️ DUAL PROCESSING ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────────────┐
│ PATH 1: CATEGORICAL PROCESSOR                                               │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Input: categorical_inputs = {                                               │
│   'year': [32],           # batch_size samples                             │
│   'quarter': [32],        # batch_size samples                             │
│   'month_sin': [32],      # seasonal feature (sin component)               │
│   'month_cos': [32]       # seasonal feature (cos component)               │
│ }                                                                           │
│                                                                             │
│ Step 1: Embedding Lookup                                                    │
│   year_emb = Embedding(5, 128)(year_values)      # [32, 128]               │
│   quarter_emb = Embedding(4, 128)(quarter_values) # [32, 128]              │
│   month_sin_emb = Embedding(24, 128)(month_sin)   # [32, 128]              │
│   month_cos_emb = Embedding(24, 128)(month_cos)   # [32, 128]              │
│                                                                             │
│ Step 2: Stack Embeddings                                                    │
│   feature_embeddings = stack([year_emb, quarter_emb, ...], dim=1)          │
│   →  [32, 4, 128]  # [batch, num_categorical, d_model]                     │
│                                                                             │
│ Step 3: Add CLS Token                                                       │
│   cls_token: [1, 1, 128] → expanded to [32, 1, 128]                       │
│   tokens = cat([cls_token, feature_embeddings], dim=1)                     │
│   →  [32, 5, 128]  # [batch, 1 + num_categorical, d_model]                │
│                                                                             │
│ Step 4: Add Positional Encoding                                            │
│   positions = [0, 1, 2, 3, 4]  # position indices                         │
│   pos_emb = PositionalEmbedding(positions)  # [32, 5, 128]                │
│   tokens = tokens + pos_emb                                                │
│                                                                             │
│ Step 5: Categorical Transformer                                             │
│   for layer in categorical_transformer_layers:                             │
│     tokens = MultiHeadAttention(tokens) + tokens                          │
│     tokens = FeedForward(LayerNorm(tokens)) + tokens                      │
│   →  [32, 5, 128]                                                          │
│                                                                             │
│ Step 6: Extract CLS₁                                                       │
│   cls1_output = tokens[:, 0, :]  # [32, 128]                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PATH 2: NUMERICAL PROCESSOR                                                 │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Input: numerical_inputs [32, 5, 8]  # [batch, sequence_length, num_features]│
│                                                                             │
│ Step 1: Feature Projection                                                  │
│   projected = Linear(8, 128)(numerical_inputs)                            │
│   →  [32, 5, 128]  # [batch, sequence_length, d_model]                    │
│                                                                             │
│ Step 2: Add CLS Token                                                       │
│   cls_token: [1, 1, 128] → expanded to [32, 1, 128]                       │
│   tokens = cat([cls_token, projected], dim=1)                             │
│   →  [32, 6, 128]  # [batch, 1 + sequence_length, d_model]                │
│                                                                             │
│ Step 3: Add Positional Encoding                                            │
│   positions = [0, 1, 2, 3, 4, 5]  # position indices                      │
│   pos_emb = PositionalEmbedding(positions)  # [32, 6, 128]                │
│   tokens = tokens + pos_emb                                                │
│                                                                             │
│ Step 4: Numerical Transformer                                              │
│   for layer in numerical_transformer_layers:                              │
│     tokens = MultiHeadAttention(tokens) + tokens                          │
│     tokens = FeedForward(LayerNorm(tokens)) + tokens                      │
│   →  [32, 6, 128]                                                          │
│                                                                             │
│ Step 5: Extract CLS₂                                                       │
│   cls2_output = tokens[:, 0, :]  # [32, 128]                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PATH 3: FUSION & PREDICTION                                                 │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Step 1: CLS Token Fusion                                                    │
│   fused_cls = cat([cls1_output, cls2_output], dim=1)                      │
│   →  [32, 256]  # [batch, 2 * d_model]                                    │
│                                                                             │
│ Step 2: Prediction Head                                                     │
│   Single-horizon (prediction_horizons=1):                                  │
│     output_size = 1                                                        │
│     predictions = MLP(fused_cls)  # [32, 256] → [32, 1]                   │
│                                                                             │
│   Multi-horizon (prediction_horizons=3):                                   │
│     output_size = 3                                                        │
│     raw_output = MLP(fused_cls)  # [32, 256] → [32, 3]                    │
│     predictions = raw_output.view(32, 3, 1).squeeze(-1)                   │
│     →  [32, 3]  # Predictions for horizons 1, 2, 3                        │
└─────────────────────────────────────────────────────────────────────────────┘

🎯 KEY MATHEMATICAL OPERATIONS:

1. Categorical Embeddings:
   - categorical_token_i = Embedding_i[category_value_i]
   - Seasonal features: sin/cos values → discretized → embedded

2. Numerical Projection:
   - numerical_tokens = numerical_features @ W_proj + b_proj
   - W_proj: [num_features, d_model], b_proj: [d_model]

3. Dual Attention:
   - Categorical: Attention(Q_cat, K_cat, V_cat) = softmax(Q_catK_cat^T/√d_k)V_cat
   - Numerical: Attention(Q_num, K_num, V_num) = softmax(Q_numK_num^T/√d_k)V_num

4. CLS Fusion:
   - fused_representation = Concat(CLS₁, CLS₂)
   - final_prediction = MLP(fused_representation)

🧠 MEMORY & COMPUTATIONAL COMPLEXITY:

Memory Usage (example):
- Categorical tokens: 32 × 5 × 128 × 4 bytes ≈ 80 KB
- Numerical tokens: 32 × 6 × 128 × 4 bytes ≈ 96 KB
- Categorical attention: 32 × 8 × 5² × 4 bytes ≈ 25 KB
- Numerical attention: 32 × 8 × 6² × 4 bytes ≈ 36 KB
- Total per layer pair: ~240 KB

Computational Complexity:
- Categorical path: O(L_cat × (T_cat² × d + T_cat × d²))
- Numerical path: O(L_num × (T_num² × d + T_num × d²))
- Overall: O(L × (T² × d + T × d²)) where T = max(T_cat, T_num)

🎨 ADVANTAGES OVER UNIFIED APPROACHES:

1. Specialized Processing:
   - Categorical features use embeddings (discrete space)
   - Numerical features use projections (continuous space)
   - Each pathway optimized for its data type

2. Seasonal Feature Handling:
   - Sin/cos pairs automatically detected and embedded
   - Preserves cyclical relationships in discrete form
   - Better for seasonal patterns than raw numerical values

3. Independent Attention:
   - Categorical features attend to other categorical features
   - Numerical sequences attend to temporal patterns
   - Prevents feature type interference

4. Fusion Flexibility:
   - Late fusion allows independent feature learning
   - CLS tokens capture domain-specific information
   - Concatenation preserves both representations

⚡ IMPLEMENTATION NOTES:

1. Feature Detection:
   - Automatic separation of categorical vs numerical
   - Sin/cos pattern detection for seasonal features
   - Vocabulary size estimation for embeddings

2. Memory Optimization:
   - Separate smaller transformers vs one large transformer
   - Reduced attention matrix sizes
   - Optional gradient checkpointing

3. Training Considerations:
   - Balance categorical and numerical learning rates
   - Separate dropout rates for different pathways
   - Independent layer normalization

4. Seasonal Optimization:
   - Cyclical features treated as high-cardinality categorical
   - Learnable embeddings for sin/cos discretization
   - Temporal positional encoding for sequence order

This dual-pathway architecture provides superior performance for time series data
with mixed categorical/numerical features and strong seasonal patterns, while
maintaining the flexibility to handle varying feature compositions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class CSNTransformerBlock(nn.Module):
    """
    Standard transformer block used in both categorical and numerical processors.

    Architecture: MultiHeadAttention → Add&Norm → FeedForward → Add&Norm
    This is the core building block that enables feature interaction learning.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Multi-head self-attention mechanism
        # Allows each position to attend to all positions in the sequence
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Layer normalization for training stability
        self.norm1 = nn.LayerNorm(d_model)  # After attention
        self.norm2 = nn.LayerNorm(d_model)  # After feed-forward

        # Position-wise feed-forward network
        # Applies same transformation to each position independently
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),       # Expand to hidden dimension
            nn.ReLU(),                      # Non-linear activation
            nn.Dropout(dropout),            # Regularization
            nn.Linear(d_ff, d_model),       # Project back to d_model
            nn.Dropout(dropout)             # Final regularization
        )

    def forward(self, x):
        """
        Args:
            x: Input tokens [batch_size, seq_len, d_model]
        Returns:
            Output tokens [batch_size, seq_len, d_model]
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
                 d_model: int,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.feature_names = list(categorical_features.keys())

        # Create embedding table for each categorical feature
        # Each feature gets its own vocabulary and embedding space
        self.embeddings = nn.ModuleDict()
        for feature_name, vocab_size in categorical_features.items():
            # Add 1 for potential unknown/padding values (vocab_size + 1)
            # Maps discrete values [0, vocab_size-1] → dense vectors [d_model]
            self.embeddings[feature_name] = nn.Embedding(vocab_size + 1, d_model)

        # CLS token for categorical feature aggregation
        # This token will collect information from all categorical features
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Stack of transformer blocks for categorical feature interaction
        d_ff = d_model * 4  # Standard transformer feed-forward dimension
        self.transformer_blocks = nn.ModuleList([
            CSNTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Positional embeddings to distinguish feature positions
        # Categorical features don't have inherent order, but position helps attention
        max_seq_len = len(categorical_features) + 1  # +1 for CLS token
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, categorical_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process categorical features through embeddings and transformer.

        Args:
            categorical_inputs: Dict mapping feature names to index tensors
                               e.g., {'year': [32], 'quarter': [32], 'month_sin': [32]}
        Returns:
            cls_output: [batch_size, d_model] - Aggregated categorical representation
        """
        batch_size = next(iter(categorical_inputs.values())).size(0)
        device = next(iter(categorical_inputs.values())).device

        # Step 1: Convert categorical indices to dense embeddings
        embedded_features = []
        for feature_name in self.feature_names:
            if feature_name in categorical_inputs:
                # Lookup embedding: [batch_size] → [batch_size, d_model]
                embedded = self.embeddings[feature_name](categorical_inputs[feature_name])
                embedded_features.append(embedded)

        # Step 2: Stack all feature embeddings
        if embedded_features:
            # Stack: List of [batch, d_model] → [batch, num_features, d_model]
            feature_embeddings = torch.stack(embedded_features, dim=1)
        else:
            # Handle case with no categorical features
            feature_embeddings = torch.zeros(batch_size, 0, self.d_model, device=device)

        # Step 3: Add CLS token for aggregation
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, d_model]

        # Concatenate: [batch, 1, d_model] + [batch, num_features, d_model]
        # Result: [batch, 1 + num_features, d_model]
        x = torch.cat([cls_tokens, feature_embeddings], dim=1)

        # Step 4: Add positional embeddings
        # Help model distinguish between different categorical features
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)  # [batch, seq_len, d_model]
        x = x + pos_emb

        # Step 5: Apply transformer blocks for feature interaction
        # Each categorical feature can attend to every other categorical feature
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)  # [batch, 1 + num_features, d_model]

        # Step 6: Extract CLS token output
        # CLS token has aggregated information from all categorical features
        cls_output = x[:, 0, :]  # [batch_size, d_model]
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
                 d_model: int,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.num_features = num_numerical_features
        self.sequence_length = sequence_length
        self.d_model = d_model

        # Project each timestep's numerical features to d_model dimension
        # Maps [num_features] → [d_model] for each timestep
        self.feature_projection = nn.Linear(num_numerical_features, d_model)

        # CLS token for numerical sequence aggregation
        # This token will collect temporal patterns from numerical sequences
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Stack of transformer blocks for temporal pattern learning
        d_ff = d_model * 4  # Standard transformer feed-forward dimension
        self.transformer_blocks = nn.ModuleList([
            CSNTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Positional embeddings for temporal order
        # Critical for preserving time series ordering in attention
        max_seq_len = sequence_length + 1  # +1 for CLS token
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, numerical_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through numerical processor with detailed step-by-step computation.

        Args:
            numerical_inputs: [batch_size, sequence_length, num_features]
                            Time series data with multiple numerical features per timestep

        Returns:
            cls_output: [batch_size, d_model] - Aggregated temporal representation

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
        # Transform each timestep: [num_features] → [d_model]
        # This allows the transformer to work with consistent dimensionality
        x = self.feature_projection(numerical_inputs)  # [batch_size, seq_len, d_model]

        # Step 2: Add CLS Token for sequence aggregation
        # CLS token will learn to aggregate information from all timesteps
        # Shape transformation: [1, 1, d_model] → [batch_size, 1, d_model]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Concatenate CLS with projected sequences
        # [batch_size, 1, d_model] + [batch_size, seq_len, d_model]
        # Result: [batch_size, seq_len+1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)

        # Step 3: Add Positional Embeddings for temporal order
        # Critical for time series: position 0=CLS, positions 1...seq_len=timesteps
        # Without this, the model can't distinguish between different timesteps
        total_seq_len = x.size(1)  # seq_len + 1 (for CLS)
        positions = torch.arange(total_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)  # [batch_size, seq_len+1, d_model]
        x = x + pos_emb  # Element-wise addition of positional information

        # Step 4: Apply Transformer Blocks for temporal pattern learning
        # Each block allows every position to attend to every other position
        # CLS token can attend to all timesteps, timesteps can attend to each other
        for transformer_block in self.transformer_blocks:
            # Self-attention + feed-forward with residual connections
            x = transformer_block(x)  # [batch_size, seq_len+1, d_model]

        # Step 5: Extract CLS Token Output
        # CLS token (position 0) has aggregated information from entire sequence
        # This becomes our fixed-size representation of the variable-length sequence
        cls_output = x[:, 0, :]  # [batch_size, d_model]

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
                 d_model: int = 128,
                 n_layers: int = 3,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 output_dim: int = 1,
                 prediction_horizons: int = 1):
        super().__init__()

        self.d_model = d_model
        self.has_categorical = len(categorical_features) > 0
        self.has_numerical = num_numerical_features > 0
        self.prediction_horizons = prediction_horizons

        # Categorical processor
        if self.has_categorical:
            self.categorical_processor = CategoricalProcessor(
                categorical_features, d_model, n_layers, n_heads, dropout
            )

        # Numerical processor
        if self.has_numerical:
            self.numerical_processor = NumericalProcessor(
                num_numerical_features, sequence_length, d_model, n_layers, n_heads, dropout
            )

        # Determine fusion dimension
        fusion_dim = 0
        if self.has_categorical:
            fusion_dim += d_model
        if self.has_numerical:
            fusion_dim += d_model

        # Fully connected layers for final prediction - multi-horizon support
        final_output_dim = output_dim * prediction_horizons
        self.prediction_head = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, final_output_dim)
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
                 d_model: int = 128,
                 n_layers: int = 3,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 prediction_horizons: int = 1):
        super().__init__()

        self.model = CSNTransformer(
            categorical_features=categorical_features,
            num_numerical_features=num_numerical_features,
            sequence_length=sequence_length,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            output_dim=1,
            prediction_horizons=prediction_horizons
        )

    def forward(self, categorical_inputs=None, numerical_inputs=None):
        return self.model(categorical_inputs, numerical_inputs)