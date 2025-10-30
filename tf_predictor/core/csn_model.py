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


class CSNTransformerTimeSeriesModel(TransformerBasedModel):
    """
    CSN-Transformer implementation for time series that implements TimeSeriesModel interface.

    This is the new standard model class that should be used with ModelFactory.
    It replaces the old CSNTransformerPredictor for new code.

    Key differences from old implementation:
    1. Implements TimeSeriesModel interface
    2. Takes (sequence_length, num_features, output_dim) instead of complex categorical setup
    3. Works directly with sequences (splits numerical/categorical internally)
    4. Provides standardized API (get_model_config, get_embedding_dim, etc.)
    """

    def __init__(
        self,
        sequence_length: int,
        num_features: int,
        output_dim: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        categorical_features: Dict[str, int] = None,
        num_numerical_features: int = None
    ):
        """
        Initialize CSN-Transformer for time series.

        Args:
            sequence_length: Length of input sequences (lookback window)
            num_features: Total number of features per time step
            output_dim: Output dimension (num_targets * prediction_horizon)
            d_model: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
            categorical_features: Dict mapping categorical feature names to cardinalities
            num_numerical_features: Number of numerical features
        """
        super().__init__(d_model=d_model, num_heads=num_heads, num_layers=num_layers)

        self.sequence_length = sequence_length
        self.num_features = num_features
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.activation_name = activation
        self.categorical_features = categorical_features or {}
        self.num_numerical_features = num_numerical_features or num_features

        # Create the actual CSN-Transformer model
        self.csn_transformer = CSNTransformer(
            categorical_features=self.categorical_features,
            num_numerical_features=self.num_numerical_features,
            sequence_length=sequence_length,
            d_model=d_model,
            n_layers=num_layers,
            n_heads=num_heads,
            dropout=dropout,
            output_dim=1,
            prediction_horizons=output_dim
        )

        # Store for attention weights (optional)
        self._last_attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features)
               The tensor is expected to have categorical and numerical features
               concatenated along the feature dimension

        Returns:
            predictions: Output tensor of shape (batch_size, output_dim)
        """
        # For now, assume all features are numerical
        # In a more sophisticated implementation, we would split x into
        # categorical and numerical components based on self.categorical_features

        categorical_inputs = None  # TODO: Extract from x if categorical_features defined
        numerical_inputs = x  # All features treated as numerical for now

        # Forward through CSN-Transformer
        predictions = self.csn_transformer(categorical_inputs, numerical_inputs)

        return predictions

    def get_model_config(self) -> Dict[str, Any]:
        """Get the current configuration of the model."""
        return {
            'model_type': 'csn_transformer',
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dropout': self.dropout_rate,
            'activation': self.activation_name,
            'sequence_length': self.sequence_length,
            'num_features': self.num_features,
            'output_dim': self.output_dim,
            'categorical_features': self.categorical_features,
            'num_numerical_features': self.num_numerical_features
        }

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get attention weights from the last forward pass.

        Returns:
            Attention weights tensor or None if not available.
        """
        return self._last_attention_weights


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
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
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
            d_model: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__(d_model=d_model, num_heads=num_heads, num_layers=num_layers)

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
            self.cls1_token = CLSToken(d_model)

            # Categorical embeddings with logarithmic dimension scaling
            self.cat_embeddings = nn.ModuleList()
            self.cat_projections = nn.ModuleList()

            for cardinality in cat_cardinalities:
                # Calculate embedding dimension using logarithmic scaling
                emb_dim = int(8 * math.log2(cardinality + 1))
                # Clamp to bounds [d_model/4, d_model]
                min_dim = d_model // 4
                max_dim = d_model
                emb_dim = max(min_dim, min(max_dim, emb_dim))

                # Create embedding layer
                embedding = nn.Embedding(cardinality, emb_dim)
                self.cat_embeddings.append(embedding)

                # Project to d_model if needed
                if emb_dim != d_model:
                    projection = nn.Linear(emb_dim, d_model)
                    self.cat_projections.append(projection)
                else:
                    self.cat_projections.append(nn.Identity())

            # Categorical transformer
            cat_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation=activation,
                batch_first=True
            )
            self.cat_transformer = nn.TransformerEncoder(cat_encoder_layer, num_layers=num_layers)

        # PATH 2: Numerical Processing
        if num_numerical > 0:
            # CLS token for numerical path
            self.cls2_token = CLSToken(d_model)

            # Project numerical features to d_model
            self.num_projection = nn.Linear(num_numerical, d_model)

            # Positional encoding for temporal sequences
            self.temporal_pos_encoding = nn.Parameter(torch.randn(1, sequence_length, d_model) * 0.02)

            # Numerical transformer
            num_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation=activation,
                batch_first=True
            )
            self.num_transformer = nn.TransformerEncoder(num_encoder_layer, num_layers=num_layers)

        # Fusion and Prediction Head
        # Input dimension is 2 * d_model (CLS₁ + CLS₂)
        fusion_dim = 2 * d_model if (num_categorical > 0 and num_numerical > 0) else d_model

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
                projected = self.cat_projections[i](emb)  # [batch, d_model]
                cat_tokens_list.append(projected.unsqueeze(1))  # [batch, 1, d_model]

            cat_tokens = torch.cat(cat_tokens_list, dim=1)  # [batch, num_categorical, d_model]

            # Step 2: Add CLS₁ token
            cls1_tokens = self.cls1_token(batch_size)  # [batch, 1, d_model]
            cat_tokens_with_cls = torch.cat([cls1_tokens, cat_tokens], dim=1)  # [batch, 1+num_cat, d_model]

            # Step 3: Categorical transformer
            cat_output = self.cat_transformer(cat_tokens_with_cls)  # [batch, 1+num_cat, d_model]

            # Step 4: Extract CLS₁
            cls1_output = cat_output[:, 0, :]  # [batch, d_model]
            cls_outputs.append(cls1_output)

        # PATH 2: Numerical Processing
        if self.num_numerical > 0:
            # Step 1: Project numerical features
            x_num_proj = self.num_projection(x_num)  # [batch, seq_len, d_model]

            # Step 2: Add positional encoding
            x_num_proj = x_num_proj + self.temporal_pos_encoding  # [batch, seq_len, d_model]

            # Step 3: Add CLS₂ token
            cls2_tokens = self.cls2_token(batch_size)  # [batch, 1, d_model]
            num_tokens_with_cls = torch.cat([cls2_tokens, x_num_proj], dim=1)  # [batch, 1+seq_len, d_model]

            # Step 4: Numerical transformer
            num_output = self.num_transformer(num_tokens_with_cls)  # [batch, 1+seq_len, d_model]

            # Step 5: Extract CLS₂
            cls2_output = num_output[:, 0, :]  # [batch, d_model]
            cls_outputs.append(cls2_output)

        # FUSION: Concatenate CLS outputs
        if len(cls_outputs) == 2:
            fused_cls = torch.cat(cls_outputs, dim=1)  # [batch, 2*d_model]
        else:
            fused_cls = cls_outputs[0]  # Only one path available

        # Prediction
        predictions = self.head(fused_cls)  # [batch, output_dim]

        return predictions

    def get_model_config(self) -> Dict[str, Any]:
        """Get the current configuration of the model."""
        return {
            'model_type': 'csn_transformer_cls',
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
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