"""
Core FT-Transformer implementation.

🧠 FT-TRANSFORMER (FEATURE TOKENIZER TRANSFORMER) ARCHITECTURE
============================================================

The FT-Transformer is a state-of-the-art architecture specifically designed for tabular data.
It converts heterogeneous features (numerical and categorical) into a unified token representation,
then applies self-attention mechanisms to capture complex feature interactions.

📊 ARCHITECTURE OVERVIEW:
┌─────────────────────────────────────────────────────────────┐
│  Raw Features → FeatureTokenizer → Transformer → Prediction │
│  [num, cat]   → [tokens]        → [attention] → [output]    │
└─────────────────────────────────────────────────────────────┘

🔄 DATA FLOW WITH MATRIX DIMENSIONS:

Example Configuration:
- batch_size = 32
- num_numerical = 10 (price, volume, etc.)
- num_categorical = 4 (year, quarter, etc.)
- cat_cardinalities = [5, 4, 12, 7] (unique values per categorical feature)
- d_token = 192 (embedding dimension)

Step 1: Feature Tokenization
┌─────────────────────────────────────────────────────────────┐
│ Input:                                                      │
│   x_num: [32, 10]     # Numerical features                 │
│   x_cat: [32, 4]      # Categorical features               │
│                                                             │
│ Numerical Tokenization: f(x) = x * W + b                   │
│   num_weights: [10, 192]                                   │
│   num_biases:  [10, 192]                                   │
│   →  num_tokens: [32, 10, 192]                             │
│                                                             │
│ Categorical Tokenization: Embedding Lookup                 │
│   cat_embeddings[0]: Embedding(5, 192)   # year           │
│   cat_embeddings[1]: Embedding(4, 192)   # quarter        │
│   cat_embeddings[2]: Embedding(12, 192)  # month          │
│   cat_embeddings[3]: Embedding(7, 192)   # weekday        │
│   →  cat_tokens: [32, 4, 192]                              │
│                                                             │
│ Token Combination:                                          │
│   feature_tokens = cat([num_tokens, cat_tokens], dim=1)    │
│   →  [32, 14, 192]  # Combined features                    │
│                                                             │
│ Add CLS Token:                                              │
│   cls_token: [1, 1, 192] → expanded to [32, 1, 192]       │
│   final_tokens = cat([cls_token, feature_tokens], dim=1)   │
│   →  [32, 15, 192]  # Final tokenized input                │
└─────────────────────────────────────────────────────────────┘

Step 2: Transformer Processing
┌─────────────────────────────────────────────────────────────┐
│ Multi-Head Self-Attention:                                 │
│   n_heads = 8, d_head = 192 // 8 = 24                     │
│                                                             │
│   Q = tokens @ W_q  # [32, 15, 192] @ [192, 192]          │
│   K = tokens @ W_k  # [32, 15, 192] @ [192, 192]          │
│   V = tokens @ W_v  # [32, 15, 192] @ [192, 192]          │
│   →  Q, K, V: [32, 15, 192]                               │
│                                                             │
│   Reshape for multi-head:                                  │
│   Q = Q.view(32, 15, 8, 24).transpose(1, 2)               │
│   →  [32, 8, 15, 24]  # [batch, heads, seq, d_head]       │
│                                                             │
│   Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V       │
│   scores = Q @ K.transpose(-2, -1)  # [32, 8, 15, 15]     │
│   attn_weights = softmax(scores / √24)                     │
│   attn_output = attn_weights @ V    # [32, 8, 15, 24]     │
│                                                             │
│   Concatenate heads: [32, 8, 15, 24] → [32, 15, 192]      │
│                                                             │
│ Feed-Forward Network:                                       │
│   d_ffn = 4 * 192 = 768                                   │
│   ff_output = LayerNorm(tokens + attn_output)             │
│   ff_intermediate = Linear(ff_output, 768) + GELU()       │
│   ff_final = Linear(ff_intermediate, 192)                 │
│   output = LayerNorm(ff_output + ff_final)                │
│   →  [32, 15, 192]                                         │
│                                                             │
│ Repeat for n_layers (typically 3-6 times)                 │
└─────────────────────────────────────────────────────────────┘

Step 3: Prediction
┌─────────────────────────────────────────────────────────────┐
│ CLS Token Extraction:                                       │
│   cls_output = transformer_output[:, 0, :]                 │
│   →  [32, 192]  # CLS token representation                 │
│                                                             │
│ Prediction Head:                                            │
│   Single-horizon (prediction_horizons=1):                  │
│     output_size = 1                                        │
│     predictions = LinearHead(cls_output)                   │
│     →  [32, 1]                                             │
│                                                             │
│   Multi-horizon (prediction_horizons=3):                   │
│     output_size = 3                                        │
│     raw_output = LinearHead(cls_output)                    │
│     →  [32, 3]                                             │
│     predictions = raw_output.view(32, 3, 1).squeeze(-1)   │
│     →  [32, 3]  # Predictions for horizons 1, 2, 3        │
└─────────────────────────────────────────────────────────────┘

🚀 TEMPORAL FT-TRANSFORMER (SequenceFTTransformer):

For time series data with sequences:
┌─────────────────────────────────────────────────────────────┐
│ Input: x_seq [32, 5, 14]  # [batch, sequence_len, features]│
│                                                             │
│ Process each timestep independently:                       │
│   for t in range(5):                                       │
│     x_num_t = x_seq[:, t, :10]   # [32, 10] at timestep t │
│     x_cat_t = x_seq[:, t, 10:]   # [32, 4] at timestep t  │
│     tokens_t = tokenize(x_num_t, x_cat_t)  # [32, 14, 192] │
│     tokens_t += temporal_pos_embedding[t]  # Add time info │
│                                                             │
│ Concatenate all timesteps:                                 │
│   sequence_tokens = cat(all_tokens, dim=1)                │
│   →  [32, 70, 192]  # 5 timesteps × 14 tokens            │
│                                                             │
│ Add CLS token: [32, 71, 192]                              │
│ Process through transformer: same as above                 │
│ Extract CLS: [32, 192] → predictions                      │
└─────────────────────────────────────────────────────────────┘

🎯 KEY MATHEMATICAL OPERATIONS:

1. Feature Tokenization:
   - Numerical: token_ij = x_i * W_ij + b_ij
   - Categorical: token_i = Embedding[category_value_i]

2. Multi-Head Attention:
   - Attention(Q,K,V) = softmax(QK^T / √d_k)V
   - MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O

3. Position Encoding (Temporal):
   - Learnable embeddings: pos_emb_t ∈ ℝ^{d_token}

🧠 MEMORY & COMPUTATIONAL COMPLEXITY:

Memory Usage (example):
- Tokens: 32 × 15 × 192 × 4 bytes ≈ 360 KB
- Attention: 32 × 8 × 15² × 4 bytes ≈ 225 KB
- Total per layer: ~600 KB

Computational Complexity:
- Tokenization: O(num_features × d_token)
- Self-Attention: O(num_tokens² × d_token)
- Feed-Forward: O(num_tokens × d_token × d_ffn)
- Overall: O(L × (T² × d + T × d²)) where L=layers, T=tokens, d=d_token

🎨 ADVANTAGES:

1. Unified Feature Representation:
   - Converts all features to same dimensional space
   - Enables direct comparison between different feature types

2. Automatic Feature Interaction Discovery:
   - Self-attention learns complex, non-linear interactions
   - No manual feature engineering needed

3. Scalability:
   - Handles varying numbers of features gracefully
   - Attention mechanism scales to large feature sets

4. Temporal Modeling:
   - SequenceFTTransformer naturally handles time series
   - Positional encodings preserve temporal relationships

⚡ IMPLEMENTATION NOTES:

1. Memory Optimization:
   - Use gradient checkpointing for deep models
   - Mixed precision training with autocast()

2. Batch Size Considerations:
   - Attention memory scales quadratically with sequence length
   - Reduce batch size for long sequences

3. Feature Scaling:
   - Numerical features should be normalized
   - Categorical features use embeddings (no scaling needed)

This implementation provides a robust, scalable foundation for tabular time series
forecasting with automatic feature interaction discovery and temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import numpy as np


class FeatureTokenizer(nn.Module):
    """Converts tabular features into transformer tokens."""
    
    def __init__(self, num_numerical: int, cat_cardinalities: List[int], d_token: int):
        super().__init__()
        
        # Validation
        if num_numerical + len(cat_cardinalities) == 0:
            raise ValueError("Must have at least one feature")
        
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token
        
        # CLS token for prediction - learnable parameter that aggregates all feature information
        # Shape: [1, 1, d_token] → will be expanded to [batch_size, 1, d_token]
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        # Numerical feature tokenization: f(x) = x * W + b
        # Each numerical feature gets its own weight vector and bias vector
        # This allows each feature to be mapped to its own subspace in d_token dimensions
        if num_numerical > 0:
            self.num_weights = nn.Parameter(torch.randn(num_numerical, d_token))  # [num_features, d_token]
            self.num_biases = nn.Parameter(torch.randn(num_numerical, d_token))   # [num_features, d_token]

        # Categorical feature embeddings - standard embedding lookup tables
        # Each categorical feature gets its own embedding table: vocab_size → d_token
        # Example: year feature with 5 unique values gets Embedding(5, d_token)
        if cat_cardinalities:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(cardinality, d_token)
                for cardinality in cat_cardinalities
            ])
    
    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Convert raw features into tokens for transformer processing.

        Args:
            x_num: Numerical features [batch_size, num_numerical]
            x_cat: Categorical features [batch_size, num_categorical]

        Returns:
            tokens: [batch_size, 1 + num_features, d_token]
                   First token is CLS, followed by feature tokens
        """
        batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
        tokens = []

        # Process numerical features using learnable linear transformation
        if self.num_numerical > 0 and x_num is not None:
            # Transform: f(x) = x * W + b for each feature independently
            # x_num: [batch, num_features] → [batch, num_features, 1]
            # weights: [num_features, d_token] → [1, num_features, d_token]
            # Result: [batch, num_features, d_token]
            num_tokens = (x_num.unsqueeze(-1) * self.num_weights.unsqueeze(0) +
                         self.num_biases.unsqueeze(0))
            tokens.append(num_tokens)

        # Process categorical features using embedding lookup
        if self.cat_cardinalities and x_cat is not None:
            # For each categorical feature, lookup its embedding
            # x_cat[:, i]: [batch] → cat_embeddings[i]: [batch, d_token]
            # Stack all: [batch, num_categorical, d_token]
            cat_tokens = torch.stack([
                self.cat_embeddings[i](x_cat[:, i])
                for i in range(len(self.cat_cardinalities))
            ], dim=1)
            tokens.append(cat_tokens)

        # Combine all feature tokens along feature dimension
        if not tokens:
            raise ValueError("No valid input features provided")

        feature_tokens = torch.cat(tokens, dim=1)  # [batch, total_features, d_token]

        # Prepend CLS token for final prediction aggregation
        # CLS token will collect information from all features via self-attention
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, d_token]
        return torch.cat([cls_tokens, feature_tokens], dim=1)    # [batch, 1+features, d_token]


class SequenceFeatureTokenizer(nn.Module):
    """Tokenizes sequences of tabular features for temporal FT-Transformer."""
    
    def __init__(self, num_numerical: int, cat_cardinalities: List[int], d_token: int, sequence_length: int):
        super().__init__()
        
        # Validation
        if num_numerical + len(cat_cardinalities) == 0:
            raise ValueError("Must have at least one feature")
        
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token
        self.sequence_length = sequence_length
        
        # CLS token for prediction
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # Numerical feature tokenization: f(x) = x * W + b (same for all timesteps)
        if num_numerical > 0:
            self.num_weights = nn.Parameter(torch.randn(num_numerical, d_token))
            self.num_biases = nn.Parameter(torch.randn(num_numerical, d_token))
        
        # Categorical feature embeddings (same for all timesteps)
        if cat_cardinalities:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(cardinality, d_token) 
                for cardinality in cat_cardinalities
            ])
        
        # Temporal positional encoding - learnable position embeddings for each timestep
        self.temporal_pos_embedding = nn.Parameter(torch.randn(sequence_length, d_token))
    
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: Sequential features [batch_size, sequence_length, num_features]
            
        Returns:
            tokens: [batch_size, 1 + sequence_length * tokens_per_timestep, d_token]
        """
        batch_size, seq_len, total_features = x_seq.shape
        
        # Split into numerical and categorical features
        if self.num_numerical > 0 and len(self.cat_cardinalities) > 0:
            x_num_seq = x_seq[:, :, :self.num_numerical]  # [batch, seq_len, num_numerical]  
            x_cat_seq = x_seq[:, :, self.num_numerical:]  # [batch, seq_len, num_categorical]
        elif self.num_numerical > 0:
            x_num_seq = x_seq
            x_cat_seq = None
        else:
            x_num_seq = None
            x_cat_seq = x_seq
            
        all_tokens = []
        
        # Process each timestep
        for t in range(seq_len):
            timestep_tokens = []
            
            # Process numerical features for this timestep
            if self.num_numerical > 0 and x_num_seq is not None:
                x_num_t = x_num_seq[:, t, :]  # [batch, num_numerical]
                # Apply tokenization: value * weight + bias
                num_tokens = (x_num_t.unsqueeze(-1) * self.num_weights.unsqueeze(0) + 
                             self.num_biases.unsqueeze(0))  # [batch, num_numerical, d_token]
                timestep_tokens.append(num_tokens)
            
            # Process categorical features for this timestep  
            if self.cat_cardinalities and x_cat_seq is not None:
                x_cat_t = x_cat_seq[:, t, :].long()  # [batch, num_categorical]
                cat_tokens = torch.stack([
                    self.cat_embeddings[i](x_cat_t[:, i]) 
                    for i in range(len(self.cat_cardinalities))
                ], dim=1)  # [batch, num_categorical, d_token]
                timestep_tokens.append(cat_tokens)
            
            # Combine tokens for this timestep
            if timestep_tokens:
                timestep_combined = torch.cat(timestep_tokens, dim=1)  # [batch, tokens_per_timestep, d_token]
                
                # Add temporal positional encoding to all tokens of this timestep
                timestep_combined = timestep_combined + self.temporal_pos_embedding[t].unsqueeze(0).unsqueeze(0)
                
                all_tokens.append(timestep_combined)
        
        # Concatenate all timesteps: [batch, seq_len * tokens_per_timestep, d_token]
        if all_tokens:
            sequence_tokens = torch.cat(all_tokens, dim=1)
        else:
            # Fallback (shouldn't happen with validation)
            device = self.cls_token.device
            sequence_tokens = torch.empty(batch_size, 0, self.d_token, device=device)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_tokens, sequence_tokens], dim=1)


class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer for tabular data."""
    
    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: List[int],
        d_token: int = 192,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ffn: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        if d_ffn is None:
            d_ffn = 4 * d_token
            
        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(num_numerical, cat_cardinalities, d_token)
        
        # Transformer encoder - stack of identical layers with self-attention + FFN
        # Each layer: MultiHeadAttention → Add&Norm → FeedForward → Add&Norm
        # norm_first=True uses pre-normalization for better gradient flow
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,          # Token dimension (input/output size)
            nhead=n_heads,            # Number of attention heads
            dim_feedforward=d_ffn,    # Hidden size in feed-forward network
            dropout=dropout,          # Dropout rate
            activation=activation,    # Activation function (GELU/ReLU)
            batch_first=True,         # Input format: [batch, seq, features]
            norm_first=True           # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output dimension for downstream prediction head
        self.d_token = d_token
    
    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x_num: Numerical features [batch_size, num_numerical]
            x_cat: Categorical features [batch_size, num_categorical]
            
        Returns:
            cls_output: [batch_size, d_token] - CLS token representation
        """
        # Step 1: Convert features to tokens
        # tokens: [batch_size, 1 + num_features, d_token]
        tokens = self.tokenizer(x_num, x_cat)

        # Step 2: Apply self-attention across all tokens
        # Each token can attend to every other token, learning feature interactions
        # transformer_out: [batch_size, 1 + num_features, d_token]
        transformer_out = self.transformer(tokens)

        # Step 3: Extract CLS token for prediction
        # CLS token (first token) has aggregated information from all features
        return transformer_out[:, 0, :]  # [batch_size, d_token]


class SequenceFTTransformer(nn.Module):
    """Feature Tokenizer Transformer for sequential tabular data."""
    
    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: List[int],
        sequence_length: int,
        d_token: int = 192,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ffn: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        if d_ffn is None:
            d_ffn = 4 * d_token
            
        # Sequence feature tokenizer
        self.tokenizer = SequenceFeatureTokenizer(num_numerical, cat_cardinalities, d_token, sequence_length)
        
        # Transformer encoder - same as before, but will process more tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head will be added by wrapper based on task
        self.d_token = d_token
    
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: Sequential features [batch_size, sequence_length, num_features]
            
        Returns:
            cls_output: [batch_size, d_token] - CLS token representation
        """
        # Tokenize sequence features  
        tokens = self.tokenizer(x_seq)
        
        # Pass through transformer
        transformer_out = self.transformer(tokens)
        
        # Return CLS token output
        return transformer_out[:, 0, :]  # [batch_size, d_token]


class FTTransformerPredictor(nn.Module):
    """Complete FT-Transformer with prediction head."""

    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: List[int],
        n_classes: int = 1,
        prediction_horizons: int = 1,
        **transformer_kwargs
    ):
        super().__init__()

        self.ft_transformer = FTTransformer(num_numerical, cat_cardinalities, **transformer_kwargs)
        self.prediction_horizons = prediction_horizons

        # Prediction head - maps CLS token representation to final predictions
        d_token = self.ft_transformer.d_token
        output_size = n_classes * prediction_horizons  # Multi-horizon: predict multiple time steps

        # Simple MLP head: Normalize → Activate → Regularize → Project
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),                                    # Stabilize input
            nn.ReLU(),                                               # Non-linearity
            nn.Dropout(transformer_kwargs.get('dropout', 0.1)),     # Regularization
            nn.Linear(d_token, output_size)                         # Final projection
        )

        self.n_classes = n_classes
    
    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x_num: Numerical features
            x_cat: Categorical features

        Returns:
            predictions: [batch_size, n_classes * prediction_horizons]
                       For regression: [batch_size, prediction_horizons]
        """
        # Get CLS token representation from transformer
        cls_output = self.ft_transformer(x_num, x_cat)  # [batch, d_token]

        # Apply prediction head
        output = self.head(cls_output)  # [batch, output_size]

        # Reshape for multi-horizon predictions
        if self.prediction_horizons > 1:
            # Convert [batch, horizons * classes] → [batch, horizons, classes] → [batch, horizons]
            batch_size = output.size(0)
            return output.view(batch_size, self.prediction_horizons, self.n_classes).squeeze(-1)
        return output  # [batch, n_classes] for single horizon


class SequenceFTTransformerPredictor(nn.Module):
    """Complete Sequential FT-Transformer with prediction head."""

    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: List[int],
        sequence_length: int,
        n_classes: int = 1,
        prediction_horizons: int = 1,
        **transformer_kwargs
    ):
        super().__init__()

        self.sequence_ft_transformer = SequenceFTTransformer(
            num_numerical, cat_cardinalities, sequence_length, **transformer_kwargs
        )
        self.prediction_horizons = prediction_horizons

        # Prediction head - conditional based on horizons
        d_token = self.sequence_ft_transformer.d_token
        output_size = n_classes * prediction_horizons  # Multi-horizon output

        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Dropout(transformer_kwargs.get('dropout', 0.1)),
            nn.Linear(d_token, output_size)
        )

        self.n_classes = n_classes
    
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: Sequential features [batch_size, sequence_length, num_features]

        Returns:
            predictions: [batch_size, n_classes * prediction_horizons]
                       For regression: [batch_size, prediction_horizons]
        """
        cls_output = self.sequence_ft_transformer(x_seq)
        output = self.head(cls_output)

        # Reshape to separate horizons if multi-horizon
        if self.prediction_horizons > 1:
            batch_size = output.size(0)
            return output.view(batch_size, self.prediction_horizons, self.n_classes).squeeze(-1)
        return output