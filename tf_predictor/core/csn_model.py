"""
CSNTransformer: Categorical-Seasonal-Numerical Transformer for Time Series Prediction.

Architecture:
- Separates features into categorical and numerical
- Seasonal features (sin/cos pairs) are treated as categorical
- Two separate transformer blocks for each feature type
- CLS token fusion for final prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class CSNTransformerBlock(nn.Module):
    """Standard transformer block for CSNTransformer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


class CategoricalProcessor(nn.Module):
    """Processes categorical features with embeddings and transformer."""

    def __init__(self,
                 categorical_features: Dict[str, int],  # feature_name: num_unique_values
                 d_model: int,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.feature_names = list(categorical_features.keys())

        # Create embeddings for each categorical feature
        self.embeddings = nn.ModuleDict()
        for feature_name, vocab_size in categorical_features.items():
            # Add 1 for potential unknown/padding values
            self.embeddings[feature_name] = nn.Embedding(vocab_size + 1, d_model)

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer blocks
        d_ff = d_model * 4
        self.transformer_blocks = nn.ModuleList([
            CSNTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Position embeddings (if needed)
        max_seq_len = len(categorical_features) + 1  # +1 for CLS token
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, categorical_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = next(iter(categorical_inputs.values())).size(0)
        device = next(iter(categorical_inputs.values())).device

        # Create embeddings for each categorical feature
        embedded_features = []
        for feature_name in self.feature_names:
            if feature_name in categorical_inputs:
                embedded = self.embeddings[feature_name](categorical_inputs[feature_name])
                embedded_features.append(embedded)

        # Stack embeddings: [batch_size, num_features, d_model]
        if embedded_features:
            feature_embeddings = torch.stack(embedded_features, dim=1)
        else:
            # If no categorical features, create zero tensor
            feature_embeddings = torch.zeros(batch_size, 0, self.d_model, device=device)

        # Add CLS token: [batch_size, 1, d_model]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Concatenate CLS token with feature embeddings
        x = torch.cat([cls_tokens, feature_embeddings], dim=1)

        # Add positional embeddings
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb

        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Return CLS token: [batch_size, d_model]
        cls_output = x[:, 0, :]
        return cls_output


class NumericalProcessor(nn.Module):
    """Processes numerical features with sequence modeling and transformer."""

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

        # Project numerical features to d_model
        self.feature_projection = nn.Linear(num_numerical_features, d_model)

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer blocks
        d_ff = d_model * 4
        self.transformer_blocks = nn.ModuleList([
            CSNTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Position embeddings
        max_seq_len = sequence_length + 1  # +1 for CLS token
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Linear projection from sequence_length to d_model
        self.cls_projection = nn.Linear(sequence_length, d_model)

    def forward(self, numerical_inputs: torch.Tensor) -> torch.Tensor:
        # numerical_inputs: [batch_size, sequence_length, num_features]
        batch_size, seq_len, num_features = numerical_inputs.shape
        device = numerical_inputs.device

        # Project each timestep's features to d_model
        x = self.feature_projection(numerical_inputs)  # [batch_size, seq_len, d_model]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, seq_len+1, d_model]

        # Add positional embeddings
        total_seq_len = x.size(1)
        positions = torch.arange(total_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb

        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Get CLS token and project to d_model
        cls_raw = x[:, 0, :]  # [batch_size, d_model]

        # Note: The CLS token is already d_model dimensional, so no projection needed
        # The original requirement mentioned projecting from sequence_length to d_model,
        # but that would apply if we were using the sequence directly, not the CLS token
        return cls_raw


class CSNTransformer(nn.Module):
    """
    Categorical-Seasonal-Numerical Transformer for time series prediction.

    Architecture:
    1. Categorical features → Embeddings → Categorical Transformer → CLS₁
    2. Numerical features → Sequence → Numerical Transformer → CLS₂
    3. Concatenate CLS₁ + CLS₂ → Fully Connected → Prediction
    """

    def __init__(self,
                 categorical_features: Dict[str, int],
                 num_numerical_features: int,
                 sequence_length: int,
                 d_model: int = 128,
                 n_layers: int = 3,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 output_dim: int = 1):
        super().__init__()

        self.d_model = d_model
        self.has_categorical = len(categorical_features) > 0
        self.has_numerical = num_numerical_features > 0

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

        # Fully connected layers for final prediction
        self.prediction_head = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self,
                categorical_inputs: Optional[Dict[str, torch.Tensor]] = None,
                numerical_inputs: Optional[torch.Tensor] = None) -> torch.Tensor:

        cls_tokens = []

        # Process categorical features
        if self.has_categorical and categorical_inputs is not None:
            cat_cls = self.categorical_processor(categorical_inputs)
            cls_tokens.append(cat_cls)

        # Process numerical features
        if self.has_numerical and numerical_inputs is not None:
            num_cls = self.numerical_processor(numerical_inputs)
            cls_tokens.append(num_cls)

        # Concatenate CLS tokens
        if not cls_tokens:
            raise ValueError("At least one of categorical or numerical inputs must be provided")

        fused_cls = torch.cat(cls_tokens, dim=1)  # [batch_size, fusion_dim]

        # Final prediction
        output = self.prediction_head(fused_cls)
        return output


# For compatibility with existing FT-Transformer interface
class CSNTransformerPredictor(nn.Module):
    """
    Wrapper for CSNTransformer to match FTTransformerPredictor interface.
    """

    def __init__(self,
                 categorical_features: Dict[str, int],
                 num_numerical_features: int,
                 sequence_length: int,
                 d_model: int = 128,
                 n_layers: int = 3,
                 n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.model = CSNTransformer(
            categorical_features=categorical_features,
            num_numerical_features=num_numerical_features,
            sequence_length=sequence_length,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            output_dim=1
        )

    def forward(self, categorical_inputs=None, numerical_inputs=None):
        return self.model(categorical_inputs, numerical_inputs)