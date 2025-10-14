"""
Shared embedding and tokenization components for transformer models.

This module provides reusable components for both FT-Transformer and CSN-Transformer:
- CLSToken: Learnable classification token for aggregation
- NumericalEmbedding: Projection for numerical features
- CategoricalEmbedding: Embedding tables for categorical features
- PositionalEncoding: Position information for sequences
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class CLSToken(nn.Module):
    """
    Learnable CLS (classification) token for sequence aggregation.

    The CLS token is prepended to the sequence and used to aggregate information
    from all other tokens through self-attention. After transformer processing,
    the CLS token representation is used for prediction.

    Args:
        d_model: Dimension of the token embedding
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Initialize as learnable parameter
        self.token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.token, std=0.02)  # Small random initialization

    def expand(self, batch_size: int) -> torch.Tensor:
        """
        Expand CLS token for a batch.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Expanded CLS token of shape [batch_size, 1, d_model]
        """
        return self.token.expand(batch_size, -1, -1)


class NumericalEmbedding(nn.Module):
    """
    Linear projection for numerical features.

    Projects numerical features to a higher-dimensional embedding space.
    Used by both FT-Transformer (per-feature projection) and CSN-Transformer
    (sequence projection).

    Args:
        input_dim: Number of input numerical features
        d_model: Output embedding dimension
        use_bias: Whether to use bias in linear projection
    """

    def __init__(self, input_dim: int, d_model: int, use_bias: bool = True):
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model, bias=use_bias)
        # Initialize with small weights for stable training
        nn.init.xavier_uniform_(self.projection.weight)
        if use_bias:
            nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project numerical features.

        Args:
            x: Input features of shape [batch, ...] where last dim is input_dim

        Returns:
            Projected features with last dim = d_model
        """
        return self.projection(x)


class CategoricalEmbedding(nn.Module):
    """
    Embedding tables for categorical features.

    Creates separate embedding tables for each categorical feature.
    Each unique category value is mapped to a learned embedding vector.

    Args:
        cardinalities: List of cardinalities (num unique values) for each categorical feature
        d_model: Embedding dimension
    """

    def __init__(self, cardinalities: list[int], d_model: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, d_model)
            for cardinality in cardinalities
        ])

        # Initialize embeddings
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Embed categorical features.

        Args:
            x_cat: Categorical feature indices of shape [batch, num_categorical]

        Returns:
            Embedded features of shape [batch, num_categorical, d_model]
        """
        # Apply each embedding to its corresponding column
        embeddings = [
            self.embeddings[i](x_cat[:, i])
            for i in range(x_cat.shape[1])
        ]
        # Stack along feature dimension: [batch, num_categorical, d_model]
        return torch.stack(embeddings, dim=1)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequence modeling.

    Adds position information to token embeddings using either:
    1. Sinusoidal encoding (fixed, from "Attention is All You Need")
    2. Learnable encoding (trainable position embeddings)

    Args:
        d_model: Dimension of the model
        max_seq_length: Maximum sequence length to support
        learnable: If True, use learnable embeddings; if False, use sinusoidal
        dropout: Dropout rate applied after adding positional encoding
    """

    def __init__(
        self,
        d_model: int,
        max_seq_length: int = 5000,
        learnable: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.learnable = learnable

        if learnable:
            # Learnable positional embeddings
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
            nn.init.normal_(self.pos_embedding, std=0.02)
        else:
            # Fixed sinusoidal positional encoding
            position = torch.arange(max_seq_length).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

            pe = torch.zeros(1, max_seq_length, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)

            # Register as buffer (not a parameter, but saved with model)
            self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]

        Returns:
            Tensor with positional encoding added, same shape as input
        """
        seq_len = x.size(1)

        if self.learnable:
            # Add learnable position embeddings
            x = x + self.pos_embedding[:, :seq_len, :]
        else:
            # Add sinusoidal position encoding
            x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)


class NumericalTokenizer(nn.Module):
    """
    FT-Transformer style numerical tokenizer.

    Converts each numerical feature to a separate token using
    feature-specific linear transformations: token_i = x_i * W_i + b_i

    This differs from NumericalEmbedding which projects all features together.

    Args:
        num_features: Number of numerical features
        d_token: Token embedding dimension
    """

    def __init__(self, num_features: int, d_token: int):
        super().__init__()
        # Each feature gets its own weight and bias
        self.weights = nn.Parameter(torch.empty(num_features, d_token))
        self.biases = nn.Parameter(torch.empty(num_features, d_token))

        # Initialize
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.biases)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        """
        Tokenize numerical features.

        Args:
            x_num: Numerical features of shape [batch, num_features]

        Returns:
            Tokens of shape [batch, num_features, d_token]
        """
        # Expand dimensions for broadcasting: [batch, num_features, 1]
        x_expanded = x_num.unsqueeze(-1)

        # Apply feature-specific linear transformation
        # [batch, num_features, 1] * [num_features, d_token] + [num_features, d_token]
        tokens = x_expanded * self.weights + self.biases

        return tokens
