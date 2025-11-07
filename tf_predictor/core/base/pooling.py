"""
Pooling modules for aggregating transformer sequence outputs.

This module provides various strategies for aggregating a sequence of token representations
[batch, seq_len, d_token] into a single fixed-size vector [batch, d_token].

Available pooling strategies:
1. CLSTokenPooling: Extract CLS token at position 0
2. SingleHeadAttentionPooling: Single-head attention with learnable query
3. MultiHeadAttentionPooling: Multi-head attention with learnable query
4. WeightedAveragePooling: Learnable weighted average over positions
5. TemporalMultiHeadAttentionPooling: Multi-head attention with temporal/recency bias

Author: TF-Predictor Team
Date: 2025-01-06
"""

import torch
import torch.nn as nn
from typing import Optional


# Valid pooling type identifiers
VALID_POOLING_TYPES = [
    'cls',
    'singlehead_attention',
    'multihead_attention',
    'weighted_avg',
    'temporal_multihead_attention'
]


class PoolingModule(nn.Module):
    """
    Base class for all pooling strategies.

    All pooling modules aggregate a sequence of tokens into a single vector.
    """

    def __init__(self, d_token: int):
        """
        Args:
            d_token: Token embedding dimension
        """
        super().__init__()
        self.d_token = d_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate sequence of tokens into a single vector.

        Args:
            x: Input tokens [batch_size, seq_len, d_token]

        Returns:
            output: Aggregated representation [batch_size, d_token]
        """
        raise NotImplementedError("Subclasses must implement forward()")


class CLSTokenPooling(PoolingModule):
    """
    Extract CLS token at position 0.

    This is the simplest pooling strategy - just extract the first token which
    is assumed to be a special CLS (classification) token that has aggregated
    information from all other tokens during transformer processing.

    Parameters: 0 (no learnable parameters)
    """

    def __init__(self, d_token: int):
        """
        Args:
            d_token: Token embedding dimension
        """
        super().__init__(d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS token at position 0.

        Args:
            x: [batch_size, seq_len, d_token] where position 0 is CLS token

        Returns:
            output: [batch_size, d_token] - the CLS token

        Note:
            Assumes that position 0 contains a CLS token that was prepended
            before the transformer layers.
        """
        return x[:, 0, :]  # Extract first token


class SingleHeadAttentionPooling(PoolingModule):
    """
    Single-head attention pooling with learnable query.

    Uses a learnable query vector that attends to all tokens in the sequence.
    The attention weights determine how much each token contributes to the
    final aggregated representation.

    Architecture:
        Query: [1, d_token] - learnable parameter
        Keys, Values: Projected from input tokens
        Output: Weighted sum of values based on query-key similarity

    Parameters: ~3 * d_token^2 (for Q, K, V projections in attention)
    """

    def __init__(self, d_token: int, dropout: float = 0.1):
        """
        Args:
            d_token: Token embedding dimension
            dropout: Dropout probability for attention weights
        """
        super().__init__(d_token)

        # Learnable query vector - "what should I pay attention to?"
        self.query = nn.Parameter(torch.randn(1, 1, d_token))

        # Single-head attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=1,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply single-head attention pooling.

        Args:
            x: [batch_size, seq_len, d_token]

        Returns:
            output: [batch_size, d_token]

        Process:
            1. Expand query for batch: [1, 1, d_token] → [batch, 1, d_token]
            2. Query attends to all tokens (keys, values = x)
            3. Get weighted sum: [batch, 1, d_token]
            4. Squeeze: [batch, d_token]
        """
        batch_size = x.size(0)

        # Expand query for batch processing
        query = self.query.expand(batch_size, -1, -1)  # [batch, 1, d_token]

        # Attention: query attends to sequence
        # output: [batch, 1, d_token], attn_weights: [batch, 1, seq_len]
        output, _ = self.attention(query, x, x)

        # Remove the singleton dimension
        return output.squeeze(1)  # [batch, d_token]


class MultiHeadAttentionPooling(PoolingModule):
    """
    Multi-head attention pooling with learnable query.

    Similar to single-head attention but uses multiple attention heads to capture
    different aspects of the sequence. Each head learns to attend to different
    patterns, and their outputs are combined.

    Architecture:
        Query: [1, d_token] - learnable parameter
        Heads: n_heads parallel attention mechanisms
        Output: Combined outputs from all heads

    Parameters: ~3 * d_token^2 + additional parameters for multi-head projection
    """

    def __init__(self, d_token: int, n_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            d_token: Token embedding dimension
            n_heads: Number of attention heads
            dropout: Dropout probability for attention weights
        """
        super().__init__(d_token)

        if d_token % n_heads != 0:
            raise ValueError(f"d_token ({d_token}) must be divisible by n_heads ({n_heads})")

        self.n_heads = n_heads

        # Learnable query vector
        self.query = nn.Parameter(torch.randn(1, 1, d_token))

        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention pooling.

        Args:
            x: [batch_size, seq_len, d_token]

        Returns:
            output: [batch_size, d_token]

        Process:
            1. Expand query for batch
            2. Multi-head attention with query attending to all tokens
            3. Each head focuses on different aspects
            4. Outputs combined and projected back to d_token
        """
        batch_size = x.size(0)

        # Expand query for batch processing
        query = self.query.expand(batch_size, -1, -1)  # [batch, 1, d_token]

        # Multi-head attention
        output, _ = self.attention(query, x, x)  # [batch, 1, d_token]

        return output.squeeze(1)  # [batch, d_token]


class WeightedAveragePooling(PoolingModule):
    """
    Learnable weighted average over sequence positions.

    Learns a weight for each position in the sequence, applies softmax to normalize,
    then computes weighted average. Unlike attention, these weights are position-based
    and don't depend on the content of the tokens.

    Architecture:
        Weights: [max_seq_len] - learnable position weights
        Output: Weighted sum of tokens based on position

    Parameters: max_seq_len (one weight per position)

    Use case: When position in sequence is more important than content
              (e.g., recent timesteps always more important)
    """

    def __init__(self, d_token: int, max_seq_len: int = 100):
        """
        Args:
            d_token: Token embedding dimension
            max_seq_len: Maximum sequence length to support
        """
        super().__init__(d_token)
        self.max_seq_len = max_seq_len

        # Learnable position weights
        # Shape: [1, max_seq_len, 1] for broadcasting
        self.weights = nn.Parameter(torch.randn(1, max_seq_len, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable weighted average.

        Args:
            x: [batch_size, seq_len, d_token]

        Returns:
            output: [batch_size, d_token]

        Process:
            1. Get position weights for current sequence length
            2. Apply softmax to normalize (sum to 1)
            3. Weighted sum: sum(weights * tokens)
        """
        batch_size, seq_len, d_token = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}. "
                f"Initialize with larger max_seq_len."
            )

        # Get weights for current sequence length
        w = self.weights[:, :seq_len, :]  # [1, seq_len, 1]

        # Normalize weights using softmax
        w = torch.softmax(w, dim=1)  # Sum to 1 across seq_len dimension

        # Weighted average: [batch, seq_len, d_token] * [1, seq_len, 1]
        # Broadcasting: weights apply to each d_token dimension
        output = (x * w).sum(dim=1)  # [batch, d_token]

        return output


class TemporalMultiHeadAttentionPooling(PoolingModule):
    """
    Multi-head attention pooling with temporal/recency bias.

    Similar to multi-head attention but adds a learnable temporal bias that can
    give more weight to recent timesteps. Useful for time series where recent
    observations are typically more relevant for prediction.

    Architecture:
        Query: [1, d_token] - learnable query
        Temporal Bias: [max_seq_len, 1] - learnable position-dependent bias
        Attention: Multi-head with biased keys

    Parameters: ~3 * d_token^2 + max_seq_len (temporal bias)

    Key difference from standard multi-head:
        - Adds learnable temporal bias to keys before attention
        - Bias can encode recency (recent positions weighted higher)
    """

    def __init__(
        self,
        d_token: int,
        n_heads: int = 8,
        max_seq_len: int = 100,
        dropout: float = 0.1
    ):
        """
        Args:
            d_token: Token embedding dimension
            n_heads: Number of attention heads
            max_seq_len: Maximum sequence length to support
            dropout: Dropout probability for attention weights
        """
        super().__init__(d_token)

        if d_token % n_heads != 0:
            raise ValueError(f"d_token ({d_token}) must be divisible by n_heads ({n_heads})")

        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        # Learnable query vector
        self.query = nn.Parameter(torch.randn(1, 1, d_token))

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Learnable temporal bias - can encode recency
        # Shape: [1, max_seq_len, 1] for broadcasting across d_token
        self.temporal_bias = nn.Parameter(torch.randn(1, max_seq_len, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal multi-head attention pooling.

        Args:
            x: [batch_size, seq_len, d_token]

        Returns:
            output: [batch_size, d_token]

        Process:
            1. Add temporal bias to input tokens (keys)
            2. Apply multi-head attention with biased keys
            3. Query attends to biased keys, retrieves original values
            4. Temporal bias influences which positions get more attention
        """
        batch_size, seq_len, d_token = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}. "
                f"Initialize with larger max_seq_len."
            )

        # Get temporal bias for current sequence length
        bias = self.temporal_bias[:, :seq_len, :]  # [1, seq_len, 1]

        # Add temporal bias to keys (modifies what we attend to)
        # Broadcast across batch and d_token dimensions
        x_biased = x + bias  # [batch, seq_len, d_token]

        # Expand query for batch processing
        query = self.query.expand(batch_size, -1, -1)  # [batch, 1, d_token]

        # Attention with biased keys but original values
        # Query attends to biased keys, retrieves original values
        output, _ = self.attention(query, x_biased, x)  # [batch, 1, d_token]

        return output.squeeze(1)  # [batch, d_token]


def create_pooling_module(
    pooling_type: str,
    d_token: int,
    n_heads: Optional[int] = None,
    max_seq_len: int = 100,
    dropout: float = 0.1
) -> PoolingModule:
    """
    Factory function to create pooling modules with validation.

    Args:
        pooling_type: Type of pooling strategy
        d_token: Token embedding dimension
        n_heads: Number of attention heads (required for multihead pooling)
        max_seq_len: Maximum sequence length (for position-based pooling)
        dropout: Dropout probability for attention-based pooling

    Returns:
        PoolingModule instance

    Raises:
        ValueError: If pooling_type is invalid or required parameters are missing

    Examples:
        >>> # CLS token pooling
        >>> pooling = create_pooling_module('cls', d_token=128)

        >>> # Single-head attention
        >>> pooling = create_pooling_module('singlehead_attention', d_token=128)

        >>> # Multi-head attention (requires n_heads)
        >>> pooling = create_pooling_module('multihead_attention', d_token=128, n_heads=8)

        >>> # Weighted average
        >>> pooling = create_pooling_module('weighted_avg', d_token=128, max_seq_len=50)

        >>> # Temporal multi-head attention
        >>> pooling = create_pooling_module('temporal_multihead_attention',
        ...                                 d_token=128, n_heads=8, max_seq_len=50)
    """
    # Validate pooling type
    if pooling_type not in VALID_POOLING_TYPES:
        raise ValueError(
            f"Invalid pooling_type '{pooling_type}'. "
            f"Must be one of: {VALID_POOLING_TYPES}"
        )

    # Create appropriate pooling module
    if pooling_type == 'cls':
        return CLSTokenPooling(d_token=d_token)

    elif pooling_type == 'singlehead_attention':
        return SingleHeadAttentionPooling(d_token=d_token, dropout=dropout)

    elif pooling_type == 'multihead_attention':
        # Validate n_heads is provided
        if n_heads is None:
            raise ValueError(
                "pooling_type='multihead_attention' requires n_heads parameter"
            )
        if n_heads < 2:
            raise ValueError(
                f"pooling_type='multihead_attention' requires n_heads >= 2, got {n_heads}. "
                f"Use 'singlehead_attention' for single head."
            )
        if d_token % n_heads != 0:
            raise ValueError(
                f"d_token ({d_token}) must be divisible by n_heads ({n_heads})"
            )

        return MultiHeadAttentionPooling(
            d_token=d_token,
            n_heads=n_heads,
            dropout=dropout
        )

    elif pooling_type == 'weighted_avg':
        return WeightedAveragePooling(
            d_token=d_token,
            max_seq_len=max_seq_len
        )

    elif pooling_type == 'temporal_multihead_attention':
        # Validate n_heads is provided
        if n_heads is None:
            raise ValueError(
                "pooling_type='temporal_multihead_attention' requires n_heads parameter"
            )
        if n_heads < 2:
            raise ValueError(
                f"pooling_type='temporal_multihead_attention' requires n_heads >= 2, got {n_heads}"
            )
        if d_token % n_heads != 0:
            raise ValueError(
                f"d_token ({d_token}) must be divisible by n_heads ({n_heads})"
            )

        return TemporalMultiHeadAttentionPooling(
            d_token=d_token,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

    # Should never reach here due to validation above
    raise ValueError(f"Unhandled pooling_type: {pooling_type}")


def get_pooling_info(pooling_type: str) -> dict:
    """
    Get information about a pooling strategy.

    Args:
        pooling_type: Type of pooling strategy

    Returns:
        Dictionary with pooling strategy information:
            - name: Full name
            - requires_cls_token: Whether CLS token must be prepended
            - learnable_params: Description of learnable parameters
            - use_case: When to use this strategy

    Examples:
        >>> info = get_pooling_info('multihead_attention')
        >>> print(info['name'])
        'Multi-Head Attention Pooling'
    """
    info_dict = {
        'cls': {
            'name': 'CLS Token Pooling',
            'requires_cls_token': True,
            'learnable_params': 'None (0 parameters)',
            'use_case': 'Simple baseline, assumes CLS token aggregates during transformer'
        },
        'singlehead_attention': {
            'name': 'Single-Head Attention Pooling',
            'requires_cls_token': False,
            'learnable_params': '~3*d_token^2 (Q, K, V projections)',
            'use_case': 'Learn single attention pattern over sequence'
        },
        'multihead_attention': {
            'name': 'Multi-Head Attention Pooling',
            'requires_cls_token': False,
            'learnable_params': '~3*d_token^2 + multi-head projections',
            'use_case': 'Learn multiple attention patterns (most flexible)'
        },
        'weighted_avg': {
            'name': 'Weighted Average Pooling',
            'requires_cls_token': False,
            'learnable_params': 'max_seq_len position weights',
            'use_case': 'Position-based weighting (e.g., recent timesteps more important)'
        },
        'temporal_multihead_attention': {
            'name': 'Temporal Multi-Head Attention Pooling',
            'requires_cls_token': False,
            'learnable_params': '~3*d_token^2 + temporal bias',
            'use_case': 'Time series with recency bias'
        }
    }

    if pooling_type not in info_dict:
        raise ValueError(f"Unknown pooling_type: {pooling_type}")

    return info_dict[pooling_type]
