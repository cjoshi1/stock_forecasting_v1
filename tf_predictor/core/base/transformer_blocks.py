"""
Shared transformer building blocks for both FT and CSN models.

This module provides reusable transformer components:
- BaseTransformerBlock: Generic self-attention + feed-forward block
- Utility functions for attention masks
"""

import torch
import torch.nn as nn
from typing import Optional


class BaseTransformerBlock(nn.Module):
    """
    Generic transformer block with multi-head self-attention and feed-forward network.

    This is the fundamental building block used in both FT-Transformer and CSN-Transformer.
    Architecture: MultiHeadAttention → Add&Norm → FeedForward → Add&Norm

    The block can be configured with different activation functions (GELU/ReLU) and
    normalization strategies (pre-norm/post-norm).

    Args:
        d_model: Model dimension (input and output size)
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension (typically 4 * d_model)
        dropout: Dropout rate for regularization
        activation: Activation function ('gelu' or 'relu')
        norm_first: If True, use pre-normalization (more stable); if False, post-norm
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        norm_first: bool = True
    ):
        super().__init__()

        # Multi-head self-attention
        # batch_first=True means input/output shape is [batch, seq, features]
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization for training stability
        self.norm1 = nn.LayerNorm(d_model)  # After attention
        self.norm2 = nn.LayerNorm(d_model)  # After feed-forward

        # Feed-forward network
        # Two-layer MLP with expansion and contraction
        activation_fn = nn.GELU() if activation.lower() == 'gelu' else nn.ReLU()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),       # Expand
            activation_fn,                   # Non-linearity
            nn.Dropout(dropout),             # Regularization
            nn.Linear(d_ff, d_model),        # Contract back to d_model
            nn.Dropout(dropout)              # Final regularization
        )

        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            attn_mask: Attention mask (e.g., for causal attention)
            key_padding_mask: Padding mask for variable-length sequences

        Returns:
            Output tensor of shape [batch, seq_len, d_model]
        """
        if self.norm_first:
            # Pre-normalization (more stable, commonly used in modern transformers)
            # Attention sub-layer
            x_norm = self.norm1(x)
            attn_out, _ = self.attention(
                x_norm, x_norm, x_norm,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask
            )
            x = x + self.dropout(attn_out)  # Residual connection

            # Feed-forward sub-layer
            x_norm = self.norm2(x)
            ff_out = self.feed_forward(x_norm)
            x = x + ff_out  # Residual connection (dropout already in feed_forward)
        else:
            # Post-normalization (original Transformer paper)
            # Attention sub-layer
            attn_out, _ = self.attention(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask
            )
            x = self.norm1(x + self.dropout(attn_out))  # Add & Norm

            # Feed-forward sub-layer
            ff_out = self.feed_forward(x)
            x = self.norm2(x + ff_out)  # Add & Norm

        return x


class TransformerStack(nn.Module):
    """
    Stack of transformer blocks.

    A simple container for multiple transformer layers that can be used
    as an alternative to nn.TransformerEncoder.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        n_layers: Number of transformer blocks to stack
        dropout: Dropout rate
        activation: Activation function ('gelu' or 'relu')
        norm_first: Whether to use pre-normalization
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        norm_first: bool = True
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            BaseTransformerBlock(
                d_model, n_heads, d_ff, dropout, activation, norm_first
            )
            for _ in range(n_layers)
        ])

        # Optional final layer norm (common in pre-norm architectures)
        self.final_norm = nn.LayerNorm(d_model) if norm_first else None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through all transformer layers.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            attn_mask: Attention mask
            key_padding_mask: Padding mask

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, attn_mask, key_padding_mask)

        if self.final_norm is not None:
            x = self.final_norm(x)

        return x


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal (autoregressive) attention mask.

    Prevents positions from attending to subsequent positions.
    Used for autoregressive modeling where future tokens should not be visible.

    Args:
        seq_len: Sequence length
        device: Device to create tensor on

    Returns:
        Boolean mask of shape [seq_len, seq_len] where True means "masked out"
    """
    # Upper triangular matrix of ones (excluding diagonal)
    # True values will be masked (not attended to)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Create padding mask for variable-length sequences.

    Args:
        lengths: Actual lengths of sequences in batch [batch_size]
        max_len: Maximum sequence length (padded length)

    Returns:
        Boolean mask of shape [batch_size, max_len] where True means "padded position"
    """
    batch_size = lengths.size(0)
    device = lengths.device

    # Create position indices [0, 1, 2, ..., max_len-1]
    positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # Create mask: positions >= length are padded
    mask = positions >= lengths.unsqueeze(1)
    return mask


def create_attention_mask(
    query_len: int,
    key_len: int,
    causal: bool = False,
    device: Optional[torch.device] = None
) -> Optional[torch.Tensor]:
    """
    Create attention mask for transformer.

    Args:
        query_len: Length of query sequence
        key_len: Length of key sequence (typically same as query_len for self-attention)
        causal: If True, create causal mask for autoregressive modeling
        device: Device to create tensor on

    Returns:
        Attention mask or None if no masking needed
    """
    if not causal:
        return None

    if device is None:
        device = torch.device('cpu')

    # For causal masking, query and key lengths should match (self-attention)
    assert query_len == key_len, "Causal masking requires self-attention (query_len == key_len)"

    return create_causal_mask(query_len, device)
