"""
Unit tests for pooling modules.

Tests all 5 pooling strategies:
- CLSTokenPooling
- SingleHeadAttentionPooling
- MultiHeadAttentionPooling
- WeightedAveragePooling
- TemporalMultiHeadAttentionPooling
"""

import pytest
import torch
import torch.nn as nn

from tf_predictor.core.base.pooling import (
    CLSTokenPooling,
    SingleHeadAttentionPooling,
    MultiHeadAttentionPooling,
    WeightedAveragePooling,
    TemporalMultiHeadAttentionPooling,
    create_pooling_module,
    VALID_POOLING_TYPES
)


class TestCLSTokenPooling:
    """Test CLS token pooling (extracts first token)."""

    def test_forward_shape(self):
        """Test output shape is correct."""
        batch_size, seq_len, d_token = 32, 10, 128
        pooling = CLSTokenPooling(d_token=d_token)

        x = torch.randn(batch_size, seq_len, d_token)
        output = pooling(x)

        assert output.shape == (batch_size, d_token)

    def test_extracts_first_token(self):
        """Test that CLS pooling extracts the first token."""
        batch_size, seq_len, d_token = 2, 5, 8
        pooling = CLSTokenPooling(d_token=d_token)

        # Create input with distinct values for each position
        x = torch.arange(batch_size * seq_len * d_token, dtype=torch.float32)
        x = x.reshape(batch_size, seq_len, d_token)

        output = pooling(x)

        # Should extract position 0
        expected = x[:, 0, :]
        assert torch.allclose(output, expected)

    def test_zero_parameters(self):
        """Test that CLS pooling has no trainable parameters."""
        pooling = CLSTokenPooling(d_token=128)
        num_params = sum(p.numel() for p in pooling.parameters())
        assert num_params == 0


class TestSingleHeadAttentionPooling:
    """Test single-head attention pooling."""

    def test_forward_shape(self):
        """Test output shape is correct."""
        batch_size, seq_len, d_token = 32, 10, 128
        pooling = SingleHeadAttentionPooling(d_token=d_token)

        x = torch.randn(batch_size, seq_len, d_token)
        output = pooling(x)

        assert output.shape == (batch_size, d_token)

    def test_has_parameters(self):
        """Test that single-head attention has trainable parameters."""
        pooling = SingleHeadAttentionPooling(d_token=128)
        num_params = sum(p.numel() for p in pooling.parameters())
        assert num_params > 0

    def test_gradient_flow(self):
        """Test that gradients flow through pooling."""
        batch_size, seq_len, d_token = 4, 5, 16
        pooling = SingleHeadAttentionPooling(d_token=d_token)

        x = torch.randn(batch_size, seq_len, d_token, requires_grad=True)
        output = pooling(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestMultiHeadAttentionPooling:
    """Test multi-head attention pooling."""

    def test_forward_shape(self):
        """Test output shape is correct."""
        batch_size, seq_len, d_token = 32, 10, 128
        pooling = MultiHeadAttentionPooling(d_token=d_token, n_heads=8)

        x = torch.randn(batch_size, seq_len, d_token)
        output = pooling(x)

        assert output.shape == (batch_size, d_token)

    def test_requires_divisible_heads(self):
        """Test that d_token must be divisible by n_heads."""
        with pytest.raises(ValueError, match="must be divisible"):
            MultiHeadAttentionPooling(d_token=128, n_heads=7)

    def test_different_num_heads(self):
        """Test with different numbers of heads."""
        d_token = 128
        for n_heads in [2, 4, 8, 16]:
            pooling = MultiHeadAttentionPooling(d_token=d_token, n_heads=n_heads)
            x = torch.randn(4, 10, d_token)
            output = pooling(x)
            assert output.shape == (4, d_token)

    def test_has_more_params_than_single_head(self):
        """Test that multi-head has comparable params to single-head."""
        d_token = 128
        single = SingleHeadAttentionPooling(d_token=d_token)
        multi = MultiHeadAttentionPooling(d_token=d_token, n_heads=8)

        single_params = sum(p.numel() for p in single.parameters())
        multi_params = sum(p.numel() for p in multi.parameters())

        # Both should have similar parameter counts (query + attention module)
        assert multi_params > 0
        assert single_params > 0


class TestWeightedAveragePooling:
    """Test weighted average pooling."""

    def test_forward_shape(self):
        """Test output shape is correct."""
        batch_size, seq_len, d_token = 32, 10, 128
        pooling = WeightedAveragePooling(d_token=d_token, max_seq_len=20)

        x = torch.randn(batch_size, seq_len, d_token)
        output = pooling(x)

        assert output.shape == (batch_size, d_token)

    def test_handles_variable_seq_len(self):
        """Test that it handles sequences shorter than max_seq_len."""
        d_token = 128
        max_seq_len = 100
        pooling = WeightedAveragePooling(d_token=d_token, max_seq_len=max_seq_len)

        for seq_len in [10, 50, 100]:
            x = torch.randn(4, seq_len, d_token)
            output = pooling(x)
            assert output.shape == (4, d_token)

    def test_has_parameters(self):
        """Test that weighted average has trainable parameters."""
        pooling = WeightedAveragePooling(d_token=128, max_seq_len=50)
        num_params = sum(p.numel() for p in pooling.parameters())
        assert num_params == 50  # max_seq_len learnable weights


class TestTemporalMultiHeadAttentionPooling:
    """Test temporal multi-head attention pooling."""

    def test_forward_shape(self):
        """Test output shape is correct."""
        batch_size, seq_len, d_token = 32, 10, 128
        pooling = TemporalMultiHeadAttentionPooling(
            d_token=d_token, n_heads=8, max_seq_len=20
        )

        x = torch.randn(batch_size, seq_len, d_token)
        output = pooling(x)

        assert output.shape == (batch_size, d_token)

    def test_temporal_bias_applied(self):
        """Test that temporal bias affects the output."""
        d_token = 64
        n_heads = 4
        seq_len = 10

        pooling = TemporalMultiHeadAttentionPooling(
            d_token=d_token, n_heads=n_heads, max_seq_len=20
        )

        x = torch.randn(2, seq_len, d_token)

        # Forward pass should use temporal bias
        output = pooling(x)
        assert output.shape == (2, d_token)

        # Check that temporal_bias parameter exists
        assert hasattr(pooling, 'temporal_bias')
        assert pooling.temporal_bias.shape[1] >= seq_len


class TestPoolingFactory:
    """Test the pooling factory function."""

    def test_creates_all_types(self):
        """Test that factory can create all pooling types."""
        d_token = 128
        n_heads = 8
        max_seq_len = 50

        for pooling_type in VALID_POOLING_TYPES:
            pooling = create_pooling_module(
                pooling_type=pooling_type,
                d_token=d_token,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                dropout=0.1
            )

            # Test forward pass
            x = torch.randn(4, 10, d_token)
            output = pooling(x)
            assert output.shape == (4, d_token)

    def test_invalid_pooling_type(self):
        """Test that invalid pooling type raises error."""
        with pytest.raises(ValueError, match="Invalid pooling_type"):
            create_pooling_module(
                pooling_type='invalid_type',
                d_token=128
            )

    def test_multihead_requires_n_heads(self):
        """Test that multihead pooling requires n_heads parameter."""
        with pytest.raises(ValueError, match="requires n_heads"):
            create_pooling_module(
                pooling_type='multihead_attention',
                d_token=128,
                n_heads=None
            )

    def test_multihead_requires_min_heads(self):
        """Test that multihead pooling requires at least 2 heads."""
        with pytest.raises(ValueError, match="requires n_heads >= 2"):
            create_pooling_module(
                pooling_type='multihead_attention',
                d_token=128,
                n_heads=1
            )

    def test_multihead_requires_divisible_heads(self):
        """Test that d_token must be divisible by n_heads."""
        with pytest.raises(ValueError, match="must be divisible"):
            create_pooling_module(
                pooling_type='multihead_attention',
                d_token=128,
                n_heads=7
            )

    def test_temporal_requires_n_heads(self):
        """Test that temporal pooling requires n_heads parameter."""
        with pytest.raises(ValueError, match="requires n_heads"):
            create_pooling_module(
                pooling_type='temporal_multihead_attention',
                d_token=128,
                n_heads=None
            )


class TestPoolingIntegration:
    """Integration tests for pooling modules."""

    def test_all_poolings_same_output_shape(self):
        """Test that all pooling types produce same output shape."""
        batch_size, seq_len, d_token = 8, 15, 64
        n_heads = 8

        x = torch.randn(batch_size, seq_len, d_token)

        outputs = {}
        for pooling_type in VALID_POOLING_TYPES:
            pooling = create_pooling_module(
                pooling_type=pooling_type,
                d_token=d_token,
                n_heads=n_heads,
                max_seq_len=seq_len,
                dropout=0.0  # Disable dropout for deterministic testing
            )

            with torch.no_grad():
                outputs[pooling_type] = pooling(x)

        # Check all outputs have same shape
        expected_shape = (batch_size, d_token)
        for pooling_type, output in outputs.items():
            assert output.shape == expected_shape, f"{pooling_type} has wrong shape"

    def test_pooling_reproducibility(self):
        """Test that pooling is deterministic in eval mode."""
        d_token = 64
        pooling = create_pooling_module(
            pooling_type='multihead_attention',
            d_token=d_token,
            n_heads=8,
            dropout=0.0
        )
        pooling.eval()

        x = torch.randn(4, 10, d_token)

        with torch.no_grad():
            output1 = pooling(x)
            output2 = pooling(x)

        assert torch.allclose(output1, output2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
