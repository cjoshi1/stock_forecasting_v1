"""
Integration tests for FT-Transformer and CSN-Transformer with pooling.

Tests that both architectures work correctly with all pooling strategies.
"""

import pytest
import torch
import torch.nn as nn

from tf_predictor.core.ft_model import FTTransformerCLSModel
from tf_predictor.core.csn_model import CSNTransformerCLSModel
from tf_predictor.core.base.pooling import VALID_POOLING_TYPES


class TestFTTransformerPooling:
    """Test FT-Transformer with different pooling strategies."""

    @pytest.mark.parametrize("pooling_type", VALID_POOLING_TYPES)
    def test_all_pooling_types(self, pooling_type):
        """Test that FT-Transformer works with all pooling types."""
        model = FTTransformerCLSModel(
            sequence_length=10,
            num_numerical=8,
            num_categorical=2,
            cat_cardinalities=[100, 5],
            output_dim=1,
            d_token=64,
            n_heads=8,
            n_layers=2,
            pooling_type=pooling_type,
            dropout=0.1
        )

        batch_size = 4
        x_num = torch.randn(batch_size, 10, 8)
        x_cat = torch.randint(0, 100, (batch_size, 2))
        x_cat[:, 1] = torch.randint(0, 5, (batch_size,))  # Second feature has cardinality 5

        # Forward pass
        output = model(x_num, x_cat)

        # Check output shape
        assert output.shape == (batch_size, 1)

    def test_numerical_only(self):
        """Test FT-Transformer with only numerical features."""
        model = FTTransformerCLSModel(
            sequence_length=10,
            num_numerical=8,
            num_categorical=0,
            cat_cardinalities=[],
            output_dim=1,
            d_token=64,
            n_heads=8,
            n_layers=2,
            pooling_type='multihead_attention'
        )

        batch_size = 4
        x_num = torch.randn(batch_size, 10, 8)

        output = model(x_num, x_cat=None)
        assert output.shape == (batch_size, 1)

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        for seq_len in [5, 10, 20]:
            model = FTTransformerCLSModel(
                sequence_length=seq_len,
                num_numerical=4,
                num_categorical=0,
                cat_cardinalities=[],
                output_dim=1,
                d_token=32,
                n_heads=4,
                n_layers=2,
                pooling_type='weighted_avg'
            )

            x_num = torch.randn(2, seq_len, 4)
            output = model(x_num)
            assert output.shape == (2, 1)

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = FTTransformerCLSModel(
            sequence_length=5,
            num_numerical=4,
            num_categorical=0,
            cat_cardinalities=[],
            output_dim=1,
            d_token=32,
            n_heads=4,
            n_layers=1,
            pooling_type='multihead_attention'
        )

        x_num = torch.randn(2, 5, 4, requires_grad=True)
        output = model(x_num)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x_num.grad is not None
        assert x_num.grad.abs().sum() > 0

    def test_get_model_config(self):
        """Test that get_model_config includes pooling_type."""
        model = FTTransformerCLSModel(
            sequence_length=10,
            num_numerical=8,
            num_categorical=2,
            cat_cardinalities=[100, 5],
            output_dim=1,
            pooling_type='singlehead_attention'
        )

        config = model.get_model_config()

        assert 'pooling_type' in config
        assert config['pooling_type'] == 'singlehead_attention'
        assert config['model_type'] == 'ft_transformer'  # No '_cls' suffix


class TestCSNTransformerPooling:
    """Test CSN-Transformer with different pooling strategies."""

    @pytest.mark.parametrize("pooling_type", VALID_POOLING_TYPES)
    def test_all_pooling_types(self, pooling_type):
        """Test that CSN-Transformer works with all pooling types."""
        model = CSNTransformerCLSModel(
            sequence_length=10,
            num_numerical=8,
            num_categorical=2,
            cat_cardinalities=[50, 3],
            output_dim=1,
            d_token=64,
            n_heads=8,
            n_layers=2,
            pooling_type=pooling_type,
            dropout=0.1
        )

        batch_size = 4
        x_num = torch.randn(batch_size, 10, 8)
        x_cat = torch.randint(0, 50, (batch_size, 2))
        x_cat[:, 1] = torch.randint(0, 3, (batch_size,))  # Second feature has cardinality 3

        # Forward pass
        output = model(x_num, x_cat)

        # Check output shape
        assert output.shape == (batch_size, 1)

    def test_same_pooling_both_pathways(self):
        """Test that both pathways use the same pooling strategy."""
        pooling_type = 'multihead_attention'
        model = CSNTransformerCLSModel(
            sequence_length=10,
            num_numerical=8,
            num_categorical=2,
            cat_cardinalities=[50, 3],
            output_dim=1,
            d_token=64,
            n_heads=8,
            n_layers=2,
            pooling_type=pooling_type
        )

        # Both pooling modules should exist and be of the same type
        assert hasattr(model, 'cat_pooling')
        assert hasattr(model, 'num_pooling')
        assert model.cat_pooling.__class__.__name__ == model.num_pooling.__class__.__name__

    def test_numerical_only(self):
        """Test CSN-Transformer with only numerical features."""
        model = CSNTransformerCLSModel(
            sequence_length=10,
            num_numerical=8,
            num_categorical=0,
            cat_cardinalities=[],
            output_dim=1,
            d_token=64,
            n_heads=8,
            n_layers=2,
            pooling_type='temporal_multihead_attention'
        )

        batch_size = 4
        x_num = torch.randn(batch_size, 10, 8)

        output = model(x_num, x_cat=None)
        assert output.shape == (batch_size, 1)

    def test_get_model_config(self):
        """Test that get_model_config includes pooling_type."""
        model = CSNTransformerCLSModel(
            sequence_length=10,
            num_numerical=8,
            num_categorical=2,
            cat_cardinalities=[100, 5],
            output_dim=1,
            pooling_type='weighted_avg'
        )

        config = model.get_model_config()

        assert 'pooling_type' in config
        assert config['pooling_type'] == 'weighted_avg'
        assert config['model_type'] == 'csn_transformer'  # No '_cls' suffix


class TestPoolingComparison:
    """Compare different pooling strategies."""

    def test_cls_vs_multihead_different_outputs(self):
        """Test that CLS and multihead pooling produce different outputs."""
        # Create two models with different pooling
        model_cls = FTTransformerCLSModel(
            sequence_length=5,
            num_numerical=4,
            num_categorical=0,
            cat_cardinalities=[],
            output_dim=1,
            d_token=32,
            n_heads=4,
            n_layers=1,
            pooling_type='cls'
        )

        model_multihead = FTTransformerCLSModel(
            sequence_length=5,
            num_numerical=4,
            num_categorical=0,
            cat_cardinalities=[],
            output_dim=1,
            d_token=32,
            n_heads=4,
            n_layers=1,
            pooling_type='multihead_attention'
        )

        # Same input
        x = torch.randn(2, 5, 4)

        with torch.no_grad():
            output_cls = model_cls(x)
            output_multihead = model_multihead(x)

        # Outputs should be different (different pooling strategies)
        assert not torch.allclose(output_cls, output_multihead, atol=1e-6)

    def test_parameter_count_differences(self):
        """Test that different pooling types have different parameter counts."""
        d_token = 64
        n_heads = 8

        param_counts = {}
        for pooling_type in ['cls', 'singlehead_attention', 'multihead_attention', 'weighted_avg']:
            model = FTTransformerCLSModel(
                sequence_length=10,
                num_numerical=4,
                num_categorical=0,
                cat_cardinalities=[],
                output_dim=1,
                d_token=d_token,
                n_heads=n_heads,
                n_layers=2,
                pooling_type=pooling_type
            )
            param_counts[pooling_type] = model.get_num_parameters()

        # CLS should have fewest parameters (no pooling params)
        assert param_counts['cls'] < param_counts['singlehead_attention']
        assert param_counts['cls'] < param_counts['multihead_attention']
        assert param_counts['cls'] < param_counts['weighted_avg']


class TestModelFactory:
    """Test model creation through the factory."""

    def test_factory_creates_models_with_pooling(self):
        """Test that ModelFactory can create models with pooling_type parameter."""
        from tf_predictor.core.base.model_factory import ModelFactory

        # Test FT-Transformer
        model_ft = ModelFactory.create_model(
            model_type='ft_transformer',
            sequence_length=10,
            num_numerical=8,
            num_categorical=2,
            cat_cardinalities=[100, 5],
            output_dim=1,
            d_token=64,
            n_heads=8,
            n_layers=2,
            pooling_type='singlehead_attention'
        )

        assert isinstance(model_ft, FTTransformerCLSModel)
        config = model_ft.get_model_config()
        assert config['pooling_type'] == 'singlehead_attention'

        # Test CSN-Transformer
        model_csn = ModelFactory.create_model(
            model_type='csn_transformer',
            sequence_length=10,
            num_numerical=8,
            num_categorical=2,
            cat_cardinalities=[50, 3],
            output_dim=1,
            d_token=64,
            n_heads=8,
            n_layers=2,
            pooling_type='weighted_avg'
        )

        assert isinstance(model_csn, CSNTransformerCLSModel)
        config = model_csn.get_model_config()
        assert config['pooling_type'] == 'weighted_avg'

    def test_default_pooling_type(self):
        """Test that default pooling_type is multihead_attention."""
        from tf_predictor.core.base.model_factory import get_default_model_params

        ft_defaults = get_default_model_params('ft_transformer')
        assert ft_defaults['pooling_type'] == 'multihead_attention'

        csn_defaults = get_default_model_params('csn_transformer')
        assert csn_defaults['pooling_type'] == 'multihead_attention'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
