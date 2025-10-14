"""
Base components shared by FT-Transformer and CSN-Transformer.

This module provides reusable building blocks:
- Embeddings: CLSToken, NumericalEmbedding, CategoricalEmbedding, PositionalEncoding
- Transformer blocks: BaseTransformerBlock, TransformerStack
- Prediction heads: MultiHorizonHead, RegressionHead, ClassificationHead
"""

from .embeddings import (
    CLSToken,
    NumericalEmbedding,
    CategoricalEmbedding,
    PositionalEncoding,
    NumericalTokenizer
)

from .transformer_blocks import (
    BaseTransformerBlock,
    TransformerStack,
    create_causal_mask,
    create_padding_mask,
    create_attention_mask
)

from .prediction_heads import (
    MultiHorizonHead,
    RegressionHead,
    ClassificationHead,
    MultiTaskHead,
    DistributionHead
)

__all__ = [
    # Embeddings
    'CLSToken',
    'NumericalEmbedding',
    'CategoricalEmbedding',
    'PositionalEncoding',
    'NumericalTokenizer',

    # Transformer blocks
    'BaseTransformerBlock',
    'TransformerStack',
    'create_causal_mask',
    'create_padding_mask',
    'create_attention_mask',

    # Prediction heads
    'MultiHorizonHead',
    'RegressionHead',
    'ClassificationHead',
    'MultiTaskHead',
    'DistributionHead',
]
