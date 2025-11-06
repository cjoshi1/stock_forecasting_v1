"""
Abstract interface for time series prediction models.

This module defines the contract that all time series models must implement,
enabling pluggable model architectures in the prediction pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


class TimeSeriesModel(ABC, nn.Module):
    """
    Abstract base class for time series prediction models.

    All concrete model implementations (FT-Transformer, CSN-Transformer, LSTM, etc.)
    must inherit from this class and implement its abstract methods.

    This interface enables:
    1. Pluggable model architectures via factory pattern
    2. Consistent API across different model types
    3. Easy addition of new models without modifying predictor code
    4. Type checking and IDE support
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features)
               For single-step prediction: sequence of historical observations
               For multi-step prediction: same, but output dimension increases

        Returns:
            Output tensor of shape:
            - Single target, single horizon: (batch_size, 1)
            - Single target, multi-horizon: (batch_size, prediction_horizon)
            - Multi-target, single horizon: (batch_size, num_targets)
            - Multi-target, multi-horizon: (batch_size, num_targets * prediction_horizon)
        """
        pass

    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the model.

        Returns:
            Dictionary containing all model hyperparameters and settings.
            This should include everything needed to reconstruct the model.

        Example:
            {
                'model_type': 'ft_transformer',
                'd_token': 64,
                'n_heads': 4,
                'n_layers': 3,
                'sequence_length': 10,
                'num_features': 5,
                'output_dim': 1
            }
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Get the embedding dimension used by the model.

        Returns:
            Integer representing the model's internal embedding dimension (d_model).
            This is useful for understanding model capacity and debugging.
        """
        pass

    def get_num_parameters(self) -> int:
        """
        Get the total number of trainable parameters in the model.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model.

        Returns:
            Dictionary containing model type, config, parameter count, etc.
        """
        return {
            'config': self.get_model_config(),
            'num_parameters': self.get_num_parameters(),
            'embedding_dim': self.get_embedding_dim()
        }


class TransformerBasedModel(TimeSeriesModel):
    """
    Abstract base class for Transformer-based models.

    Provides common functionality for Transformer variants like:
    - FT-Transformer (Feature Tokenizer Transformer)
    - CSN-Transformer (Column-wise Split Network Transformer)
    - Vanilla Transformer
    - Temporal Fusion Transformer
    """

    def __init__(self, d_token: int, n_heads: int, n_layers: int):
        """
        Initialize transformer-based model.

        Args:
            d_token: Token embedding dimension (formerly d_model)
            n_heads: Number of attention heads (formerly num_heads)
            n_layers: Number of transformer layers (formerly num_layers)
        """
        super().__init__()
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_layers = n_layers

    def get_embedding_dim(self) -> int:
        """Get the token embedding dimension (d_token)."""
        return self.d_token

    @abstractmethod
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get attention weights from the last forward pass.

        Returns:
            Attention weights tensor or None if not available.
            Shape: (batch_size, num_heads, sequence_length, sequence_length)
        """
        pass


class RecurrentBasedModel(TimeSeriesModel):
    """
    Abstract base class for RNN-based models.

    Provides common functionality for recurrent variants like:
    - LSTM
    - GRU
    - Bidirectional LSTM/GRU
    """

    def __init__(self, hidden_size: int, num_layers: int):
        """
        Initialize recurrent-based model.

        Args:
            hidden_size: Hidden state dimension
            num_layers: Number of recurrent layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def get_embedding_dim(self) -> int:
        """Get the hidden state dimension."""
        return self.hidden_size

    @abstractmethod
    def get_hidden_states(self) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Get hidden states from the last forward pass.

        Returns:
            Tuple of hidden state tensors or None if not available.
            For LSTM: (h_n, c_n) where each is (num_layers, batch_size, hidden_size)
            For GRU: (h_n,) where h_n is (num_layers, batch_size, hidden_size)
        """
        pass
