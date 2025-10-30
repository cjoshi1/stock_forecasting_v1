"""
Scaler Factory for Time Series Preprocessing

Provides a factory pattern for creating different types of scalers
for feature and target normalization.
"""

from typing import Any
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler
)
from sklearn.base import BaseEstimator, TransformerMixin


class OnlyMaxScaler(BaseEstimator, TransformerMixin):
    """
    Scale features by dividing by maximum value only (no shifting).

    Formula: X_scaled = X / X_max

    This keeps the original data distribution but scales it so the maximum
    value becomes 1. Unlike MinMaxScaler, this does NOT shift the minimum to 0.

    Attributes:
        data_max_: Maximum value per feature (fitted)

    Examples:
        >>> scaler = OnlyMaxScaler()
        >>> # Data [5, 10, 15] -> [5/15, 10/15, 15/15] = [0.33, 0.67, 1.0]
        >>> # Min stays at 0.33, max becomes 1.0
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Compute the maximum for later scaling."""
        X = np.asarray(X)
        self.data_max_ = np.max(X, axis=0)

        # Handle constant features or zero max
        self.data_max_[self.data_max_ == 0] = 1.0

        return self

    def transform(self, X):
        """Scale features by dividing by max."""
        X = np.asarray(X)
        X_scaled = X / self.data_max_
        return X_scaled

    def inverse_transform(self, X):
        """Undo the scaling to original range."""
        X = np.asarray(X)
        X_original = X * self.data_max_
        return X_original


class ScalerFactory:
    """
    Factory for creating sklearn scalers.

    Supports:
    - 'standard': StandardScaler (mean=0, std=1)
    - 'minmax': MinMaxScaler (range [0, 1] or custom)
    - 'robust': RobustScaler (median and IQR, robust to outliers)
    - 'maxabs': MaxAbsScaler (range [-1, 1], preserves sparsity)
    - 'onlymax': OnlyMaxScaler (divide by max only, no shifting)

    Usage:
        >>> scaler = ScalerFactory.create_scaler('standard')
        >>> scaler = ScalerFactory.create_scaler('minmax', feature_range=(-1, 1))
        >>> scaler = ScalerFactory.create_scaler('robust', quantile_range=(25, 75))
        >>> scaler = ScalerFactory.create_scaler('onlymax')
    """

    _SCALER_REGISTRY = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler,
        'maxabs': MaxAbsScaler,
        'onlymax': OnlyMaxScaler,
    }

    @classmethod
    def create_scaler(cls, scaler_type: str = 'standard', **kwargs) -> Any:
        """
        Create a scaler instance.

        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust', 'maxabs')
            **kwargs: Additional arguments passed to the scaler constructor

        Returns:
            Scaler instance

        Raises:
            ValueError: If scaler_type is not recognized

        Examples:
            >>> # Standard scaler (default)
            >>> scaler = ScalerFactory.create_scaler('standard')

            >>> # MinMax scaler with custom range
            >>> scaler = ScalerFactory.create_scaler('minmax', feature_range=(-1, 1))

            >>> # Robust scaler with custom quantile range
            >>> scaler = ScalerFactory.create_scaler('robust', quantile_range=(10, 90))

            >>> # MaxAbs scaler (preserves zeros, good for sparse data)
            >>> scaler = ScalerFactory.create_scaler('maxabs')

            >>> # OnlyMax scaler (divide by max only)
            >>> scaler = ScalerFactory.create_scaler('onlymax')
        """
        scaler_type = scaler_type.lower()

        if scaler_type not in cls._SCALER_REGISTRY:
            available = ', '.join(cls._SCALER_REGISTRY.keys())
            raise ValueError(
                f"Unknown scaler type '{scaler_type}'. "
                f"Available options: {available}"
            )

        scaler_class = cls._SCALER_REGISTRY[scaler_type]
        return scaler_class(**kwargs)

    @classmethod
    def get_available_scalers(cls) -> list:
        """Return list of available scaler types."""
        return list(cls._SCALER_REGISTRY.keys())

    @classmethod
    def register_scaler(cls, name: str, scaler_class: type):
        """
        Register a custom scaler type.

        Args:
            name: Name for the scaler
            scaler_class: Scaler class (must have fit/transform/inverse_transform methods)
        """
        cls._SCALER_REGISTRY[name.lower()] = scaler_class


# Usage notes for different scaler types
SCALER_USE_CASES = {
    'standard': {
        'description': 'Standardize features by removing mean and scaling to unit variance',
        'when_to_use': [
            'Default choice for most cases',
            'When features are roughly Gaussian distributed',
            'When you want zero mean and unit variance',
        ],
        'pros': ['Works well with most ML algorithms', 'Preserves outliers (can be good or bad)'],
        'cons': ['Sensitive to outliers', 'Not bounded to specific range'],
    },
    'minmax': {
        'description': 'Scale features to a given range (default [0, 1])',
        'when_to_use': [
            'When you need features in a specific range',
            'For neural networks with bounded activations',
            'When relative distances matter',
        ],
        'pros': ['Bounded output range', 'Preserves zero values if range includes 0'],
        'cons': ['Very sensitive to outliers', 'Can compress most values to small range'],
    },
    'robust': {
        'description': 'Scale using median and IQR (Interquartile Range)',
        'when_to_use': [
            'When data has many outliers',
            'For financial or sensor data with anomalies',
            'When you want robust statistics',
        ],
        'pros': ['Robust to outliers', 'Uses percentiles instead of mean/std'],
        'cons': ['Not bounded to specific range', 'Slower than StandardScaler'],
    },
    'maxabs': {
        'description': 'Scale by maximum absolute value to range [-1, 1]',
        'when_to_use': [
            'For sparse data (preserves sparsity)',
            'When data is already centered at zero',
            'When you want [-1, 1] range',
        ],
        'pros': ['Preserves sparsity', 'Simple and fast', 'Symmetric around 0'],
        'cons': ['Sensitive to outliers', 'Assumes data is centered'],
    },
    'onlymax': {
        'description': 'Scale by dividing by maximum value only (no shifting)',
        'when_to_use': [
            'When you want to preserve the original data distribution',
            'For data where minimum should NOT be shifted to 0',
            'When you need simple max-based normalization',
        ],
        'pros': ['Preserves original distribution shape', 'Simple divide-by-max operation', 'No shifting of minimum'],
        'cons': ['Sensitive to outliers', 'Minimum not normalized to specific value'],
    },
}
