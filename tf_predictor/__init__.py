"""
Generic FT-Transformer library for time series forecasting.

This library provides a reusable FT-Transformer implementation that can be
extended for any time series forecasting domain.
"""

from .core.predictor import TimeSeriesPredictor
from .core.ft_model import (
    FTTransformerPredictor,
    SequenceFTTransformerPredictor,
    FeatureTokenizer
)
from .core.utils import calculate_metrics, load_time_series_data, split_time_series
from .preprocessing.time_features import (
    create_date_features,
    create_cyclical_features,
    create_lag_features, 
    create_rolling_features,
    create_sequences
)

__version__ = "1.0.0"

__all__ = [
    "TimeSeriesPredictor",
    "FTTransformerPredictor", 
    "SequenceFTTransformerPredictor",
    "FeatureTokenizer",
    "calculate_metrics",
    "load_time_series_data",
    "split_time_series",
    "create_date_features",
    "create_cyclical_features",
    "create_lag_features",
    "create_rolling_features", 
    "create_sequences"
]