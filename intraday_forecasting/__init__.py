"""
Intraday Forecasting with FT-Transformer

A specialized application for high-frequency trading forecasting using the 
FT-Transformer (Feature Tokenizer Transformer) architecture. This package 
extends the generic `tf_predictor` library with intraday-specific features 
and workflows.

Features:
- Multiple timeframes: 1min, 5min, 15min, 1h
- Market hours filtering
- Intraday pattern recognition
- Volume and microstructure features  
- Time-of-day effects modeling
"""

from .predictor import IntradayPredictor
from .preprocessing.market_data import (
    load_intraday_data,
    create_sample_intraday_data,
    prepare_intraday_for_training,
    validate_intraday_data
)
from .preprocessing.intraday_features import (
    create_intraday_features,
    create_intraday_time_features,
    create_intraday_price_features,
    create_intraday_volume_features
)
from .preprocessing.timeframe_utils import (
    filter_market_hours,
    resample_ohlcv,
    get_timeframe_config,
    get_supported_timeframes,
    get_country_market_hours,
    prepare_intraday_data
)

__version__ = "1.0.0"

__all__ = [
    # Core predictor
    "IntradayPredictor",
    
    # Data loading and preparation
    "load_intraday_data",
    "create_sample_intraday_data", 
    "prepare_intraday_for_training",
    "validate_intraday_data",
    
    # Feature engineering
    "create_intraday_features",
    "create_intraday_time_features",
    "create_intraday_price_features", 
    "create_intraday_volume_features",
    
    # Timeframe utilities
    "filter_market_hours",
    "resample_ohlcv",
    "get_timeframe_config",
    "get_supported_timeframes",
    "get_country_market_hours",
    "prepare_intraday_data"
]