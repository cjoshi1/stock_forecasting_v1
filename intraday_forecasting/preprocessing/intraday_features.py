"""
Intraday-specific feature engineering for high-frequency trading data.

Creates features that capture intraday patterns, market microstructure,
and time-of-day effects specific to minute/hourly trading data.
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import time

# Import base time series features
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tf_predictor.preprocessing.time_features import (
    create_cyclical_features, create_lag_features, create_rolling_features
)

# Import country market hours
from .timeframe_utils import get_country_market_hours


def create_intraday_time_features(df: pd.DataFrame, timestamp_col: str = 'timestamp', 
                                  country: str = 'US') -> pd.DataFrame:
    """
    Create intraday time-based features for specified country.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        country: Country code ('US' or 'INDIA')
        
    Returns:
        DataFrame with intraday time features added
    """
    df_processed = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_processed[timestamp_col]):
        df_processed[timestamp_col] = pd.to_datetime(df_processed[timestamp_col])
    
    # Get country-specific market hours
    market_config = get_country_market_hours(country)
    market_open = market_config['open']
    market_close = market_config['close']
    
    # Use centralized cyclical encoding for intraday features
    # This includes hour, minute, dayofweek and their cyclical encodings
    df_processed = create_cyclical_features(
        df_processed, timestamp_col, 
        include_features=['hour', 'minute', 'dayofweek']
    )
    
    # Time since market open (in minutes)
    market_open_today = (df_processed[timestamp_col].dt.normalize() + 
                        pd.Timedelta(hours=market_open.hour, minutes=market_open.minute))
    df_processed['minutes_since_open'] = (df_processed[timestamp_col] - market_open_today).dt.total_seconds() / 60
    
    # Time until market close (in minutes) 
    market_close_today = (df_processed[timestamp_col].dt.normalize() + 
                         pd.Timedelta(hours=market_close.hour, minutes=market_close.minute))
    df_processed['minutes_until_close'] = (market_close_today - df_processed[timestamp_col]).dt.total_seconds() / 60
    
    # Market session indicators
    df_processed['is_market_open'] = (
        (df_processed[timestamp_col].dt.time >= market_open) & 
        (df_processed[timestamp_col].dt.time < market_close)
    ).astype(int)
    
    # Country-specific intraday session periods
    if country == 'US':
        # US market sessions (9:30 AM - 4:00 PM ET)
        df_processed['is_opening_hour'] = (df_processed['hour'] == 9).astype(int)  # 9:30-10:30 AM
        df_processed['is_lunch_hour'] = (df_processed['hour'].between(12, 13)).astype(int)  # 12:00-2:00 PM
        df_processed['is_closing_hour'] = (df_processed['hour'] >= 15).astype(int)  # 3:00-4:00 PM
        df_processed['is_power_hour'] = (df_processed['hour'] == 15).astype(int)  # 3:00-4:00 PM (last hour)
    elif country == 'INDIA':
        # Indian market sessions (9:15 AM - 3:30 PM IST)
        df_processed['is_opening_hour'] = (df_processed['hour'] == 9).astype(int)  # 9:15-10:15 AM
        df_processed['is_lunch_hour'] = (df_processed['hour'].between(12, 13)).astype(int)  # 12:00-2:00 PM
        df_processed['is_closing_hour'] = (df_processed['hour'] >= 14).astype(int)  # 2:00-3:30 PM
        df_processed['is_power_hour'] = (df_processed['hour'] >= 14).astype(int)  # Last 1.5 hours
    
    return df_processed


def create_intraday_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create intraday price-based features.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with intraday price features added
    """
    df_processed = df.copy()
    
    # Basic price features
    df_processed['returns'] = df_processed['close'].pct_change()
    df_processed['log_returns'] = np.log(df_processed['close'] / df_processed['close'].shift(1))
    
    # Intraday price ranges and ratios
    df_processed['high_low_ratio'] = df_processed['high'] / df_processed['low']
    df_processed['close_open_ratio'] = df_processed['close'] / df_processed['open'] 
    df_processed['range_pct'] = (df_processed['high'] - df_processed['low']) / df_processed['open'] * 100
    
    # Volume-weighted average price proxy
    df_processed['typical_price'] = (df_processed['high'] + df_processed['low'] + df_processed['close']) / 3
    df_processed['volume_ratio'] = df_processed['volume'] / df_processed['volume'].rolling(window=20, min_periods=1).mean()
    
    # Intraday volatility (short-term)
    df_processed['volatility_5'] = df_processed['returns'].rolling(window=5, min_periods=1).std() * np.sqrt(5)
    df_processed['volatility_15'] = df_processed['returns'].rolling(window=15, min_periods=1).std() * np.sqrt(15)
    
    # Price momentum indicators
    df_processed['momentum_5'] = df_processed['close'] / df_processed['close'].shift(5) - 1
    df_processed['momentum_15'] = df_processed['close'] / df_processed['close'].shift(15) - 1
    
    # Moving averages (short-term for intraday)
    for window in [5, 10, 20]:
        df_processed[f'sma_{window}'] = df_processed['close'].rolling(window=window, min_periods=1).mean()
        df_processed[f'close_sma_{window}_ratio'] = df_processed['close'] / df_processed[f'sma_{window}']
    
    return df_processed


def create_intraday_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create intraday volume and market microstructure features.
    
    Args:
        df: DataFrame with volume data
        
    Returns:
        DataFrame with volume features added
    """
    df_processed = df.copy()
    
    # Volume-based features
    df_processed['volume_sma_5'] = df_processed['volume'].rolling(window=5, min_periods=1).mean()
    df_processed['volume_sma_20'] = df_processed['volume'].rolling(window=20, min_periods=1).mean()
    df_processed['volume_momentum'] = df_processed['volume'] / (df_processed['volume'].shift(5) + 1e-8)
    
    # Volume rate (volume per minute equivalent)
    df_processed['volume_rate'] = df_processed['volume']  # Already per-period volume
    
    # Price-volume relationship
    df_processed['price_volume_trend'] = np.where(
        df_processed['returns'] > 0, 
        df_processed['volume'], 
        -df_processed['volume']
    )
    
    # Volume percentiles (relative volume strength)
    df_processed['volume_percentile'] = df_processed['volume'].rolling(window=50, min_periods=10).rank(pct=True) * 100
    
    return df_processed


def create_open_interest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create open interest features (if available).
    
    Args:
        df: DataFrame with open_interest column
        
    Returns:
        DataFrame with OI features added
    """
    if 'open_interest' not in df.columns:
        return df
    
    df_processed = df.copy()
    
    # Open interest changes
    df_processed['oi_change'] = df_processed['open_interest'].diff()
    df_processed['oi_pct_change'] = df_processed['open_interest'].pct_change()
    
    # OI to volume ratio
    df_processed['oi_volume_ratio'] = df_processed['open_interest'] / (df_processed['volume'] + 1e-8)
    
    # OI momentum
    df_processed['oi_momentum_5'] = (
        df_processed['open_interest'] / df_processed['open_interest'].shift(5) - 1
    )
    
    return df_processed


def create_intraday_features(df: pd.DataFrame, target_column: str = 'close', 
                           timestamp_col: str = 'timestamp', country: str = 'US',
                           verbose: bool = False) -> pd.DataFrame:
    """
    Create comprehensive intraday feature set for specified country.
    
    Args:
        df: DataFrame with intraday OHLCV data
        target_column: Target column name
        timestamp_col: Timestamp column name
        country: Country code ('US' or 'INDIA')  
        verbose: Whether to print feature creation steps
        
    Returns:
        DataFrame with all intraday features
    """
    if verbose:
        print(f"Creating intraday features for {len(df)} samples ({country} market)...")
    
    df_processed = df.copy()
    
    # 1. Time-based features
    if verbose:
        print("  ✓ Creating time-based features...")
    df_processed = create_intraday_time_features(df_processed, timestamp_col, country)
    
    # 2. Price features
    if verbose:
        print("  ✓ Creating price-based features...")
    df_processed = create_intraday_price_features(df_processed)
    
    # 3. Volume features
    if verbose:
        print("  ✓ Creating volume features...")
    df_processed = create_intraday_volume_features(df_processed)
    
    # 4. Open interest features (if available)
    if 'open_interest' in df_processed.columns:
        if verbose:
            print("  ✓ Creating open interest features...")
        df_processed = create_open_interest_features(df_processed)
    
    # 5. Lag features (shorter lags for intraday)
    if verbose:
        print("  ✓ Creating lag features...")
    lag_periods = [1, 3, 5, 10, 15]
    df_processed = create_lag_features(df_processed, 'close', lag_periods)
    df_processed = create_lag_features(df_processed, 'volume', [1, 5])
    df_processed = create_lag_features(df_processed, 'returns', [1, 3, 5])
    
    # 6. Rolling statistics (shorter windows for intraday)
    if verbose:
        print("  ✓ Creating rolling statistics...")
    rolling_windows = [5, 10, 20]
    df_processed = create_rolling_features(df_processed, 'close', rolling_windows)
    df_processed = create_rolling_features(df_processed, 'volume', rolling_windows)
    
    # 7. Additional percentage changes (shorter periods for intraday)
    if verbose:
        print("  ✓ Creating additional percentage change features...")
    # Additional percentage change periods (beyond what's already created in price features)
    additional_pct_periods = [10, 15, 30]  # 10min to 30min changes
    for period in additional_pct_periods:
        if len(df_processed) > period:
            df_processed[f'pct_change_{period}d'] = (
                (df_processed['close'] - df_processed['close'].shift(period)) / 
                df_processed['close'].shift(period)
            ) * 100
    
    # Handle missing values
    df_processed = df_processed.bfill().fillna(0)
    
    if verbose:
        feature_count = len([col for col in df_processed.columns if col != timestamp_col])
        print(f"  ✓ Created {feature_count} total features")
    
    return df_processed