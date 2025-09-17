"""
Intraday-specific feature engineering for high-frequency trading data.

Creates features that capture intraday patterns, market microstructure,
and time-of-day effects specific to minute/hourly trading data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
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


def categorize_features(all_features: List[str], essential_features: List[str]) -> Dict[str, List[str]]:
    """
    Categorize features into essential and non-essential.

    Args:
        all_features: List of all feature column names
        essential_features: List of essential feature column names

    Returns:
        Dictionary with 'essential' and 'non_essential' feature lists
    """
    non_essential = [f for f in all_features if f not in essential_features]
    return {
        'essential': essential_features,
        'non_essential': non_essential
    }


def create_essential_features(df: pd.DataFrame, timestamp_col: str = 'timestamp', timeframe: str = '5min') -> pd.DataFrame:
    """
    Create essential features: volume + lnvwap + cyclical time features.

    Args:
        df: DataFrame with OHLCV data and timestamp
        timestamp_col: Name of timestamp column
        timeframe: Trading timeframe ('1min', '5min', '15min', '1h')

    Returns:
        DataFrame with essential features added
    """
    df_processed = df.copy()

    # 1. Keep volume as is (already essential)
    # Volume is already in df, no need to create

    # 2. Add lnvwap = ln(average(high, low, close)) - only if OHLC columns exist
    if all(col in df_processed.columns for col in ['high', 'low', 'close']):
        typical_price = (df_processed['high'] + df_processed['low'] + df_processed['close']) / 3
        df_processed['lnvwap'] = np.log(typical_price)
    elif 'lnvwap' not in df_processed.columns:
        raise ValueError("Cannot create lnvwap: OHLC columns not found and lnvwap not already present")

    # 3. Add essential cyclical time features by calling tf_predictor
    # For hourly timeframe, exclude minute features (always minute=0)
    if timeframe == '1h':
        include_features = ['hour', 'dayofweek']
    else:
        include_features = ['minute', 'hour', 'dayofweek']

    df_processed = create_cyclical_features(
        df_processed,
        timestamp_col,
        include_features=include_features
    )

    # Keep only the cyclical encodings, drop raw time features
    time_features_to_drop = ['minute', 'hour', 'dayofweek', 'is_weekend']
    df_processed = df_processed.drop(columns=[col for col in time_features_to_drop if col in df_processed.columns])

    # 4. Create future target column (lnvwap shifted by +1 timestamp)
    df_processed['lnvwap_target'] = df_processed['lnvwap'].shift(-1)

    # Keep only essential features + timestamp + target (conditional on timeframe)
    if timeframe == '1h':
        essential_features = ['volume', 'lnvwap', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']
    else:
        essential_features = ['volume', 'lnvwap', 'minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']

    columns_to_keep = [timestamp_col] + essential_features + ['lnvwap_target']
    df_processed = df_processed[columns_to_keep]

    return df_processed


def create_intraday_time_features(df: pd.DataFrame, timestamp_col: str = 'timestamp',
                                  country: str = 'US', return_categories: bool = False) -> Tuple[pd.DataFrame, Dict[str, List[str]]] | pd.DataFrame:
    """
    Create intraday time-based features for specified country.

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        country: Country code ('US' or 'INDIA')
        return_categories: If True, return feature categories along with DataFrame

    Returns:
        DataFrame with intraday time features added (and optionally feature categories dict)
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

    if return_categories:
        new_features = [col for col in df_processed.columns if col not in df.columns]
        categories = categorize_features(new_features, [])  # All time features are non-essential
        return df_processed, categories

    return df_processed


def create_intraday_price_features(df: pd.DataFrame, return_categories: bool = False) -> Tuple[pd.DataFrame, Dict[str, List[str]]] | pd.DataFrame:
    """
    Create intraday price-based features.

    Args:
        df: DataFrame with OHLCV data
        return_categories: If True, return feature categories along with DataFrame

    Returns:
        DataFrame with intraday price features added (and optionally feature categories dict)
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

    if return_categories:
        new_features = [col for col in df_processed.columns if col not in df.columns]
        categories = categorize_features(new_features, [])  # All price features are non-essential
        return df_processed, categories

    return df_processed


def create_intraday_volume_features(df: pd.DataFrame, return_categories: bool = False) -> Tuple[pd.DataFrame, Dict[str, List[str]]] | pd.DataFrame:
    """
    Create intraday volume and market microstructure features.

    Args:
        df: DataFrame with volume data
        return_categories: If True, return feature categories along with DataFrame

    Returns:
        DataFrame with volume features added (and optionally feature categories dict)
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

    if return_categories:
        new_features = [col for col in df_processed.columns if col not in df.columns]
        # Basic volume features are essential, advanced ones are non-essential
        essential = [f for f in new_features if f in ['volume_sma_5', 'volume_momentum']]
        categories = categorize_features(new_features, essential)
        return df_processed, categories

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
                           timeframe: str = '5min', verbose: bool = False, use_essential_only: bool = True,
                           return_categories: bool = False) -> Tuple[pd.DataFrame, Dict[str, List[str]]] | pd.DataFrame:
    """
    Create comprehensive intraday feature set for specified country.

    Args:
        df: DataFrame with intraday OHLCV data
        target_column: Target column name
        timestamp_col: Timestamp column name
        country: Country code ('US' or 'INDIA')
        timeframe: Trading timeframe ('1min', '5min', '15min', '1h')
        verbose: Whether to print feature creation steps
        use_essential_only: If True, only create essential features (volume + lnvwap + cyclical time)
        return_categories: If True, return feature categories along with DataFrame

    Returns:
        DataFrame with features (and optionally feature categories dict)
    """
    if verbose:
        feature_type = "essential" if use_essential_only else "all"
        print(f"Creating {feature_type} intraday features for {len(df)} samples ({country} market)...")

    df_processed = df.copy()
    all_categories = {'essential': [], 'non_essential': []}

    if use_essential_only:
        # Only create essential features
        if verbose:
            print("  ✓ Creating essential features (volume + lnvwap + cyclical time)...")
        df_processed = create_essential_features(df_processed, timestamp_col, timeframe)
        # Essential features depend on timeframe
        if timeframe == '1h':
            all_categories['essential'] = ['volume', 'lnvwap', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']
        else:
            all_categories['essential'] = ['volume', 'lnvwap', 'minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']
    else:
        # Create all features (existing logic)
        # 1. Essential features first
        if verbose:
            print("  ✓ Creating essential features...")
        df_processed = create_essential_features(df_processed, timestamp_col, timeframe)
        # Essential features depend on timeframe
        if timeframe == '1h':
            all_categories['essential'] = ['volume', 'lnvwap', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']
        else:
            all_categories['essential'] = ['volume', 'lnvwap', 'minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']

        # 2. Time-based features
        if verbose:
            print("  ✓ Creating time-based features...")
        df_processed, time_cats = create_intraday_time_features(df_processed, timestamp_col, country, return_categories=True)
        all_categories['non_essential'].extend(time_cats['non_essential'])

        # 3. Price features
        if verbose:
            print("  ✓ Creating price-based features...")
        df_processed, price_cats = create_intraday_price_features(df_processed, return_categories=True)
        all_categories['non_essential'].extend(price_cats['non_essential'])

        # 4. Volume features
        if verbose:
            print("  ✓ Creating volume features...")
        df_processed, vol_cats = create_intraday_volume_features(df_processed, return_categories=True)
        all_categories['essential'].extend(vol_cats['essential'])
        all_categories['non_essential'].extend(vol_cats['non_essential'])

        # 5. Open interest features (if available)
        if 'open_interest' in df_processed.columns:
            if verbose:
                print("  ✓ Creating open interest features...")
            df_processed = create_open_interest_features(df_processed)
            # All OI features are non-essential
            oi_features = [col for col in df_processed.columns if 'oi_' in col or 'open_interest' in col]
            all_categories['non_essential'].extend(oi_features)

        # 6. Lag features (shorter lags for intraday)
        if verbose:
            print("  ✓ Creating lag features...")
        lag_periods = [1, 3, 5, 10, 15]
        # Create lag features and categorize them manually
        original_columns = df_processed.columns.tolist()
        df_processed = create_lag_features(df_processed, 'close', lag_periods)
        df_processed = create_lag_features(df_processed, 'volume', [1, 5])
        df_processed = create_lag_features(df_processed, 'returns', [1, 3, 5])

        # Categorize lag features
        lag_features = [col for col in df_processed.columns if col not in original_columns]
        volume_lag_features = [f for f in lag_features if 'volume_lag' in f]
        other_lag_features = [f for f in lag_features if f not in volume_lag_features]

        all_categories['essential'].extend(volume_lag_features)
        all_categories['non_essential'].extend(other_lag_features)

        # 7. Rolling statistics (shorter windows for intraday)
        if verbose:
            print("  ✓ Creating rolling statistics...")
        rolling_windows = [5, 10, 20]

        # Create rolling features and categorize them manually
        original_columns = df_processed.columns.tolist()
        df_processed = create_rolling_features(df_processed, 'close', rolling_windows)
        df_processed = create_rolling_features(df_processed, 'volume', rolling_windows)

        # Categorize rolling features (all are non-essential)
        rolling_features = [col for col in df_processed.columns if col not in original_columns]
        all_categories['non_essential'].extend(rolling_features)

        # 8. Additional percentage changes
        if verbose:
            print("  ✓ Creating additional percentage change features...")
        additional_pct_periods = [10, 15, 30]
        pct_features = []
        for period in additional_pct_periods:
            if len(df_processed) > period:
                feature_name = f'pct_change_{period}d'
                df_processed[feature_name] = (
                    (df_processed['close'] - df_processed['close'].shift(period)) /
                    df_processed['close'].shift(period)
                ) * 100
                pct_features.append(feature_name)
        all_categories['non_essential'].extend(pct_features)

    # Handle missing values
    df_processed = df_processed.bfill().fillna(0)

    if verbose:
        essential_count = len(all_categories['essential'])
        non_essential_count = len(all_categories['non_essential'])
        total_features = essential_count + non_essential_count
        print(f"  ✓ Created {total_features} total features ({essential_count} essential, {non_essential_count} non-essential)")

    if return_categories:
        return df_processed, all_categories

    return df_processed