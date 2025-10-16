"""
Intraday-specific feature engineering for high-frequency trading data.

Creates essential features for intraday patterns: volume, vwap, and cyclical time encodings.
"""

import pandas as pd
import numpy as np
from typing import Optional

# Import base time series features
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tf_predictor.preprocessing.time_features import create_cyclical_features


def create_intraday_features(df: pd.DataFrame, target_column = 'close',
                            timestamp_col: str = 'timestamp', country: str = 'US',
                            timeframe: str = '5min', prediction_horizon: int = 1,
                            verbose: bool = False, group_column: Optional[str] = None) -> pd.DataFrame:
    """
    Create intraday feature set with volume, vwap, and cyclical time features.

    Args:
        df: DataFrame with OHLCV data and timestamp
        target_column: Name(s) of target column(s) to predict
                      - str: Single target (e.g., 'close')
                      - List[str]: Multiple targets (e.g., ['close', 'volume'])
        timestamp_col: Name of timestamp column
        country: Country code ('US', 'INDIA', or 'CRYPTO')
        timeframe: Trading timeframe ('1min', '5min', '15min', '1h')
        prediction_horizon: Number of steps ahead to predict (1=single, >1=multi-horizon)
        verbose: Whether to print feature creation steps
        group_column: Optional column for group-based scaling (e.g., 'symbol')

    Returns:
        DataFrame with features
    """
    # Normalize target_column to list
    if isinstance(target_column, str):
        target_columns_list = [target_column]
    else:
        target_columns_list = list(target_column)

    if verbose:
        horizon_text = f"{prediction_horizon}-step" if prediction_horizon > 1 else "single-step"
        targets_text = ', '.join(target_columns_list)
        print(f"Creating essential intraday features for {len(df)} samples ({country} market, {horizon_text})...")
        print(f"  Target variable(s): {targets_text}")

    df_processed = df.copy()

    # 1. Keep volume as is (already in dataframe)
    if 'volume' not in df_processed.columns:
        raise ValueError("Volume column not found in dataframe")

    # 2. Add vwap = average(high, low, close) - only if OHLC columns exist
    if all(col in df_processed.columns for col in ['high', 'low', 'close']):
        typical_price = (df_processed['high'] + df_processed['low'] + df_processed['close']) / 3
        df_processed['vwap'] = typical_price
    elif 'vwap' not in df_processed.columns:
        raise ValueError("Cannot create vwap: OHLC columns not found and vwap not already present")

    # 3. Add cyclical time features
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



    # 4. Create future target column(s) for each target variable
    # Validate all target columns exist
    for target_col in target_columns_list:
        if target_col not in df_processed.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe. Available: {list(df_processed.columns)}")

    all_shifted_targets = []

    for target_col in target_columns_list:
        if prediction_horizon == 1:
            # Single horizon: create one target column
            shifted_target_name = f"{target_col}_target_h1"
            if group_column and group_column in df_processed.columns:
                df_processed[shifted_target_name] = df_processed.groupby(group_column)[target_col].shift(-1)
            else:
                df_processed[shifted_target_name] = df_processed[target_col].shift(-1)
            all_shifted_targets.append(shifted_target_name)
        else:
            # Multi-horizon: create multiple target columns
            for h in range(1, prediction_horizon + 1):
                col_name = f"{target_col}_target_h{h}"
                if group_column and group_column in df_processed.columns:
                    df_processed[col_name] = df_processed.groupby(group_column)[target_col].shift(-h)
                else:
                    df_processed[col_name] = df_processed[target_col].shift(-h)
                all_shifted_targets.append(col_name)

    # Drop rows where ANY target is NaN
    df_processed = df_processed.dropna(subset=all_shifted_targets)



    # Handle missing values
    df_processed = df_processed.bfill().fillna(0)

    if verbose:
        total_features = len(features)
        print(f"  ✓ Creating essential features (volume + vwap + cyclical time)...")
        print(f"  ✓ Created {total_features} total features")

    return df_processed
