"""
Stock-specific feature engineering for OHLCV data.

Creates essential features: volume, vwap (typical_price), and cyclical time encodings.
"""

import pandas as pd
import numpy as np
from tf_predictor.preprocessing.time_features import create_cyclical_features


def create_stock_features(df: pd.DataFrame, target_column, verbose: bool = False,
                         prediction_horizon: int = 1, asset_type: str = 'stock',
                         group_column: str = None) -> pd.DataFrame:
    """
    Create stock-specific features: volume, vwap, and seasonal features.

    Args:
        df: DataFrame with OHLCV data and optional date column
        target_column: Target column name(s)
                      - str: Single target (e.g., 'close')
                      - List[str]: Multiple targets (e.g., ['close', 'volume'])
        verbose: Whether to print verbose information
        prediction_horizon: Number of time steps to shift target (1 = predict next step)
        asset_type: Type of asset - 'stock' (5-day week) or 'crypto' (7-day week)
        group_column: Optional column for grouping (e.g., 'symbol' for multi-stock datasets)

    Returns:
        processed_df: DataFrame with engineered features and shifted target
    """
    # Normalize target_column to list
    if isinstance(target_column, str):
        target_columns_list = [target_column]
    else:
        target_columns_list = list(target_column)

    df_processed = df.copy()

    # 1. Calculate vwap (typical_price) if we have OHLC data
    if all(col in df_processed.columns for col in ['high', 'low', 'close']):
        df_processed['vwap'] = (df_processed['high'] + df_processed['low'] + df_processed['close']) / 3
        if verbose:
            print("   Added vwap: (high + low + close) / 3")

    # 2. Extract seasonal features from date column using tf_predictor
    if 'date' in df_processed.columns:
        # Use tf_predictor's cyclical features function with the date column
        df_processed = create_cyclical_features(df_processed, 'date', ['month', 'dayofweek'])


        if verbose:
            print("   Added cyclical seasonal features: month_sin, month_cos, dayofweek_sin, dayofweek_cos")
            if asset_type == 'crypto':
                print("   Crypto mode: dayofweek uses 7-day week (0=Sunday, 6=Saturday)")
            else:
                print("   Stock mode: dayofweek uses 5-day week (0=Monday, 4=Friday)")
            print("   Removed original date column after extraction")



    # Fill NaN values
    df_processed = df_processed.bfill().fillna(0)

    # 4. Create shifted target variables for each target column
    df_processed = _create_shifted_target(df_processed, target_columns_list, prediction_horizon, verbose)

    if verbose:
        num_features = len([f for f in available_features if f not in target_columns_list and f != group_column])
        print(f"   Created {num_features} features")

    return df_processed


def _create_shifted_target(df: pd.DataFrame, target_column, prediction_horizon: int, verbose: bool = False) -> pd.DataFrame:
    """
    Create shifted target variable(s) for time series prediction.
    Supports both single and multiple target variables, with single or multi-horizon prediction.

    Args:
        df: DataFrame with features
        target_column: Name(s) of the original target column(s)
                      - str: Single target
                      - List[str]: Multiple targets
        prediction_horizon: Number of steps to predict (1 = single, >1 = multi-horizon)
        verbose: Whether to print information

    Returns:
        df: DataFrame with shifted target column(s)
    """
    df = df.copy()

    # Normalize target_column to list
    if isinstance(target_column, str):
        target_columns_list = [target_column]
    else:
        target_columns_list = list(target_column)

    # Validate all target columns exist
    for target_col in target_columns_list:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe. Available: {list(df.columns)}")

    all_shifted_targets = []

    # Create shifted targets for each target variable
    for target_col in target_columns_list:
        if prediction_horizon == 1:
            # Single horizon
            shifted_target_name = f"{target_col}_target_h1"
            df[shifted_target_name] = df[target_col].shift(-1)
            all_shifted_targets.append(shifted_target_name)
        else:
            # Multi-horizon
            for h in range(1, prediction_horizon + 1):
                col_name = f"{target_col}_target_h{h}"
                df[col_name] = df[target_col].shift(-h)
                all_shifted_targets.append(col_name)

    # Remove rows where ANY target is NaN
    df = df.dropna(subset=all_shifted_targets)

    if verbose:
        targets_text = ', '.join(target_columns_list)
        if prediction_horizon == 1:
            print(f"   Created single-horizon targets for: {targets_text}")
            print(f"   Prediction horizon: 1 step ahead")
        else:
            print(f"   Created multi-horizon targets for: {targets_text}")
            print(f"   Prediction horizons: 1 to {prediction_horizon} steps ahead")
        print(f"   Remaining samples after shift: {len(df)}")

    return df
