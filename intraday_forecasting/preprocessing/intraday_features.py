"""
Intraday-specific feature engineering for high-frequency trading data.

Creates intraday-specific features: volume and vwap.
Time-series features (cyclical encoding) and target shifting are handled by the base TimeSeriesPredictor class.
"""

import pandas as pd
import numpy as np
from typing import Optional


def create_intraday_features(df: pd.DataFrame,
                            timestamp_col: str = 'timestamp',
                            country: str = 'US',
                            timeframe: str = '5min',
                            verbose: bool = False,
                            group_column: Optional[str] = None) -> pd.DataFrame:
    """
    Create intraday-specific features: volume and vwap only.

    Time-series features (cyclical encoding) and target shifting are handled
    by the base TimeSeriesPredictor class.

    Args:
        df: DataFrame with OHLCV data and timestamp
        timestamp_col: Name of timestamp column
        country: Country code ('US', 'INDIA', or 'CRYPTO')
        timeframe: Trading timeframe ('1min', '5min', '15min', '1h')
        verbose: Whether to print feature creation steps
        group_column: Optional column for group-based scaling (e.g., 'symbol')

    Returns:
        DataFrame with intraday-specific features (volume + vwap)
    """
    if verbose:
        print(f"Creating intraday features for {len(df)} samples ({country} market)...")

    df_processed = df.copy()

    # 1. Keep volume as is (already in dataframe)
    if 'volume' not in df_processed.columns:
        raise ValueError("Volume column not found in dataframe")

    # 2. Add vwap = average(high, low, close) - only if OHLC columns exist
    if all(col in df_processed.columns for col in ['high', 'low', 'close']):
        typical_price = (df_processed['high'] + df_processed['low'] + df_processed['close']) / 3
        df_processed['vwap'] = typical_price
        if verbose:
            print("   Added vwap: (high + low + close) / 3")
    elif 'vwap' not in df_processed.columns:
        raise ValueError("Cannot create vwap: OHLC columns not found and vwap not already present")

    # 3. Handle missing values
    df_processed = df_processed.bfill().fillna(0)

    if verbose:
        # Count feature columns (exclude timestamp and group column)
        exclude_cols = [timestamp_col]
        if group_column and group_column in df_processed.columns:
            exclude_cols.append(group_column)
        feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
        print(f"  ✓ Created intraday features (volume + vwap)")
        print(f"  ✓ Total columns: {len(feature_cols)}")

    return df_processed
