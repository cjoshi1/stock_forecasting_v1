"""
Stock-specific feature engineering for OHLCV data.

Creates domain-specific features: vwap (typical_price).
Time-series features are handled by the base TimeSeriesPredictor class.
"""

import pandas as pd
import numpy as np


def create_stock_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Create stock-specific features: vwap only.

    Time-series features (cyclical encoding) and target shifting are now handled
    by the base TimeSeriesPredictor class.

    Args:
        df: DataFrame with OHLCV data
        verbose: Whether to print verbose information

    Returns:
        processed_df: DataFrame with vwap feature added
    """
    df_processed = df.copy()

    # Calculate vwap (typical_price) if we have OHLC data
    if all(col in df_processed.columns for col in ['high', 'low', 'close']):
        df_processed['vwap'] = (df_processed['high'] + df_processed['low'] + df_processed['close']) / 3
        if verbose:
            print("   Added vwap: (high + low + close) / 3")

    # Fill NaN values
    df_processed = df_processed.bfill().fillna(0)

    return df_processed
