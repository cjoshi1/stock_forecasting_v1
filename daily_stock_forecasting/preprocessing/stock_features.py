"""
Stock-specific feature engineering for OHLCV data.

Performs basic preprocessing (NaN handling).
Time-series features are handled by the base TimeSeriesPredictor class.
"""

import pandas as pd
import numpy as np


def create_stock_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Create stock-specific features.

    Time-series features (cyclical encoding) and target shifting are now handled
    by the base TimeSeriesPredictor class.

    Args:
        df: DataFrame with OHLCV data
        verbose: Whether to print verbose information

    Returns:
        processed_df: DataFrame with basic preprocessing applied
    """
    df_processed = df.copy()

    # Fill NaN values
    df_processed = df_processed.bfill().fillna(0)

    return df_processed
