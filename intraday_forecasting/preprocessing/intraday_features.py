"""
Intraday-specific feature engineering for high-frequency trading data.

Performs basic preprocessing (NaN handling).
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
    Create intraday-specific features (flexible - only creates features if source columns exist).

    Time-series features (cyclical encoding) and target shifting are handled
    by the base TimeSeriesPredictor class.

    Args:
        df: DataFrame with timestamp and numeric columns
        timestamp_col: Name of timestamp column
        country: Country code ('US', 'INDIA', or 'CRYPTO')
        timeframe: Trading timeframe ('1min', '5min', '15min', '1h')
        verbose: Whether to print feature creation steps
        group_column: Optional column for group-based scaling (e.g., 'symbol')

    Returns:
        DataFrame with basic preprocessing applied
    """
    if verbose:
        print(f"Creating intraday features for {len(df)} samples ({country} market)...")

    df_processed = df.copy()

    # Handle missing values
    df_processed = df_processed.bfill().fillna(0)

    if verbose:
        # Count feature columns (exclude timestamp and group column)
        exclude_cols = [timestamp_col]
        if group_column and group_column in df_processed.columns:
            exclude_cols.append(group_column)
        feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
        print(f"  ✓ Basic preprocessing completed")
        print(f"  ✓ Total columns: {len(feature_cols)}")

    return df_processed
