"""
Generic time series preprocessing utilities.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def create_cyclical_features(df: pd.DataFrame, datetime_col: str, 
                            include_features: list = None) -> pd.DataFrame:
    """
    Create cyclical encoding features from datetime column.
    
    Args:
        df: DataFrame with datetime column
        datetime_col: Name of the datetime column  
        include_features: List of features to include. Options: 
                         ['year', 'month', 'day', 'hour', 'minute', 'dayofweek', 'quarter']
                         If None, includes month and dayofweek (default behavior)
        
    Returns:
        df: DataFrame with cyclical encoding features added
    """
    df = df.copy()
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Default features for backward compatibility
    if include_features is None:
        include_features = ['month', 'dayofweek']
    
    # Extract base temporal features first
    if 'year' in include_features:
        df['year'] = df[datetime_col].dt.year
    if 'month' in include_features:
        df['month'] = df[datetime_col].dt.month
    if 'day' in include_features:
        df['day'] = df[datetime_col].dt.day
    if 'hour' in include_features:
        df['hour'] = df[datetime_col].dt.hour
    if 'minute' in include_features:
        df['minute'] = df[datetime_col].dt.minute
    if 'dayofweek' in include_features:
        df['dayofweek'] = df[datetime_col].dt.dayofweek
    if 'quarter' in include_features:
        df['quarter'] = df[datetime_col].dt.quarter
    
    # Create cyclical encodings
    if 'month' in include_features:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    if 'dayofweek' in include_features:
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        # Add weekend indicator (Saturday=5, Sunday=6)
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    if 'hour' in include_features:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    if 'minute' in include_features:
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    
    if 'day' in include_features:
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)  # Approximate for month days
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    if 'quarter' in include_features:
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
    return df


def create_date_features(df: pd.DataFrame, date_column: str, group_column: str = None) -> pd.DataFrame:
    """
    Create temporal features from a date column.

    Args:
        df: DataFrame with date column
        date_column: Name of the date column
        group_column: Optional column for group-based sorting (e.g., 'symbol' for multi-stock datasets)
                     When provided, data is sorted by group first, then by date within each group

    Returns:
        df: DataFrame with additional date features
    """
    df = df.copy()

    # Convert to datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Sort data chronologically
    if group_column and group_column in df.columns:
        # Group-based sorting: sort by group, then by date within each group
        is_sorted = (df.groupby(group_column)[date_column]
                     .apply(lambda x: x.is_monotonic_increasing)
                     .all())

        if not is_sorted:
            df = df.sort_values([group_column, date_column]).reset_index(drop=True)
            print(f"   Sorted data by '{group_column}' and '{date_column}' to ensure temporal order within groups")
        else:
            print(f"   Data already chronologically sorted within groups by '{date_column}'")
    else:
        # Global sorting: sort by date only
        is_sorted = df[date_column].is_monotonic_increasing

        if not is_sorted:
            df = df.sort_values(date_column).reset_index(drop=True)
            print(f"   Sorted data by date: {df[date_column].iloc[0]} to {df[date_column].iloc[-1]}")
        else:
            print(f"   Data already chronologically sorted: {df[date_column].iloc[0]} to {df[date_column].iloc[-1]}")

    # Extract basic date features
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['quarter'] = df[date_column].dt.quarter

    # Use the new cyclical encoding function (includes dayofweek and is_weekend)
    df = create_cyclical_features(df, date_column, ['month', 'dayofweek'])

    return df


def create_rolling_features(df: pd.DataFrame, column: str, windows: list) -> pd.DataFrame:
    """
    Create rolling window features for time series.
    
    Args:
        df: DataFrame with time series data
        column: Column to create rolling features for
        windows: List of window sizes (e.g., [3, 7, 14, 30])
        
    Returns:
        df: DataFrame with additional rolling features
    """
    df = df.copy()
    
    for window in windows:
        if len(df) >= window:
            df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window).mean()
            df[f'{column}_rolling_std_{window}'] = df[column].rolling(window).std()
            df[f'{column}_rolling_min_{window}'] = df[column].rolling(window).min()
            df[f'{column}_rolling_max_{window}'] = df[column].rolling(window).max()
    
    return df


def create_shifted_targets(df: pd.DataFrame, target_column, prediction_horizon: int = 1,
                          group_column = None, verbose: bool = False) -> pd.DataFrame:
    """
    Create shifted target variable(s) for time series prediction.
    Supports both single and multiple target variables, with single or multi-horizon prediction.

    This is core time-series functionality for creating future prediction targets by shifting
    the original target column(s) backwards in time.

    Args:
        df: DataFrame with features
        target_column: Name(s) of the original target column(s)
                      - str: Single target (e.g., 'close')
                      - List[str]: Multiple targets (e.g., ['close', 'volume'])
        prediction_horizon: Number of steps to predict ahead (1 = single, >1 = multi-horizon)
        group_column: Optional column(s) for group-based shifting
                     - str: Single column (e.g., 'symbol')
                     - List[str]: Multiple columns (e.g., ['symbol', 'sector'])
                     When provided, shifting is done within each group to avoid data leakage
        verbose: Whether to print information

    Returns:
        df: DataFrame with shifted target column(s) added
            - Single target, single horizon: adds '{target}_target_h1' column
            - Single target, multi-horizon: adds '{target}_target_h1', '{target}_target_h2', etc.
            - Multiple targets: adds columns for each target with appropriate horizon suffixes
            - Rows with NaN in any target column are automatically removed
    Examples:
        >>> # Single target, single horizon
        >>> df = create_shifted_targets(df, 'close', prediction_horizon=1)
        >>> # Creates: close_target_h1

        >>> # Single target, multi-horizon
        >>> df = create_shifted_targets(df, 'close', prediction_horizon=3)
        >>> # Creates: close_target_h1, close_target_h2, close_target_h3

        >>> # Multiple targets, single horizon
        >>> df = create_shifted_targets(df, ['close', 'volume'], prediction_horizon=1)
        >>> # Creates: close_target_h1, volume_target_h1

        >>> # Multiple targets, multi-horizon with grouping
        >>> df = create_shifted_targets(df, ['close', 'volume'], prediction_horizon=2, group_column='symbol')
        >>> # Creates: close_target_h1, close_target_h2, volume_target_h1, volume_target_h2
        >>> # Shifting is done per symbol to avoid cross-contamination

        >>> # Multi-column grouping
        >>> df = create_shifted_targets(df, 'close', prediction_horizon=1, group_column=['symbol', 'sector'])
        >>> # Shifting is done per (symbol, sector) combination
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

    # Determine if we should use groupby
    use_groupby = False
    if group_column is not None:
        if isinstance(group_column, str):
            # Single group column
            use_groupby = group_column in df.columns
        elif isinstance(group_column, list):
            # Multiple group columns - check all exist and list is not empty
            use_groupby = len(group_column) > 0 and all(col in df.columns for col in group_column)
        else:
            raise ValueError(f"group_column must be str or list, got {type(group_column)}")

    # Create shifted targets for each target variable
    for target_col in target_columns_list:
        if prediction_horizon == 1:
            # Single horizon
            shifted_target_name = f"{target_col}_target_h1"
            if use_groupby:
                df[shifted_target_name] = df.groupby(group_column)[target_col].shift(-1)
            else:
                df[shifted_target_name] = df[target_col].shift(-1)
            all_shifted_targets.append(shifted_target_name)
        else:
            # Multi-horizon
            for h in range(1, prediction_horizon + 1):
                col_name = f"{target_col}_target_h{h}"
                if use_groupby:
                    df[col_name] = df.groupby(group_column)[target_col].shift(-h)
                else:
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
        if group_column:
            print(f"   Group-based shifting applied using column: {group_column}")
        print(f"   Remaining samples after shift: {len(df)}")

    return df


def create_input_variable_sequence(
    df: pd.DataFrame,
    sequence_length: int,
    feature_columns: list = None,
    exclude_columns: list = None
) -> np.ndarray:
    """
    Create sliding window sequences from input variables only.

    This function creates the X (input sequences) for time series prediction.
    Target variables (Y) should be created separately using create_shifted_targets().

    Args:
        df: DataFrame with features
        sequence_length: Length of input sequences (lookback window)
        feature_columns: Explicit list of feature columns to use.
                        If None, auto-detects all numeric columns except excluded ones.
        exclude_columns: List of columns to exclude from features (e.g., target columns,
                        date columns, group columns, shifted target columns).
                        Only used when feature_columns=None.

    Returns:
        sequences: Array of shape (n_samples, sequence_length, n_features)
                  where n_samples = len(df) - sequence_length

    Examples:
        >>> # Auto-detect features, exclude targets and metadata
        >>> sequences = create_input_variable_sequence(
        ...     df,
        ...     sequence_length=10,
        ...     exclude_columns=['close', 'volume', 'close_target_h1',
        ...                      'volume_target_h1', 'date', 'symbol']
        ... )

        >>> # Explicit feature list
        >>> sequences = create_input_variable_sequence(
        ...     df,
        ...     sequence_length=10,
        ...     feature_columns=['open', 'high', 'low', 'volume', 'vwap']
        ... )

        >>> # Multi-horizon targets - exclude all shifted columns
        >>> sequences = create_input_variable_sequence(
        ...     df,
        ...     sequence_length=10,
        ...     exclude_columns=['close', 'volume',
        ...                      'close_target_h1', 'close_target_h2', 'close_target_h3',
        ...                      'volume_target_h1', 'volume_target_h2', 'volume_target_h3',
        ...                      'date', 'symbol']
        ... )
    """
    if len(df) <= sequence_length:
        raise ValueError(f"DataFrame length ({len(df)}) must be greater than sequence_length ({sequence_length})")

    # Determine which feature columns to use
    if feature_columns is None:
        # Auto-detect features: all numeric columns except excluded ones
        if exclude_columns is None:
            exclude_columns = []

        exclude_set = set(exclude_columns)
        feature_columns = []

        for col in df.columns:
            if col not in exclude_set and pd.api.types.is_numeric_dtype(df[col]):
                feature_columns.append(col)

    if len(feature_columns) == 0:
        raise ValueError("No numeric feature columns found for sequence creation")

    features = df[feature_columns].values

    sequences = []

    # Create sequences
    for i in range(sequence_length, len(df)):
        # Sequence of features (look back)
        seq = features[i-sequence_length:i]
        sequences.append(seq)

    return np.array(sequences)


