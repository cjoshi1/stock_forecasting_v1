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


def create_date_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Create temporal features from a date column.
    
    Args:
        df: DataFrame with date column
        date_column: Name of the date column
        
    Returns:
        df: DataFrame with additional date features
    """
    df = df.copy()
    
    # Convert to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Check if data is already sorted (ascending)
    is_sorted = df[date_column].is_monotonic_increasing
    
    if not is_sorted:
        # Only sort if not already sorted (to preserve chronological splits)
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


def create_lag_features(df: pd.DataFrame, column: str, lags: list) -> pd.DataFrame:
    """
    Create lag features for time series.
    
    Args:
        df: DataFrame with time series data
        column: Column to create lags for
        lags: List of lag periods (e.g., [1, 2, 3, 7])
        
    Returns:
        df: DataFrame with additional lag features
    """
    df = df.copy()
    
    for lag in lags:
        if len(df) > lag:
            df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    
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


def create_sequences(df: pd.DataFrame, sequence_length: int, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences from time series data.
    
    Args:
        df: DataFrame with features and target
        sequence_length: Length of input sequences
        target_column: Name of target column
        
    Returns:
        sequences: Array of shape (n_samples, sequence_length, n_features)
        targets: Array of shape (n_samples,)
    """
    if len(df) <= sequence_length:
        raise ValueError(f"DataFrame length ({len(df)}) must be greater than sequence_length ({sequence_length})")
    
    # Features (exclude target column and non-numeric columns)
    feature_columns = []
    for col in df.columns:
        if col != target_column and pd.api.types.is_numeric_dtype(df[col]):
            feature_columns.append(col)
    
    if len(feature_columns) == 0:
        raise ValueError("No numeric feature columns found for sequence creation")
    
    features = df[feature_columns].values
    targets = df[target_column].values
    
    sequences = []
    sequence_targets = []
    
    # Create sequences
    for i in range(sequence_length, len(df)):
        # Sequence of features (look back)
        seq = features[i-sequence_length:i]
        # Target at current time step
        target = targets[i]
        
        sequences.append(seq)
        sequence_targets.append(target)
    
    return np.array(sequences), np.array(sequence_targets)


