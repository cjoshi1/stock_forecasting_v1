"""
Generic utilities for time series processing and evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary with calculated metrics
    """
    # Handle edge cases
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'MAE': float('nan'),
            'MSE': float('nan'), 
            'RMSE': float('nan'),
            'MAPE': float('nan'),
            'R2': float('nan'),
            'Directional_Accuracy': float('nan')
        }
    
    # Basic error metrics
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # MAPE (handle division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mape = np.nan_to_num(mape, nan=float('inf'))
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        r2 = 0.0 if ss_res == 0 else float('-inf')
    else:
        r2 = 1 - (ss_res / ss_tot)
    
    # Directional accuracy (for sequences)
    if len(y_true) > 1:
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
    else:
        directional_accuracy = float('nan')
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse, 
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    }


def load_time_series_data(file_path: str, date_column: str = 'date') -> pd.DataFrame:
    """
    Load and validate generic time series data.
    
    Args:
        file_path: Path to CSV file
        date_column: Name of date column
        
    Returns:
        df: Validated and sorted DataFrame
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Check for date column and sort chronologically
    if date_column in df.columns:
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            # Sort by date (oldest first for proper time series analysis)
            df = df.sort_values(date_column).reset_index(drop=True)
            print(f"   Sorted data chronologically: {df[date_column].iloc[0]} to {df[date_column].iloc[-1]}")
        except:
            print(f"Warning: Could not parse date column '{date_column}'")
    
    # Remove rows with all NaN values
    before_len = len(df)
    df = df.dropna(how='all')
    after_len = len(df)
    
    if before_len != after_len:
        print(f"Removed {before_len - after_len} empty rows")
    
    if len(df) == 0:
        raise ValueError("No valid data remaining after cleaning")
    
    return df


def split_time_series(df: pd.DataFrame, test_size: int = 30, val_size: int = None) -> tuple:
    """
    Split time series data maintaining temporal order.
    
    Args:
        df: DataFrame sorted by time
        test_size: Number of samples for test set
        val_size: Number of samples for validation set (optional)
        
    Returns:
        train_df, val_df, test_df (val_df is None if val_size is None)
    """
    if len(df) <= test_size:
        print(f"Warning: Dataset has only {len(df)} samples, cannot create test split of {test_size}")
        return df, None, None
    
    # Test split
    test_df = df.iloc[-test_size:].copy()
    remaining_df = df.iloc[:-test_size].copy()
    
    # Validation split
    if val_size is not None and len(remaining_df) > val_size:
        val_df = remaining_df.iloc[-val_size:].copy()
        train_df = remaining_df.iloc[:-val_size].copy()
    else:
        val_df = None
        train_df = remaining_df.copy()
    
    return train_df, val_df, test_df