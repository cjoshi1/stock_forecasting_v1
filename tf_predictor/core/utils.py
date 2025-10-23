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

    # MAPE (handle division by zero by excluding zero values)
    # Only calculate MAPE for non-zero actual values to avoid division by zero
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape_values = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
        mape = np.mean(mape_values) * 100
    else:
        # If all actual values are zero, MAPE is undefined
        mape = float('nan')

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


def calculate_metrics_multi_horizon(
    y_true_base: np.ndarray,
    y_pred: np.ndarray,
    prediction_horizon: int
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for multi-horizon predictions.

    For each horizon h, compares predictions against actual values h steps ahead.
    Example: For horizon 2 (t+2), compares pred[:, 1] against actual values 2 steps ahead.

    Args:
        y_true_base: Base actual values aligned with first horizon (1D array)
        y_pred: Predicted values (2D array: n_samples x horizons)
        prediction_horizon: Number of prediction horizons

    Returns:
        Nested dict with overall and per-horizon metrics:
        {
            'overall': {...},      # Average metrics across all horizons
            'horizon_1': {...},    # Metrics for t+1 predictions
            'horizon_2': {...},    # Metrics for t+2 predictions
            ...
        }
    """
    if len(y_pred.shape) != 2:
        raise ValueError(f"y_pred must be 2D array for multi-horizon, got shape {y_pred.shape}")

    if y_pred.shape[1] != prediction_horizon:
        raise ValueError(f"y_pred has {y_pred.shape[1]} horizons but expected {prediction_horizon}")

    # Calculate per-horizon metrics
    horizon_metrics = {}
    all_maes = []
    all_rmses = []
    all_mapes = []
    all_r2s = []
    all_dir_accs = []

    for h in range(prediction_horizon):
        horizon_num = h + 1  # 1-indexed for user display

        # Get predictions for this horizon
        horizon_preds = y_pred[:, h]

        # Get actual values h steps ahead
        # For horizon h (0-indexed), we need actual values shifted by h+1
        # Since y_true_base is aligned with horizon 0 (t+1),
        # horizon h needs actual values at index [h:]
        if h < len(y_true_base):
            # We can only evaluate where we have both predictions and future actual values
            max_samples = min(len(horizon_preds), len(y_true_base) - h)
            y_true_horizon = y_true_base[h:h + max_samples]
            y_pred_horizon = horizon_preds[:max_samples]
        else:
            # Not enough data for this horizon
            y_true_horizon = np.array([])
            y_pred_horizon = np.array([])

        # Calculate metrics for this horizon
        metrics = calculate_metrics(y_true_horizon, y_pred_horizon)
        horizon_metrics[f'horizon_{horizon_num}'] = metrics

        # Collect for overall calculation
        if not np.isnan(metrics['MAE']):
            all_maes.append(metrics['MAE'])
            all_rmses.append(metrics['RMSE'])
            all_mapes.append(metrics['MAPE'])
            all_r2s.append(metrics['R2'])
            all_dir_accs.append(metrics['Directional_Accuracy'])

    # Calculate overall metrics (average across horizons)
    overall_metrics = {
        'MAE': np.mean(all_maes) if all_maes else float('nan'),
        'MSE': np.mean([m**2 for m in all_rmses]) if all_rmses else float('nan'),
        'RMSE': np.mean(all_rmses) if all_rmses else float('nan'),
        'MAPE': np.mean(all_mapes) if all_mapes else float('nan'),
        'R2': np.mean(all_r2s) if all_r2s else float('nan'),
        'Directional_Accuracy': np.mean(all_dir_accs) if all_dir_accs else float('nan')
    }

    # Return nested structure with overall first
    result = {'overall': overall_metrics}
    result.update(horizon_metrics)

    return result


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


def split_time_series(df: pd.DataFrame, test_size: int = 30, val_size: int = None,
                      group_column: str = None, time_column: str = None,
                      sequence_length: int = 1) -> tuple:
    """
    Split time series data maintaining temporal order.

    When group_column is specified, splits each group separately to ensure:
    - Equal representation from each group in train/val/test
    - Temporal order within each group (earliest data in train, most recent in test)

    Args:
        df: DataFrame sorted by time
        test_size: Number of samples for test set (per group if group_column specified)
        val_size: Number of samples for validation set (per group if group_column specified)
        group_column: Optional column name to split by groups (e.g., 'symbol' for stocks)
        time_column: Time column for sorting within groups. Auto-detected if None.
        sequence_length: Required sequence length for model (used to calculate minimum samples needed)

    Returns:
        train_df, val_df, test_df (val_df is None if val_size is None)
    """
    if group_column is None:
        # Original behavior: simple temporal split
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

    else:
        # Group-wise splitting: split each group separately
        if group_column not in df.columns:
            raise ValueError(f"Group column '{group_column}' not found in dataframe")

        # Auto-detect time column if not provided
        if time_column is None:
            possible_time_cols = ['timestamp', 'date', 'datetime', 'time', 'Date', 'Timestamp', 'DateTime']
            for col in possible_time_cols:
                if col in df.columns:
                    time_column = col
                    break
            if time_column is None:
                print(f"Warning: No time column found, assuming data is already sorted")

        # Sort by group and time to ensure temporal order within each group
        if time_column:
            df_sorted = df.sort_values([group_column, time_column]).reset_index(drop=True)
        else:
            df_sorted = df.copy()

        # Split each group separately
        train_dfs = []
        val_dfs = []
        test_dfs = []

        unique_groups = df_sorted[group_column].unique()

        for group_value in unique_groups:
            group_mask = df_sorted[group_column] == group_value
            group_df = df_sorted[group_mask].copy()

            # Check if group has enough data
            # Need: test_size + val_size + (sequence_length + 1) for at least 1 training sequence
            # Plus some buffer for target shifting (add sequence_length again to be safe)
            min_train_samples = sequence_length * 2 + 10  # Need at least this many for meaningful training
            min_required = test_size + (val_size if val_size else 0) + min_train_samples
            if len(group_df) < min_required:
                print(f"Warning: Group '{group_value}' has only {len(group_df)} samples, skipping (needs >= {min_required} with sequence_length={sequence_length})")
                continue

            # Split this group: most recent data goes to test, earliest to train
            group_test = group_df.iloc[-test_size:].copy()
            group_remaining = group_df.iloc[:-test_size].copy()

            # Validation split for this group
            if val_size is not None and len(group_remaining) > val_size:
                group_val = group_remaining.iloc[-val_size:].copy()
                group_train = group_remaining.iloc[:-val_size].copy()
            else:
                group_val = None
                group_train = group_remaining.copy()

            # Collect splits
            train_dfs.append(group_train)
            if group_val is not None:
                val_dfs.append(group_val)
            test_dfs.append(group_test)

        # Combine all groups
        if len(train_dfs) == 0:
            error_msg = (
                f"ERROR: No groups had sufficient data for splitting!\n"
                f"  Total groups: {len(unique_groups)}\n"
                f"  Required samples per group: {min_required}\n"
                f"    = {test_size} (test) + {val_size if val_size else 0} (val) + {min_train_samples} (train)\n"
                f"    with sequence_length={sequence_length}\n"
                f"  Group sizes: {dict(df_sorted[group_column].value_counts().items())}\n"
                f"\n"
                f"Suggestions:\n"
                f"  1. Reduce test_size (currently {test_size})\n"
                f"  2. Reduce val_size (currently {val_size})\n"
                f"  3. Reduce sequence_length (currently {sequence_length})\n"
                f"  4. Use more data per group\n"
                f"  5. Remove groups with insufficient data before splitting"
            )
            raise ValueError(error_msg)

        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else None
        test_df = pd.concat(test_dfs, ignore_index=True)

        # Print split summary
        print(f"   Group-wise split summary:")
        print(f"   - {len(unique_groups)} groups processed")
        print(f"   - Train: {len(train_df)} samples ({len(train_dfs)} groups × ~{len(train_df)//len(train_dfs)} samples)")
        if val_df is not None:
            print(f"   - Val: {len(val_df)} samples ({len(val_dfs)} groups × ~{len(val_df)//len(val_dfs)} samples)")
        print(f"   - Test: {len(test_df)} samples ({len(test_dfs)} groups × {test_size} samples)")

        return train_df, val_df, test_df