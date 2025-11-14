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
                      sequence_length: int = 1, include_overlap: bool = True) -> tuple:
    """
    Split time series data maintaining temporal order with optional sequence overlap.

    When group_column is specified, splits each group separately to ensure:
    - Equal representation from each group in train/val/test
    - Temporal order within each group (earliest data in train, most recent in test)

    When include_overlap=True (default), adds overlap between splits:
    - Validation set prepends (sequence_length - 1) rows from end of training set
    - Test set prepends (sequence_length - 1) rows from end of validation (or training) set
    This ensures the first row of val/test can generate predictions by having sufficient context.

    Args:
        df: DataFrame sorted by time
        test_size: Number of samples for test set (per group if group_column specified)
        val_size: Number of samples for validation set (per group if group_column specified)
        group_column: Optional column name to split by groups (e.g., 'symbol' for stocks)
        time_column: Time column for sorting within groups. Auto-detected if None.
        sequence_length: Required sequence length for model (used to calculate minimum samples needed
                        and for overlap calculation)
        include_overlap: If True (default), includes (sequence_length - 1) overlap rows between splits
                        to ensure first row of each split can form complete sequences

    Returns:
        train_df, val_df, test_df (val_df is None if val_size is None)
        Note: If include_overlap=True, val_df and test_df include context rows from previous splits
    """
    # Calculate overlap size for sequence context
    overlap_size = max(0, sequence_length - 1) if include_overlap else 0

    if group_column is None:
        # Non-grouped path: simple temporal split
        if test_size > 0 and len(df) <= test_size:
            print(f"Warning: Dataset has only {len(df)} samples, cannot create test split of {test_size}")
            return df, None, None

        # Create base splits (without overlap)
        # Handle test_size=0 edge case (pandas iloc[:-0] returns empty DataFrame)
        if test_size == 0:
            test_df_base = pd.DataFrame()
            remaining_df = df.copy()
        else:
            test_df_base = df.iloc[-test_size:].copy()
            remaining_df = df.iloc[:-test_size].copy()

        # Validation split
        # Handle val_size edge cases
        if val_size is not None and val_size > 0 and len(remaining_df) > val_size:
            val_df_base = remaining_df.iloc[-val_size:].copy()
            train_df = remaining_df.iloc[:-val_size].copy()
        else:
            val_df_base = None
            train_df = remaining_df.copy()

        # Add overlap if requested
        if include_overlap and overlap_size > 0:
            # Val gets overlap from train
            if val_df_base is not None:
                if len(train_df) >= overlap_size:
                    overlap_for_val = train_df.iloc[-overlap_size:].copy()
                    val_df = pd.concat([overlap_for_val, val_df_base], ignore_index=True)
                else:
                    val_df = val_df_base  # Not enough data for overlap
            else:
                val_df = None

            # Test gets overlap from val (or train if no val)
            source_for_test = val_df_base if val_df_base is not None else train_df
            if len(source_for_test) >= overlap_size:
                overlap_for_test = source_for_test.iloc[-overlap_size:].copy()
                test_df = pd.concat([overlap_for_test, test_df_base], ignore_index=True)
            else:
                test_df = test_df_base  # Not enough data for overlap
        else:
            # No overlap requested
            val_df = val_df_base
            test_df = test_df_base

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
            # Create base splits (without overlap)
            # Handle test_size=0 edge case (pandas iloc[:-0] returns empty DataFrame)
            if test_size == 0:
                group_test_base = pd.DataFrame()
                group_remaining = group_df.copy()
            else:
                group_test_base = group_df.iloc[-test_size:].copy()
                group_remaining = group_df.iloc[:-test_size].copy()

            # Validation split for this group
            # Handle val_size edge cases
            if val_size is not None and val_size > 0 and len(group_remaining) > val_size:
                group_val_base = group_remaining.iloc[-val_size:].copy()
                group_train = group_remaining.iloc[:-val_size].copy()
            else:
                group_val_base = None
                group_train = group_remaining.copy()

            # Add overlap if requested (within this group only)
            if include_overlap and overlap_size > 0:
                # Val gets overlap from train
                if group_val_base is not None:
                    if len(group_train) >= overlap_size:
                        overlap_for_val = group_train.iloc[-overlap_size:].copy()
                        group_val = pd.concat([overlap_for_val, group_val_base], ignore_index=True)
                    else:
                        group_val = group_val_base
                else:
                    group_val = None

                # Test gets overlap from val (or train if no val)
                source_for_test = group_val_base if group_val_base is not None else group_train
                if len(source_for_test) >= overlap_size:
                    overlap_for_test = source_for_test.iloc[-overlap_size:].copy()
                    group_test = pd.concat([overlap_for_test, group_test_base], ignore_index=True)
                else:
                    group_test = group_test_base
            else:
                # No overlap requested
                group_val = group_val_base
                group_test = group_test_base

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
        if include_overlap and overlap_size > 0:
            # With overlap
            print(f"   - Train: {len(train_df)} samples ({len(train_dfs)} groups × ~{len(train_df)//len(train_dfs)} samples)")
            if val_df is not None:
                print(f"   - Val: {len(val_df)} samples ({len(val_dfs)} groups × ~{len(val_df)//len(val_dfs)} samples, includes {overlap_size} overlap)")
            print(f"   - Test: {len(test_df)} samples ({len(test_dfs)} groups × ~{len(test_df)//len(test_dfs)} samples, includes {overlap_size} overlap)")
            print(f"   - Overlap: {overlap_size} rows prepended to val/test for sequence context")
        else:
            # Without overlap
            print(f"   - Train: {len(train_df)} samples ({len(train_dfs)} groups × ~{len(train_df)//len(train_dfs)} samples)")
            if val_df is not None:
                print(f"   - Val: {len(val_df)} samples ({len(val_dfs)} groups × ~{len(val_df)//len(val_dfs)} samples)")
            print(f"   - Test: {len(test_df)} samples ({len(test_dfs)} groups × {test_size} samples)")

        return train_df, val_df, test_df