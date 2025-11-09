"""
Forward return calculation for multi-horizon return forecasting.

Calculates holding period returns for multiple forward horizons.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def calculate_forward_returns(
    df: pd.DataFrame,
    price_column: str = 'close',
    horizons: List[int] = [1, 2, 3, 4, 5],
    return_type: str = 'percentage',
    group_column: Optional[str] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Calculate forward returns for multiple horizons.

    Creates target columns for return prediction:
    - return_1d: (price[t+1] - price[t]) / price[t] * 100
    - return_2d: (price[t+2] - price[t]) / price[t] * 100
    - ... and so on

    All returns are holding period returns from the current day's price.

    Args:
        df: DataFrame with price data
        price_column: Column to use for return calculation (default: 'close')
        horizons: List of forward horizons in days (default: [1,2,3,4,5])
        return_type: Type of return - 'percentage' or 'log' (default: 'percentage')
        group_column: Optional column to group by (e.g., 'symbol' for multi-stock data)
                     If provided, returns are calculated separately for each group
        verbose: Whether to print calculation info

    Returns:
        DataFrame with additional return columns (return_1d, return_2d, etc.)

    Example:
        If today's close = 100:
        - return_1d = (tomorrow_close - 100) / 100 * 100
        - return_2d = (close_in_2days - 100) / 100 * 100
    """
    # Verify price column exists
    if price_column not in df.columns:
        raise ValueError(f"Price column '{price_column}' not found in dataframe")

    if verbose:
        print(f"\n📈 Calculating Forward Returns:")
        print(f"   Price Column: {price_column}")
        print(f"   Return Type: {return_type}")
        print(f"   Horizons: {horizons}")
        if group_column:
            print(f"   Grouping by: {group_column}")

    def _calculate_returns_for_group(group_df):
        """Helper function to calculate returns for a single group."""
        group_out = group_df.copy()
        prices = group_df[price_column]

        # Calculate returns for each horizon
        for horizon in horizons:
            col_name = f'return_{horizon}d'

            if return_type == 'percentage':
                # Simple percentage return: (future - current) / current * 100
                future_price = prices.shift(-horizon)
                returns = (future_price - prices) / prices * 100

            elif return_type == 'log':
                # Log return: ln(future / current) * 100
                future_price = prices.shift(-horizon)
                returns = np.log(future_price / prices) * 100

            else:
                raise ValueError(f"Unknown return_type: {return_type}. Use 'percentage' or 'log'")

            group_out[col_name] = returns

        return group_out

    # Calculate returns (grouped or ungrouped)
    if group_column:
        if group_column not in df.columns:
            raise ValueError(f"Group column '{group_column}' not found in dataframe")

        # Calculate returns separately for each group
        # Note: We use include_groups=True (pandas 2.2+) to keep the grouping column
        # For older pandas, this parameter is ignored and grouping columns are included by default
        try:
            df_out = df.groupby(group_column, group_keys=False).apply(_calculate_returns_for_group, include_groups=True)
        except TypeError:
            # Fallback for older pandas versions that don't support include_groups
            df_out = df.groupby(group_column, group_keys=False).apply(_calculate_returns_for_group)
    else:
        # Calculate returns for entire dataframe
        df_out = _calculate_returns_for_group(df)

    if verbose:
        # Show statistics for each horizon
        for horizon in horizons:
            col_name = f'return_{horizon}d'
            valid_count = df_out[col_name].notna().sum()
            nan_count = df_out[col_name].isna().sum()
            print(f"   - {col_name}: {valid_count} valid, {nan_count} NaN (last {horizon} rows)")

        # Show statistics for first return
        first_return_col = f'return_{horizons[0]}d'
        returns_stats = df_out[first_return_col].describe()
        print(f"\n   Statistics for {first_return_col}:")
        print(f"   - Mean: {returns_stats['mean']:.3f}%")
        print(f"   - Std:  {returns_stats['std']:.3f}%")
        print(f"   - Min:  {returns_stats['min']:.3f}%")
        print(f"   - Max:  {returns_stats['max']:.3f}%")

    return df_out


def get_return_column_names(horizons: List[int] = [1, 2, 3, 4, 5]) -> List[str]:
    """
    Get the column names for return targets.

    Args:
        horizons: List of forward horizons

    Returns:
        List of return column names
    """
    return [f'return_{h}d' for h in horizons]


def validate_return_targets(df: pd.DataFrame, horizons: List[int] = [1, 2, 3, 4, 5]) -> bool:
    """
    Validate that return target columns exist and have valid data.

    Args:
        df: DataFrame to validate
        horizons: Expected horizons

    Returns:
        True if all return columns are valid

    Raises:
        ValueError: If validation fails
    """
    return_cols = get_return_column_names(horizons)

    # Check all columns exist
    missing_cols = [col for col in return_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing return columns: {missing_cols}")

    # Check for sufficient valid data
    for col in return_cols:
        valid_count = df[col].notna().sum()
        if valid_count == 0:
            raise ValueError(f"Return column '{col}' has no valid data")

        valid_pct = valid_count / len(df) * 100
        if valid_pct < 10:
            raise ValueError(f"Return column '{col}' has insufficient valid data ({valid_pct:.1f}%)")

    return True


def calculate_cumulative_returns(
    df: pd.DataFrame,
    price_column: str = 'close',
    verbose: bool = False
) -> pd.DataFrame:
    """
    Calculate cumulative returns from the start of the series.

    Useful for visualizing overall performance.

    Args:
        df: DataFrame with price data
        price_column: Column to use for return calculation
        verbose: Whether to print info

    Returns:
        DataFrame with cumulative_return column
    """
    df_out = df.copy()

    if price_column not in df.columns:
        raise ValueError(f"Price column '{price_column}' not found")

    # Calculate cumulative return from first price
    first_price = df[price_column].iloc[0]
    df_out['cumulative_return'] = (df[price_column] - first_price) / first_price * 100

    if verbose:
        total_return = df_out['cumulative_return'].iloc[-1]
        print(f"\n   Cumulative Return: {total_return:.2f}%")

    return df_out
