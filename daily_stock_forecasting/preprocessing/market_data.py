"""
Stock market data loading and validation utilities.
"""

import logging
import pandas as pd
import numpy as np
from tf_predictor.core.utils import load_time_series_data


def load_stock_data(file_path: str, date_column: str = 'date', asset_type: str = 'stock', group_column: str = None) -> pd.DataFrame:
    """
    Load and validate stock or crypto data.

    Args:
        file_path: Path to CSV file
        date_column: Name of date column
        asset_type: Type of asset - 'stock' (5-day week) or 'crypto' (7-day week)
        group_column: Optional group column to preserve (e.g., 'symbol' for multi-stock datasets)

    Returns:
        df: Validated DataFrame
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {file_path}", exc_info=True)
        raise e

    # Standardize column names to lowercase for consistency
    df.columns = df.columns.str.lower()
    date_column = date_column.lower()
    if group_column is not None:
        group_column = group_column.lower()

    # Remove any non-numeric columns that we don't need (except group column)
    # Keep only OHLCV + date + group column + any other useful columns
    cols_to_keep = []

    # Always keep OHLCV columns if they exist
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in ohlcv_cols:
        if col in df.columns:
            cols_to_keep.append(col)

    # Keep date column if it exists
    if date_column in df.columns:
        cols_to_keep.append(date_column)

    # Keep group column if specified and exists
    if group_column is not None and group_column in df.columns:
        cols_to_keep.append(group_column)

    # Keep any other numeric columns (like vwap, change, etc.)
    for col in df.columns:
        if col not in cols_to_keep:
            # Try to convert to numeric, keep if successful
            try:
                pd.to_numeric(df[col], errors='raise')
                cols_to_keep.append(col)
            except (ValueError, TypeError):
                # Skip non-numeric columns (unless it's the group column which we already handled)
                if col != group_column:
                    print(f"Skipping non-numeric column: '{col}'")
                pass

    # Filter to only the columns we want
    df = df[cols_to_keep].copy()
    
    # Expected OHLCV columns
    expected_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    
    if missing_cols:
        logging.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate and convert data types for OHLCV columns
    for col in expected_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                raise ValueError(f"Column '{col}' contains non-numeric values")
    
    # Handle other numeric columns
    numeric_cols = [col for col in df.columns if col not in expected_cols + [date_column]]
    for col in numeric_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            print(f"Warning: Could not convert column '{col}' to numeric, leaving it as is.")
    
    # Check for date column and sort chronologically
    if date_column in df.columns:
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            # Sort by date (oldest first for proper time series analysis)
            df = df.sort_values(date_column).reset_index(drop=True)
            print(f"   Sorted data chronologically: {df[date_column].iloc[0]} to {df[date_column].iloc[-1]}")

            # Validate trading days based on asset type
            if asset_type == 'crypto':
                # Crypto: Expect data for all 7 days of the week
                print(f"   Asset type: Cryptocurrency (24/7 trading)")
            else:
                # Stock: Check for weekend data (might indicate data issues)
                weekend_mask = df[date_column].dt.dayofweek >= 5
                if weekend_mask.any():
                    print(f"   Warning: Found {weekend_mask.sum()} weekend rows in stock data")
                print(f"   Asset type: Traditional stock (weekday trading)")
        except:
            print(f"Warning: Could not parse date column '{date_column}'")
    
    # Remove rows with NaN in critical columns
    before_len = len(df)
    df = df.dropna(subset=expected_cols)
    after_len = len(df)
    
    if before_len != after_len:
        print(f"Removed {before_len - after_len} rows with missing values")
    
    if len(df) == 0:
        raise ValueError("No valid data remaining after cleaning")
    
    return df


def validate_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate stock data for common issues.
    
    Args:
        df: DataFrame with stock data
        
    Returns:
        df: Cleaned DataFrame
    """
    df = df.copy()
    
    # Check for negative prices
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df.columns:
            negative_mask = df[col] <= 0
            if negative_mask.any():
                print(f"Warning: Found {negative_mask.sum()} negative/zero values in {col}")
                # Replace with forward fill
                df.loc[negative_mask, col] = np.nan
                df[col] = df[col].fillna(method='ffill')
    
    # Check for high > low consistency
    if all(col in df.columns for col in ['high', 'low']):
        inconsistent_mask = df['high'] < df['low']
        if inconsistent_mask.any():
            print(f"Warning: Found {inconsistent_mask.sum()} rows where high < low")
            # Swap values
            df.loc[inconsistent_mask, ['high', 'low']] = df.loc[inconsistent_mask, ['low', 'high']].values
    
    # Check for negative volume
    if 'volume' in df.columns:
        negative_vol_mask = df['volume'] < 0
        if negative_vol_mask.any():
            print(f"Warning: Found {negative_vol_mask.sum()} negative volume values")
            df.loc[negative_vol_mask, 'volume'] = 0
    
    return df


def create_sample_stock_data(n_samples: int = 300, start_price: float = 100.0, asset_type: str = 'stock') -> pd.DataFrame:
    """
    Create synthetic stock or crypto data for testing purposes.

    Args:
        n_samples: Number of data points to generate
        start_price: Starting price for the synthetic data
        asset_type: Type of asset - 'stock' (5-day week) or 'crypto' (7-day week)

    Returns:
        df: DataFrame with synthetic OHLCV data
    """
    if asset_type == 'crypto':
        # Crypto: Generate continuous daily data (all 7 days of the week)
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    else:
        # Stock: Generate business days only (Monday-Friday, skip weekends)
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='B')
    
    # Generate price data with some volatility and trend
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0.001, 0.02, n_samples)  # Daily returns
    
    prices = [start_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Generate OHLC data
    daily_volatility = np.random.uniform(0.01, 0.05, n_samples)
    
    opens = prices * (1 + np.random.normal(0, daily_volatility * 0.5))
    highs = np.maximum(opens, prices) * (1 + np.random.uniform(0, daily_volatility))
    lows = np.minimum(opens, prices) * (1 - np.random.uniform(0, daily_volatility))
    closes = prices
    
    # Generate volume data
    base_volume = 1000000
    volumes = np.random.lognormal(np.log(base_volume), 0.5, n_samples).astype(int)
    
    df = pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    return df
