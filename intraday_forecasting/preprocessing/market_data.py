"""
Intraday market data loading and validation utilities.

Handles loading minute-level OHLCV data, validation, and sample data generation
for intraday trading applications.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import os

from .timeframe_utils import prepare_intraday_data, get_timeframe_config
from .intraday_features import create_intraday_features


def load_intraday_data(file_path: str, timestamp_col: str = 'timestamp',
                      validate: bool = True, group_column: Optional[str] = None) -> pd.DataFrame:
    """
    Load intraday time series data from CSV file (flexible - works with any numeric columns).

    Supports various formats including:
    - Standard OHLCV format with timestamp column
    - Custom formats with any numeric columns
    - BTC-USD format with Datetime, OHLCV, Dividends, Stock Splits columns
    - Yahoo Finance format

    Args:
        file_path: Path to CSV file with intraday data
        timestamp_col: Name of timestamp column
        validate: Whether to validate data format
        group_column: Optional group column to preserve (e.g., 'symbol')

    Returns:
        DataFrame with loaded data and all numeric columns preserved
    """
    if not os.path.exists(file_path):
        logging.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Load CSV
    df = pd.read_csv(file_path)

    # Map common column name variations to standard lowercase names
    column_mapping = {
        'Datetime': 'timestamp',
        'Date': 'timestamp',
        'Time': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'Vol': 'volume',
        'Symbol': 'symbol'
    }

    # Apply column mapping
    df.rename(columns=column_mapping, inplace=True)


    # Check for typical OHLCV columns (informational only - not required)
    expected_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    present_cols = [col for col in expected_cols if col in df.columns]

    # Inform user about available columns
    if missing_cols and present_cols:
        print(f"   Info: Missing typical OHLCV columns: {missing_cols}")
        print(f"   Found OHLCV columns: {present_cols}")
    elif not present_cols:
        print(f"   Info: No typical OHLCV columns found - using custom numeric columns")

    # Ensure timestamp column exists
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found. Available columns: {list(df.columns)}")

    # Convert timestamp to datetime and handle timezone-aware timestamps
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Convert timezone-aware timestamps to UTC and then remove timezone for consistency
    if df[timestamp_col].dt.tz is not None:
        df[timestamp_col] = df[timestamp_col].dt.tz_convert('UTC').dt.tz_localize(None)

    if validate:
        df = validate_intraday_data(df, timestamp_col, group_column)

    return df


def validate_intraday_data(df: pd.DataFrame, timestamp_col: str = 'timestamp', group_column: Optional[str] = None) -> pd.DataFrame:
    """
    Validate intraday data quality and format.

    Only validates columns that are present in the DataFrame.
    Safe to use with partial OHLCV data or custom columns.

    Args:
        df: DataFrame with intraday data
        timestamp_col: Name of timestamp column
        group_column: Optional group column to preserve

    Returns:
        Validated and cleaned DataFrame
    """
    df_clean = df.copy()

    # Sort by timestamp
    df_clean = df_clean.sort_values(timestamp_col).reset_index(drop=True)

    # Check which OHLCV columns are present
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    present_numeric_cols = [col for col in numeric_cols if col in df_clean.columns]
    
    # Check for non-positive prices (only for columns that exist)
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df_clean.columns:
            invalid_mask = (df_clean[col] <= 0) | (df_clean[col].isna())
            if invalid_mask.sum() > 0:
                print(f"Warning: Removing {invalid_mask.sum()} rows with invalid {col} values")
                df_clean = df_clean[~invalid_mask]

    # Check for negative volume (only if column exists)
    if 'volume' in df_clean.columns:
        invalid_volume = (df_clean['volume'] < 0) | (df_clean['volume'].isna())
        if invalid_volume.sum() > 0:
            print(f"Warning: Removing {invalid_volume.sum()} rows with invalid volume values")
            df_clean = df_clean[~invalid_volume]

    # Validate OHLC relationships (only if all OHLC columns exist)
    if all(col in df_clean.columns for col in ['open', 'high', 'low', 'close']):
        invalid_ohlc = (
            (df_clean['high'] < df_clean['low']) |
            (df_clean['high'] < df_clean['open']) |
            (df_clean['high'] < df_clean['close']) |
            (df_clean['low'] > df_clean['open']) |
            (df_clean['low'] > df_clean['close'])
        )
        if invalid_ohlc.sum() > 0:
            print(f"Warning: Removing {invalid_ohlc.sum()} rows with invalid OHLC relationships")
            df_clean = df_clean[~invalid_ohlc]
    
    # Remove duplicates
    # If group_column is specified, deduplicate on both timestamp and group_column
    # Otherwise, deduplicate only on timestamp
    before_dedup = len(df_clean)
    if group_column is not None and group_column in df_clean.columns:
        dedup_cols = [timestamp_col, group_column]
    else:
        dedup_cols = [timestamp_col]

    df_clean = df_clean.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    after_dedup = len(df_clean)
    if before_dedup != after_dedup:
        print(f"Warning: Removed {before_dedup - after_dedup} duplicate timestamps")

    return df_clean


def create_sample_intraday_data(n_days: int = 5, start_date: str = "2023-01-01", 
                               symbol: str = "SAMPLE", include_oi: bool = False) -> pd.DataFrame:
    """
    Generate sample minute-level intraday data for testing.
    
    Args:
        n_days: Number of trading days to generate
        start_date: Starting date for data generation
        symbol: Symbol name for the data
        include_oi: Whether to include open interest column
        
    Returns:
        DataFrame with sample intraday data
    """
    start_dt = pd.to_datetime(start_date)
    
    # Generate trading days (skip weekends)
    trading_days = []
    current_date = start_dt
    days_added = 0
    
    while days_added < n_days:
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            trading_days.append(current_date)
            days_added += 1
        current_date += timedelta(days=1)
    
    # Generate minute-level data for each trading day
    all_data = []
    base_price = 100.0
    
    for day in trading_days:
        # Market hours: 9:30 AM to 4:00 PM (390 minutes)
        market_open = day.replace(hour=9, minute=30)
        market_close = day.replace(hour=16, minute=0)
        
        # Generate minute timestamps
        timestamps = pd.date_range(market_open, market_close, freq='1T')[:-1]  # Exclude 4:00 PM
        
        # Random walk for prices with some intraday patterns
        n_minutes = len(timestamps)
        returns = np.random.normal(0, 0.001, n_minutes)  # Small random returns
        
        # Add some intraday patterns
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            minute = ts.minute
            
            # Opening volatility
            if hour == 9:
                returns[i] += np.random.normal(0, 0.002)
            # Lunch lull (lower volatility)
            elif hour in [12, 13]:
                returns[i] *= 0.5
            # Closing volatility
            elif hour == 15:
                returns[i] += np.random.normal(0, 0.0015)
        
        # Generate OHLCV data
        cumulative_returns = np.cumsum(returns)
        prices = base_price * np.exp(cumulative_returns)
        
        day_data = []
        for i, (ts, price) in enumerate(zip(timestamps, prices)):
            # Generate OHLC from price with some noise
            noise = np.random.normal(0, 0.0005, 4)
            open_price = price + noise[0] * price
            close_price = price + noise[1] * price
            high_price = max(open_price, close_price) + abs(noise[2]) * price
            low_price = min(open_price, close_price) - abs(noise[3]) * price
            
            # Ensure OHLC relationships
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate volume with intraday patterns
            base_volume = 10000
            hour_multiplier = {
                9: 2.0, 10: 1.8, 11: 1.2, 12: 0.8, 13: 0.8, 
                14: 1.0, 15: 1.5, 16: 0.1
            }.get(ts.hour, 1.0)
            
            volume = int(base_volume * hour_multiplier * np.random.lognormal(0, 0.3))
            
            row = {
                'timestamp': ts,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            }
            
            # Add open interest if requested
            if include_oi:
                if i == 0:
                    oi = 50000  # Starting OI
                else:
                    oi_change = np.random.randint(-100, 101)
                    oi = max(0, day_data[-1]['open_interest'] + oi_change)
                row['open_interest'] = oi
            
            day_data.append(row)
        
        all_data.extend(day_data)
        base_price = prices[-1]  # Continue from last price for next day
    
    df = pd.DataFrame(all_data)
    
    # Add symbol column
    df['symbol'] = symbol
    
    return df


def prepare_intraday_for_training(df: pd.DataFrame, target_column: str = 'close',
                                timeframe: str = '5min', timestamp_col: str = 'timestamp',
                                country: str = 'US', prediction_horizon: int = 1,
                                group_column: Optional[str] = None,
                                verbose: bool = False) -> Dict[str, Any]:
    """
    Complete pipeline to prepare intraday data for training.

    Args:
        df: Raw minute-level DataFrame
        target_column: Column to predict
        timeframe: Target timeframe for resampling
        timestamp_col: Name of timestamp column
        country: Country code ('US' or 'INDIA')
        prediction_horizon: Number of steps ahead to predict (1=single, >1=multi-horizon)
        group_column: Optional column for group-based scaling (e.g., 'symbol' for multi-stock datasets)
        verbose: Whether to print processing steps

    Returns:
        Dictionary with processed data and metadata
    """
    if verbose:
        horizon_text = f"{prediction_horizon}-step" if prediction_horizon > 1 else "single-step"
        print(f"Preparing intraday data for {timeframe} forecasting ({country} market, {horizon_text})...")
        print(f"Original data: {len(df)} minute bars")

    # Step 1: Prepare data (market hours + resampling)
    df_processed = prepare_intraday_data(df, timeframe, timestamp_col, country, group_column)

    if verbose:
        print(f"After resampling to {timeframe}: {len(df_processed)} bars")

    # Step 2: Create features
    df_features = create_intraday_features(
        df_processed, target_column, timestamp_col, country,
        timeframe, prediction_horizon, verbose, group_column=group_column
    )

    # Step 3: Get timeframe configuration
    config = get_timeframe_config(timeframe)

    if verbose:
        print(f"Suggested sequence length: {config['sequence_length']} bars")
        feature_cols = [col for col in df_features.columns if col != timestamp_col]
        print(f"Total features created: {len(feature_cols)}")

    return {
        'data': df_features,
        'config': config,
        'original_length': len(df),
        'processed_length': len(df_features),
        'timeframe': timeframe,
        'target_column': target_column,
        'prediction_horizon': prediction_horizon,
        'country': country
    }
