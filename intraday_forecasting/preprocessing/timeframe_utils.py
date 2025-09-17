"""
Timeframe resampling utilities for intraday data.

Handles conversion from minute-level data to various trading timeframes
with proper market hours filtering for multiple countries (US and India).
"""

import pandas as pd
from datetime import time, datetime
from typing import Dict, Any


# Market hours configuration by country
MARKET_HOURS_CONFIG = {
    'US': {
        'open': time(9, 30),   # 9:30 AM ET
        'close': time(16, 0),  # 4:00 PM ET
        'timezone': 'America/New_York',
        'description': 'US Eastern Time (NYSE/NASDAQ)',
        'filter_hours': True
    },
    'INDIA': {
        'open': time(9, 15),   # 9:15 AM IST
        'close': time(15, 30), # 3:30 PM IST
        'timezone': 'Asia/Kolkata',
        'description': 'India Standard Time (NSE/BSE)',
        'filter_hours': True
    },
    'CRYPTO': {
        'open': time(0, 0),    # 24/7 trading
        'close': time(23, 59), # 24/7 trading
        'timezone': 'UTC',
        'description': 'Cryptocurrency 24/7 Trading',
        'filter_hours': False  # No market hours filtering for crypto
    }
}

# Backward compatibility
MARKET_OPEN = MARKET_HOURS_CONFIG['US']['open']
MARKET_CLOSE = MARKET_HOURS_CONFIG['US']['close']

# Timeframe configurations
TIMEFRAME_CONFIG = {
    '1min': {
        'resample_rule': '1T',
        'sequence_length': 240,  # 4 hours of 1-min bars
        'description': '1-minute bars'
    },
    '5min': {
        'resample_rule': '5T', 
        'sequence_length': 96,   # 8 hours of 5-min bars
        'description': '5-minute bars'
    },
    '15min': {
        'resample_rule': '15T',
        'sequence_length': 32,   # 8 hours of 15-min bars  
        'description': '15-minute bars'
    },
    '1h': {
        'resample_rule': '1h',
        'sequence_length': 12,   # 12 hours of hourly bars
        'description': '1-hour bars'
    }
}


def filter_market_hours(df: pd.DataFrame, timestamp_col: str = 'timestamp',
                        country: str = 'US') -> pd.DataFrame:
    """
    Filter data to regular market hours for specified country.

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        country: Country code ('US', 'INDIA', or 'CRYPTO')

    Returns:
        Filtered DataFrame with only market hours data
    """
    if country not in MARKET_HOURS_CONFIG:
        raise ValueError(f"Unsupported country: {country}. Use: {list(MARKET_HOURS_CONFIG.keys())}")

    df = df.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Get market hours for country
    market_config = MARKET_HOURS_CONFIG[country]

    # Skip market hours filtering for crypto (24/7 trading)
    if not market_config.get('filter_hours', True):
        return df

    market_open = market_config['open']
    market_close = market_config['close']

    # Filter to market hours
    df_market = df[
        (df[timestamp_col].dt.time >= market_open) &
        (df[timestamp_col].dt.time < market_close)
    ].copy()

    return df_market


def resample_ohlcv(df: pd.DataFrame, timeframe: str, timestamp_col: str = 'timestamp', 
                   country: str = 'US') -> pd.DataFrame:
    """
    Resample minute-level OHLCV data to specified timeframe.
    
    Args:
        df: DataFrame with timestamp, open, high, low, close, volume columns
        timeframe: Target timeframe ('1min', '5min', '15min', '1h')
        timestamp_col: Name of timestamp column
        
    Returns:
        Resampled DataFrame
    """
    if timeframe not in TIMEFRAME_CONFIG:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Use: {list(TIMEFRAME_CONFIG.keys())}")
    
    df = df.copy()
    
    # Ensure timestamp is datetime and set as index
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    df.set_index(timestamp_col, inplace=True)
    df.sort_index(inplace=True)
    
    # Get resampling rule
    rule = TIMEFRAME_CONFIG[timeframe]['resample_rule']
    
    # Define aggregation functions
    agg_funcs = {
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Add open_interest if present
    if 'open_interest' in df.columns:
        agg_funcs['open_interest'] = 'last'  # Use last value for OI
    
    # Resample data
    df_resampled = df.resample(rule).agg(agg_funcs)
    
    # Drop rows with NaN (periods with no data)
    df_resampled.dropna(inplace=True)
    
    # Reset index to get timestamp as column
    df_resampled.reset_index(inplace=True)
    df_resampled.rename(columns={'index': timestamp_col}, inplace=True)
    
    return df_resampled


def get_timeframe_config(timeframe: str) -> Dict[str, Any]:
    """
    Get configuration for specified timeframe.
    
    Args:
        timeframe: Target timeframe
        
    Returns:
        Configuration dictionary
    """
    if timeframe not in TIMEFRAME_CONFIG:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Use: {list(TIMEFRAME_CONFIG.keys())}")
    
    return TIMEFRAME_CONFIG[timeframe].copy()


def validate_timeframe(timeframe: str) -> bool:
    """
    Validate if timeframe is supported.
    
    Args:
        timeframe: Timeframe to validate
        
    Returns:
        True if valid, False otherwise
    """
    return timeframe in TIMEFRAME_CONFIG


def get_supported_timeframes() -> list:
    """
    Get list of supported timeframes.
    
    Returns:
        List of supported timeframe strings
    """
    return list(TIMEFRAME_CONFIG.keys())


def get_supported_countries() -> list:
    """
    Get list of supported countries.
    
    Returns:
        List of supported country codes
    """
    return list(MARKET_HOURS_CONFIG.keys())


def get_country_market_hours(country: str) -> Dict[str, Any]:
    """
    Get market hours configuration for specified country.
    
    Args:
        country: Country code ('US' or 'INDIA')
        
    Returns:
        Dictionary with market hours configuration
    """
    if country not in MARKET_HOURS_CONFIG:
        raise ValueError(f"Unsupported country: {country}. Use: {list(MARKET_HOURS_CONFIG.keys())}")
    
    return MARKET_HOURS_CONFIG[country].copy()


def validate_country(country: str) -> bool:
    """
    Validate if country is supported.
    
    Args:
        country: Country code to validate
        
    Returns:
        True if valid, False otherwise
    """
    return country in MARKET_HOURS_CONFIG


def prepare_intraday_data(df: pd.DataFrame, timeframe: str = '5min', 
                         timestamp_col: str = 'timestamp', country: str = 'US') -> pd.DataFrame:
    """
    Complete pipeline to prepare intraday data for training.
    
    Args:
        df: Raw minute-level DataFrame
        timeframe: Target timeframe for resampling
        timestamp_col: Name of timestamp column
        country: Country code ('US' or 'INDIA')
        
    Returns:
        Processed DataFrame ready for feature engineering
    """
    # Step 1: Filter to market hours for specified country
    df_market = filter_market_hours(df, timestamp_col, country)
    
    # Step 2: Resample to target timeframe (skip if already 1min)
    if timeframe != '1min':
        df_processed = resample_ohlcv(df_market, timeframe, timestamp_col, country)
    else:
        df_processed = df_market.copy()
    
    # Step 3: Ensure chronological order
    df_processed = df_processed.sort_values(timestamp_col).reset_index(drop=True)
    
    return df_processed