"""
Stock-specific feature engineering for OHLCV data.
"""

import pandas as pd
import numpy as np
from tf_predictor.preprocessing.time_features import create_date_features, create_rolling_features


def create_stock_features(df: pd.DataFrame, target_column: str, verbose: bool = False) -> pd.DataFrame:
    """
    Create comprehensive stock-specific features from OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data and optional date column
        target_column: Target column name
        verbose: Whether to print verbose information
        
    Returns:
        processed_df: DataFrame with engineered features
    """
    df_processed = df.copy()
    
    # Create date features if date column exists
    if 'date' in df_processed.columns:
        df_processed = create_date_features(df_processed, 'date')
    
    # Basic OHLCV features (excluding target)
    base_features = ['open', 'high', 'low', 'volume']
    if target_column != 'close':
        base_features.append('close')
    if target_column != 'open':  
        base_features = [f for f in base_features if f != 'open'] + ['open']
    if target_column not in ['high', 'low', 'volume']:
        base_features = [f for f in base_features if f not in ['high', 'low', 'volume']] + \
                      [f for f in ['high', 'low', 'volume'] if f != target_column]
    
    # Add technical indicators
    if 'close' in df_processed.columns:
        # Simple returns
        df_processed['returns'] = df_processed['close'].pct_change()
        df_processed['log_returns'] = np.log(df_processed['close'] / df_processed['close'].shift(1))
        
        # Custom percentage change calculations with specific periods (optimized selection)
        pct_change_periods = [1, 3, 5, 10]  # Focused on key periods to reduce feature count
        for period in pct_change_periods:
            if len(df_processed) > period:
                # Calculate percentage change over specific periods
                df_processed[f'pct_change_{period}d'] = ((df_processed['close'] - df_processed['close'].shift(period)) 
                                                       / df_processed['close'].shift(period)) * 100
                
                # Only calculate for close price to reduce feature complexity
                # Other OHLC columns can be added later if needed
        
        # Additional technical indicators (simplified)
        # Basic volatility (rolling standard deviation of 1-day returns)
        if len(df_processed) >= 10:
            df_processed['volatility_10d'] = df_processed['pct_change_1d'].rolling(10).std()
        
        # Simple momentum indicators
        if len(df_processed) >= 5:
            df_processed['momentum_5d'] = df_processed['pct_change_1d'].rolling(5).mean()
        
        # Price ratios
        df_processed['high_low_ratio'] = df_processed['high'] / df_processed['low']
        df_processed['close_open_ratio'] = df_processed['close'] / df_processed['open']
    
    # Use standardized rolling features from tf_predictor
    rolling_windows = [5]
    
    # Rolling features for volume
    if 'volume' in df_processed.columns:
        df_processed = create_rolling_features(df_processed, 'volume', rolling_windows)
        # Volume ratio using standardized rolling mean
        df_processed['volume_ratio'] = df_processed['volume'] / df_processed['volume_rolling_mean_5']
    
    # Rolling features for price columns (if we have enough data)
    for col in ['open', 'high', 'low', 'close']:
        if col in df_processed.columns and len(df_processed) >= 5:
            df_processed = create_rolling_features(df_processed, col, rolling_windows)
            
    # Fill NaN values
    df_processed = df_processed.bfill().fillna(0)
    
    return df_processed


def create_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced technical indicators for stock data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        df: DataFrame with additional technical indicators
    """
    df = df.copy()
    
    # RSI (Relative Strength Index)
    if 'close' in df.columns:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    if 'close' in df.columns:
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    if 'close' in df.columns:
        rolling_mean = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = rolling_mean + (rolling_std * 2)
        df['bb_lower'] = rolling_mean - (rolling_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
    
    return df