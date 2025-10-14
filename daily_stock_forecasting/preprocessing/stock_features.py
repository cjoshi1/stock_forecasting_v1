"""
Stock-specific feature engineering for OHLCV data.
"""

import pandas as pd
import numpy as np
from tf_predictor.preprocessing.time_features import create_date_features, create_rolling_features


def create_stock_features(df: pd.DataFrame, target_column: str, verbose: bool = False, use_essential_only: bool = False, prediction_horizon: int = 1, asset_type: str = 'stock') -> pd.DataFrame:
    """
    Create comprehensive stock-specific features from OHLCV data.

    Args:
        df: DataFrame with OHLCV data and optional date column
        target_column: Target column name
        verbose: Whether to print verbose information
        use_essential_only: If True, only create essential features (volume, typical_price, seasonal)
        prediction_horizon: Number of time steps to shift target (1 = predict next step)
        asset_type: Type of asset - 'stock' (5-day week) or 'crypto' (7-day week)

    Returns:
        processed_df: DataFrame with engineered features and shifted target
    """
    df_processed = df.copy()

    # Always calculate typical_price if we have OHLC data
    if all(col in df_processed.columns for col in ['high', 'low', 'close']):
        df_processed['typical_price'] = (df_processed['high'] + df_processed['low'] + df_processed['close']) / 3
        if verbose:
            print("   Added typical_price: (high + low + close) / 3")

    # Handle essential features mode
    if use_essential_only:
        return _create_essential_features(df_processed, target_column, verbose, prediction_horizon, asset_type)

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

    # Create shifted target variable
    df_processed = _create_shifted_target(df_processed, target_column, prediction_horizon, verbose)

    return df_processed


def _create_essential_features(df: pd.DataFrame, target_column: str, verbose: bool = False, prediction_horizon: int = 1, asset_type: str = 'stock') -> pd.DataFrame:
    """
    Create only essential features: volume, typical_price, and seasonal features.

    Args:
        df: DataFrame with OHLC data and optional date column
        target_column: Target column name
        verbose: Whether to print verbose information
        prediction_horizon: Number of time steps to shift target
        asset_type: Type of asset - 'stock' (5-day week) or 'crypto' (7-day week)

    Returns:
        processed_df: DataFrame with essential features only and shifted target
    """
    from tf_predictor.preprocessing.time_features import create_cyclical_features

    df_processed = df.copy()

    # Extract seasonal features from date column using tf_predictor
    if 'date' in df_processed.columns:
        # Use tf_predictor's cyclical features function with the date column
        df_processed = create_cyclical_features(df_processed, 'date', ['month', 'dayofweek', 'year'])

        # Add crypto-specific features if needed
        if asset_type == 'crypto':
            # Add weekend indicator for crypto (different behavior on weekends)
            df_processed['is_weekend'] = df_processed['date'].dt.dayofweek.isin([5, 6]).astype(int)
            if verbose:
                print("   Added crypto-specific feature: is_weekend")

        # Remove original temporal columns and date column (keep year as requested)
        df_processed = df_processed.drop(['date', 'month', 'dayofweek'], axis=1)

        if verbose:
            print("   Added cyclical seasonal features: month_sin, month_cos, dayofweek_sin, dayofweek_cos")
            if asset_type == 'crypto':
                print("   Crypto mode: dayofweek uses 7-day week (0=Sunday, 6=Saturday)")
            else:
                print("   Stock mode: dayofweek uses 5-day week (0=Monday, 4=Friday)")
            print("   Kept year as raw value (not cyclical)")
            print("   Removed original date column after extraction")

    # Keep only essential features: volume, typical_price, and seasonal features
    essential_features = ['volume', 'typical_price', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'year']

    # Add crypto-specific features to essential list
    if asset_type == 'crypto':
        essential_features.append('is_weekend')

    # Always include target column in essential features (target can be volume, typical_price, etc.)
    if target_column not in essential_features:
        essential_features.append(target_column)

    # Filter to only essential features that exist in the dataframe
    available_essential = [col for col in essential_features if col in df_processed.columns]
    df_processed = df_processed[available_essential]

    # Fill NaN values
    df_processed = df_processed.bfill().fillna(0)

    # Create shifted target variable
    df_processed = _create_shifted_target(df_processed, target_column, prediction_horizon, verbose)

    if verbose:
        print(f"   Essential features only: {len(available_essential)} features")
        print(f"   Features: {available_essential}")

    return df_processed


def _create_shifted_target(df: pd.DataFrame, target_column: str, prediction_horizon: int, verbose: bool = False) -> pd.DataFrame:
    """
    Create shifted target variable(s) for time series prediction.
    Single horizon: Creates one target column
    Multi-horizon: Creates multiple target columns for horizons 1 to N

    Args:
        df: DataFrame with features
        target_column: Name of the original target column
        prediction_horizon: Number of steps to predict (1 = single, >1 = multi-horizon)
        verbose: Whether to print information

    Returns:
        df: DataFrame with shifted target column(s)
    """
    df = df.copy()

    # Validate target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    if prediction_horizon == 1:
        # Single horizon (existing behavior)
        shifted_target_name = f"{target_column}_target_h1"
        df[shifted_target_name] = df[target_column].shift(-1)
        df = df.dropna(subset=[shifted_target_name])

        if verbose:
            print(f"   Created single target: {shifted_target_name}")
            print(f"   Prediction horizon: 1 step ahead")
            print(f"   Remaining samples after shift: {len(df)}")

    else:
        # Multi-horizon (new behavior)
        target_columns = []
        for h in range(1, prediction_horizon + 1):
            col_name = f"{target_column}_target_h{h}"
            df[col_name] = df[target_column].shift(-h)
            target_columns.append(col_name)

        # Remove rows where ANY target is NaN
        df = df.dropna(subset=target_columns)

        if verbose:
            print(f"   Created multi-horizon targets: {target_columns}")
            print(f"   Prediction horizons: 1 to {prediction_horizon} steps ahead")
            print(f"   Remaining samples after shift: {len(df)}")

    return df


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