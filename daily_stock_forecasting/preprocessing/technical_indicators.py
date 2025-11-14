"""
Technical indicator calculations for stock trading.

Implements standard technical indicators used in trading analysis:
- RSI (Relative Strength Index)
- Bollinger Bands
- Relative Volume
- Intraday Momentum
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.

    Args:
        prices: Series of closing prices
        period: Number of periods for RSI calculation (default: 14)

    Returns:
        Series of RSI values (0-100)
    """
    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # Calculate average gains and losses using exponential moving average
    avg_gains = gains.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_losses = losses.ewm(span=period, min_periods=period, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    # Handle division by zero (when avg_losses = 0)
    rsi = rsi.fillna(100)

    return rsi


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands.

    Bollinger Bands consist of a middle band (SMA) and upper/lower bands
    at specified standard deviations from the middle.

    Args:
        prices: Series of closing prices
        period: Number of periods for moving average (default: 20)
        num_std: Number of standard deviations for bands (default: 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    # Middle band is simple moving average
    middle_band = prices.rolling(window=period).mean()

    # Calculate standard deviation
    std_dev = prices.rolling(window=period).std()

    # Upper and lower bands
    upper_band = middle_band + (num_std * std_dev)
    lower_band = middle_band - (num_std * std_dev)

    return upper_band, middle_band, lower_band


def calculate_bb_position(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Calculate Bollinger Band Position.

    Normalized position of price within the Bollinger Bands:
    - 0.0 = at lower band
    - 0.5 = at middle band
    - 1.0 = at upper band

    Args:
        prices: Series of closing prices
        period: Number of periods for Bollinger Bands (default: 20)
        num_std: Number of standard deviations (default: 2.0)

    Returns:
        Series of BB position values (typically 0-1, but can exceed)
    """
    upper_band, middle_band, lower_band = calculate_bollinger_bands(prices, period, num_std)

    # Calculate position within bands
    band_width = upper_band - lower_band
    bb_position = (prices - lower_band) / band_width

    # Handle division by zero (when bands collapse)
    bb_position = bb_position.fillna(0.5)

    return bb_position


def calculate_relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Relative Volume.

    Ratio of current volume to its moving average.
    Values > 1.0 indicate above-average volume.

    Args:
        volume: Series of volume data
        period: Number of periods for moving average (default: 20)

    Returns:
        Series of relative volume values
    """
    # Calculate moving average of volume
    avg_volume = volume.rolling(window=period).mean()

    # Calculate relative volume
    relative_vol = volume / avg_volume

    # Handle division by zero
    relative_vol = relative_vol.fillna(1.0)

    return relative_vol


def calculate_intraday_momentum(open_prices: pd.Series, close_prices: pd.Series) -> pd.Series:
    """
    Calculate Intraday Momentum.

    Percentage change from open to close within the same day.
    Positive values indicate bullish intraday action.

    Args:
        open_prices: Series of opening prices
        close_prices: Series of closing prices

    Returns:
        Series of intraday momentum values (as percentage)
    """
    # Calculate (close - open) / open
    momentum = (close_prices - open_prices) / open_prices

    # Handle division by zero
    momentum = momentum.fillna(0.0)

    return momentum


def calculate_technical_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    vol_period: int = 20,
    group_column: Optional[str] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Calculate all technical indicators for stock data.

    Adds the following columns to the dataframe:
    - relative_volume: Volume relative to its MA
    - intraday_momentum: (close - open) / open
    - rsi_14: Relative Strength Index
    - bb_position: Position within Bollinger Bands

    Args:
        df: DataFrame with OHLCV data (must have: open, high, low, close, volume)
        rsi_period: Period for RSI calculation (default: 14)
        bb_period: Period for Bollinger Bands (default: 20)
        bb_std: Standard deviations for Bollinger Bands (default: 2.0)
        vol_period: Period for volume MA (default: 20)
        group_column: Optional column to group by (e.g., 'symbol' for multi-stock data)
                     If provided, indicators are calculated separately for each group
        verbose: Whether to print calculation info

    Returns:
        DataFrame with additional technical indicator columns
    """
    # Verify required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if verbose:
        print(f"\n📊 Calculating Technical Indicators:")
        print(f"   RSI Period: {rsi_period}")
        print(f"   Bollinger Bands: {bb_period} days, {bb_std}σ")
        print(f"   Relative Volume: {vol_period} days MA")
        if group_column:
            print(f"   Grouping by: {group_column}")

    def _calculate_indicators_for_group(group_df):
        """Helper function to calculate indicators for a single group."""
        group_out = group_df.copy()

        # Calculate each indicator
        group_out['relative_volume'] = calculate_relative_volume(group_df['volume'], period=vol_period)
        group_out['intraday_momentum'] = calculate_intraday_momentum(group_df['open'], group_df['close'])
        group_out['rsi_14'] = calculate_rsi(group_df['close'], period=rsi_period)
        group_out['bb_position'] = calculate_bb_position(group_df['close'], period=bb_period, num_std=bb_std)

        return group_out

    # Calculate indicators (grouped or ungrouped)
    if group_column:
        if group_column not in df.columns:
            raise ValueError(f"Group column '{group_column}' not found in dataframe")

        # Calculate indicators separately for each group
        # Note: We use include_groups=True (pandas 2.2+) to keep the grouping column
        # For older pandas, this parameter is ignored and grouping columns are included by default
        try:
            df_out = df.groupby(group_column, group_keys=False).apply(_calculate_indicators_for_group, include_groups=True)
        except TypeError:
            # Fallback for older pandas versions that don't support include_groups
            df_out = df.groupby(group_column, group_keys=False).apply(_calculate_indicators_for_group)
    else:
        # Calculate indicators for entire dataframe
        df_out = _calculate_indicators_for_group(df)

    if verbose:
        # Count how many valid (non-NaN) values we have for each indicator
        print(f"\n   Indicators calculated:")
        print(f"   - relative_volume: {df_out['relative_volume'].notna().sum()}/{len(df_out)} valid values")
        print(f"   - intraday_momentum: {df_out['intraday_momentum'].notna().sum()}/{len(df_out)} valid values")
        print(f"   - rsi_14: {df_out['rsi_14'].notna().sum()}/{len(df_out)} valid values")
        print(f"   - bb_position: {df_out['bb_position'].notna().sum()}/{len(df_out)} valid values")

        # Show first few NaN rows (warming up period)
        first_valid_idx = df_out[['relative_volume', 'rsi_14', 'bb_position']].notna().all(axis=1).idxmax()
        if first_valid_idx > 0:
            print(f"   ⚠️  First {first_valid_idx} rows have NaN (indicator warm-up period)")

    return df_out
