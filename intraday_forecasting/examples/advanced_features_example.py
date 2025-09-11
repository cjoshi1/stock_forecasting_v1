"""
Advanced intraday features example.

This example demonstrates:
1. Advanced feature engineering for intraday data
2. Market session analysis
3. Time-of-day effect modeling  
4. Volume microstructure features
5. Custom feature creation
"""

import pandas as pd
import numpy as np
from datetime import datetime, time

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from intraday_forecasting import (
    IntradayPredictor,
    create_sample_intraday_data,
    create_intraday_features,
    create_intraday_time_features,
    create_intraday_volume_features
)
from tf_predictor.core.utils import split_time_series


def analyze_market_sessions(df):
    """
    Analyze trading patterns across different market sessions.
    
    Args:
        df: DataFrame with intraday data
        
    Returns:
        Dictionary with session analysis
    """
    print("\n📊 Market Session Analysis:")
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add hour for analysis
    df['hour'] = df['timestamp'].dt.hour
    
    # Define market sessions
    sessions = {
        'Opening (9:30-10:30)': df[df['hour'] == 9],
        'Morning (10:30-12:00)': df[df['hour'].between(10, 11)],
        'Lunch (12:00-14:00)': df[df['hour'].between(12, 13)], 
        'Afternoon (14:00-15:00)': df[df['hour'] == 14],
        'Power Hour (15:00-16:00)': df[df['hour'] == 15]
    }
    
    session_stats = {}
    
    for session_name, session_data in sessions.items():
        if len(session_data) == 0:
            continue
            
        # Calculate session statistics
        avg_volume = session_data['volume'].mean()
        avg_volatility = session_data['close'].pct_change().std() * 100
        avg_range = ((session_data['high'] - session_data['low']) / session_data['open'] * 100).mean()
        
        session_stats[session_name] = {
            'avg_volume': avg_volume,
            'avg_volatility': avg_volatility,
            'avg_range': avg_range,
            'sample_count': len(session_data)
        }
        
        print(f"   {session_name}:")
        print(f"     - Avg Volume: {avg_volume:,.0f}")
        print(f"     - Avg Volatility: {avg_volatility:.3f}%")
        print(f"     - Avg Range: {avg_range:.3f}%")
        print(f"     - Samples: {len(session_data)}")
    
    return session_stats


def create_custom_microstructure_features(df):
    """
    Create custom microstructure features for high-frequency trading.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional microstructure features
    """
    print("\n🔬 Creating custom microstructure features...")
    
    df_enhanced = df.copy()
    
    # 1. Price impact measures
    df_enhanced['price_impact_5'] = (
        df_enhanced['close'] - df_enhanced['close'].shift(5)
    ) / (df_enhanced['volume'].rolling(5).sum() / 1e6)  # Normalized by volume
    
    # 2. Volume-weighted return
    df_enhanced['vwap_return'] = df_enhanced['close'].pct_change() * df_enhanced['volume']
    df_enhanced['vwap_return_5'] = df_enhanced['vwap_return'].rolling(5).sum()
    
    # 3. Bid-ask spread proxy (using high-low)
    df_enhanced['spread_proxy'] = (df_enhanced['high'] - df_enhanced['low']) / df_enhanced['close'] * 10000  # in basis points
    df_enhanced['spread_ma'] = df_enhanced['spread_proxy'].rolling(10).mean()
    df_enhanced['spread_ratio'] = df_enhanced['spread_proxy'] / df_enhanced['spread_ma']
    
    # 4. Order flow proxy
    # Up moves with volume vs down moves with volume
    up_condition = df_enhanced['close'] > df_enhanced['close'].shift(1)
    df_enhanced['buy_volume_proxy'] = np.where(up_condition, df_enhanced['volume'], 0)
    df_enhanced['sell_volume_proxy'] = np.where(~up_condition, df_enhanced['volume'], 0)
    
    df_enhanced['buy_sell_ratio'] = (
        df_enhanced['buy_volume_proxy'].rolling(10).sum() / 
        (df_enhanced['sell_volume_proxy'].rolling(10).sum() + 1)
    )
    
    # 5. Tick-by-tick intensity
    df_enhanced['tick_intensity'] = df_enhanced['volume'] / (
        abs(df_enhanced['close'] - df_enhanced['open']) + 1e-8
    )
    df_enhanced['tick_intensity_ma'] = df_enhanced['tick_intensity'].rolling(15).mean()
    
    # 6. Market maker vs taker activity proxy
    # High volume with small price moves = market making
    # High volume with large price moves = taking
    price_move = abs(df_enhanced['close'] - df_enhanced['open']) / df_enhanced['open']
    volume_normalized = df_enhanced['volume'] / df_enhanced['volume'].rolling(20).mean()
    
    df_enhanced['market_making_proxy'] = np.where(
        (volume_normalized > 1.2) & (price_move < 0.001),
        volume_normalized, 0
    )
    df_enhanced['taking_proxy'] = np.where(
        (volume_normalized > 1.0) & (price_move > 0.002),
        volume_normalized, 0
    )
    
    # Fill NaN values
    df_enhanced = df_enhanced.fillna(method='bfill').fillna(0)
    
    new_features = [col for col in df_enhanced.columns if col not in df.columns]
    print(f"   Created {len(new_features)} custom features:")
    for feature in new_features[:8]:  # Show first 8
        print(f"     - {feature}")
    if len(new_features) > 8:
        print(f"     ... and {len(new_features) - 8} more")
    
    return df_enhanced


def time_of_day_prediction_example():
    """
    Example focusing on time-of-day effects in predictions.
    """
    print("="*60)
    print("🕐 Time-of-Day Effects Example")
    print("="*60)
    
    # Generate longer sample data for better time patterns
    print("\n📊 Generating extended sample data...")
    df = create_sample_intraday_data(n_days=10)
    print(f"   Generated {len(df)} minute-level samples")
    
    # Analyze market sessions
    session_stats = analyze_market_sessions(df)
    
    # Add basic intraday features
    print("\n🔧 Adding time-based features...")
    df_with_time = create_intraday_time_features(df)
    
    # Resample to 5-minute for analysis
    print("\n🔄 Resampling to 5-minute bars...")
    from intraday_forecasting.preprocessing.timeframe_utils import prepare_intraday_data
    df_5min = prepare_intraday_data(df_with_time, '5min')
    
    # Add all intraday features
    df_features = create_intraday_features(df_5min, verbose=True)
    
    # Split data
    train_df, val_df, test_df = split_time_series(df_features, test_size=100, val_size=50)
    
    # Train model with time-aware features
    print("\n🧠 Training time-aware predictor...")
    predictor = IntradayPredictor(
        target_column='close',
        timeframe='5min',
        d_token=96,
        n_layers=3,
        n_heads=6,
        verbose=True
    )
    
    predictor.fit(
        df=train_df, val_df=val_df,
        epochs=30, batch_size=16,
        verbose=True
    )
    
    # Evaluate model
    test_metrics = predictor.evaluate(test_df)
    print(f"\n📈 Time-Aware Model Results:")
    for metric, value in test_metrics.items():
        if not np.isnan(value):
            print(f"   - {metric}: {value:.4f}")
    
    # Analyze predictions by time of day
    print("\n🕐 Analyzing predictions by time of day...")
    predictions = predictor.predict(test_df)
    test_processed = predictor.prepare_features(test_df, fit_scaler=False)
    
    # Get timestamps and actual values
    test_timestamps = test_df['timestamp'].iloc[predictor.sequence_length:]
    actual_values = test_processed[predictor.target_column].iloc[predictor.sequence_length:]
    
    # Group by hour
    prediction_df = pd.DataFrame({
        'timestamp': test_timestamps.reset_index(drop=True),
        'actual': actual_values.reset_index(drop=True),
        'predicted': predictions[:len(actual_values)]
    })
    
    prediction_df['hour'] = prediction_df['timestamp'].dt.hour
    prediction_df['error'] = abs(prediction_df['actual'] - prediction_df['predicted'])
    
    hourly_performance = prediction_df.groupby('hour').agg({
        'error': ['mean', 'std', 'count'],
        'actual': 'mean',
        'predicted': 'mean'
    }).round(4)
    
    print(f"\n   Hourly Prediction Performance:")
    print(f"   {'Hour':<6} {'Samples':<8} {'Avg Error':<10} {'Std Error':<10}")
    print(f"   {'-'*40}")
    
    for hour in sorted(prediction_df['hour'].unique()):
        hour_data = hourly_performance.loc[hour]
        samples = int(hour_data[('error', 'count')])
        avg_error = hour_data[('error', 'mean')]
        std_error = hour_data[('error', 'std')]
        
        print(f"   {hour:<6} {samples:<8} {avg_error:<10.4f} {std_error:<10.4f}")
    
    return predictor, session_stats


def microstructure_prediction_example():
    """
    Example using advanced microstructure features.
    """
    print("\n" + "="*60)
    print("🔬 Microstructure Features Example")
    print("="*60)
    
    # Generate sample data
    print("\n📊 Generating sample data...")
    df = create_sample_intraday_data(n_days=8)
    
    # Create microstructure features
    df_micro = create_custom_microstructure_features(df)
    
    # Prepare for training
    from intraday_forecasting.preprocessing.timeframe_utils import prepare_intraday_data
    df_5min = prepare_intraday_data(df_micro, '5min')
    
    # Add standard intraday features
    df_features = create_intraday_features(df_5min, verbose=False)
    
    print(f"   Total features: {len(df_features.columns) - 1}")  # Subtract timestamp
    
    # Split data
    train_df, val_df, test_df = split_time_series(df_features, test_size=80, val_size=40)
    
    # Train microstructure-aware model
    print("\n🧠 Training microstructure-aware predictor...")
    
    micro_predictor = IntradayPredictor(
        target_column='close',
        timeframe='5min',
        d_token=128,
        n_layers=4,
        n_heads=8,
        dropout=0.15,  # More regularization for complex features
        verbose=True
    )
    
    micro_predictor.fit(
        df=train_df, val_df=val_df,
        epochs=40, batch_size=12,
        learning_rate=5e-4,  # Lower learning rate
        verbose=True
    )
    
    # Evaluate microstructure model
    test_metrics = micro_predictor.evaluate(test_df)
    print(f"\n📈 Microstructure Model Results:")
    for metric, value in test_metrics.items():
        if not np.isnan(value):
            print(f"   - {metric}: {value:.4f}")
    
    # Compare with baseline (no microstructure features)
    print("\n📊 Comparing with baseline model...")
    
    # Create baseline data (standard features only)
    df_baseline = create_intraday_features(df_5min[['timestamp', 'open', 'high', 'low', 'close', 'volume']], verbose=False)
    train_base, val_base, test_base = split_time_series(df_baseline, test_size=80, val_size=40)
    
    baseline_predictor = IntradayPredictor(
        target_column='close',
        timeframe='5min',
        d_token=128,
        n_layers=4,
        n_heads=8,
        verbose=False
    )
    
    baseline_predictor.fit(
        df=train_base, val_df=val_base,
        epochs=40, batch_size=12,
        learning_rate=5e-4,
        verbose=False
    )
    
    baseline_metrics = baseline_predictor.evaluate(test_base)
    
    print(f"\n   Model Comparison:")
    print(f"   {'Metric':<20} {'Microstructure':<15} {'Baseline':<15} {'Improvement':<15}")
    print(f"   {'-'*70}")
    
    for metric in ['MAE', 'RMSE', 'MAPE']:
        if metric in test_metrics and metric in baseline_metrics:
            micro_val = test_metrics[metric]
            base_val = baseline_metrics[metric]
            improvement = ((base_val - micro_val) / base_val * 100) if base_val != 0 else 0
            
            print(f"   {metric:<20} {micro_val:<15.4f} {base_val:<15.4f} {improvement:<15.2f}%")
    
    return micro_predictor, baseline_predictor


if __name__ == "__main__":
    # Run advanced examples
    
    # 1. Time-of-day effects
    time_predictor, session_stats = time_of_day_prediction_example()
    
    # 2. Microstructure features
    micro_predictor, baseline_predictor = microstructure_prediction_example()
    
    print("\n" + "="*60)
    print("🎉 Advanced features examples completed!")
    print("="*60)
    
    print(f"\nKey takeaways:")
    print(f"1. Different market sessions have distinct patterns")
    print(f"2. Time-of-day features can improve prediction accuracy")
    print(f"3. Microstructure features add predictive value for intraday")
    print(f"4. More complex models require careful regularization")
    
    print(f"\nNext steps:")
    print(f"1. Experiment with feature selection techniques")
    print(f"2. Try ensemble methods combining multiple timeframes")
    print(f"3. Implement real-time feature calculation")
    print(f"4. Add regime detection (high/low volatility periods)")