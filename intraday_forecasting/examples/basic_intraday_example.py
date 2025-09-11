"""
Basic intraday forecasting example.

This example demonstrates how to:
1. Generate sample intraday data
2. Train an intraday predictor
3. Make predictions and evaluate performance
4. Visualize results
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from intraday_forecasting import (
    IntradayPredictor,
    create_sample_intraday_data,
    prepare_intraday_for_training
)
from tf_predictor.core.utils import split_time_series


def basic_intraday_prediction_example():
    """
    Basic example: Train a 5-minute predictor on sample data.
    """
    print("="*60)
    print("🚀 Basic Intraday Prediction Example")
    print("="*60)
    
    # 1. Generate sample data
    print("\n📊 Generating sample intraday data...")
    df = create_sample_intraday_data(n_days=7)  # 7 days of minute-level data
    print(f"   Generated {len(df)} minute-level samples")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # 2. Prepare data for 5-minute forecasting
    print("\n🔄 Preparing data for 5-minute forecasting...")
    preparation_result = prepare_intraday_for_training(
        df, 
        target_column='close',
        timeframe='5min',
        country='US',  # Default to US market
        verbose=True
    )
    
    df_processed = preparation_result['data']
    print(f"   Processed {len(df_processed)} 5-minute bars")
    
    # 3. Split data
    print("\n🔄 Splitting data...")
    train_df, val_df, test_df = split_time_series(
        df_processed, 
        test_size=50, 
        val_size=30
    )
    
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val: {len(val_df) if val_df is not None else 0} samples") 
    print(f"   Test: {len(test_df) if test_df is not None else 0} samples")
    
    # 4. Initialize and train model
    print("\n🧠 Training IntradayPredictor...")
    
    predictor = IntradayPredictor(
        target_column='close',
        timeframe='5min',
        d_token=64,      # Smaller model for quick example
        n_layers=2,      
        n_heads=4,
        verbose=True
    )
    
    # Train with fewer epochs for quick example
    predictor.fit(
        df=train_df,
        val_df=val_df,
        epochs=20,
        batch_size=16,
        learning_rate=1e-3,
        verbose=True
    )
    
    # 5. Evaluate model
    print("\n📈 Evaluating model...")
    
    train_metrics = predictor.evaluate(train_df)
    print(f"\n   Training Metrics:")
    for metric, value in train_metrics.items():
        if not np.isnan(value):
            print(f"   - {metric}: {value:.4f}")
    
    if test_df is not None:
        test_metrics = predictor.evaluate(test_df)
        print(f"\n   Test Metrics:")
        for metric, value in test_metrics.items():
            if not np.isnan(value):
                print(f"   - {metric}: {value:.4f}")
    
    # 6. Make predictions on new data
    print("\n🔮 Making predictions...")
    
    predictions = predictor.predict(test_df)
    print(f"   Generated {len(predictions)} predictions")
    
    # Show sample predictions
    if len(predictions) >= 5:
        print(f"\n   Sample predictions (first 5):")
        test_processed = predictor.prepare_features(test_df, fit_scaler=False)
        test_actual = test_processed[predictor.target_column].iloc[predictor.sequence_length:predictor.sequence_length+5]
        test_timestamps = test_df[predictor.timestamp_col].iloc[predictor.sequence_length:predictor.sequence_length+5]
        
        for i, (ts, actual, pred) in enumerate(zip(test_timestamps, test_actual, predictions[:5])):
            print(f"   {ts.strftime('%Y-%m-%d %H:%M')}: Actual={actual:.2f}, Predicted={pred:.2f}")
    
    # 7. Future predictions
    print("\n🔮 Predicting next 3 periods...")
    
    future_predictions = predictor.predict_next_bars(test_df, n_predictions=3)
    print(f"   Future predictions:")
    for _, row in future_predictions.iterrows():
        print(f"   {row[predictor.timestamp_col].strftime('%Y-%m-%d %H:%M')}: {row[f'predicted_{predictor.target_column}']:.2f}")
    
    print("\n✅ Basic example completed successfully!")
    print("="*60)
    
    return predictor, train_df, test_df


def multi_timeframe_example():
    """
    Example: Compare predictions across different timeframes.
    """
    print("\n" + "="*60)
    print("🕐 Multi-Timeframe Comparison Example")
    print("="*60)
    
    # Generate sample data
    print("\n📊 Generating sample data...")
    df = create_sample_intraday_data(n_days=10)
    
    timeframes = ['5min', '15min', '1h']
    results = {}
    
    for timeframe in timeframes:
        print(f"\n🔄 Training {timeframe} predictor...")
        
        # Prepare data
        preparation_result = prepare_intraday_for_training(
            df, target_column='close', timeframe=timeframe, country='US', verbose=False
        )
        df_processed = preparation_result['data']
        
        # Split data
        train_df, val_df, test_df = split_time_series(
            df_processed, test_size=20, val_size=15
        )
        
        if train_df is None or len(train_df) < 50:
            print(f"   ⚠️  Insufficient data for {timeframe}")
            continue
        
        # Train model
        predictor = IntradayPredictor(
            target_column='close',
            timeframe=timeframe,
            country='US',  # US market
            d_token=32,  # Small model for speed
            n_layers=2,
            verbose=False
        )
        
        predictor.fit(
            df=train_df, val_df=val_df, 
            epochs=15, batch_size=8, verbose=False
        )
        
        # Evaluate
        test_metrics = predictor.evaluate(test_df)
        mape = test_metrics.get('MAPE', np.nan)
        
        results[timeframe] = {
            'predictor': predictor,
            'mape': mape,
            'samples': len(df_processed)
        }
        
        print(f"   {timeframe} MAPE: {mape:.2f}%")
    
    # Compare results
    print(f"\n📊 Timeframe Comparison:")
    for timeframe, result in results.items():
        config = result['predictor'].get_timeframe_info()
        print(f"   {timeframe}: MAPE={result['mape']:.2f}%, "
              f"Samples={result['samples']}, "
              f"Sequence={config['sequence_length']}")
    
    print("\n✅ Multi-timeframe comparison completed!")
    
    return results


def multi_country_example():
    """
    Example: Compare predictions between US and India markets.
    """
    print("\n" + "="*60)
    print("🌍 Multi-Country Comparison Example")
    print("="*60)
    
    # Generate sample data
    print("\n📊 Generating sample data...")
    df = create_sample_intraday_data(n_days=7)
    
    countries = ['US', 'INDIA']
    results = {}
    
    for country in countries:
        print(f"\n🔄 Training predictor for {country} market...")
        
        # Prepare data for specific country
        preparation_result = prepare_intraday_for_training(
            df, target_column='close', timeframe='5min', country=country, verbose=True
        )
        df_processed = preparation_result['data']
        
        # Split data
        train_df, val_df, test_df = split_time_series(
            df_processed, test_size=30, val_size=20
        )
        
        if train_df is None or len(train_df) < 50:
            print(f"   ⚠️  Insufficient data for {country}")
            continue
        
        # Train country-specific model
        predictor = IntradayPredictor(
            target_column='close',
            timeframe='5min',
            country=country,
            d_token=64,
            n_layers=2,
            verbose=True
        )
        
        predictor.fit(
            df=train_df, val_df=val_df, 
            epochs=20, batch_size=16, verbose=True
        )
        
        # Evaluate
        test_metrics = predictor.evaluate(test_df)
        mape = test_metrics.get('MAPE', np.nan)
        
        results[country] = {
            'predictor': predictor,
            'mape': mape,
            'market_info': predictor.get_timeframe_info()
        }
        
        print(f"   {country} market MAPE: {mape:.2f}%")
    
    # Compare results
    print(f"\n🌍 Country Market Comparison:")
    for country, result in results.items():
        info = result['market_info']
        print(f"   {country} Market:")
        print(f"     - MAPE: {result['mape']:.2f}%")
        print(f"     - Market Hours: {info['market_hours']}")
        print(f"     - Timezone: {info['timezone']}")
    
    print("\n✅ Multi-country comparison completed!")
    
    return results


def volume_analysis_example():
    """
    Example: Focus on volume-based predictions.
    """
    print("\n" + "="*60)
    print("📊 Volume Analysis Example")
    print("="*60)
    
    # Generate data with more pronounced volume patterns
    print("\n📊 Generating volume-focused sample data...")
    df = create_sample_intraday_data(n_days=5)
    
    # Prepare data for volume prediction
    preparation_result = prepare_intraday_for_training(
        df, target_column='volume', timeframe='5min', country='US', verbose=True
    )
    
    df_processed = preparation_result['data']
    
    # Split data
    train_df, val_df, test_df = split_time_series(df_processed, test_size=40, val_size=25)
    
    # Train volume predictor
    print("\n🧠 Training volume predictor...")
    
    volume_predictor = IntradayPredictor(
        target_column='volume',
        timeframe='5min',
        country='US',
        d_token=64,
        n_layers=2,
        verbose=True
    )
    
    volume_predictor.fit(
        df=train_df, val_df=val_df,
        epochs=25, batch_size=16,
        verbose=True
    )
    
    # Evaluate volume prediction
    test_metrics = volume_predictor.evaluate(test_df)
    print(f"\n📈 Volume Prediction Results:")
    for metric, value in test_metrics.items():
        if not np.isnan(value):
            print(f"   - {metric}: {value:.4f}")
    
    # Analyze volume patterns
    print(f"\n📊 Volume Pattern Analysis:")
    
    # Get predictions
    predictions = volume_predictor.predict(test_df)
    test_processed = volume_predictor.prepare_features(test_df, fit_scaler=False)
    actual_volumes = test_processed['volume'].iloc[volume_predictor.sequence_length:]
    
    # Basic volume statistics
    print(f"   Actual volume range: {actual_volumes.min():.0f} - {actual_volumes.max():.0f}")
    print(f"   Predicted volume range: {min(predictions):.0f} - {max(predictions):.0f}")
    print(f"   Mean actual volume: {actual_volumes.mean():.0f}")
    print(f"   Mean predicted volume: {np.mean(predictions):.0f}")
    
    print("\n✅ Volume analysis completed!")
    
    return volume_predictor


if __name__ == "__main__":
    # Run all examples
    
    # 1. Basic example
    predictor, train_df, test_df = basic_intraday_prediction_example()
    
    # 2. Multi-timeframe example
    multi_results = multi_timeframe_example()
    
    # 3. Multi-country example
    country_results = multi_country_example()
    
    # 4. Volume analysis example
    volume_predictor = volume_analysis_example()
    
    print("\n" + "="*60)
    print("🎉 All examples completed successfully!")
    print("="*60)
    
    print(f"\nNext steps:")
    print(f"1. Try different countries: --country US or --country INDIA")
    print(f"2. Try different timeframes: 1min, 15min, 1h")
    print(f"3. Experiment with model architecture (d_token, n_layers)")
    print(f"4. Use real intraday data with load_intraday_data()")
    print(f"5. Run the main CLI: python main.py --use_sample_data --timeframe 5min --country US")