#!/usr/bin/env python3
"""
Stock Price Prediction Example using Stock-Specific FT-Transformer

This example demonstrates how to use the stock_forecasting library for 
stock price and return prediction. It shows:
1. Loading and validating stock data
2. Using the StockPredictor with stock-specific features
3. Training models for different targets (price, returns, percentage changes)
4. Evaluating and visualizing results
5. Making future predictions
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from daily_stock_forecasting.predictor import StockPredictor
from daily_stock_forecasting.preprocessing.market_data import load_stock_data, create_sample_stock_data
from daily_stock_forecasting.preprocessing.stock_features import create_stock_features
from tf_predictor.core.utils import split_time_series, calculate_metrics


def demonstrate_basic_stock_prediction():
    """Basic stock price prediction example."""
    print("\\n" + "="*60)
    print("📈 Basic Stock Price Prediction")
    print("="*60)
    
    # Create sample stock data
    print("\\n📊 Creating synthetic stock data...")
    df = create_sample_stock_data(n_samples=500)  # About 2 years of trading days
    print(f"   Generated {len(df)} trading days of OHLCV data")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Split data
    train_df, val_df, test_df = split_time_series(df, test_size=60, val_size=30)
    print(f"   Train: {len(train_df)} days, Val: {len(val_df)} days, Test: {len(test_df)} days")
    
    # Initialize stock predictor
    predictor = StockPredictor(
        target_column='close',
        sequence_length=10,  # Use 10 days of history
        d_token=64,         # Smaller for faster training
        n_layers=2,
        n_heads=4,
        verbose=True
    )
    
    print(f"\\n🧠 Training model to predict: {predictor.target_column}")
    
    # Train model
    predictor.fit(
        df=train_df,
        val_df=val_df,
        epochs=5,  # Quick training for example
        batch_size=32,
        learning_rate=1e-3,
        verbose=True
    )
    
    # Evaluate
    test_metrics = predictor.evaluate(test_df)
    print("\\n📊 Test Results:")
    for metric, value in test_metrics.items():
        if not np.isnan(value):
            unit = "$" if metric in ['MAE', 'RMSE'] else ("%" if 'MAPE' in metric else "")
            print(f"   {metric}: {value:.3f}{unit}")
    
    return predictor, test_df


def demonstrate_return_prediction():
    """Demonstrate predicting stock returns instead of prices."""
    print("\\n" + "="*60)
    print("📊 Stock Return Prediction")  
    print("="*60)
    
    # Create sample data
    df = create_sample_stock_data(n_samples=400)
    train_df, val_df, test_df = split_time_series(df, test_size=50, val_size=25)
    
    print(f"\\n🎯 Training model to predict 1-day percentage change...")
    
    # Initialize predictor for percentage changes
    return_predictor = StockPredictor(
        target_column='pct_change_1d',  # Predict 1-day percentage change
        sequence_length=7,
        d_token=64,
        n_layers=2,
        verbose=True
    )
    
    # Train model
    return_predictor.fit(
        df=train_df,
        val_df=val_df,
        epochs=5,
        batch_size=32,
        verbose=True
    )
    
    # Evaluate
    test_metrics = return_predictor.evaluate(test_df)
    print("\\n📊 Percentage Change Prediction Results:")
    for metric, value in test_metrics.items():
        if not np.isnan(value):
            if metric == 'MAPE':
                print(f"   {metric}: {value:.1f}% (high MAPE is expected for percentage changes)")
            else:
                print(f"   {metric}: {value:.3f}")
    
    return return_predictor


def demonstrate_feature_engineering():
    """Show the stock-specific features being created."""
    print("\\n" + "="*60)
    print("🔧 Stock Feature Engineering")
    print("="*60)
    
    # Create small sample to show features
    df = create_sample_stock_data(n_samples=50)
    print(f"\\n📊 Original data columns: {list(df.columns)}")
    
    # Create stock features
    df_features = create_stock_features(df, target_column='close', verbose=True)
    
    print(f"\\n🎯 After feature engineering: {len(df_features.columns)} total columns")
    
    # Categorize features
    feature_names = list(df_features.columns)
    
    # Original OHLCV features
    ohlcv_features = [f for f in feature_names if f in ['open', 'high', 'low', 'close', 'volume', 'date']]
    
    # Date-based features
    date_features = [f for f in feature_names if any(x in f for x in ['year', 'month', 'day', 'weekend', 'quarter', '_sin', '_cos'])]
    
    # Price-based technical features  
    price_features = [f for f in feature_names if any(x in f for x in ['returns', 'pct_change', 'volatility', 'momentum'])]
    
    # Ratio features
    ratio_features = [f for f in feature_names if 'ratio' in f]
    
    # Moving average features
    ma_features = [f for f in feature_names if '_ma_' in f]
    
    # Volume features
    volume_features = [f for f in feature_names if f.startswith('volume') and f != 'volume']
    
    print(f"\\n📋 Feature Categories:")
    print(f"   📊 OHLCV features ({len(ohlcv_features)}): {', '.join(ohlcv_features)}")
    print(f"   📅 Date features ({len(date_features)}): {', '.join(date_features[:3])}{'...' if len(date_features) > 3 else ''}")
    print(f"   📈 Price technical ({len(price_features)}): {', '.join(price_features[:3])}{'...' if len(price_features) > 3 else ''}")
    print(f"   ⚖️  Ratios ({len(ratio_features)}): {', '.join(ratio_features)}")
    print(f"   📊 Moving averages ({len(ma_features)}): {', '.join(ma_features[:3])}{'...' if len(ma_features) > 3 else ''}")
    print(f"   📦 Volume features ({len(volume_features)}): {', '.join(volume_features)}")
    
    # Show sample values for key features
    print(f"\\n💡 Sample Feature Values (last 5 days):")
    key_features = ['close', 'pct_change_1d', 'volatility_10d', 'high_low_ratio', 'returns']
    display_features = [f for f in key_features if f in df_features.columns]
    
    sample_data = df_features[display_features].tail()
    for col in display_features:
        values = sample_data[col].values
        print(f"   {col:15}: {' '.join(f'{v:7.3f}' for v in values)}")


def demonstrate_real_data_usage():
    """Show how to use the library with real stock data."""
    print("\\n" + "="*60)
    print("🏢 Real Stock Data Usage")
    print("="*60)
    
    # Check if sample data exists
    sample_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample', 'MSFT_sample.csv')
    
    if os.path.exists(sample_file):
        print(f"\\n📈 Loading real MSFT data from: {sample_file}")
        
        try:
            df = load_stock_data(sample_file)
            print(f"   Loaded {len(df)} trading days")
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            
            # Show data quality
            print(f"\\n📊 Data Quality:")
            print(f"   Missing values: {df.isnull().sum().sum()}")
            print(f"   Duplicate dates: {df['date'].duplicated().sum()}")
            
            # Quick prediction example
            if len(df) >= 100:  # Ensure we have enough data
                print(f"\\n🚀 Quick prediction example with real data...")
                
                # Use smaller dataset for quick demo
                recent_df = df.tail(100)  # Last 100 days
                train_df, val_df, test_df = split_time_series(recent_df, test_size=15, val_size=10)
                
                quick_predictor = StockPredictor(
                    target_column='close',
                    sequence_length=5,
                    d_token=32,  # Very small for quick demo
                    n_layers=1,
                    verbose=False
                )
                
                # Quick training
                quick_predictor.fit(
                    df=train_df,
                    val_df=val_df,
                    epochs=2,
                    batch_size=16,
                    verbose=False
                )
                
                metrics = quick_predictor.evaluate(test_df)
                print(f"   Quick model MAPE: {metrics.get('MAPE', 'N/A'):.2f}%")
                
            else:
                print(f"   Not enough data for prediction demo (need at least 100 samples)")
                
        except Exception as e:
            print(f"   Error loading data: {e}")
            
    else:
        print(f"\\n💡 Real data usage:")
        print(f"   1. Place your CSV file in: stock_forecasting/data/raw/your_data.csv")
        print(f"   2. Ensure columns: date, open, high, low, close, volume")
        print(f"   3. Load with: df = load_stock_data('path/to/your_data.csv')")
        print(f"   4. Use StockPredictor as shown in other examples")


def main():
    print("🚀 Stock Forecasting Examples with FT-Transformer")
    print("=" * 80)
    print("This example demonstrates the stock_forecasting library capabilities")
    
    # Run examples
    try:
        # Basic stock prediction
        basic_predictor, test_df = demonstrate_basic_stock_prediction()
        
        # Return prediction
        return_predictor = demonstrate_return_prediction()
        
        # Feature engineering demo
        demonstrate_feature_engineering()
        
        # Real data usage
        demonstrate_real_data_usage()
        
        print("\\n" + "="*80)
        print("🎉 All examples completed successfully!")
        print("=" * 80)
        
        print("\\n💡 Next Steps:")
        print("   1. Try different target columns: 'open', 'high', 'low', 'pct_change_3d', etc.")
        print("   2. Experiment with sequence_length (5-20 typically work well)")
        print("   3. Adjust model size with d_token and n_layers for your data size")
        print("   4. Use your own data with load_stock_data('your_file.csv')")
        print("   5. Add technical indicators with create_technical_indicators()")
        
    except Exception as e:
        print(f"\\n❌ Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()