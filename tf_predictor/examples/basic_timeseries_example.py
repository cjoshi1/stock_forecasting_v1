#!/usr/bin/env python3
"""
Basic Time Series Prediction Example using Generic FT-Transformer

This example shows how to use the generic tf_predictor library for any time series
prediction task. It demonstrates:
1. Creating a custom predictor by extending TimeSeriesPredictor
2. Implementing domain-specific feature engineering
3. Training and evaluating the model
4. Making predictions
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tf_predictor.core.predictor import TimeSeriesPredictor
from tf_predictor.core.utils import split_time_series, calculate_metrics
from tf_predictor.preprocessing.time_features import (
    create_date_features, create_lag_features, create_rolling_features
)


class EnergyConsumptionPredictor(TimeSeriesPredictor):
    """
    Example predictor for energy consumption forecasting.
    
    This demonstrates how to extend TimeSeriesPredictor for a specific domain.
    """
    
    def create_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """Create energy consumption specific features."""
        df_processed = df.copy()
        
        if self.verbose:
            print(f"   Creating features for {len(df_processed)} samples...")
        
        # Add date features if date column exists
        if 'date' in df_processed.columns:
            df_processed = create_date_features(df_processed, 'date')
            
        # Energy consumption specific features
        if 'consumption' in df_processed.columns:
            # Add lag features (previous days' consumption)
            df_processed = create_lag_features(df_processed, 'consumption', [1, 2, 7])
            
            # Add rolling statistics
            df_processed = create_rolling_features(df_processed, 'consumption', [3, 7])
            
            # Percentage change features
            df_processed['consumption_pct_change'] = df_processed['consumption'].pct_change() * 100
        
        # Weather features (if available)
        if 'temperature' in df_processed.columns:
            df_processed = create_rolling_features(df_processed, 'temperature', [3, 7])
            
        # Special time-based features for energy
        if 'date' in df_processed.columns:
            # Working day indicator (energy consumption is different on weekends)
            df_processed['is_working_day'] = (df_processed['dayofweek'] < 5).astype(int)
            
            # Season-based features (heating/cooling demand)
            df_processed['is_winter'] = df_processed['month'].isin([12, 1, 2]).astype(int)
            df_processed['is_summer'] = df_processed['month'].isin([6, 7, 8]).astype(int)
        
        # Fill NaN values
        df_processed = df_processed.bfill().fillna(0)
        
        if self.verbose:
            print(f"   Created {len(df_processed.columns)} features")
        
        return df_processed


def create_sample_energy_data(n_samples=365, start_date='2020-01-01'):
    """Create synthetic energy consumption data for the example."""
    dates = pd.date_range(start=start_date, periods=n_samples, freq='D')
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Base consumption with weekly and yearly patterns
    t = np.arange(n_samples)
    
    # Base level
    base_consumption = 100
    
    # Yearly seasonality (higher in winter and summer due to heating/cooling)
    yearly_pattern = 20 * np.sin(2 * np.pi * t / 365.25 - np.pi/2)  # Peak in winter
    summer_boost = 10 * np.maximum(0, np.sin(2 * np.pi * t / 365.25))  # Summer cooling
    
    # Weekly pattern (lower on weekends)
    weekly_pattern = -5 * np.sin(2 * np.pi * t / 7)
    weekend_effect = -10 * ((t % 7) >= 5)  # Lower on weekends
    
    # Temperature effect
    temperature = 15 + 10 * np.sin(2 * np.pi * t / 365.25) + np.random.normal(0, 3, n_samples)
    temp_effect = 0.5 * np.abs(temperature - 20)  # More energy when temp far from 20C
    
    # Random noise
    noise = np.random.normal(0, 5, n_samples)
    
    # Combine all effects
    consumption = (base_consumption + yearly_pattern + summer_boost + 
                  weekly_pattern + weekend_effect + temp_effect + noise)
    
    # Ensure positive values
    consumption = np.maximum(consumption, 10)
    
    return pd.DataFrame({
        'date': dates,
        'consumption': consumption,
        'temperature': temperature,
        'humidity': np.random.uniform(30, 90, n_samples),  # Additional feature
        'day_of_year': dates.dayofyear
    })


def main():
    print("=" * 80)
    print("🔋 Energy Consumption Forecasting with Generic FT-Transformer")
    print("=" * 80)
    
    # 1. Create synthetic energy consumption data
    print("\\n📊 Creating synthetic energy consumption data...")
    df = create_sample_energy_data(n_samples=365 * 2)  # 2 years of daily data
    print(f"   Generated {len(df)} days of energy consumption data")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Consumption range: {df['consumption'].min():.1f} - {df['consumption'].max():.1f} kWh")
    
    # 2. Split data chronologically
    print("\\n🔄 Splitting data...")
    train_df, val_df, test_df = split_time_series(
        df, 
        test_size=60,   # 2 months for testing
        val_size=30     # 1 month for validation
    )
    
    print(f"   Train samples: {len(train_df)} ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"   Validation samples: {len(val_df)} ({val_df['date'].min()} to {val_df['date'].max()})")
    print(f"   Test samples: {len(test_df)} ({test_df['date'].min()} to {test_df['date'].max()})")
    
    # 3. Initialize custom predictor
    print("\\n🧠 Initializing Energy Consumption Predictor...")
    predictor = EnergyConsumptionPredictor(
        target_column='consumption',
        sequence_length=7,  # Use past 7 days to predict next day
        d_token=64,         # Smaller model for faster training
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        verbose=True
    )
    
    print(f"   Target: {predictor.target_column}")
    print(f"   Sequence length: {predictor.sequence_length} days")
    print(f"   Model parameters: d_token={predictor.ft_kwargs.get('d_token', 192)}, n_layers={predictor.ft_kwargs.get('n_layers', 3)}")
    
    # 4. Train model
    print("\\n🏋️ Training model...")
    predictor.fit(
        df=train_df,
        val_df=val_df,
        epochs=10,  # Reduced for example
        batch_size=32,
        learning_rate=1e-3,
        patience=5,
        verbose=True
    )
    
    # 5. Evaluate on test set
    print("\\n📈 Evaluating model...")
    
    # Get predictions
    test_predictions = predictor.predict(test_df)
    print(f"   Generated {len(test_predictions)} predictions")
    
    # Calculate metrics
    test_metrics = predictor.evaluate(test_df)
    print("\\n   📊 Test Metrics:")
    for metric, value in test_metrics.items():
        if not np.isnan(value):
            unit = "kWh" if metric in ['MAE', 'RMSE'] else ("%" if metric in ['MAPE'] else "")
            print(f"   - {metric}: {value:.3f}{unit}")
    
    # 6. Show sample predictions
    print("\\n🔮 Sample Predictions vs Actual:")
    print("   Date       | Actual  | Predicted | Error")
    print("   " + "-" * 45)
    
    # Get actual values for comparison
    test_processed = predictor.prepare_features(test_df, fit_scaler=False)
    actual_values = test_processed[predictor.target_column].values[predictor.sequence_length:]
    
    for i in range(min(10, len(test_predictions))):
        date_idx = predictor.sequence_length + i
        if date_idx < len(test_df):
            date = test_df.iloc[date_idx]['date'].strftime('%Y-%m-%d')
            actual = actual_values[i]
            predicted = test_predictions[i]
            error = abs(actual - predicted)
            print(f"   {date} | {actual:7.1f} | {predicted:9.1f} | {error:5.1f}")
    
    # 7. Feature importance (show what features were created)
    print("\\n🎯 Generated Features:")
    sample_features = predictor.prepare_features(train_df.head(50), fit_scaler=False)
    feature_names = [col for col in sample_features.columns if col != 'consumption']
    print(f"   Total features: {len(feature_names)}")
    
    # Show feature categories
    date_features = [f for f in feature_names if f in ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend']]
    lag_features = [f for f in feature_names if 'lag_' in f]
    rolling_features = [f for f in feature_names if 'rolling_' in f]
    other_features = [f for f in feature_names if f not in date_features + lag_features + rolling_features]
    
    if date_features:
        print(f"   Date features ({len(date_features)}): {', '.join(date_features[:5])}")
    if lag_features:
        print(f"   Lag features ({len(lag_features)}): {', '.join(lag_features)}")
    if rolling_features:
        print(f"   Rolling features ({len(rolling_features)}): {', '.join(rolling_features[:5])}{'...' if len(rolling_features) > 5 else ''}")
    if other_features:
        print(f"   Other features ({len(other_features)}): {', '.join(other_features[:5])}{'...' if len(other_features) > 5 else ''}")
    
    # 8. Save model (optional)
    model_path = "energy_predictor.pt"
    print(f"\\n💾 Saving model to {model_path}...")
    predictor.save(model_path)
    
    print("\\n🎉 Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()