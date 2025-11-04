#!/usr/bin/env python3
"""
Test that the stock predictor works with custom features/columns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from daily_stock_forecasting.predictor import StockPredictor
from daily_stock_forecasting.preprocessing.market_data import load_stock_data

print("="*80)
print("🧪 Testing Stock Predictor with Custom Features")
print("="*80)

# Create data with custom features
print("\n1️⃣  Creating dataset with custom features...")
dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
close_prices = 100 + np.cumsum(np.random.randn(100) * 2)

# Add custom features
df = pd.DataFrame({
    'date': dates,
    'close': close_prices,
    'feature_1': np.sin(np.arange(100) * 0.1),  # Cyclical pattern
    'feature_2': np.random.randn(100),          # Random noise
    'feature_3': close_prices * 0.01,           # Derived from close
    'ma_5': pd.Series(close_prices).rolling(5).mean().fillna(method='bfill'),  # Moving average
    'volatility': pd.Series(close_prices).rolling(10).std().fillna(0),  # Volatility
})

print(f"   Created {len(df)} rows")
print(f"   Columns: {list(df.columns)}")
print(f"   Data shape: {df.shape}")
print(f"\n   First 5 rows:")
print(df.head())

# Save to CSV and reload
print("\n2️⃣  Saving and reloading via load_stock_data()...")
temp_file = '/tmp/test_custom_features.csv'
df.to_csv(temp_file, index=False)

try:
    df_loaded = load_stock_data(temp_file)
    print(f"   ✅ Successfully loaded data with custom features")
    print(f"   Loaded columns: {list(df_loaded.columns)}")
    print(f"   Number of numeric features: {len([c for c in df_loaded.columns if c != 'date'])}")
except Exception as e:
    print(f"   ❌ Failed to load: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with StockPredictor
print("\n3️⃣  Creating StockPredictor...")
try:
    predictor = StockPredictor(
        target_column='close',
        sequence_length=5,
        prediction_horizon=1,
        d_model=32,
        num_layers=2,
        num_heads=2,
        verbose=True
    )
    print("   ✅ Predictor created successfully")
    print(f"   All numeric columns will be used as features (including custom ones)")
except Exception as e:
    print(f"   ❌ Failed to create predictor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Split data
print("\n4️⃣  Splitting data...")
from tf_predictor.core.utils import split_time_series
train_df, val_df, test_df = split_time_series(df_loaded, test_size=15, val_size=10)
print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Train
print("\n5️⃣  Training model (quick test - 3 epochs)...")
try:
    predictor.fit(
        df=train_df,
        val_df=val_df,
        epochs=3,
        batch_size=8,
        verbose=False
    )
    print("   ✅ Training completed")

    # Check what features were actually used
    print(f"\n   📊 Model Input Features:")
    print(f"   - Numerical columns: {predictor.numerical_columns}")
    print(f"   - Total features used: {len(predictor.numerical_columns)}")
    print(f"   - This includes: original columns + cyclical date features")

except Exception as e:
    print(f"   ❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Evaluate
print("\n6️⃣  Evaluating on test set...")
try:
    metrics = predictor.evaluate(test_df)
    print("   ✅ Evaluation completed")
    print(f"   Test MAPE: {metrics.get('MAPE', 0):.2f}%")
    print(f"   Test MAE: {metrics.get('MAE', 0):.4f}")
except Exception as e:
    print(f"   ❌ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Predict
print("\n7️⃣  Making predictions...")
try:
    predictions = predictor.predict(test_df)
    print(f"   ✅ Predictions generated, shape: {predictions.shape}")
    print(f"   Sample predictions: {predictions[:3]}")
except Exception as e:
    print(f"   ❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("🎉 SUCCESS! Stock predictor works with custom features")
print("="*80)
print("\n💡 Key Takeaway:")
print("   You can add ANY numeric columns to your data and they will")
print("   automatically be used as features by the model!")
print("   Examples: technical indicators, sentiment scores, macro data, etc.")
print("="*80)
