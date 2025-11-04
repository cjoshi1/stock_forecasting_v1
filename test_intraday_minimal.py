#!/usr/bin/env python3
"""
Test that the intraday predictor works with minimal data (timestamp + close only).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intraday_forecasting.predictor import IntradayPredictor
from intraday_forecasting.preprocessing.market_data import load_intraday_data

print("="*80)
print("🧪 Testing Intraday Predictor with Minimal Data (timestamp + close only)")
print("="*80)

# Create minimal synthetic intraday data - just timestamp and close
print("\n1️⃣  Creating minimal dataset (timestamp + close only)...")

# Generate 5-minute bars for one trading day (9:30 AM to 4:00 PM)
start_time = datetime(2023, 1, 3, 9, 30)  # Tuesday (not Monday to avoid weekend)
timestamps = pd.date_range(start=start_time, end=start_time.replace(hour=16), freq='5T')[:-1]

# Random walk for prices
np.random.seed(42)
close_prices = 100 + np.cumsum(np.random.randn(len(timestamps)) * 0.5)

df = pd.DataFrame({
    'timestamp': timestamps,
    'close': close_prices
})

print(f"   Created {len(df)} 5-minute bars")
print(f"   Columns: {list(df.columns)}")
print(f"   Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print(f"\n   First 5 rows:")
print(df.head())

# Save to CSV and reload to test load_intraday_data
print("\n2️⃣  Saving and reloading via load_intraday_data()...")
temp_file = '/tmp/test_minimal_intraday.csv'
df.to_csv(temp_file, index=False)

try:
    df_loaded = load_intraday_data(temp_file, validate=False)  # Skip validation for simplicity
    print(f"   ✅ Successfully loaded minimal data")
    print(f"   Columns after loading: {list(df_loaded.columns)}")
except Exception as e:
    print(f"   ❌ Failed to load: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with IntradayPredictor
print("\n3️⃣  Creating IntradayPredictor...")
try:
    predictor = IntradayPredictor(
        target_column='close',
        timeframe='5min',
        timestamp_col='timestamp',
        country='US',
        prediction_horizon=1,
        sequence_length=10,  # Override default (96) for small test dataset
        d_model=32,
        num_layers=2,
        num_heads=2,
        verbose=True
    )
    print("   ✅ Predictor created successfully")
except Exception as e:
    print(f"   ❌ Failed to create predictor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Split data (use simple split since we have sequential time series)
print("\n4️⃣  Splitting data...")
train_size = int(len(df_loaded) * 0.7)
val_size = int(len(df_loaded) * 0.15)

train_df = df_loaded.iloc[:train_size]
val_df = df_loaded.iloc[train_size:train_size+val_size]
test_df = df_loaded.iloc[train_size+val_size:]

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
print("🎉 SUCCESS! Intraday predictor works with minimal data (timestamp + close only)")
print("="*80)
