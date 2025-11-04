#!/usr/bin/env python3
"""
Test that the stock predictor works with minimal data (just date + close).
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
print("🧪 Testing Stock Predictor with Minimal Data (date + close only)")
print("="*80)

# Create minimal synthetic data - just date and close
print("\n1️⃣  Creating minimal dataset (date + close only)...")
dates = pd.date_range(start='2020-01-01', periods=100, freq='B')  # 100 business days
close_prices = 100 + np.cumsum(np.random.randn(100) * 2)  # Random walk

df = pd.DataFrame({
    'date': dates,
    'close': close_prices
})

print(f"   Created {len(df)} rows")
print(f"   Columns: {list(df.columns)}")
print(f"   First 5 rows:")
print(df.head())

# Save to CSV and reload to test load_stock_data
print("\n2️⃣  Saving and reloading via load_stock_data()...")
temp_file = '/tmp/test_minimal_stock.csv'
df.to_csv(temp_file, index=False)

try:
    df_loaded = load_stock_data(temp_file)
    print(f"   ✅ Successfully loaded minimal data")
    print(f"   Columns after loading: {list(df_loaded.columns)}")
except Exception as e:
    print(f"   ❌ Failed to load: {e}")
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
except Exception as e:
    print(f"   ❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Evaluate with CSV export
print("\n6️⃣  Evaluating on test set and exporting CSV...")
try:
    metrics = predictor.evaluate(test_df, export_csv='outputs/test_export.csv')
    print("   ✅ Evaluation completed")
    print(f"   Test MAPE: {metrics.get('MAPE', 0):.2f}%")
    print(f"   Test MAE: {metrics.get('MAE', 0):.4f}")
except Exception as e:
    print(f"   ❌ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Show CSV format
print("\n7️⃣  Reading exported CSV...")
csv_content = pd.read_csv('outputs/test_export.csv', nrows=20)
print(f"\n   First 20 rows of CSV (showing both sections):")
print(csv_content)

# Predict
print("\n8️⃣  Making predictions...")
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
print("🎉 SUCCESS! Stock predictor works with minimal data (date + close only)")
print("="*80)
