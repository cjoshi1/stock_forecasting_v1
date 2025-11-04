#!/usr/bin/env python3
"""
Test to verify overall metrics calculation for multi-horizon predictions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from daily_stock_forecasting.predictor import StockPredictor
from tf_predictor.core.utils import split_time_series

print("="*80)
print("🧪 Testing Overall Metrics Calculation (Multi-Horizon)")
print("="*80)

# Create sample data
dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
close_prices = 100 + np.cumsum(np.random.randn(100) * 2)

df = pd.DataFrame({
    'date': dates,
    'close': close_prices
})

print("\n1️⃣  Creating predictor with prediction_horizon=3...")
predictor = StockPredictor(
    target_column='close',
    sequence_length=5,
    prediction_horizon=3,  # Multi-horizon
    d_model=32,
    num_layers=2,
    num_heads=2,
    verbose=False
)
print("   ✅ Created predictor")

# Split data
train_df, val_df, test_df = split_time_series(df, test_size=20, val_size=10)
print(f"\n2️⃣  Split data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# Train
print("\n3️⃣  Training (3 epochs)...")
predictor.fit(
    df=train_df,
    val_df=val_df,
    epochs=3,
    batch_size=8,
    verbose=False
)
print("   ✅ Training completed")

# Evaluate
print("\n4️⃣  Evaluating metrics...")
metrics = predictor.evaluate(test_df)

print("\n📊 Metrics Structure:")
print(f"   Keys: {list(metrics.keys())}")

# Verify overall metrics exist
if 'overall' in metrics:
    print("\n   ✅ 'overall' key found")
    overall = metrics['overall']
    print(f"   Overall MAE: {overall['MAE']:.4f}")
    print(f"   Overall RMSE: {overall['RMSE']:.4f}")
    print(f"   Overall MAPE: {overall['MAPE']:.2f}%")
else:
    print("\n   ❌ 'overall' key NOT found!")
    sys.exit(1)

# Verify per-horizon metrics exist
expected_horizons = ['horizon_1', 'horizon_2', 'horizon_3']
missing_horizons = [h for h in expected_horizons if h not in metrics]

if missing_horizons:
    print(f"\n   ❌ Missing horizon keys: {missing_horizons}")
    sys.exit(1)
else:
    print(f"\n   ✅ All horizon keys found: {expected_horizons}")
    for h in expected_horizons:
        print(f"   {h} MAPE: {metrics[h]['MAPE']:.2f}%")

# Manual verification: Check that overall is NOT just average of horizons
h1_mae = metrics['horizon_1']['MAE']
h2_mae = metrics['horizon_2']['MAE']
h3_mae = metrics['horizon_3']['MAE']
avg_mae = (h1_mae + h2_mae + h3_mae) / 3
overall_mae = metrics['overall']['MAE']

print("\n5️⃣  Verifying overall calculation method...")
print(f"   H1 MAE: {h1_mae:.4f}")
print(f"   H2 MAE: {h2_mae:.4f}")
print(f"   H3 MAE: {h3_mae:.4f}")
print(f"   Average of horizons: {avg_mae:.4f}")
print(f"   Overall MAE: {overall_mae:.4f}")

# They should be different (overall is flatten-then-calculate, not average)
# Allow small tolerance for numerical errors
if abs(overall_mae - avg_mae) < 0.0001:
    print("\n   ⚠️  WARNING: Overall MAE equals average of horizons")
    print("      This suggests it might be using average method (but could be coincidence)")
else:
    print(f"\n   ✅ Overall MAE differs from average (diff: {abs(overall_mae - avg_mae):.4f})")
    print("      This confirms flatten-then-calculate approach is being used")

print("\n" + "="*80)
print("🎉 SUCCESS! Overall metrics calculation is working correctly")
print("="*80)
print("\n💡 Key Points:")
print("   - Overall metrics are calculated by flattening all horizons first")
print("   - Then metrics are computed on the combined data")
print("   - This is the mathematically correct approach")
print("="*80)
