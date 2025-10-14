#!/usr/bin/env python3
"""
Quick test for group-based scaling combined with multi-horizon predictions.

Verifies API and structure without full training.
"""

import pandas as pd
import numpy as np
from daily_stock_forecasting import StockPredictor

np.random.seed(42)

print("="*80)
print("QUICK TEST: Group + Multi-Horizon API Verification")
print("="*80)

# Create minimal synthetic data
symbols = ['AAPL', 'GOOGL']
dates = pd.date_range('2024-01-01', periods=50, freq='D')
data = []

for symbol in symbols:
    base_price = 150 if symbol == 'AAPL' else 2800
    for i, date in enumerate(dates):
        data.append({
            'date': date,
            'symbol': symbol,
            'open': base_price + i,
            'high': base_price + i + 1,
            'low': base_price + i - 1,
            'close': base_price + i,
            'volume': 1000000
        })

df = pd.DataFrame(data)

# Split
train_df = df[df['date'] < '2024-02-06'].copy()  # 35 days per symbol
test_df = df[df['date'] >= '2024-02-06'].copy()  # 15 days per symbol

print(f"\n1. Data: {len(symbols)} symbols, {len(train_df)} train, {len(test_df)} test")

# Initialize
model = StockPredictor(
    target_column='close',
    sequence_length=5,
    group_column='symbol',
    prediction_horizon=2,
    d_token=32,
    n_layers=1,
    n_heads=2,
    dropout=0.1,
    verbose=False
)

print(f"2. Model: group_column={model.group_column}, prediction_horizon={model.prediction_horizon}")

# Train minimally
print("3. Training (3 epochs, minimal)...")
model.fit(train_df, val_df=None, epochs=3, batch_size=8, verbose=False)
print("   ✅ Training completed")

# Test predictions with group info
print("\n4. Testing predict() with return_group_info...")
preds, groups = model.predict(test_df, return_group_info=True)
print(f"   Predictions shape: {preds.shape} (expected: (n_samples, 2))")
print(f"   Group indices count: {len(groups)}")
print(f"   Unique groups: {sorted(set(groups))}")

assert preds.shape[1] == 2, f"Expected 2 horizons, got {preds.shape[1]}"
assert set(groups) == set(symbols), f"Group mismatch: {set(groups)} vs {set(symbols)}"
print("   ✅ Predictions with group info work correctly")

# Test per-group evaluation
print("\n5. Testing evaluate() with per_group=True...")
metrics = model.evaluate(test_df, per_group=True)

print(f"   Metric keys: {list(metrics.keys())}")
assert 'overall' in metrics, "Missing 'overall'"
assert 'AAPL' in metrics, "Missing 'AAPL'"
assert 'GOOGL' in metrics, "Missing 'GOOGL'"

overall = metrics['overall']
print(f"   Overall structure: {list(overall.keys())}")
assert 'overall' in overall, "Missing overall->overall"
assert 'horizon_1' in overall, "Missing horizon_1"
assert 'horizon_2' in overall, "Missing horizon_2"

aapl_metrics = metrics['AAPL']
print(f"   AAPL structure: {list(aapl_metrics.keys())}")
assert 'overall' in aapl_metrics, "Missing AAPL->overall"
assert 'horizon_1' in aapl_metrics, "Missing AAPL->horizon_1"

print("   ✅ Per-group evaluation works correctly")

# Test CSV export
print("\n6. Testing CSV export with symbols...")
from daily_stock_forecasting.visualization.stock_charts import export_predictions_csv
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    csv_path = export_predictions_csv(model, train_df, test_df, tmpdir)
    csv_df = pd.read_csv(csv_path)

    print(f"   CSV shape: {csv_df.shape}")
    print(f"   CSV columns: {list(csv_df.columns)}")

    assert 'symbol' in csv_df.columns, "Symbol column missing!"
    assert 'pred_h1' in csv_df.columns, "pred_h1 missing!"
    assert 'pred_h2' in csv_df.columns, "pred_h2 missing!"

    csv_symbols = sorted(csv_df['symbol'].unique())
    assert csv_symbols == symbols, f"Symbol mismatch: {csv_symbols} vs {symbols}"

    print("   ✅ CSV export with symbols works correctly")

    # Show sample
    print("\n   Sample CSV rows:")
    print(csv_df[['date', 'symbol', 'actual', 'pred_h1', 'pred_h2', 'dataset']].head(6).to_string())

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nVerified:")
print("  ✓ predict() returns group indices when requested")
print("  ✓ evaluate(per_group=True) returns nested dict structure")
print("  ✓ CSV export includes symbol column and horizon predictions")
print("  ✓ Multi-symbol + multi-horizon combination works end-to-end")
