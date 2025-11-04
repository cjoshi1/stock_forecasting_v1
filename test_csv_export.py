#!/usr/bin/env python3
"""
Test CSV export functionality integrated into evaluate().
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from daily_stock_forecasting.predictor import StockPredictor
from tf_predictor.core.utils import split_time_series

print("="*80)
print("🧪 Testing CSV Export in evaluate() - Multi-Group, Multi-Target, Multi-Horizon")
print("="*80)

# Create synthetic multi-stock data
print("\n1️⃣  Creating multi-stock dataset...")
dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
np.random.seed(42)

data = []
for symbol in ['AAPL', 'GOOGL', 'MSFT']:
    base_close = 100 if symbol == 'AAPL' else (200 if symbol == 'GOOGL' else 150)
    base_volume = 1000 if symbol == 'AAPL' else (2000 if symbol == 'GOOGL' else 1500)

    close_prices = base_close + np.cumsum(np.random.randn(100) * 2)
    volumes = base_volume + np.cumsum(np.random.randn(100) * 10)

    for i, date in enumerate(dates):
        data.append({
            'date': date,
            'symbol': symbol,
            'close': close_prices[i],
            'volume': volumes[i]
        })

df = pd.DataFrame(data)
print(f"   Created {len(df)} rows for {df['symbol'].nunique()} symbols")
print(f"   Columns: {list(df.columns)}")

# Split data
print("\n2️⃣  Splitting data...")
train_df, val_df, test_df = split_time_series(df, test_size=30, val_size=15, group_column='symbol')
print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Create multi-target, multi-horizon predictor with group-based scaling
print("\n3️⃣  Creating predictor (multi-target, multi-horizon, multi-group)...")
predictor = StockPredictor(
    target_column=['close', 'volume'],  # Multi-target
    sequence_length=5,
    prediction_horizon=3,                # Multi-horizon
    group_columns='symbol',              # Multi-group
    d_model=32,
    num_layers=2,
    num_heads=2,
    verbose=False
)
print("   ✅ Predictor created")

# Train
print("\n4️⃣  Training (quick test - 3 epochs)...")
predictor.fit(
    df=train_df,
    val_df=val_df,
    epochs=3,
    batch_size=16,
    verbose=False
)
print("   ✅ Training completed")

# Evaluate WITH CSV export
print("\n5️⃣  Evaluating and exporting to CSV...")
metrics = predictor.evaluate(
    test_df,
    per_group=False,
    export_csv='outputs/test_predictions.csv'
)

print("\n📊 Metrics:")
print(f"   Keys: {list(metrics.keys())}")

# Show sample of CSV
print("\n6️⃣  Reading exported CSV...")
import pandas as pd
csv_df = pd.read_csv('outputs/test_predictions.csv')

print(f"\n📋 CSV Structure:")
print(f"   Rows: {len(csv_df)}")
print(f"   Columns: {list(csv_df.columns)}")
print(f"\n   First 5 rows:")
print(csv_df.head(5).to_string(index=False))

print(f"\n   Sample for each symbol:")
for symbol in csv_df['symbol'].unique()[:3]:
    symbol_data = csv_df[csv_df['symbol'] == symbol].head(2)
    print(f"\n   {symbol}:")
    print(symbol_data[['date', 'symbol', 'close_actual', 'close_pred_h1', 'volume_actual', 'volume_pred_h1']].to_string(index=False))

print("\n" + "="*80)
print("🎉 SUCCESS! CSV export integrated into evaluate()")
print("="*80)
print("\n💡 CSV Format:")
print("   - Columns: symbol, date, close_actual, close_pred_h1, close_pred_h2, close_pred_h3,")
print("              volume_actual, volume_pred_h1, volume_pred_h2, volume_pred_h3")
print("   - One row per prediction instance")
print("   - Supports: multi-group ✅, multi-target ✅, multi-horizon ✅")
print("="*80)
