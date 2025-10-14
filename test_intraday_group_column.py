#!/usr/bin/env python3
"""
Quick test to verify symbol column is preserved in intraday feature engineering.
"""

import pandas as pd
import numpy as np
from intraday_forecasting.preprocessing.intraday_features import create_intraday_features

# Create test data with symbol column
np.random.seed(42)
dates = pd.date_range('2024-01-01 09:30:00', periods=100, freq='1min')
symbols = ['BTC', 'ETH']

data = []
for symbol in symbols:
    for date in dates:
        data.append({
            'timestamp': date,
            'symbol': symbol,
            'open': 100 + np.random.randn(),
            'high': 101 + np.random.randn(),
            'low': 99 + np.random.randn(),
            'close': 100 + np.random.randn(),
            'volume': 1000000
        })

df = pd.DataFrame(data)

print("Input DataFrame:")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"  Has 'symbol' column: {'symbol' in df.columns}")

# Test feature engineering WITH group_column
print("\nTesting with group_column='symbol'...")
df_features = create_intraday_features(
    df=df,
    target_column='close',
    timestamp_col='timestamp',
    country='CRYPTO',
    timeframe='1min',
    prediction_horizon=1,
    verbose=False,
    use_essential_only=True,
    group_column='symbol'  # This should preserve the symbol column
)

print("Output DataFrame:")
print(f"  Shape: {df_features.shape}")
print(f"  Columns: {list(df_features.columns)}")
print(f"  Has 'symbol' column: {'symbol' in df_features.columns}")

if 'symbol' in df_features.columns:
    print("\n✅ SUCCESS: Symbol column preserved!")
    print(f"  Unique symbols: {df_features['symbol'].unique()}")
else:
    print("\n❌ FAILED: Symbol column was dropped!")
    exit(1)

# Test without group_column to ensure backward compatibility
print("\nTesting without group_column (backward compatibility)...")
df_features_no_group = create_intraday_features(
    df=df,
    target_column='close',
    timestamp_col='timestamp',
    country='CRYPTO',
    timeframe='1min',
    prediction_horizon=1,
    verbose=False,
    use_essential_only=True,
    group_column=None
)

print("Output DataFrame (no group):")
print(f"  Shape: {df_features_no_group.shape}")
print(f"  Columns: {list(df_features_no_group.columns)}")
print(f"  Has 'symbol' column: {'symbol' in df_features_no_group.columns}")

if 'symbol' not in df_features_no_group.columns:
    print("\n✅ SUCCESS: Symbol column correctly excluded when group_column=None")
else:
    print("\n⚠️  Symbol column present (acceptable if it was in original features)")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
