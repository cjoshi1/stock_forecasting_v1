#!/usr/bin/env python3
"""
Feature List Extractor

This script shows exactly what 78 features are being created.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create a small sample dataset to see all features
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
sample_data = pd.DataFrame({
    'timestamp': dates,
    'open': np.random.uniform(100, 110, 100),
    'high': np.random.uniform(110, 120, 100),
    'low': np.random.uniform(90, 100, 100),
    'close': np.random.uniform(95, 115, 100),
    'volume': np.random.uniform(1000, 10000, 100)
})

# Import feature creation functions
from intraday_forecasting.preprocessing.intraday_features import create_intraday_features

print("="*80)
print("📊 COMPLETE LIST OF 78 INTRADAY FEATURES")
print("="*80)

# Create features
df_with_features = create_intraday_features(sample_data, 'close', 'timestamp', 'CRYPTO', verbose=False)

# Get all features (excluding timestamp)
all_columns = list(df_with_features.columns)
feature_columns = [col for col in all_columns if col != 'timestamp']

print(f"\n🔢 Total Features Created: {len(feature_columns)}")
print(f"📈 Original OHLCV Columns: {['open', 'high', 'low', 'close', 'volume']}")
print(f"⚙️  Engineered Features: {len(feature_columns) - 5}")

# Categorize features
time_features = []
price_features = []
volume_features = []
lag_features = []
rolling_features = []
other_features = []

for col in feature_columns:
    if any(time_word in col.lower() for time_word in ['hour', 'minute', 'day', 'week', 'sin', 'cos']):
        time_features.append(col)
    elif 'lag' in col.lower():
        lag_features.append(col)
    elif any(roll_word in col.lower() for roll_word in ['mean', 'std', 'sma', 'rolling']):
        rolling_features.append(col)
    elif any(vol_word in col.lower() for vol_word in ['volume', 'vol']):
        volume_features.append(col)
    elif any(price_word in col.lower() for price_word in ['open', 'high', 'low', 'close', 'price', 'return', 'ratio', 'momentum', 'pct', 'range']):
        price_features.append(col)
    else:
        other_features.append(col)

print(f"\n📊 FEATURE BREAKDOWN:")
print(f"├── Time-based features: {len(time_features)}")
print(f"├── Price-based features: {len(price_features)}")
print(f"├── Volume-based features: {len(volume_features)}")
print(f"├── Lag features: {len(lag_features)}")
print(f"├── Rolling statistics: {len(rolling_features)}")
print(f"└── Other features: {len(other_features)}")

print(f"\n⏰ TIME-BASED FEATURES ({len(time_features)}):")
for i, feature in enumerate(time_features, 1):
    print(f"  {i:2d}. {feature}")

print(f"\n💰 PRICE-BASED FEATURES ({len(price_features)}):")
for i, feature in enumerate(price_features, 1):
    print(f"  {i:2d}. {feature}")

print(f"\n📦 VOLUME-BASED FEATURES ({len(volume_features)}):")
for i, feature in enumerate(volume_features, 1):
    print(f"  {i:2d}. {feature}")

print(f"\n⏮️  LAG FEATURES ({len(lag_features)}):")
for i, feature in enumerate(lag_features, 1):
    print(f"  {i:2d}. {feature}")

print(f"\n📈 ROLLING STATISTICS ({len(rolling_features)}):")
for i, feature in enumerate(rolling_features, 1):
    print(f"  {i:2d}. {feature}")

if other_features:
    print(f"\n🔧 OTHER FEATURES ({len(other_features)}):")
    for i, feature in enumerate(other_features, 1):
        print(f"  {i:2d}. {feature}")

print(f"\n💡 FEATURE COMPLEXITY ANALYSIS:")
print(f"├── High-value features (price/volume patterns): ~{len(price_features) + len(volume_features)}")
print(f"├── Time patterns (cyclical): {len(time_features)}")
print(f"├── Historical context (lags): {len(lag_features)}")
print(f"└── Statistical smoothing (rolling): {len(rolling_features)}")

print(f"\n🚀 OPTIMIZATION SUGGESTIONS:")
print("├── Essential features (~30): OHLCV + key ratios + short lags")
print("├── Good features (~45): + time features + rolling means")
print("└── Full features (~78): + all statistical features")

print(f"\n{'='*80}")