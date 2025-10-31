"""Test if processed dataframe retains target columns."""
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(__file__))

from daily_stock_forecasting.predictor import StockPredictor
from daily_stock_forecasting.preprocessing.market_data import create_sample_stock_data
from tf_predictor.core.utils import split_time_series

print("Testing if processed dataframe has target columns...")

# Create sample data
df = create_sample_stock_data(n_samples=100, asset_type='crypto', num_symbols=2)
print(f"\n1. Original df columns: {list(df.columns)}")
print(f"   Has 'close': {'close' in df.columns}")
print(f"   Has 'volume': {'volume' in df.columns}")

# Split
train_df, val_df, test_df = split_time_series(df, test_size=20, val_size=10, 
                                               group_column='symbol', time_column='date', 
                                               sequence_length=10)

# Create predictor
model = StockPredictor(
    target_column=['close', 'volume'],
    sequence_length=10,
    prediction_horizon=3,
    asset_type='crypto',
    group_columns='symbol',
    scaler_type='standard'
)

# Call prepare_features (what gets stored in _last_processed_df)
processed_df = model.prepare_features(test_df.copy(), fit_scaler=False)

print(f"\n2. Processed df columns: {list(processed_df.columns)}")
print(f"   Has 'close': {'close' in processed_df.columns}")
print(f"   Has 'volume': {'volume' in processed_df.columns}")
print(f"   Has 'close_target_h1': {'close_target_h1' in processed_df.columns}")
print(f"   Has 'volume_target_h1': {'volume_target_h1' in processed_df.columns}")

print(f"\n3. Shape comparison:")
print(f"   Original test_df: {test_df.shape}")
print(f"   Processed df: {processed_df.shape}")

if 'close' in processed_df.columns:
    print(f"\n✅ Original target columns ARE present in processed df")
else:
    print(f"\n❌ Original target columns are MISSING from processed df")
    print(f"   This would cause the evaluation fix to fail!")
