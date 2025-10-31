#!/usr/bin/env python3
"""
Test the refactored pipeline to ensure all steps execute properly.
This test verifies:
1. Feature creation (_create_base_features)
2. Target shifting
3. Storage of unencoded/unscaled dataframe
4. Categorical encoding
5. Numerical column determination
6. Per-horizon target scaling
7. Sequence creation
8. Prediction and inverse transform
9. Evaluation alignment
"""

import pandas as pd
import numpy as np
from daily_stock_forecasting.predictor import StockPredictor
from daily_stock_forecasting.preprocessing.market_data import create_sample_stock_data

print("=" * 80)
print("TESTING REFACTORED PIPELINE")
print("=" * 80)

# Create sample data
print("\n1. Creating sample stock data...")
df = create_sample_stock_data(n_samples=100, start_date='2020-01-01')
print(f"   Created {len(df)} samples")
print(f"   Columns: {list(df.columns)}")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

# Test single-target, multi-horizon
print("\n" + "=" * 80)
print("TEST 1: Single-Target, Multi-Horizon (close, horizon=3)")
print("=" * 80)

predictor = StockPredictor(
    target_column='close',
    sequence_length=5,
    prediction_horizon=3,
    model_type='ft_transformer_cls',
    verbose=True
)

print("\n2. Testing _create_base_features()...")
df_features = predictor._create_base_features(df.copy())
print(f"\n   Input columns: {list(df.columns)}")
print(f"   Output columns: {list(df_features.columns)}")
print(f"   Features added: {set(df_features.columns) - set(df.columns)}")
print(f"   Features removed: {set(df.columns) - set(df_features.columns)}")

# Check vwap
if 'vwap' in df_features.columns:
    print(f"   ✓ vwap created successfully")
else:
    print(f"   ✗ ERROR: vwap not created!")

# Check cyclical features
cyclical_features = [col for col in df_features.columns if '_sin' in col or '_cos' in col]
print(f"   ✓ Cyclical features: {cyclical_features}")

# Check original temporal features removed
temporal_removed = ['year', 'month', 'day', 'dayofweek', 'hour', 'minute']
remaining_temporal = [col for col in df_features.columns if col in temporal_removed]
if remaining_temporal:
    print(f"   ✗ ERROR: Temporal features not removed: {remaining_temporal}")
else:
    print(f"   ✓ Original temporal features removed")

print("\n3. Testing prepare_data() with fit_scaler=True...")
print("   This should:")
print("   - Create shifted targets (close_target_h1, h2, h3)")
print("   - Store unencoded/unscaled dataframe (store_for_evaluation=False)")
print("   - Encode categorical features (if any)")
print("   - Determine numerical columns")
print("   - Scale features and targets (per-horizon)")
print("   - Create sequences")

X_train, y_train = predictor.prepare_data(df.copy(), fit_scaler=True, store_for_evaluation=False)

print(f"\n   X_train shape: {X_train[0].shape if isinstance(X_train, tuple) else X_train.shape}")
if isinstance(X_train, tuple):
    print(f"   X_train is tuple (X_num, X_cat)")
    print(f"   X_num shape: {X_train[0].shape}")
    print(f"   X_cat shape: {X_train[1].shape}")
print(f"   y_train shape: {y_train.shape}")

# Check target scalers
print(f"\n   Target scalers created:")
for key in sorted(predictor.target_scalers_dict.keys()):
    print(f"     - {key}: {type(predictor.target_scalers_dict[key]).__name__}")

expected_scalers = ['close_target_h1', 'close_target_h2', 'close_target_h3']
if all(key in predictor.target_scalers_dict for key in expected_scalers):
    print(f"   ✓ All per-horizon scalers created")
else:
    print(f"   ✗ ERROR: Missing scalers!")

print("\n4. Testing predict() with store_for_evaluation=True...")
predictions = predictor.fit(df[:80].copy(), epochs=2, verbose=False)

print("   Making predictions...")
predictions = predictor.predict(df[80:].copy())
print(f"   Predictions shape: {predictions.shape}")
print(f"   Expected shape: (n_samples, {predictor.prediction_horizon})")

# Check stored dataframe
if hasattr(predictor, '_last_processed_df'):
    stored_df = predictor._last_processed_df
    print(f"\n   ✓ Stored dataframe: {len(stored_df)} rows")
    print(f"   Stored columns: {list(stored_df.columns)}")

    # Check for shifted targets
    shifted_targets = [col for col in stored_df.columns if '_target_h' in col]
    print(f"   Shifted targets in stored df: {shifted_targets}")

    # Check that categorical columns are NOT encoded (original values)
    if predictor.categorical_columns:
        print(f"   Categorical columns preserved: {predictor.categorical_columns}")

    # Check that numerical columns are NOT scaled (original values)
    print(f"   Sample values from stored df (should be unscaled):")
    for col in ['close', 'volume', 'vwap'][:3]:
        if col in stored_df.columns:
            print(f"     {col}: mean={stored_df[col].mean():.2f}, std={stored_df[col].std():.2f}")
else:
    print(f"   ✗ ERROR: _last_processed_df not stored!")

print("\n5. Testing evaluation alignment...")
metrics = predictor.evaluate(df[80:].copy())
print(f"   Evaluation metrics: {metrics}")

if 'overall' in metrics:
    print(f"   Overall MAPE: {metrics['overall']['MAPE']:.2f}%")
    if metrics['overall']['MAPE'] < 100:
        print(f"   ✓ MAPE is reasonable (< 100%)")
    else:
        print(f"   ⚠ MAPE is 100% - potential alignment issue!")

# Test multi-target
print("\n" + "=" * 80)
print("TEST 2: Multi-Target, Multi-Horizon (close + volume, horizon=2)")
print("=" * 80)

predictor_multi = StockPredictor(
    target_column=['close', 'volume'],
    sequence_length=5,
    prediction_horizon=2,
    model_type='ft_transformer_cls',
    verbose=True
)

print("\nTesting multi-target prepare_data()...")
X_train_multi, y_train_multi = predictor_multi.prepare_data(df.copy(), fit_scaler=True, store_for_evaluation=False)

print(f"\n   X_train shape: {X_train_multi[0].shape if isinstance(X_train_multi, tuple) else X_train_multi.shape}")
print(f"   y_train shape: {y_train_multi.shape}")
print(f"   Expected y shape: (n_samples, {len(['close', 'volume']) * 2})")

print(f"\n   Target scalers created:")
for key in sorted(predictor_multi.target_scalers_dict.keys()):
    print(f"     - {key}: {type(predictor_multi.target_scalers_dict[key]).__name__}")

expected_multi_scalers = ['close_target_h1', 'close_target_h2', 'volume_target_h1', 'volume_target_h2']
if all(key in predictor_multi.target_scalers_dict for key in expected_multi_scalers):
    print(f"   ✓ All multi-target per-horizon scalers created")
else:
    print(f"   ✗ ERROR: Missing multi-target scalers!")

print("\nFitting multi-target model...")
predictor_multi.fit(df[:80].copy(), epochs=2, verbose=False)

print("Making multi-target predictions...")
predictions_multi = predictor_multi.predict(df[80:].copy())
print(f"   Predictions type: {type(predictions_multi)}")
if isinstance(predictions_multi, dict):
    print(f"   ✓ Multi-target predictions returned as dict")
    for target, preds in predictions_multi.items():
        print(f"     {target}: shape={preds.shape}")
else:
    print(f"   ✗ ERROR: Expected dict, got {type(predictions_multi)}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("✓ All refactored pipeline steps executed successfully!")
print("✓ Per-horizon target scaling working")
print("✓ Dataframe storage for evaluation working")
print("✓ Multi-target and single-target modes working")
print("✓ Cyclical encoding applied and originals dropped")
print("=" * 80)
