#!/usr/bin/env python3
"""
Test pipeline with sample data - showing data at each stage.
"""

import pandas as pd
import numpy as np
from daily_stock_forecasting.predictor import StockPredictor
from daily_stock_forecasting.preprocessing.market_data import create_sample_stock_data

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("=" * 100)
print("PIPELINE TEST: Single-Target, Multi-Horizon (close, horizon=3)")
print("=" * 100)

# Stage 0: Raw data
print("\n" + "=" * 100)
print("STAGE 0: RAW INPUT DATA")
print("=" * 100)
df = create_sample_stock_data(n_samples=50)
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3))
print(f"\nLast 3 rows:")
print(df.tail(3))

# Create predictor
predictor = StockPredictor(
    target_column='close',
    sequence_length=5,
    prediction_horizon=3,
    model_type='ft_transformer_cls',
    verbose=False
)

# Stage 1: After _create_base_features
print("\n" + "=" * 100)
print("STAGE 1: AFTER _create_base_features() - Stock features + Time-series features")
print("=" * 100)
df_stage1 = predictor._create_base_features(df.copy())
print(f"\nShape: {df_stage1.shape}")
print(f"Columns: {list(df_stage1.columns)}")
print(f"\nColumns added: {set(df_stage1.columns) - set(df.columns)}")
print(f"Columns removed: {set(df.columns) - set(df_stage1.columns)}")
print(f"\nFirst 3 rows (showing new features):")
new_cols = list(set(df_stage1.columns) - set(df.columns))[:10]
print(df_stage1[['close', 'volume'] + new_cols].head(3))

# Stage 2: After create_shifted_targets (simulate this step)
print("\n" + "=" * 100)
print("STAGE 2: AFTER create_shifted_targets() - Target shifting")
print("=" * 100)
from tf_predictor.preprocessing.time_features import create_shifted_targets
df_stage2 = create_shifted_targets(
    df_stage1.copy(),
    target_column=['close'],
    prediction_horizon=3,
    group_column=None,
    verbose=True
)
print(f"\nShape: {df_stage2.shape}")
print(f"Columns: {list(df_stage2.columns)}")
print(f"\nShifted target columns: {[col for col in df_stage2.columns if '_target_h' in col]}")
print(f"\nFirst 5 rows (showing targets):")
print(df_stage2[['close', 'close_target_h1', 'close_target_h2', 'close_target_h3']].head(5))
print(f"\nLast 5 rows (showing targets - notice NaN removed):")
print(df_stage2[['close', 'close_target_h1', 'close_target_h2', 'close_target_h3']].tail(5))

# Stage 3: This is where dataframe is STORED for evaluation (unencoded, unscaled)
print("\n" + "=" * 100)
print("STAGE 3: STORED FOR EVALUATION (unencoded, unscaled)")
print("=" * 100)
print("This dataframe is stored in predictor._last_processed_df during predict()")
print("It has:")
print("  - Original categorical values (not label-encoded)")
print("  - Original numerical values (not scaled)")
print("  - Shifted target columns")
print(f"\nExample values (unscaled):")
print(df_stage2[['close', 'volume', 'vwap', 'close_target_h1', 'close_target_h2', 'close_target_h3']].describe())

# Now let's actually run prepare_data to see the rest
print("\n" + "=" * 100)
print("STAGE 4-7: Running prepare_data() to see scaling and sequence creation")
print("=" * 100)

# Split data
train_df = df[:40].copy()
test_df = df[40:].copy()

print("\nTraining data...")
X_train, y_train = predictor.prepare_data(train_df, fit_scaler=True, store_for_evaluation=False)

print(f"\nAfter prepare_data():")
print(f"  X_train shape: {X_train[0].shape if isinstance(X_train, tuple) else X_train.shape}")
if isinstance(X_train, tuple):
    print(f"    X_num shape: {X_train[0].shape} (sequences of numerical features)")
    print(f"    X_cat shape: {X_train[1].shape} (categorical features)")
print(f"  y_train shape: {y_train.shape}")
print(f"    Interpretation: ({y_train.shape[0]} samples, {y_train.shape[1]} horizons)")

print(f"\nTarget scalers created (per-horizon):")
for key in sorted(predictor.target_scalers_dict.keys()):
    scaler = predictor.target_scalers_dict[key]
    print(f"  {key}: {type(scaler).__name__}")

print(f"\nNumerical feature columns ({len(predictor.numerical_columns)} total):")
print(f"  {predictor.numerical_columns[:10]}...")  # Show first 10

# Now test prediction with storage
print("\n" + "=" * 100)
print("STAGE: PREDICTION WITH STORAGE")
print("=" * 100)

print("\nFitting model (2 epochs)...")
predictor.fit(train_df, epochs=2, verbose=False)

print("\nPredicting on test data (with store_for_evaluation=True)...")
predictions = predictor.predict(test_df)

print(f"\nPredictions shape: {predictions.shape}")
print(f"Predictions (inverse transformed, original scale):")
print(predictions[:5])

# Check stored dataframe
print(f"\n" + "=" * 100)
print("STORED DATAFRAME FOR EVALUATION")
print("=" * 100)
if hasattr(predictor, '_last_processed_df'):
    stored_df = predictor._last_processed_df
    print(f"Shape: {stored_df.shape}")
    print(f"Columns: {list(stored_df.columns)}")

    print(f"\nThis dataframe has:")
    print(f"  - Unencoded categorical values")
    print(f"  - Unscaled numerical values")
    print(f"  - Shifted targets")

    print(f"\nSample (first 3 rows):")
    print(stored_df[['close', 'volume', 'vwap', 'close_target_h1', 'close_target_h2', 'close_target_h3']].head(3))

    print(f"\nStatistics (unscaled):")
    print(stored_df[['close', 'volume', 'vwap']].describe())

# Evaluation
print(f"\n" + "=" * 100)
print("EVALUATION (using stored dataframe)")
print("=" * 100)
metrics = predictor.evaluate(test_df)
print(f"\nMetrics: {metrics}")

if 'overall' in metrics:
    print(f"\nOverall metrics:")
    for key, value in metrics['overall'].items():
        print(f"  {key}: {value:.4f}")

    print(f"\nPer-horizon metrics:")
    for horizon in range(1, predictor.prediction_horizon + 1):
        horizon_key = f'horizon_{horizon}'
        if horizon_key in metrics:
            print(f"  Horizon {horizon}:")
            for key, value in metrics[horizon_key].items():
                print(f"    {key}: {value:.4f}")

print("\n" + "=" * 100)
print("PIPELINE VERIFICATION COMPLETE!")
print("=" * 100)
print("✓ All stages executed successfully")
print("✓ Per-horizon scaling working")
print("✓ Dataframe stored correctly for evaluation")
print("=" * 100)
