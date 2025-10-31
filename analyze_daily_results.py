"""
Diagnostic script to analyze daily forecasting results and identify the root cause of 100% MAPE.
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))

from daily_stock_forecasting.predictor import StockPredictor
from daily_stock_forecasting.preprocessing.market_data import load_stock_data
from tf_predictor.core.utils import split_time_series

# Configuration matching the problematic run
DATA_PATH = "data/crypto_portfolio.csv"  # Update this to your actual data path
TARGET = ["close", "volume"]
ASSET_TYPE = "crypto"
GROUP_COLUMNS = "symbol"
PREDICTION_HORIZON = 3
SEQUENCE_LENGTH = 20
SCALER_TYPE = "onlymax"

print("="*80)
print("DAILY FORECASTING DIAGNOSTIC ANALYSIS")
print("="*80)

# Load data
print("\n1. Loading data...")
try:
    df = load_stock_data(DATA_PATH, asset_type=ASSET_TYPE, group_column=GROUP_COLUMNS)
    print(f"   ✅ Loaded {len(df)} samples")
    print(f"   Groups: {df[GROUP_COLUMNS].unique()}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Sample data:")
    print(df.head())
except Exception as e:
    print(f"   ❌ Error loading data: {e}")
    sys.exit(1)

# Split data
print("\n2. Splitting data...")
train_df, val_df, test_df = split_time_series(
    df,
    test_size=30,
    val_size=20,
    group_column=GROUP_COLUMNS,
    time_column='date',
    sequence_length=SEQUENCE_LENGTH
)
print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Initialize and train model
print("\n3. Training model...")
model = StockPredictor(
    target_column=TARGET,
    sequence_length=SEQUENCE_LENGTH,
    prediction_horizon=PREDICTION_HORIZON,
    asset_type=ASSET_TYPE,
    group_columns=GROUP_COLUMNS,
    scaler_type=SCALER_TYPE,
    d_model=64,
    num_layers=2,
    num_heads=4,
    dropout=0.1
)

model.fit(
    df=train_df,
    val_df=val_df,
    epochs=5,  # Just 5 epochs for quick diagnosis
    batch_size=32,
    learning_rate=1e-3,
    patience=5,
    verbose=False
)
print("   ✅ Model trained")

# Make predictions and analyze
print("\n4. Analyzing predictions...")

# Get predictions on test set
predictions = model.predict(test_df)
print(f"   Predictions type: {type(predictions)}")

if isinstance(predictions, dict):
    for target_name, preds in predictions.items():
        print(f"\n   Target: {target_name}")
        print(f"   Predictions shape: {preds.shape}")
        print(f"   Predictions range: [{preds.min():.4f}, {preds.max():.4f}]")
        print(f"   Predictions mean: {preds.mean():.4f}")
        print(f"   Predictions std: {preds.std():.4f}")
        print(f"   Sample predictions: {preds[:5, 0] if len(preds.shape) > 1 else preds[:5]}")
        
        # Get actual values
        actual = test_df[target_name].values[SEQUENCE_LENGTH:]
        min_len = min(len(actual), preds.shape[0])
        actual = actual[:min_len]
        
        print(f"\n   Actual values:")
        print(f"   Actual range: [{actual.min():.4f}, {actual.max():.4f}]")
        print(f"   Actual mean: {actual.mean():.4f}")
        print(f"   Actual std: {actual.std():.4f}")
        print(f"   Sample actuals: {actual[:5]}")
        
        # Calculate error
        if PREDICTION_HORIZON == 1:
            pred_h1 = preds[:min_len]
        else:
            pred_h1 = preds[:min_len, 0]  # First horizon
        
        errors = np.abs(pred_h1 - actual)
        mape = np.mean(errors / (np.abs(actual) + 1e-8)) * 100
        
        print(f"\n   Error Analysis (Horizon 1):")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   MAE: {errors.mean():.4f}")
        print(f"   Max error: {errors.max():.4f}")
        
        # Check for scaling issues
        print(f"\n   Diagnostic checks:")
        if np.all(pred_h1 == pred_h1[0]):
            print(f"   ❌ All predictions are identical: {pred_h1[0]}")
        if np.abs(pred_h1).mean() < 0.01:
            print(f"   ❌ Predictions too small (likely not inverse-transformed)")
        if np.abs(pred_h1).mean() > 1000 * np.abs(actual).mean():
            print(f"   ❌ Predictions too large (likely double-scaled)")
        if mape > 90:
            print(f"   ❌ MAPE > 90% - severe prediction error")
            print(f"   Ratio of pred/actual means: {pred_h1.mean() / actual.mean():.4f}")

# Check scalers
print("\n5. Checking scalers...")
print(f"   Group feature scalers: {len(model.group_feature_scalers)}")
print(f"   Group target scalers: {len(model.group_target_scalers)}")

for group_val, scaler_dict in list(model.group_target_scalers.items())[:2]:
    print(f"\n   Group {group_val}:")
    for target, scaler in scaler_dict.items():
        print(f"      Target: {target}")
        if hasattr(scaler, 'scale_'):
            print(f"      Scale: {scaler.scale_}")
        if hasattr(scaler, 'min_'):
            print(f"      Min: {scaler.min_}")
        if hasattr(scaler, 'max_value'):
            print(f"      Max: {scaler.max_value}")

# Per-group analysis
print("\n6. Per-group analysis...")
predictions_with_groups, group_indices = model.predict(test_df, return_group_info=True)

if isinstance(predictions_with_groups, dict):
    for target_name in predictions_with_groups.keys():
        print(f"\n   Target: {target_name}")
        preds = predictions_with_groups[target_name]
        
        for group_val in set(group_indices):
            group_mask = np.array([g == group_val for g in group_indices])
            group_preds = preds[group_mask]
            
            # Get group name from encoder
            if model.group_columns and model.group_columns[0] in model.cat_encoders:
                encoder = model.cat_encoders[model.group_columns[0]]
                group_name = encoder.classes_[group_val]
            else:
                group_name = group_val
            
            # Get actual values for this group
            group_df = test_df[test_df[GROUP_COLUMNS] == group_name]
            group_actual = group_df[target_name].values[SEQUENCE_LENGTH:]
            
            min_len = min(len(group_actual), group_preds.shape[0])
            group_actual = group_actual[:min_len]
            
            if PREDICTION_HORIZON == 1:
                group_pred_h1 = group_preds[:min_len]
            else:
                group_pred_h1 = group_preds[:min_len, 0]
            
            errors = np.abs(group_pred_h1 - group_actual)
            mape = np.mean(errors / (np.abs(group_actual) + 1e-8)) * 100
            
            print(f"      Group {group_name}: MAPE={mape:.2f}%, pred_mean={group_pred_h1.mean():.2f}, actual_mean={group_actual.mean():.2f}")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
