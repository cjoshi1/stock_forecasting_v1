#!/usr/bin/env python3
"""
Debug script to trace evaluation alignment issues.

This script loads a trained model and manually walks through the evaluation pipeline,
printing dataframe state at each step to identify where alignment breaks.

Usage:
    python debug_alignment_detailed.py \\
        --model_path outputs/models/model.pt \\
        --data_path /path/to/test_data.csv \\
        --group_column symbol
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from daily_stock_forecasting.predictor import StockPredictor
from daily_stock_forecasting.preprocessing.market_data import load_stock_data
from debug_evaluation import debug_print_df_by_group


def main():
    parser = argparse.ArgumentParser(description='Debug evaluation alignment')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test data CSV')
    parser.add_argument('--group_column', type=str, default='symbol',
                       help='Group column name (default: symbol)')
    parser.add_argument('--asset_type', type=str, default='crypto',
                       choices=['stock', 'crypto'],
                       help='Asset type')
    parser.add_argument('--max_rows', type=int, default=10,
                       help='Max rows to display per group')

    args = parser.parse_args()

    print("="*80)
    print("🔍 DEBUGGING EVALUATION ALIGNMENT")
    print("="*80)

    # Load model
    print(f"\n📦 Loading model from: {args.model_path}")
    model = StockPredictor.load(args.model_path)
    print(f"   ✅ Model loaded")
    print(f"   - Targets: {model.target_columns}")
    print(f"   - Sequence length: {model.sequence_length}")
    print(f"   - Prediction horizon: {model.prediction_horizon}")
    print(f"   - Group columns: {model.group_columns}")

    # Load test data
    print(f"\n📊 Loading test data from: {args.data_path}")
    df = load_stock_data(args.data_path, asset_type=args.asset_type, group_column=args.group_column)
    print(f"   ✅ Loaded {len(df)} rows")
    print(f"   - Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   - Columns: {list(df.columns)}")

    # =========================================================================
    # STEP 1: Show raw input dataframe
    # =========================================================================
    debug_print_df_by_group(df, args.group_column,
                           "STEP 1: Raw Input DataFrame (df)",
                           max_rows_per_group=args.max_rows)

    # =========================================================================
    # STEP 2: Manually trace through predict() pipeline
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: Tracing through predict() pipeline...")
    print("="*80)

    # Step 2a: Create base features (sorting, date features, domain-specific)
    print("\n--- Step 2a: _create_base_features() ---")
    df_features = model._create_base_features(df.copy())
    debug_print_df_by_group(df_features, args.group_column,
                           "After _create_base_features() [sorted + date features]",
                           max_rows_per_group=args.max_rows)

    # Step 2b: Create shifted targets
    print("\n--- Step 2b: create_shifted_targets() ---")
    from tf_predictor.preprocessing.time_features import create_shifted_targets
    group_col_for_shift = model.categorical_columns if model.categorical_columns else model.group_columns
    df_with_targets = create_shifted_targets(
        df_features.copy(),
        target_column=model.target_columns,
        prediction_horizon=model.prediction_horizon,
        group_column=group_col_for_shift,
        verbose=True
    )
    debug_print_df_by_group(df_with_targets, args.group_column,
                           "After create_shifted_targets() [UNSCALED - this is stored in _last_processed_df]",
                           max_rows_per_group=args.max_rows)

    print("\n" + "🔑 KEY INSIGHT: The dataframe above is what gets stored in model._last_processed_df")
    print("   This is the reference for extracting 'actual' values during evaluation.")
    print("   Notice the row count per group after shifting.")

    # =========================================================================
    # STEP 3: Run actual predict() and show stored dataframe
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: Running model.predict() with return_group_info=True")
    print("="*80)

    predictions, group_indices = model.predict(df.copy(), return_group_info=True)

    print(f"\n✅ Predictions generated:")
    if isinstance(predictions, dict):
        for target, preds in predictions.items():
            if model.prediction_horizon == 1:
                print(f"   - {target}: {len(preds)} predictions (1D array)")
            else:
                print(f"   - {target}: {preds.shape} predictions (2D array)")
    else:
        if model.prediction_horizon == 1:
            print(f"   - Predictions: {len(predictions)} predictions (1D array)")
        else:
            print(f"   - Predictions: {predictions.shape} predictions (2D array)")

    print(f"\n✅ Group indices: {len(group_indices)} entries")
    print(f"   - Unique groups: {sorted(set(group_indices))}")

    # Show the stored processed dataframe
    if hasattr(model, '_last_processed_df') and model._last_processed_df is not None:
        debug_print_df_by_group(model._last_processed_df, args.group_column,
                               "STEP 3: model._last_processed_df (stored by predict())",
                               max_rows_per_group=args.max_rows)
    else:
        print("\n⚠️  WARNING: model._last_processed_df not found!")

    # =========================================================================
    # STEP 4: Trace through evaluation extraction
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: Simulating evaluation 'actual' value extraction")
    print("="*80)

    # Get encoder mapping
    if model.group_columns and args.group_column in model.cat_encoders:
        encoder = model.cat_encoders[args.group_column]
        group_value_to_name = {i: name for i, name in enumerate(encoder.classes_)}
        print(f"\n📋 Group encoding mapping:")
        for encoded_val, original_name in group_value_to_name.items():
            print(f"   {encoded_val} -> {original_name}")
    else:
        group_value_to_name = {gv: gv for gv in sorted(set(group_indices))}

    # For each target and group, show alignment
    unique_groups = sorted(set(group_indices))

    for target_col in model.target_columns:
        print(f"\n{'='*80}")
        print(f"TARGET: {target_col}")
        print(f"{'='*80}")

        target_predictions = predictions[target_col] if isinstance(predictions, dict) else predictions

        for group_value in unique_groups:
            group_name = group_value_to_name[group_value]

            print(f"\n--- Group {group_value}: {group_name} ---")

            # Get processed data for this group
            group_df_processed = model._last_processed_df[
                model._last_processed_df[args.group_column] == group_name
            ].copy()

            print(f"   Processed df for this group: {len(group_df_processed)} rows")

            # Extract actual values with sequence_length offset
            if model.sequence_length > 1:
                group_actual_full = group_df_processed[target_col].values[model.sequence_length:]
                print(f"   After sequence offset ({model.sequence_length}): {len(group_actual_full)} actual values")
            else:
                group_actual_full = group_df_processed[target_col].values
                print(f"   No sequence offset: {len(group_actual_full)} actual values")

            # Get predictions for this group
            group_mask = np.array([g == group_value for g in group_indices])
            num_group_preds = group_mask.sum()

            if model.prediction_horizon == 1:
                group_preds = target_predictions[group_mask]
                print(f"   Predictions for this group: {len(group_preds)} predictions (1D)")
            else:
                group_preds = target_predictions[group_mask, :]
                print(f"   Predictions for this group: {group_preds.shape} predictions (2D)")

            # Show alignment
            print(f"\n   📊 ALIGNMENT CHECK:")
            print(f"      Actual values available:  {len(group_actual_full)}")
            print(f"      Predictions available:     {num_group_preds}")

            if model.prediction_horizon == 1:
                min_len = min(len(group_actual_full), len(group_preds))
                print(f"      Will use (min):            {min_len}")
                print(f"\n      First 5 actual:     {group_actual_full[:5]}")
                print(f"      First 5 predicted:  {group_preds[:5]}")
            else:
                num_preds = group_preds.shape[0]
                needed_actuals = num_preds + model.prediction_horizon - 1
                print(f"      Needed actuals (for multi-horizon): {needed_actuals}")
                print(f"      Available: {len(group_actual_full)}, Needed: {needed_actuals}")

                if len(group_actual_full) >= needed_actuals:
                    print(f"      ✅ Sufficient actuals for all predictions")
                else:
                    valid_preds = max(0, len(group_actual_full) - model.prediction_horizon + 1)
                    print(f"      ⚠️  Only {valid_preds} predictions can be evaluated")

                print(f"\n      First 3 actual:     {group_actual_full[:3]}")
                print(f"      First 3 predicted:  {group_preds[:3]}")

    # =========================================================================
    # STEP 5: Summary and recommendations
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: SUMMARY & DIAGNOSTIC")
    print("="*80)

    print("\n📋 Pipeline flow:")
    print("   1. Raw df -> _create_base_features() [sorting, date features]")
    print("   2. -> create_shifted_targets() [adds close_target_h1, etc.]")
    print("   3. -> STORED in _last_processed_df (BEFORE encoding/scaling)")
    print("   4. -> encode_categorical() [symbol -> 0,1,2]")
    print("   5. -> scale_features() [normalize]")
    print("   6. -> create_sequences() [sliding window]")
    print("   7. -> Model predicts on sequences")
    print("   8. -> Evaluation extracts 'actual' from _last_processed_df")

    print("\n🔍 Key alignment points:")
    print(f"   - Sequence offset: {model.sequence_length} (first {model.sequence_length} rows dropped from actuals)")
    print(f"   - Prediction horizon: {model.prediction_horizon}")
    print(f"   - Groups are processed separately (no cross-contamination)")

    print("\n💡 What to check:")
    print("   1. Does _last_processed_df have the expected rows per group?")
    print("   2. After sequence offset, do actual counts match prediction counts?")
    print("   3. Are the first few actual/predicted values reasonable?")
    print("   4. Check if any groups have insufficient data (warnings above)")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
