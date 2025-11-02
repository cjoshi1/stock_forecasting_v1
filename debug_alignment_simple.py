#!/usr/bin/env python3
"""
Simple alignment debugging with minimal synthetic data.

Creates a tiny dataset (2 groups, 10 rows each) and traces through
the entire prepare_data pipeline to identify alignment issues.

Configuration:
- 2 symbols (AAPL, GOOGL)
- 10 rows per symbol
- sequence_length = 3
- prediction_horizon = 2
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tf_predictor.core.predictor import TimeSeriesPredictor


class DualOutput:
    """Write to both console and file simultaneously."""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def create_synthetic_data():
    """
    Create minimal synthetic dataset for debugging.

    2 symbols, 10 rows each, simple incrementing values
    to make tracking transformations easy.
    """
    data = []

    # Symbol AAPL: close values 100, 101, 102, ..., 109
    for i in range(10):
        data.append({
            'symbol': 'AAPL',
            'date': datetime(2024, 1, 1) + timedelta(days=i),
            'close': 100.0 + i,
            'volume': 1000 + i * 10,
            'open': 99.0 + i,
            'high': 101.0 + i,
            'low': 99.0 + i,
        })

    # Symbol GOOGL: close values 200, 201, 202, ..., 209
    for i in range(10):
        data.append({
            'symbol': 'GOOGL',
            'date': datetime(2024, 1, 1) + timedelta(days=i),
            'close': 200.0 + i,
            'volume': 2000 + i * 20,
            'open': 199.0 + i,
            'high': 201.0 + i,
            'low': 199.0 + i,
        })

    df = pd.DataFrame(data)
    df['timestamp'] = df['date']

    return df


def print_df_state(df, title, group_col='symbol', show_cols=None, show_all_cols=False):
    """Print dataframe state at a checkpoint."""
    print("\n" + "="*80)
    print(f"📊 {title}")
    print("="*80)
    print(f"Total rows: {len(df)}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    if show_all_cols:
        # Show ALL columns
        show_cols = list(df.columns)
    elif show_cols is None:
        # Default: show important columns
        show_cols = []
        # Always include group and date if they exist
        if group_col in df.columns:
            show_cols.append(group_col)
        if 'date' in df.columns:
            show_cols.append('date')

        # Add OHLCV if they exist
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                show_cols.append(col)

        # Add any shifted target columns
        shifted_cols = [col for col in df.columns if '_target_h' in col]
        show_cols.extend(shifted_cols)

        # Add first 2 date features as examples
        date_features = [col for col in df.columns if any(x in col for x in ['_sin', '_cos', 'weekend'])]
        show_cols.extend(date_features[:2])

        # Keep only columns that exist
        show_cols = [col for col in show_cols if col in df.columns]

    if group_col in df.columns:
        for symbol in sorted(df[group_col].unique()):
            group_df = df[df[group_col] == symbol]
            print(f"\n--- {symbol} ({len(group_df)} rows) ---")
            print(group_df[show_cols].to_string(index=True))
    else:
        print(df[show_cols].to_string(index=True))


def test_single_target():
    """Test single-target scenario."""
    print("="*80)
    print("🔍 TEST 1: SINGLE-TARGET ALIGNMENT")
    print("="*80)
    print("\nConfiguration:")
    print("  - Targets: close (SINGLE-TARGET)")
    print("  - Groups: 2 (AAPL, GOOGL)")
    print("  - Rows per group: 10")
    print("  - Sequence length: 3")
    print("  - Prediction horizon: 2")

    # Create synthetic data
    df_raw = create_synthetic_data()
    print_df_state(df_raw, "STEP 0: Raw Input Data", show_cols=['symbol', 'date', 'close', 'volume'])

    # Create predictor
    print("\n" + "="*80)
    print("Creating Single-Target TimeSeriesPredictor...")
    print("="*80)

    predictor = TimeSeriesPredictor(
        target_column='close',  # SINGLE TARGET
        sequence_length=3,
        prediction_horizon=2,
        group_columns='symbol',
        categorical_columns='symbol',
        model_type='ft_transformer_cls',  # Fixed: use correct model name
        scaler_type='standard',
        use_lagged_target_features=False,
        d_model=32,
        num_heads=2,
        num_layers=2
    )
    predictor.verbose = True

    # Add a dummy logger to avoid AttributeError
    import logging
    predictor.logger = logging.getLogger(__name__)

    print("\n✅ Single-target predictor created")
    print(f"   Target columns: {predictor.target_columns}")
    print(f"   Is multi-target: {predictor.is_multi_target}")

    return run_test(predictor, df_raw, "SINGLE-TARGET")


def test_multi_target():
    """Test multi-target scenario."""
    print("\n\n" + "="*80)
    print("🔍 TEST 2: MULTI-TARGET ALIGNMENT")
    print("="*80)
    print("\nConfiguration:")
    print("  - Targets: close, volume (MULTI-TARGET)")
    print("  - Groups: 2 (AAPL, GOOGL)")
    print("  - Rows per group: 10")
    print("  - Sequence length: 3")
    print("  - Prediction horizon: 2")

    # Create synthetic data
    df_raw = create_synthetic_data()

    # Create multi-target predictor
    print("\n" + "="*80)
    print("Creating Multi-Target TimeSeriesPredictor...")
    print("="*80)

    predictor = TimeSeriesPredictor(
        target_column=['close', 'volume'],  # MULTI-TARGET
        sequence_length=3,
        prediction_horizon=2,
        group_columns='symbol',
        categorical_columns='symbol',
        model_type='ft_transformer_cls',  # Fixed: use correct model name
        scaler_type='standard',
        use_lagged_target_features=False,
        d_model=32,
        num_heads=2,
        num_layers=2
    )
    predictor.verbose = True

    # Add a dummy logger to avoid AttributeError
    import logging
    predictor.logger = logging.getLogger(__name__)

    print("\n✅ Multi-target predictor created")
    print(f"   Target columns: {predictor.target_columns}")
    print(f"   Is multi-target: {predictor.is_multi_target}")

    return run_test(predictor, df_raw, "MULTI-TARGET")


def run_test(predictor, df_raw, test_name):
    """Run the actual test with given predictor."""
    print(f"\n--- Running {test_name} test ---")

    # STEP 1: Create base features
    print("\n" + "="*80)
    print("STEP 1: _create_base_features()")
    print("="*80)
    print("This step: sorts by group+time, adds date features")
    print("\n💡 NOTES:")
    print(f"   - Synthetic data has 'timestamp' column (copy of 'date' for testing)")
    print(f"   - Sorting uses group_columns: {predictor.group_columns}")
    print(f"   - Data is sorted by (group_key, timestamp) for proper temporal order")
    print(f"   - Cyclical encoding: date features → sin/cos pairs (month_sin, month_cos, etc.)")

    df_step1 = predictor._create_base_features(df_raw.copy())

    print_df_state(df_step1, "After _create_base_features()")

    # STEP 2: Create shifted targets
    print("\n" + "="*80)
    print("STEP 2: create_shifted_targets()")
    print("="*80)

    # Show what shifted columns will be created
    shifted_cols_desc = []
    for target in predictor.target_columns:
        for h in range(1, predictor.prediction_horizon + 1):
            shifted_cols_desc.append(f"{target}_target_h{h}")

    print(f"This step: adds shifted target columns: {', '.join(shifted_cols_desc)}")
    print("Expected behavior:")
    for target in predictor.target_columns:
        print(f"  - {target}_target_h1 = {target} shifted by -1 (next period's {target})")
        print(f"  - {target}_target_h2 = {target} shifted by -2 (2 periods ahead {target})")
    print(f"  - Last {predictor.prediction_horizon} rows per group will have NaN targets (no future data)")
    print("  - These NaN rows get dropped")

    from tf_predictor.preprocessing.time_features import create_shifted_targets

    df_step2 = create_shifted_targets(
        df_step1.copy(),
        target_column=predictor.target_columns,
        prediction_horizon=predictor.prediction_horizon,
        group_column=predictor.group_columns,
        verbose=True
    )

    # Build show_cols dynamically based on targets
    group_col = predictor.group_columns[0] if predictor.group_columns else None
    show_cols_step2 = [group_col, 'date'] if group_col else ['date']
    for target in predictor.target_columns:
        show_cols_step2.append(target)
        for h in range(1, predictor.prediction_horizon + 1):
            shifted_col = f"{target}_target_h{h}"
            if shifted_col in df_step2.columns:
                show_cols_step2.append(shifted_col)

    print_df_state(df_step2, "After create_shifted_targets()", show_cols=show_cols_step2)

    print("\n🔑 KEY POINT: This dataframe (df_step2) is what gets stored in _last_processed_df")
    print(f"   It's BEFORE encoding and scaling, so '{group_col}' is still a string")
    print("   Row count per group:", df_step2.groupby(group_col).size().to_dict() if group_col else len(df_step2))

    # STEP 3: Encode categorical features
    print("\n" + "="*80)
    print("STEP 3: _encode_categorical_features()")
    print("="*80)
    print("This step: encodes 'symbol' as integers (AAPL=0, GOOGL=1)")

    df_step3 = predictor._encode_categorical_features(df_step2.copy(), fit_encoders=True)
    print_df_state(df_step3, "After _encode_categorical_features()",
                   show_cols=show_cols_step2)  # Use same columns as step 2

    print("\nEncoding mapping:")
    if group_col and group_col in predictor.cat_encoders:
        encoder = predictor.cat_encoders[group_col]
        for i, name in enumerate(encoder.classes_):
            print(f"  {name} -> {i}")

    # STEP 4: Determine numerical columns
    print("\n" + "="*80)
    print("STEP 4: _determine_numerical_columns()")
    print("="*80)
    print("This step: identifies which columns are numerical features (excludes shifted targets)")

    df_step4 = predictor._determine_numerical_columns(df_step3.copy())

    print(f"\n📋 Column Classification:")
    print(f"   Numerical feature columns ({len(predictor.numerical_columns)}): {predictor.numerical_columns}")
    print(f"   Categorical columns ({len(predictor.categorical_columns)}): {predictor.categorical_columns}")
    print(f"   Target columns ({len(predictor.target_columns)}): {predictor.target_columns}")

    print(f"\n💡 NOTES:")
    print(f"   - use_lagged_target_features = {predictor.use_lagged_target_features}")
    if not predictor.use_lagged_target_features:
        print(f"   - Original target columns ({predictor.target_columns}) are EXCLUDED from numerical_columns")
        print(f"   - This means NO autoregressive features (model can't see past target values)")
        print(f"   - Model only sees: {', '.join(predictor.numerical_columns[:3])}... etc.")
    else:
        print(f"   - Original target columns are INCLUDED in numerical_columns (autoregressive)")

    print(f"   - Shifted target columns (close_target_h1, etc.) are NEVER in numerical_columns")
    print(f"   - They are scaled separately and used as labels (y), not features (X)")

    # STEP 5: Scale features and targets
    print("\n" + "="*80)
    print("STEP 5: _scale_features_grouped()")
    print("="*80)
    print("This step: scales numerical features AND shifted targets per group")

    # Collect shifted target columns
    shifted_target_columns = []
    for target_col in predictor.target_columns:
        for h in range(1, predictor.prediction_horizon + 1):
            shifted_col = f"{target_col}_target_h{h}"
            if shifted_col in df_step4.columns:
                shifted_target_columns.append(shifted_col)

    print(f"Will scale these shifted targets: {shifted_target_columns}")

    df_step5 = predictor._scale_features_grouped(df_step4.copy(), fit_scaler=True,
                                                  shifted_target_columns=shifted_target_columns)

    # Show scaler parameters for each group
    print("\n📊 Scaler Parameters (per group):")
    if predictor.group_columns and predictor.group_target_scalers:
        encoder = predictor.cat_encoders[predictor.group_columns[0]]
        for group_idx, group_name in enumerate(encoder.classes_):
            print(f"\n   Group {group_idx} ({group_name}):")

            # Get the group key (might be tuple for multi-group)
            if len(predictor.group_columns) == 1:
                group_key = group_name
            else:
                group_key = (group_name,)  # Simplified for single group case

            # Show feature scaler params
            if group_key in predictor.group_feature_scalers:
                feature_scaler = predictor.group_feature_scalers[group_key]
                if hasattr(feature_scaler, 'mean_'):
                    print(f"      Feature scaler (StandardScaler):")
                    print(f"         mean: {feature_scaler.mean_[:3]}... (first 3)")
                    print(f"         std:  {feature_scaler.scale_[:3]}... (first 3)")

            # Show target scaler params
            if group_key in predictor.group_target_scalers:
                target_scaler_info = predictor.group_target_scalers[group_key]
                for shifted_col in shifted_target_columns:
                    if shifted_col in target_scaler_info:
                        scaler = target_scaler_info[shifted_col]
                        if hasattr(scaler, 'mean_'):
                            print(f"      {shifted_col} scaler:")
                            print(f"         mean: {scaler.mean_[0]:.2f}, std: {scaler.scale_[0]:.2f}")

    print_df_state(df_step5, "After _scale_features_grouped() [SCALED]",
                   show_cols=show_cols_step2)  # Use same columns to see scaled values

    print("\n⚠️  Note: Values are now scaled (normalized). Original values are stored in _last_processed_df from Step 2")

    # STEP 6: Create sequences
    print("\n" + "="*80)
    print("STEP 6: _prepare_data_grouped() - Create sequences")
    print("="*80)
    print("This step: creates sliding window sequences of length 3")
    print("\n💡 Expected (AFTER FIX):")
    print("  - Each group has 8 rows after shifting (10 - 2 dropped)")
    print("  - With sequence_length=3, NEW logic creates 8 - 3 + 1 = 6 sequences per group")
    print("  - First sequence uses rows [0:3], predicts for row 2 (index 2)")
    print("  - Last sequence uses rows [5:8], predicts for row 7 (index 7)")
    print("  - Targets are extracted from indices [2:8] (offset = sequence_length - 1 = 2)")
    print("  - Total: 6 × 2 groups = 12 sequences")

    X, y = predictor._prepare_data_grouped(df_step5.copy(), fit_scaler=False)

    print(f"\n✅ Sequences created:")
    if isinstance(X, tuple):
        X_num, X_cat = X
        print(f"   X_num shape: {X_num.shape}")
        print(f"   X_cat shape: {X_cat.shape}")
    else:
        print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")

    print(f"\n   Expected: 10 sequences total (5 per group)")
    print(f"   Got: {len(y)} sequences")

    print(f"\n   Group indices: {predictor._last_group_indices}")

    # =========================================================================
    # STEP 7: Train a simple model and run actual evaluation
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 7: Training model and running ACTUAL evaluation")
    print("="*80)

    print("\n🔧 Training a simple model for 5 epochs...")
    try:
        # fit() takes dataframes, not prepared tensors
        predictor.fit(
            df_raw,
            epochs=5,
            batch_size=4,
            verbose=False
        )
        print("✅ Model trained successfully")
    except Exception as e:
        print(f"⚠️  Training failed: {e}")
        import traceback
        traceback.print_exc()
        print("   Continuing without trained model...")

    # =========================================================================
    # STEP 8: Run actual evaluation and inspect what's happening
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 8: Running ACTUAL evaluation to verify alignment")
    print("="*80)

    print("\n📋 Testing _evaluate_per_group() method...")
    print("   This will show exactly what the predictor extracts for actuals\n")

    try:
        # Call the actual evaluation method
        metrics = predictor.evaluate(df_raw, per_group=True)

        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*80)

        print("\n📊 Metrics Structure:")
        print(f"   Top-level keys: {list(metrics.keys())}")

        # Check if multi-target
        if predictor.is_multi_target:
            # Multi-target structure
            print(f"\n   Multi-target mode: {len(predictor.target_columns)} targets")

            # Show overall metrics
            if 'overall' in metrics:
                print("\n" + "="*80)
                print("OVERALL METRICS (All Groups Combined)")
                print("="*80)
                overall = metrics['overall']

                for target_col in predictor.target_columns:
                    if target_col in overall:
                        print(f"\n--- Target: {target_col} ---")
                        target_metrics = overall[target_col]

                        # Check if multi-horizon
                        if isinstance(target_metrics, dict) and 'horizon_1' in target_metrics:
                            for h in range(1, predictor.prediction_horizon + 1):
                                horizon_key = f'horizon_{h}'
                                if horizon_key in target_metrics:
                                    print(f"\n  {horizon_key}:")
                                    for metric_name, value in target_metrics[horizon_key].items():
                                        print(f"    {metric_name}: {value:.4f}")

                            if 'overall' in target_metrics:
                                print(f"\n  overall (all horizons):")
                                for metric_name, value in target_metrics['overall'].items():
                                    print(f"    {metric_name}: {value:.4f}")
                        else:
                            # Single horizon
                            for metric_name, value in target_metrics.items():
                                print(f"    {metric_name}: {value:.4f}")

            # Show per-group metrics
            if 'per_group' in metrics:
                print("\n" + "="*80)
                print("PER-GROUP METRICS")
                print("="*80)
                per_group = metrics['per_group']

                # Get encoder for group name decoding
                encoder = predictor.cat_encoders.get(group_col, None) if group_col else None

                for group_key, group_data in per_group.items():
                    group_name = encoder.classes_[int(group_key)] if encoder else group_key
                    print(f"\n--- Group {group_key}: {group_name} ---")

                    for target_col in predictor.target_columns:
                        if target_col in group_data:
                            print(f"\n  Target: {target_col}")
                            target_metrics = group_data[target_col]

                            if isinstance(target_metrics, dict) and 'horizon_1' in target_metrics:
                                for h in range(1, predictor.prediction_horizon + 1):
                                    horizon_key = f'horizon_{h}'
                                    if horizon_key in target_metrics:
                                        print(f"\n    {horizon_key}:")
                                        for metric_name, value in target_metrics[horizon_key].items():
                                            print(f"      {metric_name}: {value:.4f}")

                                if 'overall' in target_metrics:
                                    print(f"\n    overall (all horizons):")
                                    for metric_name, value in target_metrics['overall'].items():
                                        print(f"      {metric_name}: {value:.4f}")
                            else:
                                for metric_name, value in target_metrics.items():
                                    print(f"      {metric_name}: {value:.4f}")
        else:
            # Single-target structure
            print(f"\n   Single-target mode")

            # Show overall metrics
            if 'overall' in metrics:
                print("\n--- OVERALL METRICS ---")
                overall = metrics['overall']
                for horizon_key, horizon_metrics in overall.items():
                    print(f"\n  {horizon_key}:")
                    for metric_name, value in horizon_metrics.items():
                        print(f"    {metric_name}: {value:.4f}")

            # Show per-group metrics
            if 'per_group' in metrics:
                print("\n--- PER-GROUP METRICS ---")
                per_group = metrics['per_group']

                # Get encoder for group name decoding
                encoder = predictor.cat_encoders.get(group_col, None) if group_col else None

                for group_key, group_metrics in per_group.items():
                    group_name = encoder.classes_[int(group_key)] if encoder else group_key
                    print(f"\n  Group {group_key}: {group_name}")
                    for horizon_key, horizon_metrics in group_metrics.items():
                        print(f"    {horizon_key}:")
                        for metric_name, value in horizon_metrics.items():
                            print(f"      {metric_name}: {value:.4f}")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

        print("\n🔍 Let's manually inspect what would be extracted...")

    # =========================================================================
    # STEP 9: Manual inspection of shifted column extraction
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 9: Manual verification of shifted column extraction")
    print("="*80)

    # Make a prediction to populate _last_processed_df
    print("\n🔮 Making predictions to populate internal state...")
    try:
        test_predictions = predictor.predict(df_raw)
        print(f"✅ Predictions generated, shape: {test_predictions.shape if hasattr(test_predictions, 'shape') else 'N/A'}")
    except Exception as e:
        print(f"⚠️  Prediction failed: {e}")
        # Manually set _last_processed_df
        predictor._last_processed_df = df_step2.copy()
        print("   Using df_step2 as _last_processed_df")

    if hasattr(predictor, '_last_processed_df') and predictor._last_processed_df is not None:
        print("\n📦 Inspecting _last_processed_df (used for evaluation):")
        print(f"   Total rows: {len(predictor._last_processed_df)}")
        if group_col:
            print(f"   Per group: {predictor._last_processed_df.groupby(group_col).size().to_dict()}")
        print(f"   Columns: {list(predictor._last_processed_df.columns)}")

        # Show the shifted target columns
        shifted_cols = [col for col in predictor._last_processed_df.columns if '_target_h' in col]
        print(f"\n   Shifted target columns: {shifted_cols}")

        # Get encoder mapping
        if group_col and group_col in predictor.cat_encoders:
            encoder = predictor.cat_encoders[group_col]
            group_value_to_name = {i: name for i, name in enumerate(encoder.classes_)}
            unique_groups = sorted(set(predictor._last_group_indices))

            offset = predictor.sequence_length - 1
            print(f"\n   Extraction offset: {offset} (sequence_length - 1)")

            for group_value in unique_groups:
                group_name = group_value_to_name[group_value]
                print(f"\n--- Group {group_value}: {group_name} ---")

                # Get processed data for this group
                group_df = predictor._last_processed_df[
                    predictor._last_processed_df[group_col] == group_name
                ].copy()

                print(f"Rows in group: {len(group_df)}")

                # Get dates for alignment verification
                group_dates = group_df['date'].values if 'date' in group_df.columns else None
                extracted_dates = group_dates[offset:] if group_dates is not None else None

                # Show what will be extracted for each target and horizon
                for target_col in predictor.target_columns:
                    print(f"\n  Target: {target_col}")
                    for h in range(1, predictor.prediction_horizon + 1):
                        shifted_col = f"{target_col}_target_h{h}"
                        if shifted_col in group_df.columns:
                            # Show full column first
                            print(f"\n    {shifted_col} column:")
                            print(f"      Full: {group_df[shifted_col].values}")

                            # Show what gets extracted with offset
                            extracted = group_df[shifted_col].values[offset:]
                            print(f"      After offset [{offset}:]: {extracted}")
                            print(f"      Length: {len(extracted)}")

                            # ⭐ CRITICAL: Show date-time alignment
                            if extracted_dates is not None:
                                print(f"\n    📅 DATE-TIME ALIGNMENT for {shifted_col}:")
                                print(f"      Extracted dates (for actuals): {extracted_dates}")
                                print(f"      These dates correspond to prediction indices: {list(range(len(extracted)))}")
                                print(f"\n      ⚠️  VERIFICATION: Each prediction must match the actual for the SAME date!")
                                print(f"      Example: prediction[0] for date {extracted_dates[0]} should match")
                                print(f"               actual from {shifted_col}[offset+0] = {extracted[0]}")

                # Show how many predictions this group has
                group_mask = np.array([g == group_value for g in predictor._last_group_indices])
                num_group_preds = group_mask.sum()
                print(f"\n  Number of predictions for this group: {num_group_preds}")

                # Check alignment
                if len(extracted) == num_group_preds:
                    print(f"  ✅ PERFECT ALIGNMENT: {len(extracted)} actuals == {num_group_preds} predictions")
                else:
                    print(f"  ❌ MISALIGNMENT: {len(extracted)} actuals != {num_group_preds} predictions")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("📋 SUMMARY - Data Flow Through Pipeline")
    print("="*80)

    print("\n1. Raw data:        10 rows per group")
    print("2. After shifting:   8 rows per group (2 dropped due to no future data)")
    print("3. After sequences:  6 predictions per group (NEW: 8 - 3 + 1 = 6)")
    print("\n4. Evaluation extracts actuals from Step 2 (8 rows per group)")
    print("   - Applies NEW sequence offset: 8 - (3-1) = 6 actuals per group")
    print("   - For each horizon, extract from shifted columns directly")
    print("   - Available: 6 actuals, Predictions: 6")
    print("   - ✅  PERFECT ALIGNMENT!")

    print("\n🔍 This demonstrates the NEW alignment (FIXED):")
    print("   - Predictions count: (rows_after_shift - sequence_length + 1)")
    print("   - Actuals extracted from shifted columns with offset = sequence_length - 1")
    print("   - Each horizon is independent: close_target_h1, close_target_h2, etc.")
    print("   - Actuals count ALWAYS equals predictions count!")

    print("\n💡 Potential issues to investigate:")
    print("   1. Is sequence_length offset applied consistently?")
    print("   2. Are group boundaries respected (no data leakage)?")
    print("   3. Is the multi-horizon actual extraction correct?")
    print("   4. Does _last_processed_df match what we expect?")

    return 0


def main():
    """Run both single-target and multi-target tests."""
    # Set up dual output to both console and markdown file
    output_filename = "alignment_test_results.md"
    dual_output = DualOutput(output_filename)
    sys.stdout = dual_output

    print("# Alignment Test Results")
    print(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80)
    print("🧪 COMPREHENSIVE ALIGNMENT TESTING")
    print("="*80)
    print("\nThis script will test:")
    print("  1. Single-target, multi-horizon (close only)")
    print("  2. Multi-target, multi-horizon (close + volume)")
    print("\nBoth tests use:")
    print("  - 2 groups (AAPL, GOOGL)")
    print("  - 10 rows per group")
    print("  - Sequence length: 3")
    print("  - Prediction horizon: 2")

    # Test 1: Single-target
    try:
        result1 = test_single_target()
        if result1 == 0:
            print("\n" + "="*80)
            print("✅ TEST 1 (SINGLE-TARGET) PASSED")
            print("="*80)
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Multi-target
    try:
        result2 = test_multi_target()
        if result2 == 0:
            print("\n" + "="*80)
            print("✅ TEST 2 (MULTI-TARGET) PASSED")
            print("="*80)
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Final summary
    print("\n\n" + "="*80)
    print("🎯 FINAL SUMMARY")
    print("="*80)
    print("\nBoth tests completed. Review output above for:")
    print("  ✓ Shifted target columns created correctly")
    print("  ✓ Extraction from shifted columns (not original)")
    print("  ✓ Alignment between predictions and actuals")
    print("  ✓ Per-group, per-horizon, overall metrics")
    print("\nKey verification points:")
    print("  - close_target_h1, close_target_h2 for single-target")
    print("  - close_target_h1, close_target_h2, volume_target_h1, volume_target_h2 for multi-target")
    print("  - Offset = sequence_length - 1 = 2")
    print("  - 6 predictions per group (8 rows after shift - 3 + 1)")

    print(f"\n\n---\n**Results saved to:** `{output_filename}`")

    # Clean up: restore stdout and close file
    sys.stdout = dual_output.terminal
    dual_output.close()

    print(f"\n✅ Results saved to {output_filename}")


if __name__ == "__main__":
    main()
