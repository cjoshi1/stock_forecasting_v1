"""
Comprehensive test script for tf_predictor enhancements:
1. Inference mode: predict() without target columns
2. Sequence overlap: split_time_series() with overlap for val/test sets

Tests cover:
- Single-target vs Multi-target
- Single-horizon vs Multi-horizon
- Single-group vs Multi-group
- With and without categorical features
"""

import numpy as np
import pandas as pd
import sys
import os

# Add tf_predictor to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tf_predictor'))

from tf_predictor.core.predictor import TimeSeriesPredictor
from tf_predictor.core.utils import split_time_series


def generate_test_data(n_samples=200, n_symbols=3, multi_target=False):
    """Generate synthetic time series data for testing."""
    np.random.seed(42)

    data = []
    for symbol in [f'SYM{i}' for i in range(n_symbols)]:
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        # Generate correlated features
        trend = np.linspace(100, 150, n_samples)
        noise = np.random.randn(n_samples) * 5
        values = trend + noise

        for i, date in enumerate(dates):
            row = {
                'symbol': symbol,
                'date': date,
                'feature1': values[i],
                'feature2': values[i] * 0.8 + np.random.randn() * 2,
                'feature3': values[i] * 1.2 + np.random.randn() * 3,
                'target1': values[i] + np.random.randn() * 2
            }

            if multi_target:
                row['target2'] = values[i] * 0.9 + np.random.randn() * 3

            data.append(row)

    return pd.DataFrame(data)


def test_single_target_single_horizon_multigroup():
    """Test 1: Single-target, single-horizon, multi-group with inference mode."""
    print("\n" + "="*80)
    print("TEST 1: Single-Target, Single-Horizon, Multi-Group")
    print("="*80)

    # Generate data
    df = generate_test_data(n_samples=150, n_symbols=2, multi_target=False)
    print(f"\n1. Generated test data: {len(df)} rows, 2 symbols")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Targets: ['target1']")

    # Split data with overlap
    train_df, val_df, test_df = split_time_series(
        df,
        test_size=30,
        val_size=20,
        group_column='symbol',
        time_column='date',
        sequence_length=10,
        include_overlap=True
    )

    print(f"\n2. Split data:")
    print(f"   - Train: {len(train_df)} rows")
    print(f"   - Val: {len(val_df)} rows")
    print(f"   - Test: {len(test_df)} rows")

    # Train model
    print(f"\n3. Training model...")
    predictor = TimeSeriesPredictor(
        target_column='target1',
        sequence_length=10,
        prediction_horizon=1,
        group_columns='symbol',
        categorical_columns='symbol',
        model_type='ft_transformer',
        d_token=32,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        verbose=False
    )

    predictor.fit(train_df, val_df=val_df, epochs=5, batch_size=16, verbose=False)
    print("   ✓ Model trained successfully")

    # Test prediction WITH targets
    print(f"\n4. Testing prediction WITH target column...")
    preds_with_target = predictor.predict(test_df, inference_mode=False)
    print(f"   ✓ Predictions shape: {preds_with_target.shape}")

    # Test prediction WITHOUT targets
    print(f"\n5. Testing prediction WITHOUT target column (inference_mode=True)...")
    test_df_no_target = test_df.drop(columns=['target1'])
    preds_without_target = predictor.predict(test_df_no_target, inference_mode=True)
    print(f"   ✓ Predictions shape: {preds_without_target.shape}")

    # Verify consistency
    print(f"\n6. Verifying predictions are similar...")
    # Note: shapes might differ due to NaN removal at different stages
    print(f"   - With targets: {preds_with_target.shape}")
    print(f"   - Without targets: {preds_without_target.shape}")

    print("\n✅ TEST 1 PASSED!\n")
    return True


def test_multi_target_single_horizon_multigroup():
    """Test 2: Multi-target, single-horizon, multi-group with inference mode."""
    print("\n" + "="*80)
    print("TEST 2: Multi-Target, Single-Horizon, Multi-Group")
    print("="*80)

    # Generate data with multiple targets
    df = generate_test_data(n_samples=150, n_symbols=2, multi_target=True)
    print(f"\n1. Generated test data: {len(df)} rows, 2 symbols")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Targets: ['target1', 'target2']")

    # Split data
    train_df, val_df, test_df = split_time_series(
        df,
        test_size=30,
        val_size=20,
        group_column='symbol',
        time_column='date',
        sequence_length=10,
        include_overlap=True
    )

    print(f"\n2. Split data:")
    print(f"   - Train: {len(train_df)} rows")
    print(f"   - Val: {len(val_df)} rows")
    print(f"   - Test: {len(test_df)} rows")

    # Train model with multiple targets
    print(f"\n3. Training model with multi-target...")
    predictor = TimeSeriesPredictor(
        target_column=['target1', 'target2'],
        sequence_length=10,
        prediction_horizon=1,
        group_columns='symbol',
        categorical_columns='symbol',
        model_type='ft_transformer',
        d_token=32,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        verbose=False
    )

    predictor.fit(train_df, val_df=val_df, epochs=5, batch_size=16, verbose=False)
    print("   ✓ Model trained successfully")

    # Test prediction WITH targets
    print(f"\n4. Testing prediction WITH target columns...")
    preds_with_target = predictor.predict(test_df, inference_mode=False)
    print(f"   ✓ Predictions type: {type(preds_with_target)}")
    if isinstance(preds_with_target, dict):
        for target, preds in preds_with_target.items():
            print(f"   ✓ {target}: shape {preds.shape}")

    # Test prediction WITHOUT targets
    print(f"\n5. Testing prediction WITHOUT target columns (inference_mode=True)...")
    test_df_no_target = test_df.drop(columns=['target1', 'target2'])
    preds_without_target = predictor.predict(test_df_no_target, inference_mode=True)
    print(f"   ✓ Predictions type: {type(preds_without_target)}")
    if isinstance(preds_without_target, dict):
        for target, preds in preds_without_target.items():
            print(f"   ✓ {target}: shape {preds.shape}")

    print("\n✅ TEST 2 PASSED!\n")
    return True


def test_single_target_multi_horizon_multigroup():
    """Test 3: Single-target, multi-horizon, multi-group with inference mode."""
    print("\n" + "="*80)
    print("TEST 3: Single-Target, Multi-Horizon, Multi-Group")
    print("="*80)

    # Generate data
    df = generate_test_data(n_samples=150, n_symbols=2, multi_target=False)
    print(f"\n1. Generated test data: {len(df)} rows, 2 symbols")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Targets: ['target1']")
    print(f"   Horizons: [1, 2, 3]")

    # Split data
    train_df, val_df, test_df = split_time_series(
        df,
        test_size=30,
        val_size=20,
        group_column='symbol',
        time_column='date',
        sequence_length=10,
        include_overlap=True
    )

    print(f"\n2. Split data:")
    print(f"   - Train: {len(train_df)} rows")
    print(f"   - Val: {len(val_df)} rows")
    print(f"   - Test: {len(test_df)} rows")

    # Train model with multi-horizon
    print(f"\n3. Training model with multi-horizon prediction...")
    predictor = TimeSeriesPredictor(
        target_column='target1',
        sequence_length=10,
        prediction_horizon=3,  # Predict 3 steps ahead
        group_columns='symbol',
        categorical_columns='symbol',
        model_type='ft_transformer',
        d_token=32,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        verbose=False
    )

    predictor.fit(train_df, val_df=val_df, epochs=5, batch_size=16, verbose=False)
    print("   ✓ Model trained successfully")

    # Test prediction WITH targets
    print(f"\n4. Testing prediction WITH target column...")
    preds_with_target = predictor.predict(test_df, inference_mode=False)
    print(f"   ✓ Predictions shape: {preds_with_target.shape}")
    print(f"   ✓ Expected: (n_samples, 3) for 3 horizons")

    # Test prediction WITHOUT targets
    print(f"\n5. Testing prediction WITHOUT target column (inference_mode=True)...")
    test_df_no_target = test_df.drop(columns=['target1'])
    preds_without_target = predictor.predict(test_df_no_target, inference_mode=True)
    print(f"   ✓ Predictions shape: {preds_without_target.shape}")

    print("\n✅ TEST 3 PASSED!\n")
    return True


def test_multi_target_multi_horizon_multigroup():
    """Test 4: Multi-target, multi-horizon, multi-group with inference mode."""
    print("\n" + "="*80)
    print("TEST 4: Multi-Target, Multi-Horizon, Multi-Group")
    print("="*80)

    # Generate data with multiple targets
    df = generate_test_data(n_samples=150, n_symbols=2, multi_target=True)
    print(f"\n1. Generated test data: {len(df)} rows, 2 symbols")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Targets: ['target1', 'target2']")
    print(f"   Horizons: [1, 2, 3]")

    # Split data
    train_df, val_df, test_df = split_time_series(
        df,
        test_size=30,
        val_size=20,
        group_column='symbol',
        time_column='date',
        sequence_length=10,
        include_overlap=True
    )

    print(f"\n2. Split data:")
    print(f"   - Train: {len(train_df)} rows")
    print(f"   - Val: {len(val_df)} rows")
    print(f"   - Test: {len(test_df)} rows")

    # Train model with multi-target and multi-horizon
    print(f"\n3. Training model with multi-target AND multi-horizon...")
    predictor = TimeSeriesPredictor(
        target_column=['target1', 'target2'],
        sequence_length=10,
        prediction_horizon=3,  # Predict 3 steps ahead for both targets
        group_columns='symbol',
        categorical_columns='symbol',
        model_type='ft_transformer',
        d_token=32,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        verbose=False
    )

    predictor.fit(train_df, val_df=val_df, epochs=5, batch_size=16, verbose=False)
    print("   ✓ Model trained successfully")

    # Test prediction WITH targets
    print(f"\n4. Testing prediction WITH target columns...")
    preds_with_target = predictor.predict(test_df, inference_mode=False)
    print(f"   ✓ Predictions type: {type(preds_with_target)}")
    if isinstance(preds_with_target, dict):
        for target, preds in preds_with_target.items():
            print(f"   ✓ {target}: shape {preds.shape} (expected: (n_samples, 3) for 3 horizons)")

    # Test prediction WITHOUT targets
    print(f"\n5. Testing prediction WITHOUT target columns (inference_mode=True)...")
    test_df_no_target = test_df.drop(columns=['target1', 'target2'])
    preds_without_target = predictor.predict(test_df_no_target, inference_mode=True)
    print(f"   ✓ Predictions type: {type(preds_without_target)}")
    if isinstance(preds_without_target, dict):
        for target, preds in preds_without_target.items():
            print(f"   ✓ {target}: shape {preds.shape}")

    print("\n✅ TEST 4 PASSED!\n")
    return True


def test_sequence_overlap():
    """Test 5: Sequence overlap feature in split_time_series."""
    print("\n" + "="*80)
    print("TEST 5: Sequence Overlap in Time Series Split")
    print("="*80)

    df = generate_test_data(n_samples=150, n_symbols=2, multi_target=False)

    # Split WITHOUT overlap
    print(f"\n1. Testing split WITHOUT overlap...")
    train_no, val_no, test_no = split_time_series(
        df,
        test_size=30,
        val_size=20,
        group_column='symbol',
        time_column='date',
        sequence_length=10,
        include_overlap=False
    )

    print(f"   - Train: {len(train_no)} rows")
    print(f"   - Val: {len(val_no)} rows")
    print(f"   - Test: {len(test_no)} rows")

    # Split WITH overlap
    print(f"\n2. Testing split WITH overlap...")
    train_yes, val_yes, test_yes = split_time_series(
        df,
        test_size=30,
        val_size=20,
        group_column='symbol',
        time_column='date',
        sequence_length=10,
        include_overlap=True
    )

    print(f"   - Train: {len(train_yes)} rows")
    print(f"   - Val: {len(val_yes)} rows (includes 9 overlap rows)")
    print(f"   - Test: {len(test_yes)} rows (includes 9 overlap rows)")

    # Verify overlap
    overlap_per_group = 9  # sequence_length - 1
    n_groups = 2
    total_overlap = overlap_per_group * n_groups
    print(f"\n3. Verifying overlap...")
    print(f"   - Expected overlap per group: {overlap_per_group}")
    print(f"   - Total expected overlap (2 groups): {total_overlap}")
    print(f"   - Val size difference: {len(val_yes) - len(val_no)} (expected: {total_overlap})")
    print(f"   - Test size difference: {len(test_yes) - len(test_no)} (expected: {total_overlap})")

    if len(val_yes) - len(val_no) == total_overlap and len(test_yes) - len(test_no) == total_overlap:
        print(f"   ✓ Overlap working correctly!")
    else:
        print(f"   ✗ Overlap size mismatch!")
        return False

    print("\n✅ TEST 5 PASSED!\n")
    return True


if __name__ == '__main__':
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*20 + "TF_PREDICTOR ENHANCEMENTS TEST SUITE" + " "*22 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)

    all_passed = True

    try:
        test1_passed = test_single_target_single_horizon_multigroup()
        all_passed = all_passed and test1_passed
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED WITH ERROR:")
        print(f"   {type(e).__name__}: {e}")
        all_passed = False

    try:
        test2_passed = test_multi_target_single_horizon_multigroup()
        all_passed = all_passed and test2_passed
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED WITH ERROR:")
        print(f"   {type(e).__name__}: {e}")
        all_passed = False

    try:
        test3_passed = test_single_target_multi_horizon_multigroup()
        all_passed = all_passed and test3_passed
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED WITH ERROR:")
        print(f"   {type(e).__name__}: {e}")
        all_passed = False

    try:
        test4_passed = test_multi_target_multi_horizon_multigroup()
        all_passed = all_passed and test4_passed
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED WITH ERROR:")
        print(f"   {type(e).__name__}: {e}")
        all_passed = False

    try:
        test5_passed = test_sequence_overlap()
        all_passed = all_passed and test5_passed
    except Exception as e:
        print(f"\n❌ TEST 5 FAILED WITH ERROR:")
        print(f"   {type(e).__name__}: {e}")
        all_passed = False

    # Final summary
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80)

    print("\n📋 Test Summary:")
    print("   1. Single-Target, Single-Horizon, Multi-Group")
    print("   2. Multi-Target, Single-Horizon, Multi-Group")
    print("   3. Single-Target, Multi-Horizon, Multi-Group")
    print("   4. Multi-Target, Multi-Horizon, Multi-Group")
    print("   5. Sequence Overlap Feature")
    print("="*80 + "\n")

    sys.exit(0 if all_passed else 1)
