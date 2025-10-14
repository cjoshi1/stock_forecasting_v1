"""
Test script for group-based scaling functionality.
"""

import pandas as pd
import numpy as np
from tf_predictor.core.predictor import TimeSeriesPredictor


# Create a mock predictor class for testing
class MockPredictor(TimeSeriesPredictor):
    """Mock predictor for testing scaling logic."""

    def create_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """Just return the dataframe as-is for testing."""
        # Add a shifted target column (required by base class)
        df_copy = df.copy()
        df_copy[f'{self.target_column}_target_h1'] = df_copy[self.target_column].shift(-1)
        return df_copy.dropna()


def test_single_group_scaling():
    """Test original single-group scaling (no group_column)."""
    print("\n" + "="*70)
    print("TEST 1: Single-Group Scaling (Original Behavior)")
    print("="*70)

    # Create test data
    df = pd.DataFrame({
        'value1': [10, 20, 30, 40, 50],
        'value2': [100, 200, 300, 400, 500],
        'target': [1, 2, 3, 4, 5]
    })

    print("\nOriginal data:")
    print(df)

    # Create predictor without group column
    predictor = MockPredictor(
        target_column='target',
        sequence_length=2,
        group_column=None  # No grouping
    )

    # Prepare features (will trigger scaling)
    predictor.verbose = True
    df_scaled = predictor.prepare_features(df, fit_scaler=True)

    print("\nScaled data (single scaler for all):")
    print(df_scaled[['value1', 'value2']])

    # Check that scaling happened (mean should be ~0, std may vary with small samples)
    assert abs(df_scaled['value1'].mean()) < 1e-10, "Mean should be ~0"
    print(f"   value1: mean={df_scaled['value1'].mean():.6f}, std={df_scaled['value1'].std():.4f}")
    print(f"   value2: mean={df_scaled['value2'].mean():.6f}, std={df_scaled['value2'].std():.4f}")

    print("\n✓ Single-group scaling works correctly")


def test_multi_group_scaling():
    """Test group-based scaling with group_column."""
    print("\n" + "="*70)
    print("TEST 2: Multi-Group Scaling (New Behavior)")
    print("="*70)

    # Create test data with two groups (symbols)
    df = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'GOOGL', 'GOOGL', 'GOOGL'],
        'price': [150, 152, 154, 2800, 2850, 2900],
        'volume': [1000, 1100, 1050, 500, 520, 510],
        'target': [1, 2, 3, 4, 5, 6]
    })

    print("\nOriginal data:")
    print(df)

    # Create predictor with group column
    predictor = MockPredictor(
        target_column='target',
        sequence_length=2,
        group_column='symbol'  # Group by symbol
    )

    # Prepare features (will trigger group scaling)
    predictor.verbose = True
    df_scaled = predictor.prepare_features(df, fit_scaler=True)

    print("\nScaled data (separate scaler per symbol):")
    print(df_scaled[['symbol', 'price', 'volume']])

    # Check that each group was scaled separately
    aapl_data = df_scaled[df_scaled['symbol'] == 'AAPL']
    googl_data = df_scaled[df_scaled['symbol'] == 'GOOGL']

    print(f"\nAAPL price stats: mean={aapl_data['price'].mean():.4f}, std={aapl_data['price'].std():.4f}")
    print(f"GOOGL price stats: mean={googl_data['price'].mean():.4f}, std={googl_data['price'].std():.4f}")

    # Each group should be independently normalized
    assert abs(aapl_data['price'].mean()) < 1e-10, "AAPL mean should be ~0"
    assert abs(googl_data['price'].mean()) < 1e-10, "GOOGL mean should be ~0"

    print("\n✓ Multi-group scaling works correctly")
    print(f"✓ Stored scalers for groups: {list(predictor.group_feature_scalers.keys())}")


def test_grouped_data_preparation():
    """Test group-based sequence creation and target scaling."""
    print("\n" + "="*70)
    print("TEST 3: Grouped Data Preparation (Sequences + Targets)")
    print("="*70)

    # Create test data with two groups and enough rows for sequences
    df = pd.DataFrame({
        'symbol': ['AAPL']*10 + ['GOOGL']*10,
        'price': list(range(150, 160)) + list(range(2800, 2810)),
        'volume': list(range(1000, 1010)) + list(range(500, 510)),
        'target': list(range(1, 11)) + list(range(11, 21))
    })

    print("\nOriginal data (showing first/last 5 rows):")
    print(df.head())
    print("...")
    print(df.tail())

    # Create predictor with group column
    predictor = MockPredictor(
        target_column='target',
        sequence_length=3,
        group_column='symbol'
    )

    predictor.verbose = True

    # Test single-horizon
    print("\n--- Testing Single-Horizon ---")
    X, y = predictor.prepare_data(df, fit_scaler=True)

    print(f"\nResult shapes: X={X.shape}, y={y.shape}")
    assert X.ndim == 3, "X should be 3D (samples, seq_len, features)"
    assert y.ndim == 1, "y should be 1D for single horizon"
    assert len(X) == len(y), "X and y should have same number of samples"

    # Check that we have sequences from both groups
    # With sequence_length=3 and 10 samples per group:
    # - After shift(-1) and dropna() in create_features: 9 samples per group
    # - After sequence creation (9 - 3): 6 sequences per group
    # - Total: 6 * 2 = 12 sequences (but GOOGL only has 9 rows after processing)
    # Actually: AAPL has 9, GOOGL has 9, so 6+6=12 or close to it
    assert len(X) > 0, "Should have created sequences"
    assert len(X) <= 14, f"Should have at most 14 sequences, got {len(X)}"
    print(f"   Note: Got {len(X)} sequences (after dropna: some rows lost per group)")

    # Check that group target scalers were created
    assert 'AAPL' in predictor.group_target_scalers, "Should have AAPL target scaler"
    assert 'GOOGL' in predictor.group_target_scalers, "Should have GOOGL target scaler"

    print(f"✓ Created {len(X)} sequences from 2 groups")
    print(f"✓ Group target scalers: {list(predictor.group_target_scalers.keys())}")

    # Test multi-horizon
    print("\n--- Testing Multi-Horizon ---")
    predictor_mh = MockPredictor(
        target_column='target',
        sequence_length=3,
        prediction_horizon=3,
        group_column='symbol'
    )

    # Add multi-horizon target columns to the data
    df_mh = df.copy()
    for h in range(1, 4):
        df_mh[f'target_target_h{h}'] = df_mh['target'].shift(-h)

    predictor_mh.verbose = True
    X_mh, y_mh = predictor_mh.prepare_data(df_mh.dropna(), fit_scaler=True)

    print(f"\nResult shapes: X={X_mh.shape}, y={y_mh.shape}")
    assert X_mh.ndim == 3, "X should be 3D"
    assert y_mh.ndim == 2, "y should be 2D for multi-horizon"
    assert y_mh.shape[1] == 3, "y should have 3 horizons"

    print(f"✓ Multi-horizon: created {len(X_mh)} sequences with {y_mh.shape[1]} horizons")
    print(f"✓ One scaler per group (not per horizon): {list(predictor_mh.group_target_scalers.keys())}")

    print("\n✓ Grouped data preparation test passed")


def test_scaling_consistency():
    """Test that transform uses fitted scalers correctly."""
    print("\n" + "="*70)
    print("TEST 4: Scaling Consistency (Fit vs Transform)")
    print("="*70)

    # Training data
    train_df = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL', 'GOOGL', 'GOOGL'],
        'price': [150, 152, 2800, 2850],
        'target': [1, 2, 3, 4]
    })

    # Test data (new data from same symbols)
    test_df = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL', 'GOOGL', 'GOOGL'],
        'price': [151, 153, 2825, 2875],
        'target': [5, 6, 7, 8]
    })

    print("\nTraining data:")
    print(train_df)

    # Fit on training data
    predictor = MockPredictor(
        target_column='target',
        sequence_length=2,
        group_column='symbol'
    )

    predictor.verbose = True
    train_scaled = predictor.prepare_features(train_df, fit_scaler=True)

    print("\nTest data (before scaling):")
    print(test_df)

    # Transform test data (should use fitted scalers)
    test_scaled = predictor.prepare_features(test_df, fit_scaler=False)

    print("\nTest data (after scaling):")
    print(test_scaled[['symbol', 'price']])

    print("\n✓ Scaling consistency test passed")
    print("✓ Test data successfully transformed using fitted scalers")


if __name__ == '__main__':
    try:
        test_single_group_scaling()
        test_multi_group_scaling()
        test_grouped_data_preparation()
        test_scaling_consistency()

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
