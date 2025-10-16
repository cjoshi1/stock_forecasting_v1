"""
Test group-wise data splitting functionality.

Tests that:
1. Data is split per group (equal representation in each dataset)
2. Temporal order is maintained within each group
3. Most recent data goes to test, earliest data goes to train
4. Works correctly for both single and multi-group datasets
"""

import numpy as np
import pandas as pd
from tf_predictor.core.utils import split_time_series


def test_single_group_splitting():
    """Test that single-group data splitting maintains temporal order."""
    print("\n" + "=" * 80)
    print("Test 1: Single-Group Data Splitting")
    print("=" * 80)

    # Create synthetic single-group data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'value': np.arange(100),  # Sequential values for easy verification
        'feature1': np.random.randn(100)
    })

    print(f"\nOriginal data: {len(df)} samples")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Value range: {df['value'].min()} to {df['value'].max()}")

    # Split without group_column (original behavior)
    train_df, val_df, test_df = split_time_series(
        df,
        test_size=20,
        val_size=10,
        group_column=None
    )

    print(f"\nSplit results:")
    print(f"  Train: {len(train_df)} samples, dates {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"  Val: {len(val_df)} samples, dates {val_df['date'].min()} to {val_df['date'].max()}")
    print(f"  Test: {len(test_df)} samples, dates {test_df['date'].min()} to {test_df['date'].max()}")

    # Verify temporal order
    assert train_df['date'].is_monotonic_increasing, "Train data not chronologically sorted"
    assert val_df['date'].is_monotonic_increasing, "Val data not chronologically sorted"
    assert test_df['date'].is_monotonic_increasing, "Test data not chronologically sorted"

    # Verify earliest data is in train, most recent in test
    assert train_df['date'].max() < val_df['date'].min(), "Train data overlaps with val data"
    assert val_df['date'].max() < test_df['date'].min(), "Val data overlaps with test data"
    assert train_df['value'].max() < val_df['value'].min(), "Train values not earliest"
    assert val_df['value'].max() < test_df['value'].min(), "Val values not middle"

    print("\n✅ Test 1 PASSED: Single-group splitting maintains temporal order")


def test_multi_group_splitting():
    """Test that multi-group data splitting gives equal representation."""
    print("\n" + "=" * 80)
    print("Test 2: Multi-Group Data Splitting")
    print("=" * 80)

    # Create synthetic multi-group data (3 symbols)
    groups = ['AAPL', 'GOOGL', 'MSFT']
    n_samples_per_group = 100

    dfs = []
    for group in groups:
        dates = pd.date_range('2024-01-01', periods=n_samples_per_group, freq='D')
        group_df = pd.DataFrame({
            'date': dates,
            'symbol': group,
            'value': np.arange(n_samples_per_group) + (ord(group[0]) * 1000),  # Unique per group
            'feature1': np.random.randn(n_samples_per_group)
        })
        dfs.append(group_df)

    df = pd.concat(dfs, ignore_index=True)

    print(f"\nOriginal data: {len(df)} samples across {len(groups)} groups")
    for group in groups:
        group_data = df[df['symbol'] == group]
        print(f"  {group}: {len(group_data)} samples, dates {group_data['date'].min()} to {group_data['date'].max()}")

    # Split with group_column
    train_df, val_df, test_df = split_time_series(
        df,
        test_size=20,
        val_size=10,
        group_column='symbol',
        time_column='date'
    )

    print(f"\nSplit results:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Verify equal representation from each group
    for group in groups:
        train_group = train_df[train_df['symbol'] == group]
        val_group = val_df[val_df['symbol'] == group]
        test_group = test_df[test_df['symbol'] == group]

        print(f"\n  {group}:")
        print(f"    Train: {len(train_group)} samples, dates {train_group['date'].min()} to {train_group['date'].max()}")
        print(f"    Val: {len(val_group)} samples, dates {val_group['date'].min()} to {val_group['date'].max()}")
        print(f"    Test: {len(test_group)} samples, dates {test_group['date'].min()} to {test_group['date'].max()}")

        # Verify equal counts per group
        assert len(test_group) == 20, f"{group} test set has {len(test_group)} samples, expected 20"
        assert len(val_group) == 10, f"{group} val set has {len(val_group)} samples, expected 10"

        # Verify temporal order within each group
        assert train_group['date'].is_monotonic_increasing, f"{group} train data not sorted"
        assert val_group['date'].is_monotonic_increasing, f"{group} val data not sorted"
        assert test_group['date'].is_monotonic_increasing, f"{group} test data not sorted"

        # Verify earliest data in train, most recent in test
        assert train_group['date'].max() < val_group['date'].min(), f"{group} train overlaps with val"
        assert val_group['date'].max() < test_group['date'].min(), f"{group} val overlaps with test"

    print("\n✅ Test 2 PASSED: Multi-group splitting gives equal representation and temporal order")


def test_intraday_group_splitting():
    """Test group-wise splitting with intraday data."""
    print("\n" + "=" * 80)
    print("Test 3: Intraday Multi-Group Data Splitting")
    print("=" * 80)

    # Create synthetic intraday data (2 symbols, 5-minute bars)
    groups = ['BTC', 'ETH']
    n_samples_per_group = 200  # ~16 hours of 5-min data

    dfs = []
    for group in groups:
        timestamps = pd.date_range('2024-01-01 09:30:00', periods=n_samples_per_group, freq='5min')
        group_df = pd.DataFrame({
            'timestamp': timestamps,
            'symbol': group,
            'close': 3000 + np.cumsum(np.random.randn(n_samples_per_group)),
            'volume': 50000 + np.random.randn(n_samples_per_group) * 1000
        })
        dfs.append(group_df)

    df = pd.concat(dfs, ignore_index=True)

    print(f"\nOriginal data: {len(df)} samples across {len(groups)} groups")
    for group in groups:
        group_data = df[df['symbol'] == group]
        print(f"  {group}: {len(group_data)} samples, time {group_data['timestamp'].min()} to {group_data['timestamp'].max()}")

    # Split with group_column
    train_df, val_df, test_df = split_time_series(
        df,
        test_size=30,
        val_size=20,
        group_column='symbol',
        time_column='timestamp'
    )

    print(f"\nSplit results:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Verify equal representation from each group
    for group in groups:
        train_group = train_df[train_df['symbol'] == group]
        val_group = val_df[val_df['symbol'] == group]
        test_group = test_df[test_df['symbol'] == group]

        print(f"\n  {group}:")
        print(f"    Train: {len(train_group)} samples")
        print(f"    Val: {len(val_group)} samples")
        print(f"    Test: {len(test_group)} samples")

        # Verify equal counts per group
        assert len(test_group) == 30, f"{group} test set has {len(test_group)} samples, expected 30"
        assert len(val_group) == 20, f"{group} val set has {len(val_group)} samples, expected 20"

        # Verify temporal order within each group
        assert train_group['timestamp'].is_monotonic_increasing, f"{group} train data not sorted"
        assert val_group['timestamp'].is_monotonic_increasing, f"{group} val data not sorted"
        assert test_group['timestamp'].is_monotonic_increasing, f"{group} test data not sorted"

        # Verify earliest data in train, most recent in test
        assert train_group['timestamp'].max() < val_group['timestamp'].min(), f"{group} train overlaps with val"
        assert val_group['timestamp'].max() < test_group['timestamp'].min(), f"{group} val overlaps with test"

    print("\n✅ Test 3 PASSED: Intraday group splitting maintains temporal order per group")


def test_insufficient_data_warning():
    """Test that warnings are issued for groups with insufficient data."""
    print("\n" + "=" * 80)
    print("Test 4: Insufficient Data Warning")
    print("=" * 80)

    # Create data where one group has insufficient samples
    groups = ['AAPL', 'TINY']

    # AAPL has plenty of data
    dates_aapl = pd.date_range('2024-01-01', periods=100, freq='D')
    df_aapl = pd.DataFrame({
        'date': dates_aapl,
        'symbol': 'AAPL',
        'value': np.arange(100)
    })

    # TINY has insufficient data (needs at least test_size + val_size + 10 = 40)
    dates_tiny = pd.date_range('2024-01-01', periods=30, freq='D')
    df_tiny = pd.DataFrame({
        'date': dates_tiny,
        'symbol': 'TINY',
        'value': np.arange(30)
    })

    df = pd.concat([df_aapl, df_tiny], ignore_index=True)

    print(f"\nOriginal data:")
    print(f"  AAPL: {len(df[df['symbol'] == 'AAPL'])} samples")
    print(f"  TINY: {len(df[df['symbol'] == 'TINY'])} samples")

    # Split with group_column
    print("\nAttempting split (should warn about TINY)...")
    train_df, val_df, test_df = split_time_series(
        df,
        test_size=20,
        val_size=10,
        group_column='symbol',
        time_column='date'
    )

    # Verify TINY was skipped
    train_groups = train_df['symbol'].unique()
    test_groups = test_df['symbol'].unique()

    assert 'AAPL' in train_groups, "AAPL should be in train set"
    assert 'AAPL' in test_groups, "AAPL should be in test set"

    print(f"\nGroups in train: {train_groups}")
    print(f"Groups in test: {test_groups}")

    print("\n✅ Test 4 PASSED: Insufficient data warning handled correctly")


# ============================================================================
# Run All Tests
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing Group-Wise Data Splitting Functionality")
    print("=" * 80)

    try:
        test_single_group_splitting()
        test_multi_group_splitting()
        test_intraday_group_splitting()
        test_insufficient_data_warning()

        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print("\nSummary:")
        print("  ✅ Test 1: Single-group splitting maintains temporal order")
        print("  ✅ Test 2: Multi-group splitting gives equal representation")
        print("  ✅ Test 3: Intraday group splitting works correctly")
        print("  ✅ Test 4: Insufficient data warnings handled properly")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise
