"""
Test that automatic sorting ensures temporal order even with unsorted input data.
"""

import pandas as pd
import numpy as np
from tf_predictor.core.predictor import TimeSeriesPredictor


class MockPredictor(TimeSeriesPredictor):
    """Mock predictor for testing auto-sort."""

    def create_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """Just return the dataframe with target column."""
        df_copy = df.copy()
        df_copy[f'{self.target_column}_target_h1'] = df_copy[self.target_column].shift(-1)
        return df_copy.dropna()


def test_auto_sort_with_unsorted_data():
    """Test that automatic sorting fixes unsorted input data."""
    print("\n" + "="*70)
    print("TEST: Automatic Sorting with UNSORTED Input Data")
    print("="*70)

    # Create data that is INTENTIONALLY UNSORTED (shuffled)
    data = []
    # Generate data for two symbols with explicit timestamps
    for t in range(20):
        if t % 2 == 0:
            data.append({'symbol': 'AAPL', 'timestamp': t, 'price': 100 + t, 'target': t})
        else:
            data.append({'symbol': 'GOOGL', 'timestamp': t, 'price': 2000 + t, 'target': t})

    df = pd.DataFrame(data)

    # NOW SHUFFLE THE DATA to simulate unsorted input
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nShuffled (UNSORTED) input data:")
    print(df_shuffled.head(15))
    print("\nNotice: timestamps are OUT OF ORDER within each symbol!")

    # Create predictor with group column
    predictor = MockPredictor(
        target_column='target',
        sequence_length=3,
        group_column='symbol',
        verbose=True
    )

    print("\n" + "-"*70)
    print("Processing with automatic sorting enabled...")
    print("-"*70)

    X, y = predictor.prepare_data(df_shuffled, fit_scaler=True)

    print(f"\n✓ Created {len(X)} sequences from 2 groups")

    # Verify temporal order is NOW correct after auto-sorting
    df_processed = predictor.prepare_features(df_shuffled, fit_scaler=False)

    # Check AAPL sequences
    aapl_data = df_processed[df_processed['symbol'] == 'AAPL']
    aapl_timestamps = aapl_data['timestamp'].values
    print(f"\nAAPL timestamps after processing: {aapl_timestamps}")

    # Check if timestamps are monotonically increasing
    is_aapl_sorted = all(aapl_timestamps[i] < aapl_timestamps[i+1] for i in range(len(aapl_timestamps)-1))

    # Check GOOGL sequences
    googl_data = df_processed[df_processed['symbol'] == 'GOOGL']
    googl_timestamps = googl_data['timestamp'].values
    print(f"GOOGL timestamps after processing: {googl_timestamps}")

    is_googl_sorted = all(googl_timestamps[i] < googl_timestamps[i+1] for i in range(len(googl_timestamps)-1))

    print("\n" + "="*70)
    if is_aapl_sorted and is_googl_sorted:
        print("✓ PASS: Automatic sorting FIXED the unsorted data!")
        print("  Data is now in correct temporal order within each group")
    else:
        print("✗ FAIL: Automatic sorting did NOT work!")
        if not is_aapl_sorted:
            print("  - AAPL sequences are still out of order")
        if not is_googl_sorted:
            print("  - GOOGL sequences are still out of order")
    print("="*70)

    return is_aapl_sorted and is_googl_sorted


def test_auto_sort_without_timestamp():
    """Test behavior when no timestamp column is present."""
    print("\n" + "="*70)
    print("TEST: Auto-sort with NO timestamp column")
    print("="*70)

    # Create data WITHOUT a timestamp column
    data = []
    for i in range(20):
        if i % 2 == 0:
            data.append({'symbol': 'AAPL', 'price': 100 + i, 'target': i})
        else:
            data.append({'symbol': 'GOOGL', 'price': 2000 + i, 'target': i})

    df = pd.DataFrame(data)

    print("\nData without timestamp column:")
    print(df.head(10))

    predictor = MockPredictor(
        target_column='target',
        sequence_length=3,
        group_column='symbol',
        verbose=True
    )

    print("\n" + "-"*70)
    print("Processing without timestamp column...")
    print("-"*70)

    try:
        X, y = predictor.prepare_data(df, fit_scaler=True)
        print(f"\n✓ Successfully processed {len(X)} sequences")
        print("✓ PASS: Works correctly even without timestamp column")
        return True
    except Exception as e:
        print(f"\n✗ FAIL: Error processing data without timestamp: {e}")
        return False


if __name__ == '__main__':
    try:
        result1 = test_auto_sort_with_unsorted_data()
        result2 = test_auto_sort_without_timestamp()

        if result1 and result2:
            print("\n" + "="*70)
            print("ALL AUTO-SORT TESTS PASSED! ✓")
            print("="*70)
        else:
            print("\n✗ Some tests failed")
            exit(1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
