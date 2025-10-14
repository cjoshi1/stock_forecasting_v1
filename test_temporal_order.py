"""
Test that temporal order is maintained within each group during training.
"""

import pandas as pd
import numpy as np
from tf_predictor.core.predictor import TimeSeriesPredictor


class MockPredictor(TimeSeriesPredictor):
    """Mock predictor for testing temporal order."""

    def create_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """Just return the dataframe with target column."""
        df_copy = df.copy()
        df_copy[f'{self.target_column}_target_h1'] = df_copy[self.target_column].shift(-1)
        return df_copy.dropna()


def test_temporal_order_maintained():
    """Test that temporal order is maintained within each group."""
    print("\n" + "="*70)
    print("TEST: Temporal Order Maintenance in Group-Based Scaling")
    print("="*70)

    # Create data with EXPLICIT temporal order and TWO groups INTERLEAVED
    # This simulates real-world multi-stock data where rows alternate between symbols
    data = []
    for t in range(20):
        # Alternate between AAPL and GOOGL
        if t % 2 == 0:
            data.append({'symbol': 'AAPL', 'time': t, 'price': 100 + t, 'target': t})
        else:
            data.append({'symbol': 'GOOGL', 'time': t, 'price': 2000 + t, 'target': t})

    df = pd.DataFrame(data)

    print("\nOriginal data (interleaved symbols):")
    print(df.head(10))
    print("...")
    print(df.tail(10))

    # Create predictor with group column
    predictor = MockPredictor(
        target_column='target',
        sequence_length=3,
        group_column='symbol'
    )

    predictor.verbose = True
    X, y = predictor.prepare_data(df, fit_scaler=True)

    print(f"\n✓ Created {len(X)} sequences from 2 interleaved groups")

    # Now verify temporal order is maintained
    # Extract the processed data to check sequence order
    df_processed = predictor.prepare_features(df, fit_scaler=False)

    # Check AAPL sequences
    aapl_data = df_processed[df_processed['symbol'] == 'AAPL']
    print(f"\nAAPL extracted data (should be in temporal order):")
    print(aapl_data[['symbol', 'time', 'price']].head(10))

    # Verify AAPL times are sequential
    aapl_times = aapl_data['time'].values
    print(f"\nAAPL times: {aapl_times}")

    # Check if times are monotonically increasing
    is_aapl_sequential = all(aapl_times[i] < aapl_times[i+1] for i in range(len(aapl_times)-1))

    # Check GOOGL sequences
    googl_data = df_processed[df_processed['symbol'] == 'GOOGL']
    print(f"\nGOOGL extracted data (should be in temporal order):")
    print(googl_data[['symbol', 'time', 'price']].head(10))

    googl_times = googl_data['time'].values
    print(f"\nGOOGL times: {googl_times}")

    is_googl_sequential = all(googl_times[i] < googl_times[i+1] for i in range(len(googl_times)-1))

    print("\n" + "="*70)
    if is_aapl_sequential and is_googl_sequential:
        print("✓ PASS: Temporal order is MAINTAINED within each group")
    else:
        print("✗ FAIL: Temporal order is BROKEN!")
        if not is_aapl_sequential:
            print("  - AAPL sequences are out of order")
        if not is_googl_sequential:
            print("  - GOOGL sequences are out of order")
    print("="*70)

    return is_aapl_sequential and is_googl_sequential


if __name__ == '__main__':
    try:
        result = test_temporal_order_maintained()
        if result:
            print("\n✓ Temporal order test PASSED")
        else:
            print("\n✗ Temporal order test FAILED")
            exit(1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
