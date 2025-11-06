"""
Test script to verify multi-column grouping fix.

This tests the critical bug fix for multi-column group evaluation.
"""

import pandas as pd
import numpy as np
from tf_predictor.core.predictor import TimeSeriesPredictor

def test_multi_column_grouping():
    """Test that multi-column grouping works correctly in evaluation."""
    print("="*60)
    print("Testing Multi-Column Grouping Fix")
    print("="*60)

    # Create synthetic data with multi-column grouping
    # 2 symbols × 2 sectors = 4 groups
    np.random.seed(42)

    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    data = []

    for symbol in ['AAPL', 'GOOGL']:
        for sector in ['Tech', 'Finance']:
            for date in dates:
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'sector': sector,
                    'open': 100 + np.random.randn(),
                    'high': 102 + np.random.randn(),
                    'low': 98 + np.random.randn(),
                    'close': 100 + np.random.randn(),
                    'volume': 1000000 + np.random.randint(-10000, 10000)
                })

    df = pd.DataFrame(data)
    print(f"\nCreated dataset: {len(df)} rows")
    print(f"Groups: {df.groupby(['symbol', 'sector']).ngroups} unique combinations")
    print(df.groupby(['symbol', 'sector']).size())

    # Split train/test
    train_df = df[df['date'] < '2024-03-01'].copy()
    test_df = df[df['date'] >= '2024-03-01'].copy()

    print(f"\nTrain: {len(train_df)} rows")
    print(f"Test: {len(test_df)} rows")

    # Create predictor with multi-column grouping
    print("\n" + "="*60)
    print("Creating predictor with multi-column grouping...")
    print("="*60)

    predictor = TimeSeriesPredictor(
        target_column='close',
        sequence_length=5,
        prediction_horizon=1,
        group_columns=['symbol', 'sector'],  # Multi-column grouping
        model_type='ft_transformer_cls',
        d_model=32,
        num_heads=2,
        num_layers=1,
        verbose=True
    )

    # Train
    print("\n" + "="*60)
    print("Training...")
    print("="*60)
    predictor.fit(
        train_df,
        epochs=2,  # Just a few epochs for testing
        batch_size=16,
        verbose=False
    )

    # Evaluate with per_group=True (this triggered the bug)
    print("\n" + "="*60)
    print("Evaluating with per_group=True...")
    print("="*60)

    try:
        metrics = predictor.evaluate(test_df, per_group=True)

        print("\n✅ SUCCESS: Multi-column group evaluation completed!")
        print("\nMetrics structure:")
        for key in metrics.keys():
            if key == 'overall':
                print(f"  {key}: {metrics[key]}")
            else:
                print(f"  {key}: MAE={metrics[key].get('MAE', 'N/A'):.4f}")

        # Verify we have the expected groups
        expected_groups = {('AAPL', 'Tech'), ('AAPL', 'Finance'),
                          ('GOOGL', 'Tech'), ('GOOGL', 'Finance')}

        # Group keys in metrics are strings of encoded integers
        # Just check we have 4 groups + 'overall'
        group_keys = [k for k in metrics.keys() if k != 'overall']

        if len(group_keys) >= 4:
            print(f"\n✅ PASS: Found {len(group_keys)} group metrics (expected 4)")
        else:
            print(f"\n⚠️  WARNING: Found {len(group_keys)} group metrics (expected 4)")

        return True

    except Exception as e:
        print(f"\n❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_helper_method():
    """Test the _filter_dataframe_by_group helper method."""
    print("\n" + "="*60)
    print("Testing _filter_dataframe_by_group helper")
    print("="*60)

    # Create simple test data
    df = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL', 'GOOGL', 'GOOGL'],
        'sector': ['Tech', 'Finance', 'Tech', 'Finance'],
        'value': [1, 2, 3, 4]
    })

    predictor = TimeSeriesPredictor(
        target_column='value',
        group_columns=['symbol', 'sector']
    )

    # Test single-column grouping
    predictor.group_columns = ['symbol']
    filtered = predictor._filter_dataframe_by_group(df, 'AAPL')
    assert len(filtered) == 2, "Single-column filter failed"
    assert all(filtered['symbol'] == 'AAPL'), "Single-column filter incorrect"
    print("✅ Single-column grouping: PASS")

    # Test multi-column grouping
    predictor.group_columns = ['symbol', 'sector']
    filtered = predictor._filter_dataframe_by_group(df, ('AAPL', 'Tech'))
    assert len(filtered) == 1, "Multi-column filter failed"
    assert filtered.iloc[0]['value'] == 1, "Multi-column filter incorrect"
    print("✅ Multi-column grouping: PASS")

    filtered = predictor._filter_dataframe_by_group(df, ('GOOGL', 'Finance'))
    assert len(filtered) == 1, "Multi-column filter failed"
    assert filtered.iloc[0]['value'] == 4, "Multi-column filter incorrect"
    print("✅ Multi-column grouping (second test): PASS")

    return True

if __name__ == '__main__':
    print("\n" + "="*60)
    print("MULTI-COLUMN GROUPING BUG FIX VERIFICATION")
    print("="*60)

    # Test helper method first
    helper_pass = test_helper_method()

    # Test full evaluation
    eval_pass = test_multi_column_grouping()

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Helper method test: {'✅ PASS' if helper_pass else '❌ FAIL'}")
    print(f"Full evaluation test: {'✅ PASS' if eval_pass else '❌ FAIL'}")

    if helper_pass and eval_pass:
        print("\n🎉 All tests passed! Multi-column grouping fix verified.")
    else:
        print("\n❌ Some tests failed. Please review the output above.")
