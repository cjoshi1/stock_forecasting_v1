"""
Quick test to verify basic alignment implementation works.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add tf_predictor to path
sys.path.insert(0, str(Path(__file__).parent))

from tf_predictor.core.predictor import TimeSeriesPredictor

def test_basic_alignment():
    """Test basic alignment with simplest case: no groups, single target, single horizon"""
    print("\n" + "="*80)
    print("TEST: Basic Alignment (No groups, single target, single horizon)")
    print("="*80)

    # Generate simple synthetic data
    np.random.seed(42)
    n_rows = 100

    data = {
        'date': pd.date_range('2020-01-01', periods=n_rows, freq='D'),
        'feature_1': np.random.randn(n_rows),
        'feature_2': np.random.randn(n_rows),
        'target': np.cumsum(np.random.randn(n_rows)) + 100
    }
    df = pd.DataFrame(data)

    print(f"✓ Generated {len(df)} rows of synthetic data")

    # Create predictor
    predictor = TimeSeriesPredictor(
        target_column='target',
        sequence_length=10,
        prediction_horizon=1,
        group_columns=None,
        verbose=True
    )

    print(f"\n✓ Created predictor (seq_len=10, horizon=1, no groups)")

    # Fit
    print(f"\nTraining...")
    predictor.fit(df, epochs=2, batch_size=16, verbose=False)
    print(f"✓ Training complete")

    # Check that indices were stored
    print(f"\nChecking index tracking...")
    assert hasattr(predictor, '_last_sequence_indices'), "Missing _last_sequence_indices"
    print(f"✓ _last_sequence_indices exists: {len(predictor._last_sequence_indices)} indices")

    # Check that _last_processed_df has _original_index
    assert hasattr(predictor, '_last_processed_df'), "Missing _last_processed_df"
    assert '_original_index' in predictor._last_processed_df.columns, "Missing _original_index in _last_processed_df"
    print(f"✓ _last_processed_df has _original_index column")

    # Make predictions
    print(f"\nMaking predictions...")
    predictions = predictor.predict(df)
    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"✓ Sequence indices shape: {predictor._last_sequence_indices.shape}")

    # Verify alignment
    print(f"\nVerifying alignment...")
    assert len(predictions) == len(predictor._last_sequence_indices), \
        f"Mismatch: {len(predictions)} predictions vs {len(predictor._last_sequence_indices)} indices"
    print(f"✓ Prediction count matches index count: {len(predictions)}")

    # Evaluate
    print(f"\nEvaluating...")
    metrics = predictor.evaluate(df)
    print(f"✓ Evaluation complete")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.4f}")

    # Test return_indices parameter
    print(f"\nTesting return_indices parameter...")
    predictions_with_indices = predictor.predict(df, return_indices=True)
    assert len(predictions_with_indices) == 2, "Should return tuple of (predictions, indices)"
    preds, indices = predictions_with_indices
    print(f"✓ return_indices=True works")
    print(f"   Predictions: {len(preds)}")
    print(f"   Indices: {len(indices)}")
    print(f"   Sample indices: {indices[:5]}")

    print(f"\n" + "="*80)
    print("TEST PASSED ✓")
    print("="*80)

if __name__ == '__main__':
    test_basic_alignment()
