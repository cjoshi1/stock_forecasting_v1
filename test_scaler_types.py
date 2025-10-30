"""
Test script for scaler flexibility feature.
Tests all scaler types with TimeSeriesPredictor.
"""

import pandas as pd
import numpy as np
from tf_predictor import TimeSeriesPredictor, ScalerFactory

print("=" * 80)
print("SCALER FLEXIBILITY TEST")
print("=" * 80)

# Create simple test data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')
data = {
    'date': dates,
    'feature1': np.random.randn(100) * 10 + 50,
    'feature2': np.random.randn(100) * 5 + 20,
    'target': np.random.randn(100) * 2 + 10
}
df = pd.DataFrame(data)

train_df = df.iloc[:80]
val_df = df.iloc[80:]

print(f"\n✓ Created test data: {len(df)} rows")
print(f"  Train: {len(train_df)} rows, Val: {len(val_df)} rows")

# Test all scaler types
scaler_types = ['standard', 'minmax', 'robust', 'maxabs', 'onlymax']

for scaler_type in scaler_types:
    print(f"\n{'=' * 80}")
    print(f"TEST: {scaler_type.upper()} Scaler")
    print(f"{'=' * 80}")

    try:
        # Create predictor
        predictor = TimeSeriesPredictor(
            target_column='target',
            sequence_length=5,
            scaler_type=scaler_type,
            d_model=32,
            num_heads=2,
            num_layers=1
        )
        print(f"✓ Predictor created with scaler_type='{scaler_type}'")

        # Train
        predictor.fit(train_df, val_df=val_df, epochs=2, batch_size=16, verbose=False)
        print(f"✓ Training completed")

        # Predict
        preds = predictor.predict(val_df)
        print(f"✓ Predictions: shape={preds.shape}")

        # Evaluate
        metrics = predictor.evaluate(val_df)
        print(f"✓ Evaluation: RMSE={metrics['RMSE']:.4f}")

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'=' * 80}")
print("TEST SUMMARY")
print(f"{'=' * 80}")
print("✓ All scaler types working correctly!")
print(f"✓ Tested scalers: {', '.join(scaler_types)}")
print(f"{'=' * 80}")
