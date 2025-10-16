"""
Test script for multi-target prediction functionality.
"""
import pandas as pd
import numpy as np
from intraday_forecasting.predictor import IntradayPredictor

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("Multi-Target Prediction Test")
print("=" * 80)

# Create synthetic data
n_samples = 200
dates = pd.date_range('2024-01-01', periods=n_samples, freq='1h')

df = pd.DataFrame({
    'timestamp': dates,
    'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
    'high': 102 + np.cumsum(np.random.randn(n_samples) * 0.5),
    'low': 98 + np.cumsum(np.random.randn(n_samples) * 0.5),
    'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
    'volume': 1000000 + np.random.randint(-100000, 100000, n_samples)
})

# Split data
train_size = int(0.7 * len(df))
val_size = int(0.15 * len(df))

train_df = df[:train_size].copy()
val_df = df[train_size:train_size + val_size].copy()
test_df = df[train_size + val_size:].copy()

print(f"\nData split:")
print(f"  Train: {len(train_df)} samples")
print(f"  Val:   {len(val_df)} samples")
print(f"  Test:  {len(test_df)} samples")

# Test 1: Single-target, single-horizon (backward compatibility)
print("\n" + "=" * 80)
print("Test 1: Single-Target, Single-Horizon (Backward Compatibility)")
print("=" * 80)

try:
    model1 = IntradayPredictor(
        target_column='close',
        timeframe='1h',
        sequence_length=5,
        prediction_horizon=1,
        d_token=32,
        n_layers=2,
        n_heads=2,
        verbose=True
    )

    print("\n🔧 Training model...")
    model1.fit(train_df, val_df=val_df, epochs=3, batch_size=32, verbose=True)

    print("\n📊 Making predictions...")
    predictions = model1.predict(test_df)

    print(f"\n✅ Test 1 PASSED!")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Predictions type: {type(predictions)}")
    print(f"   Sample predictions: {predictions[:5]}")

except Exception as e:
    print(f"\n❌ Test 1 FAILED with error:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Single-target, multi-horizon (backward compatibility)
print("\n" + "=" * 80)
print("Test 2: Single-Target, Multi-Horizon (Backward Compatibility)")
print("=" * 80)

try:
    model2 = IntradayPredictor(
        target_column='close',
        timeframe='1h',
        sequence_length=5,
        prediction_horizon=3,
        d_token=32,
        n_layers=2,
        n_heads=2,
        verbose=True
    )

    print("\n🔧 Training model...")
    model2.fit(train_df, val_df=val_df, epochs=3, batch_size=32, verbose=True)

    print("\n📊 Making predictions...")
    predictions = model2.predict(test_df)

    print(f"\n✅ Test 2 PASSED!")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Predictions type: {type(predictions)}")
    print(f"   Sample predictions (first sample, all horizons): {predictions[0]}")

except Exception as e:
    print(f"\n❌ Test 2 FAILED with error:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Multi-target, single-horizon (NEW)
print("\n" + "=" * 80)
print("Test 3: Multi-Target, Single-Horizon (NEW FEATURE)")
print("=" * 80)

try:
    model3 = IntradayPredictor(
        target_column=['close', 'volume'],
        timeframe='1h',
        sequence_length=5,
        prediction_horizon=1,
        d_token=32,
        n_layers=2,
        n_heads=2,
        verbose=True
    )

    print("\n🔧 Training model...")
    model3.fit(train_df, val_df=val_df, epochs=3, batch_size=32, verbose=True)

    print("\n📊 Making predictions...")
    predictions = model3.predict(test_df)

    print(f"\n✅ Test 3 PASSED!")
    print(f"   Predictions type: {type(predictions)}")

    if isinstance(predictions, dict):
        print(f"   Predictions keys: {list(predictions.keys())}")
        for key, val in predictions.items():
            print(f"   - {key}: shape={val.shape}, sample={val[:3]}")
    else:
        print(f"   WARNING: Expected dict, got {type(predictions)}")
        print(f"   Predictions shape: {predictions.shape}")

except Exception as e:
    print(f"\n❌ Test 3 FAILED with error:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Multi-target, multi-horizon (NEW)
print("\n" + "=" * 80)
print("Test 4: Multi-Target, Multi-Horizon (NEW FEATURE)")
print("=" * 80)

try:
    model4 = IntradayPredictor(
        target_column=['close', 'volume'],
        timeframe='1h',
        sequence_length=5,
        prediction_horizon=3,
        d_token=32,
        n_layers=2,
        n_heads=2,
        verbose=True
    )

    print("\n🔧 Training model...")
    model4.fit(train_df, val_df=val_df, epochs=3, batch_size=32, verbose=True)

    print("\n📊 Making predictions...")
    predictions = model4.predict(test_df)

    print(f"\n✅ Test 4 PASSED!")
    print(f"   Predictions type: {type(predictions)}")

    if isinstance(predictions, dict):
        print(f"   Predictions keys: {list(predictions.keys())}")
        for key, val in predictions.items():
            print(f"   - {key}: shape={val.shape}")
            if len(val.shape) == 2:
                print(f"     First sample, all horizons: {val[0]}")
    else:
        print(f"   WARNING: Expected dict, got {type(predictions)}")
        print(f"   Predictions shape: {predictions.shape}")

except Exception as e:
    print(f"\n❌ Test 4 FAILED with error:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 5: predict_next_bars with multi-target
print("\n" + "=" * 80)
print("Test 5: predict_next_bars() with Multi-Target (NEW FEATURE)")
print("=" * 80)

try:
    # Reuse model3 from test 3 (multi-target, single-horizon)
    print("\n📊 Testing predict_next_bars()...")
    next_bars = model3.predict_next_bars(test_df, n_predictions=5)

    print(f"\n✅ Test 5 PASSED!")
    print(f"   Result type: {type(next_bars)}")
    print(f"   Result shape: {next_bars.shape}")
    print(f"   Columns: {list(next_bars.columns)}")
    print(f"\n   First 3 predictions:")
    print(next_bars.head(3))

except Exception as e:
    print(f"\n❌ Test 5 FAILED with error:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)
print("All tests completed. Check results above for details.")
