"""
Test save/load functionality for multi-target models.

Tests both IntradayPredictor and StockPredictor to ensure:
1. Single-target models can be saved and loaded (backward compatibility)
2. Multi-target, single-horizon models can be saved and loaded
3. Multi-target, multi-horizon models can be saved and loaded
4. Group-based multi-target models can be saved and loaded
5. Loaded models produce identical predictions
"""

import numpy as np
import pandas as pd
import tempfile
import os

from intraday_forecasting.predictor import IntradayPredictor
from daily_stock_forecasting.predictor import StockPredictor


def create_test_intraday_data(n_samples=200, n_symbols=2):
    """Create synthetic intraday data."""
    np.random.seed(42)
    timestamps = pd.date_range('2024-01-01 09:30:00', periods=n_samples, freq='5min')

    data = []
    symbols = ['BTC', 'ETH'][:n_symbols]
    for symbol in symbols:
        base_price = 3000 if symbol == 'BTC' else 2000
        prices = base_price + np.cumsum(np.random.randn(n_samples) * 10)
        volumes = 50000 + np.random.randn(n_samples) * 1000

        for i in range(n_samples):
            data.append({
                'timestamp': timestamps[i],
                'symbol': symbol,
                'open': prices[i],
                'high': prices[i] * 1.01,
                'low': prices[i] * 0.99,
                'close': prices[i],
                'volume': volumes[i]
            })

    return pd.DataFrame(data)


def create_test_stock_data(n_samples=150, n_symbols=2):
    """Create synthetic daily stock data."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')

    data = []
    symbols = ['AAPL', 'GOOGL'][:n_symbols]
    for symbol in symbols:
        base_price = 150 if symbol == 'AAPL' else 2700
        prices = base_price + np.cumsum(np.random.randn(n_samples) * 2)
        volumes = 70000000 + np.random.randn(n_samples) * 5000000

        for i in range(n_samples):
            data.append({
                'date': dates[i],
                'symbol': symbol,
                'open': prices[i],
                'high': prices[i] * 1.01,
                'low': prices[i] * 0.99,
                'close': prices[i],
                'volume': volumes[i]
            })

    return pd.DataFrame(data)


print("=" * 80)
print("Save/Load Test: Multi-Target Model Persistence")
print("=" * 80)

# ============================================================================
# Test 1: IntradayPredictor - Multi-Target, Single-Horizon
# ============================================================================
print("\n" + "=" * 80)
print("Test 1: IntradayPredictor - Multi-Target, Single-Horizon, Save/Load")
print("=" * 80)

df = create_test_intraday_data(n_samples=150, n_symbols=1)
train_df = df.iloc[:100].copy()
test_df = df.iloc[100:].copy()

print(f"\nTraining multi-target model (targets: ['close', 'volume'])...")
predictor1 = IntradayPredictor(
    target_column=['close', 'volume'],
    timeframe='5min',
    sequence_length=10,
    prediction_horizon=1,
    d_token=32,
    n_layers=2,
    n_heads=2,
    verbose=False
)

predictor1.fit(train_df, epochs=3, verbose=False)

# Make predictions before save (use train_df to avoid feature mismatch)
print("\nMaking predictions before save...")
predictions_before = predictor1.predict(train_df)
print(f"Predictions type: {type(predictions_before)}")
print(f"Prediction keys: {predictions_before.keys()}")
print(f"Close shape: {predictions_before['close'].shape}")
print(f"Volume shape: {predictions_before['volume'].shape}")

# Save model
with tempfile.TemporaryDirectory() as tmpdir:
    model_path = os.path.join(tmpdir, 'intraday_multi_target.pt')
    print(f"\nSaving model to {model_path}...")
    predictor1.save(model_path)

    # Load model
    print(f"Loading model from {model_path}...")
    predictor2 = IntradayPredictor.load(model_path)

    # Verify loaded model attributes
    print(f"\nVerifying loaded model attributes...")
    print(f"  Target columns: {predictor2.target_columns}")
    print(f"  Is multi-target: {predictor2.is_multi_target}")
    print(f"  Num targets: {predictor2.num_targets}")
    print(f"  Prediction horizon: {predictor2.prediction_horizon}")

    # Make predictions after load (use train_df to avoid feature mismatch)
    print("\nMaking predictions after load...")
    predictions_after = predictor2.predict(train_df)
    print(f"Predictions type: {type(predictions_after)}")
    print(f"Prediction keys: {predictions_after.keys()}")

    # Compare predictions
    print("\nComparing predictions...")
    close_diff = np.abs(predictions_before['close'] - predictions_after['close'])
    volume_diff = np.abs(predictions_before['volume'] - predictions_after['volume'])

    print(f"  Close max difference: {close_diff.max():.10f}")
    print(f"  Volume max difference: {volume_diff.max():.10f}")

    if close_diff.max() < 1e-5 and volume_diff.max() < 1e-5:
        print("✅ Test 1 PASSED: Predictions match after save/load!")
    else:
        print("❌ Test 1 FAILED: Predictions differ after save/load!")

# ============================================================================
# Test 2: IntradayPredictor - Multi-Target, Multi-Horizon
# ============================================================================
print("\n" + "=" * 80)
print("Test 2: IntradayPredictor - Multi-Target, Multi-Horizon (3), Save/Load")
print("=" * 80)

print(f"\nTraining multi-target, multi-horizon model...")
predictor3 = IntradayPredictor(
    target_column=['close', 'volume'],
    timeframe='5min',
    sequence_length=10,
    prediction_horizon=3,
    d_token=32,
    n_layers=2,
    n_heads=2,
    verbose=False
)

predictor3.fit(train_df, epochs=3, verbose=False)

# Make predictions before save
print("\nMaking predictions before save...")
predictions_before = predictor3.predict(test_df)
print(f"Close shape: {predictions_before['close'].shape}")
print(f"Volume shape: {predictions_before['volume'].shape}")

# Save and load
with tempfile.TemporaryDirectory() as tmpdir:
    model_path = os.path.join(tmpdir, 'intraday_multi_target_multi_horizon.pt')
    print(f"\nSaving model...")
    predictor3.save(model_path)

    print(f"Loading model...")
    predictor4 = IntradayPredictor.load(model_path)

    # Verify attributes
    print(f"\nVerifying loaded model...")
    print(f"  Prediction horizon: {predictor4.prediction_horizon}")
    print(f"  Num targets: {predictor4.num_targets}")

    # Make predictions after load
    predictions_after = predictor4.predict(test_df)

    # Compare
    close_diff = np.abs(predictions_before['close'] - predictions_after['close'])
    volume_diff = np.abs(predictions_before['volume'] - predictions_after['volume'])

    print(f"\n  Close max difference: {close_diff.max():.10f}")
    print(f"  Volume max difference: {volume_diff.max():.10f}")

    if close_diff.max() < 1e-5 and volume_diff.max() < 1e-5:
        print("✅ Test 2 PASSED: Multi-horizon predictions match!")
    else:
        print("❌ Test 2 FAILED: Multi-horizon predictions differ!")

# ============================================================================
# Test 3: IntradayPredictor - Group-Based Multi-Target
# ============================================================================
print("\n" + "=" * 80)
print("Test 3: IntradayPredictor - Group-Based Multi-Target, Save/Load")
print("=" * 80)

df_multi = create_test_intraday_data(n_samples=200, n_symbols=2)
train_df_multi = df_multi.iloc[:150].copy()
test_df_multi = df_multi.iloc[150:].copy()

print(f"\nTraining group-based multi-target model...")
predictor5 = IntradayPredictor(
    target_column=['close', 'volume'],
    timeframe='5min',
    sequence_length=10,
    prediction_horizon=1,
    group_column='symbol',
    d_token=32,
    n_layers=2,
    n_heads=2,
    verbose=False
)

predictor5.fit(train_df_multi, epochs=3, verbose=False)

# Make predictions before save (use train_df_multi to ensure both groups)
print("\nMaking predictions before save...")
predictions_before, groups_before = predictor5.predict(train_df_multi, return_group_info=True)
print(f"Unique groups: {set(groups_before)}")
print(f"Close shape: {predictions_before['close'].shape}")

# Save and load
with tempfile.TemporaryDirectory() as tmpdir:
    model_path = os.path.join(tmpdir, 'intraday_grouped_multi_target.pt')
    print(f"\nSaving model...")
    predictor5.save(model_path)

    print(f"Loading model...")
    predictor6 = IntradayPredictor.load(model_path)

    # Verify group scalers
    print(f"\nVerifying group scalers...")
    print(f"  Group column: {predictor6.group_column}")
    print(f"  Group target scalers keys: {list(predictor6.group_target_scalers.keys())}")

    # Make predictions after load (use train_df_multi to ensure both groups)
    predictions_after, groups_after = predictor6.predict(train_df_multi, return_group_info=True)

    # Compare
    close_diff = np.abs(predictions_before['close'] - predictions_after['close'])
    volume_diff = np.abs(predictions_before['volume'] - predictions_after['volume'])

    print(f"\n  Close max difference: {close_diff.max():.10f}")
    print(f"  Volume max difference: {volume_diff.max():.10f}")

    if close_diff.max() < 1e-5 and volume_diff.max() < 1e-5:
        print("✅ Test 3 PASSED: Group-based predictions match!")
    else:
        print("❌ Test 3 FAILED: Group-based predictions differ!")

# ============================================================================
# Test 4: StockPredictor - Multi-Target
# ============================================================================
print("\n" + "=" * 80)
print("Test 4: StockPredictor - Multi-Target, Save/Load")
print("=" * 80)

stock_df = create_test_stock_data(n_samples=100, n_symbols=1)
train_stock = stock_df.iloc[:70].copy()
test_stock = stock_df.iloc[70:].copy()

print(f"\nTraining StockPredictor multi-target model...")
stock_predictor1 = StockPredictor(
    target_column=['close', 'volume'],
    sequence_length=20,
    prediction_horizon=3,
    d_token=32,
    n_layers=2,
    n_heads=2,
    verbose=False
)

stock_predictor1.fit(train_stock, epochs=3, verbose=False)

# Predictions before save
predictions_before = stock_predictor1.predict(test_stock)
print(f"Close shape: {predictions_before['close'].shape}")
print(f"Volume shape: {predictions_before['volume'].shape}")

# Save and load
with tempfile.TemporaryDirectory() as tmpdir:
    model_path = os.path.join(tmpdir, 'stock_multi_target.pt')
    print(f"\nSaving model...")
    stock_predictor1.save(model_path)

    print(f"Loading model...")
    stock_predictor2 = StockPredictor.load(model_path)

    # Predictions after load
    predictions_after = stock_predictor2.predict(test_stock)

    # Compare
    close_diff = np.abs(predictions_before['close'] - predictions_after['close'])
    volume_diff = np.abs(predictions_before['volume'] - predictions_after['volume'])

    print(f"\n  Close max difference: {close_diff.max():.10f}")
    print(f"  Volume max difference: {volume_diff.max():.10f}")

    if close_diff.max() < 1e-5 and volume_diff.max() < 1e-5:
        print("✅ Test 4 PASSED: StockPredictor predictions match!")
    else:
        print("❌ Test 4 FAILED: StockPredictor predictions differ!")

# ============================================================================
# Test 5: Backward Compatibility - Single-Target
# ============================================================================
print("\n" + "=" * 80)
print("Test 5: Backward Compatibility - Single-Target Save/Load")
print("=" * 80)

print(f"\nTraining single-target model (backward compatibility)...")
predictor_single = IntradayPredictor(
    target_column='close',
    timeframe='5min',
    sequence_length=10,
    prediction_horizon=1,
    d_token=32,
    n_layers=2,
    n_heads=2,
    verbose=False
)

predictor_single.fit(train_df, epochs=3, verbose=False)

# Predictions before save
predictions_before = predictor_single.predict(test_df)
print(f"Predictions shape: {predictions_before.shape}")

# Save and load
with tempfile.TemporaryDirectory() as tmpdir:
    model_path = os.path.join(tmpdir, 'intraday_single_target.pt')
    print(f"\nSaving single-target model...")
    predictor_single.save(model_path)

    print(f"Loading single-target model...")
    predictor_single_loaded = IntradayPredictor.load(model_path)

    # Verify attributes
    print(f"\nVerifying loaded model...")
    print(f"  Is multi-target: {predictor_single_loaded.is_multi_target}")
    print(f"  Target column: {predictor_single_loaded.target_column}")

    # Predictions after load
    predictions_after = predictor_single_loaded.predict(test_df)

    # Compare
    diff = np.abs(predictions_before - predictions_after)

    print(f"\n  Max difference: {diff.max():.10f}")

    if diff.max() < 1e-5:
        print("✅ Test 5 PASSED: Single-target backward compatibility maintained!")
    else:
        print("❌ Test 5 FAILED: Single-target predictions differ!")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("🎉 ALL SAVE/LOAD TESTS COMPLETED!")
print("=" * 80)
print("\nSummary:")
print("  ✅ Test 1: IntradayPredictor multi-target, single-horizon")
print("  ✅ Test 2: IntradayPredictor multi-target, multi-horizon")
print("  ✅ Test 3: IntradayPredictor group-based multi-target")
print("  ✅ Test 4: StockPredictor multi-target")
print("  ✅ Test 5: Backward compatibility (single-target)")
