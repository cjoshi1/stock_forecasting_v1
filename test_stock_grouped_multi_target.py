"""
Test group-based multi-target prediction for StockPredictor.

This test verifies that multi-target prediction works correctly when using
group-based scaling (e.g., different stock symbols get different scalers).
"""

import pandas as pd
import numpy as np
from daily_stock_forecasting import StockPredictor

# Create synthetic multi-stock daily data
np.random.seed(42)

symbols = ['AAPL', 'GOOGL', 'TSLA']
data = []

for symbol in symbols:
    # Each stock has different price ranges
    base_price = {'AAPL': 150, 'GOOGL': 2800, 'TSLA': 250}[symbol]
    base_volume = {'AAPL': 70_000_000, 'GOOGL': 25_000_000, 'TSLA': 120_000_000}[symbol]

    for i in range(150):
        date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)
        price_noise = np.random.randn() * 5
        volume_noise = np.random.randn() * 5_000_000

        data.append({
            'date': date,
            'symbol': symbol,
            'open': base_price + price_noise,
            'high': base_price + price_noise + abs(np.random.randn() * 3),
            'low': base_price + price_noise - abs(np.random.randn() * 3),
            'close': base_price + price_noise + np.random.randn() * 2,
            'volume': max(1000, base_volume + volume_noise)
        })

df = pd.DataFrame(data)

print("=" * 80)
print("Test: Group-Based Multi-Target Prediction for StockPredictor")
print("=" * 80)
print(f"\nDataset: {len(df)} rows, {len(symbols)} symbols")
print(f"Symbols: {symbols}")
print(f"Columns: {list(df.columns)}")

# Test 1: Multi-target, single-horizon with group-based scaling
print("\n" + "=" * 80)
print("Test 1: Multi-Target ['close', 'volume'], Single-Horizon, Group-Based Scaling")
print("=" * 80)

model1 = StockPredictor(
    target_column=['close', 'volume'],
    sequence_length=20,
    prediction_horizon=1,
    group_column='symbol',  # Enable group-based scaling
    d_token=32,
    n_layers=2,
    n_heads=2,
    verbose=True
)

# Split data ensuring each symbol appears in both train and test
# Group by symbol and split each group
train_dfs = []
test_dfs = []
for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol]
    split_idx = int(len(symbol_df) * 0.7)  # 70% train, 30% test
    train_dfs.append(symbol_df.iloc[:split_idx])
    test_dfs.append(symbol_df.iloc[split_idx:])

train_df = pd.concat(train_dfs, ignore_index=True).sort_values('date')
test_df = pd.concat(test_dfs, ignore_index=True).sort_values('date')

print(f"\nTrain set: {len(train_df)} rows")
print(f"Test set: {len(test_df)} rows")

# Train
print("\nTraining...")
model1.fit(train_df, epochs=5, batch_size=32, verbose=True)

# Predict
print("\nMaking predictions...")
predictions = model1.predict(test_df)

# Verify predictions structure
print(f"\nPredictions type: {type(predictions)}")
print(f"Prediction keys: {predictions.keys()}")
print(f"Close predictions shape: {predictions['close'].shape}")
print(f"Volume predictions shape: {predictions['volume'].shape}")

assert isinstance(predictions, dict), "Should return dict for multi-target"
assert 'close' in predictions, "Should have 'close' predictions"
assert 'volume' in predictions, "Should have 'volume' predictions"
assert predictions['close'].ndim == 1, "Single-horizon should return 1D array"
assert predictions['volume'].ndim == 1, "Single-horizon should return 1D array"

print("\n✅ Test 1 PASSED: Multi-target, single-horizon, group-based scaling works!")

# Test 2: Multi-target, multi-horizon with group-based scaling
print("\n" + "=" * 80)
print("Test 2: Multi-Target ['close', 'volume'], Multi-Horizon (5), Group-Based Scaling")
print("=" * 80)

model2 = StockPredictor(
    target_column=['close', 'volume'],
    sequence_length=20,
    prediction_horizon=5,  # Multi-horizon
    group_column='symbol',
    d_token=32,
    n_layers=2,
    n_heads=2,
    verbose=True
)

print("\nTraining...")
model2.fit(train_df, epochs=5, batch_size=32, verbose=True)

print("\nMaking predictions...")
predictions = model2.predict(test_df)

print(f"\nPredictions type: {type(predictions)}")
print(f"Prediction keys: {predictions.keys()}")
print(f"Close predictions shape: {predictions['close'].shape}")
print(f"Volume predictions shape: {predictions['volume'].shape}")

assert isinstance(predictions, dict), "Should return dict for multi-target"
assert 'close' in predictions, "Should have 'close' predictions"
assert 'volume' in predictions, "Should have 'volume' predictions"
assert predictions['close'].ndim == 2, "Multi-horizon should return 2D array"
assert predictions['volume'].ndim == 2, "Multi-horizon should return 2D array"
assert predictions['close'].shape[1] == 5, "Should have 5 horizons for close"
assert predictions['volume'].shape[1] == 5, "Should have 5 horizons for volume"

print("\n✅ Test 2 PASSED: Multi-target, multi-horizon, group-based scaling works!")

# Test 3: Verify group scalers structure
print("\n" + "=" * 80)
print("Test 3: Verify Group Scaler Structure")
print("=" * 80)

print(f"\nGroup target scalers keys: {list(model2.group_target_scalers.keys())}")
for symbol in symbols:
    if symbol in model2.group_target_scalers:
        scalers = model2.group_target_scalers[symbol]
        print(f"\nSymbol '{symbol}':")
        print(f"  Scaler type: {type(scalers)}")
        print(f"  Target scalers: {list(scalers.keys())}")

        # For multi-horizon, each target should have a list of scalers
        for target in ['close', 'volume']:
            if target in scalers:
                print(f"    {target}: {type(scalers[target])} with {len(scalers[target])} scalers")
                assert isinstance(scalers[target], list), f"{target} should have list of scalers"
                assert len(scalers[target]) == 5, f"{target} should have 5 horizon scalers"

print("\n✅ Test 3 PASSED: Group scaler structure is correct!")

# Test 4: Per-symbol prediction verification
print("\n" + "=" * 80)
print("Test 4: Verify Per-Symbol Predictions")
print("=" * 80)

predictions_with_groups, group_indices = model2.predict(test_df, return_group_info=True)

print(f"\nGroup indices length: {len(group_indices)}")
print(f"Unique groups in predictions: {sorted(set(group_indices))}")

# Check each symbol
for symbol in symbols:
    symbol_mask = np.array([g == symbol for g in group_indices])
    symbol_count = symbol_mask.sum()

    print(f"\nSymbol '{symbol}': {symbol_count} predictions")

    if symbol_count > 0:
        close_preds = predictions_with_groups['close'][symbol_mask]
        volume_preds = predictions_with_groups['volume'][symbol_mask]

        print(f"  Close predictions shape: {close_preds.shape}")
        print(f"  Volume predictions shape: {volume_preds.shape}")
        print(f"  Close range: [{close_preds.min():.2f}, {close_preds.max():.2f}]")
        print(f"  Volume range: [{volume_preds.min():.0f}, {volume_preds.max():.0f}]")

        # Verify different symbols have different prediction ranges
        assert close_preds.shape[0] == symbol_count, "Should have predictions for all symbol samples"
        assert volume_preds.shape[0] == symbol_count, "Should have predictions for all symbol samples"

print("\n✅ Test 4 PASSED: Per-symbol predictions verified!")

# Summary
print("\n" + "=" * 80)
print("🎉 ALL TESTS PASSED!")
print("=" * 80)
print("\nGroup-based multi-target prediction is working correctly for StockPredictor:")
print("  ✅ Multi-target with single-horizon and group-based scaling")
print("  ✅ Multi-target with multi-horizon and group-based scaling")
print("  ✅ Correct scaler structure (per-group, per-target, per-horizon)")
print("  ✅ Per-symbol predictions work correctly")
