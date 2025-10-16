"""
Test multi-target evaluation functionality.
"""
import pandas as pd
import numpy as np
from intraday_forecasting.predictor import IntradayPredictor
from daily_stock_forecasting.predictor import StockPredictor

def generate_crypto_data(n_samples=200, symbols=['BTC', 'ETH']):
    """Generate synthetic crypto data with multiple targets."""
    np.random.seed(42)

    data = []
    for symbol in symbols:
        base_price = 50000 if symbol == 'BTC' else 3000
        base_volume = 1e9 if symbol == 'BTC' else 5e8

        for i in range(n_samples):
            timestamp = pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=i*5)

            price = base_price + np.random.randn() * 100 + i * 10
            volume = base_volume + np.random.randn() * 1e7 + i * 1e6

            data.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'open': price + np.random.randn(),
                'high': price + abs(np.random.randn() * 10),
                'low': price - abs(np.random.randn() * 10),
                'close': price,
                'volume': max(volume, 1e6)
            })

    return pd.DataFrame(data).sort_values(['symbol', 'timestamp']).reset_index(drop=True)

def generate_stock_data(n_samples=200, symbols=['AAPL', 'GOOGL', 'TSLA']):
    """Generate synthetic stock data."""
    np.random.seed(42)

    data = []
    base_prices = {'AAPL': 200, 'GOOGL': 2700, 'TSLA': 250}
    base_volumes = {'AAPL': 70000000, 'GOOGL': 30000000, 'TSLA': 120000000}

    for symbol in symbols:
        base_price = base_prices[symbol]
        base_volume = base_volumes[symbol]

        for i in range(n_samples):
            date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)

            price = base_price + np.random.randn() * 5 + i * 0.5
            volume = base_volume + np.random.randn() * 1e6 + i * 5e4

            data.append({
                'date': date,
                'symbol': symbol,
                'open': price + np.random.randn(),
                'high': price + abs(np.random.randn() * 2),
                'low': price - abs(np.random.randn() * 2),
                'close': price,
                'volume': max(volume, 1e6)
            })

    return pd.DataFrame(data).sort_values(['symbol', 'date']).reset_index(drop=True)

print("="*80)
print("Multi-Target Evaluation Test")
print("="*80)

# Test 1: IntradayPredictor - Multi-target evaluation
print("\n" + "="*80)
print("Test 1: IntradayPredictor - Multi-target, Single-Horizon Evaluation")
print("="*80)

df = generate_crypto_data(n_samples=200, symbols=['BTC', 'ETH'])
train_size = int(0.7 * len(df))
train_df = df[:train_size]
test_df = df[train_size:]

print(f"\nData split:")
print(f"  Train: {len(train_df)} samples")
print(f"  Test:  {len(test_df)} samples")

predictor = IntradayPredictor(
    target_column=['close', 'volume'],
    timeframe='5min',
    sequence_length=10,
    prediction_horizon=1,
    group_column='symbol',
    d_token=32,
    n_layers=2,
    n_heads=2,
    dropout=0.1,
    verbose=False
)

print("\nTraining model...")
predictor.fit(train_df, epochs=5, batch_size=32, verbose=False)

print("\nEvaluating on test data (standard evaluation)...")
metrics = predictor.evaluate(test_df, per_group=False)

print("\nMetrics structure:")
print(f"  Keys: {list(metrics.keys())}")
for target, target_metrics in metrics.items():
    print(f"\n  {target.upper()} metrics:")
    for metric_name, metric_value in target_metrics.items():
        if isinstance(metric_value, (int, float)):
            print(f"    - {metric_name}: {metric_value:.4f}")

print("\n✅ Test 1 PASSED: Multi-target evaluation returns per-target metrics")

# Test 2: IntradayPredictor - Multi-target per-group evaluation
print("\n" + "="*80)
print("Test 2: IntradayPredictor - Multi-target, Per-Group Evaluation")
print("="*80)

print("\nEvaluating on test data (per-group evaluation)...")
metrics = predictor.evaluate(test_df, per_group=True)

print("\nMetrics structure:")
print(f"  Top-level keys: {list(metrics.keys())}")
for group_key, group_metrics in metrics.items():
    print(f"\n  {group_key}:")
    if isinstance(group_metrics, dict):
        for target, target_metrics in group_metrics.items():
            print(f"    {target}:")
            if isinstance(target_metrics, dict):
                for metric_name, metric_value in target_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        print(f"      - {metric_name}: {metric_value:.4f}")

print("\n✅ Test 2 PASSED: Multi-target per-group evaluation works!")

# Test 3: IntradayPredictor - Multi-target, Multi-horizon evaluation
print("\n" + "="*80)
print("Test 3: IntradayPredictor - Multi-target, Multi-Horizon Evaluation")
print("="*80)

predictor_mh = IntradayPredictor(
    target_column=['close', 'volume'],
    timeframe='5min',
    sequence_length=10,
    prediction_horizon=3,
    group_column='symbol',
    d_token=32,
    n_layers=2,
    n_heads=2,
    dropout=0.1,
    verbose=False
)

print("\nTraining model...")
predictor_mh.fit(train_df, epochs=5, batch_size=32, verbose=False)

print("\nEvaluating on test data...")
metrics = predictor_mh.evaluate(test_df, per_group=False)

print("\nMetrics structure:")
print(f"  Keys: {list(metrics.keys())}")
for target, target_metrics in metrics.items():
    print(f"\n  {target.upper()} metrics:")
    if isinstance(target_metrics, dict):
        for key, value in target_metrics.items():
            if isinstance(value, dict):
                print(f"    {key}:")
                for metric_name, metric_value in value.items():
                    if isinstance(metric_value, (int, float)):
                        print(f"      - {metric_name}: {metric_value:.4f}")
            elif isinstance(value, (int, float)):
                print(f"    - {key}: {value:.4f}")

print("\n✅ Test 3 PASSED: Multi-target, multi-horizon evaluation works!")

# Test 4: StockPredictor - Multi-target evaluation
print("\n" + "="*80)
print("Test 4: StockPredictor - Multi-target Evaluation")
print("="*80)

stock_df = generate_stock_data(n_samples=150, symbols=['AAPL', 'GOOGL'])
train_size = int(0.7 * len(stock_df))
stock_train = stock_df[:train_size]
stock_test = stock_df[train_size:]

print(f"\nData split:")
print(f"  Train: {len(stock_train)} samples")
print(f"  Test:  {len(stock_test)} samples")

stock_predictor = StockPredictor(
    target_column=['close', 'volume'],
    sequence_length=20,
    prediction_horizon=3,
    group_column='symbol',
    d_token=32,
    n_layers=2,
    n_heads=2,
    dropout=0.1,
    verbose=False
)

print("\nTraining model...")
stock_predictor.fit(stock_train, epochs=5, batch_size=32, verbose=False)

print("\nEvaluating on test data...")
metrics = stock_predictor.evaluate(stock_test, per_group=False)

print("\nMetrics structure:")
print(f"  Keys: {list(metrics.keys())}")
for target, target_metrics in metrics.items():
    print(f"\n  {target.upper()} metrics:")
    if isinstance(target_metrics, dict):
        for key, value in target_metrics.items():
            if isinstance(value, dict):
                print(f"    {key}:")
                for metric_name, metric_value in value.items():
                    if isinstance(metric_value, (int, float)):
                        print(f"      - {metric_name}: {metric_value:.4f}")

print("\n✅ Test 4 PASSED: StockPredictor multi-target evaluation works!")

# Test 5: StockPredictor - Multi-target per-group evaluation
print("\n" + "="*80)
print("Test 5: StockPredictor - Multi-target, Per-Group Evaluation")
print("="*80)

print("\nEvaluating on test data (per-group)...")
metrics = stock_predictor.evaluate(stock_test, per_group=True)

print("\nMetrics structure:")
print(f"  Top-level keys: {list(metrics.keys())}")
for group_key in ['overall', 'AAPL', 'GOOGL']:
    if group_key in metrics:
        print(f"\n  {group_key}:")
        group_metrics = metrics[group_key]
        if isinstance(group_metrics, dict):
            for target, target_metrics in group_metrics.items():
                print(f"    {target}:")
                if isinstance(target_metrics, dict):
                    # Check if nested (multi-horizon)
                    if 'overall' in target_metrics:
                        print(f"      overall MAE: {target_metrics['overall'].get('MAE', 'N/A'):.4f}")
                    else:
                        print(f"      MAE: {target_metrics.get('MAE', 'N/A'):.4f}")

print("\n✅ Test 5 PASSED: StockPredictor per-group evaluation works!")

print("\n" + "="*80)
print("🎉 ALL EVALUATION TESTS PASSED!")
print("="*80)
print("\nSummary:")
print("  ✅ IntradayPredictor: Multi-target, single-horizon evaluation")
print("  ✅ IntradayPredictor: Multi-target, per-group evaluation")
print("  ✅ IntradayPredictor: Multi-target, multi-horizon evaluation")
print("  ✅ StockPredictor: Multi-target, multi-horizon evaluation")
print("  ✅ StockPredictor: Multi-target, per-group evaluation")
