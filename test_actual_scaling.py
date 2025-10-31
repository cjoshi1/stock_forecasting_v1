"""
Test the actual scaling behavior with real data to diagnose the XRP issue.
"""
import pandas as pd
import sys
sys.path.insert(0, '/home/ubuntu/code/stock_forecasting_v1')

from daily_stock_forecasting.predictor import StockPredictor

# Load data
data_path = "/home/ubuntu/code/get_stock_data/data/daily_data_BTC_USD_ETH_USD_XRP_USD_1d_1y_20251030_091134.csv"
df = pd.read_csv(data_path)
df.columns = df.columns.str.lower()

# Convert datetime
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.rename(columns={'datetime': 'date'})
df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

# Take a small subset for testing
train_df = df.iloc[:50].copy()  # Just 50 rows for quick test

print(f"Data shape: {train_df.shape}")
print(f"Symbols: {train_df['symbol'].unique()}")
print(f"\nSymbol value ranges:")
for sym in train_df['symbol'].unique():
    sym_df = train_df[train_df['symbol'] == sym]
    print(f"{sym}:")
    print(f"  close: {sym_df['close'].min():.2f} to {sym_df['close'].max():.2f}")
    print(f"  samples: {len(sym_df)}")

# Initialize predictor
predictor = StockPredictor(
    target_column=['close', 'volume'],
    sequence_length=5,
    prediction_horizon=2,
    group_columns='symbol',
    categorical_columns='symbol',
    scaler_type='standard',
    asset_type='stock',
    model_type='ft_transformer_cls',
    d_model=32,
    num_layers=1,
    num_heads=2,
    dropout=0.1
)

print("\n" + "="*60)
print("Training on small dataset")
print("="*60)

# Prepare data to inspect scalers
X_train, y_train = predictor.prepare_data(train_df, fit_scaler=True)

print(f"\nNumber of group feature scalers: {len(predictor.group_feature_scalers)}")
print(f"Group feature scaler keys: {list(predictor.group_feature_scalers.keys())}")

print(f"\nNumber of group target scalers: {len(predictor.group_target_scalers)}")
print(f"Group target scaler keys: {list(predictor.group_target_scalers.keys())}")

# Inspect the target scalers
for group_key, scaler_dict in predictor.group_target_scalers.items():
    print(f"\nGroup '{group_key}':")
    for target_name, scaler in scaler_dict.items():
        print(f"  {target_name}: mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}")

# Check if group indices match
print(f"\nLast group indices (from prepare_data): {predictor._last_group_indices[:10] if hasattr(predictor, '_last_group_indices') else 'Not set'}")

# Check the dataframe symbols
print(f"\nOriginal symbol order in train_df:")
print(train_df.groupby('symbol').size())
