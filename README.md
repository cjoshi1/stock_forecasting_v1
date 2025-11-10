# 🚀 TF Predictor: Modular Time Series Forecasting Framework

**Version:** 2.1.0 (Updated: 2025-11-07)

A comprehensive, modular framework for time series forecasting using the TF Predictor (Feature Tokenizer Transformer) architecture. This project provides a generic reusable library and specialized applications for daily and intraday stock market forecasting.

## 🎉 Version 2.1.0 Highlights

- **🎯 Configurable Pooling Strategies** ⭐ NEW: Choose from 5 pooling methods for sequence aggregation
- **🚀 Multi-Head Attention Pooling**: Default pooling strategy for better feature aggregation
- **Per-Horizon Scaling**: Each prediction horizon gets its own scaler for improved accuracy
- **Automatic Cyclical Encoding**: Temporal features automatically encoded as sin/cos
- **Evaluation Bug Fixed**: Resolved 100% MAPE bug through proper dataframe alignment
- **Cleaner Architecture**: 7-stage preprocessing pipeline with clear separation of concerns
- **Simplified API**: `_create_base_features()` replaces `prepare_features()` for better extensibility
- **Better Metrics**: Per-horizon evaluation breakdown (MAPE, RMSE, R2 for each horizon)

## 🚀 Quick Start

### For Intraday Market Prediction
```bash
# Train with sample 5-minute data for the US market
python run_intraday_forecasting.py --use_sample_data --timeframe 5min --country US --epochs 30

# Train with sample 1-minute data for the Indian market
python run_intraday_forecasting.py --use_sample_data --timeframe 1min --country INDIA --epochs 50

# Train with BTC cryptocurrency data (24/7 trading)
python run_intraday_forecasting.py --csv_path /path/to/BTC-USD_1m_max.csv --timeframe 1min --country CRYPTO --epochs 20
```

### For Cryptocurrency Prediction
```bash
# Using the simplified BTC runner (auto-detects BTC data)
python run_btc_forecasting.py --timeframe 1min --epochs 20

# 15-minute BTC forecasting with predictions saved
python run_btc_forecasting.py --timeframe 15min --epochs 50 --save_predictions

# 1-hour BTC forecasting with verbose output
python run_btc_forecasting.py --timeframe 1h --epochs 100 --verbose

# Custom BTC data file
python run_btc_forecasting.py --csv_path /path/to/your/BTC-USD_data.csv --timeframe 5min --epochs 30
```

### For Daily Stock Market Prediction
```bash
# Navigate to stock forecasting
cd daily_stock_forecasting/

# Train with sample data - single step prediction (default)
python main.py --use_sample_data --target close --asset_type stock --epochs 50

# Multi-horizon prediction - predict 3 steps ahead simultaneously
python main.py --use_sample_data --target close --asset_type stock --prediction_horizon 3 --epochs 50

# Train with your data
python main.py --data_path data/raw/AAPL.csv --target close --asset_type stock --epochs 100

# Multi-target prediction - predict multiple variables simultaneously
python main.py --use_sample_data --target "close,volume" --asset_type stock --epochs 50
```

### For Custom Time Series (Generic Library - v2.0.0 API)
```python
from tf_predictor.core.predictor import TimeSeriesPredictor

# Create your domain-specific predictor
class MyPredictor(TimeSeriesPredictor):
    def _create_base_features(self, df):
        """
        Add domain-specific features only (v2.0.0).
        Time-series features added automatically by parent.
        """
        df_processed = df.copy()
        # Add your custom features
        df_processed['custom_feature'] = df_processed['value'] * 2
        # Fill NaN
        df_processed = df_processed.fillna(0)
        # Call parent for time-series features (automatic cyclical encoding)
        return super()._create_base_features(df_processed)

predictor = MyPredictor(
    target_column='value',
    sequence_length=10,
    prediction_horizon=1,
    model_type='ft_transformer',       # or 'csn_transformer'
    pooling_type='multihead_attention', # ⭐ NEW in v2.1.0 - Pooling strategy
    d_token=128,                        # Embedding dimension
    n_layers=3,                         # Transformer layers
    n_heads=8                           # Attention heads
)
predictor.fit(train_df, val_df, epochs=100)
```

## 🎯 Pooling Strategies (v2.1.0)

Choose from 5 pooling methods to aggregate transformer sequences:

### Available Pooling Types

1. **`'multihead_attention'`** ⭐ **DEFAULT** - Multi-head attention pooling
   - Best overall performance
   - Learnable attention across sequence
   - ~3×d_token² parameters

2. **`'singlehead_attention'`** - Single-head attention pooling
   - Simpler than multihead
   - Good for smaller models
   - ~3×d_token² parameters

3. **`'temporal_multihead_attention'`** - Temporal multi-head with recency bias
   - Emphasizes recent timesteps
   - Good for time series with strong trends
   - ~3×d_token² + temporal bias parameters

4. **`'weighted_avg'`** - Learnable weighted average
   - Simplest learnable pooling
   - Fast inference
   - max_seq_len parameters

5. **`'cls'`** - CLS token pooling (legacy)
   - Classic transformer approach
   - No additional parameters
   - Use for comparison

### Usage Example

```python
from tf_predictor.core.predictor import TimeSeriesPredictor

# Use default multihead attention pooling
predictor = TimeSeriesPredictor(
    target_column='close',
    model_type='ft_transformer',
    pooling_type='multihead_attention',  # ⭐ DEFAULT
    d_token=128,
    n_heads=8,
    n_layers=3
)

# Try different pooling strategies
for pooling in ['cls', 'singlehead_attention', 'multihead_attention',
                'weighted_avg', 'temporal_multihead_attention']:
    predictor = TimeSeriesPredictor(
        target_column='close',
        pooling_type=pooling,  # ⭐ Experiment with different pooling
        d_token=128,
        n_heads=8,
        n_layers=3
    )
    predictor.fit(train_df, val_df, epochs=50)
```

**See `POOLING_VERIFICATION_RESULTS.md` for complete pooling documentation.**

## 🏗️ Project Structure

```
tf_future/
├── tf_predictor/                    # 🧠 Generic reusable time series library
│   ├── core/                          # Core transformer models and predictor base class
│   ├── preprocessing/                 # Generic time series feature engineering
│   ├── tests/                         # Comprehensive test suite
│   └── README.md                      # Library documentation
├── daily_stock_forecasting/                 # 📈 Daily stock market prediction application
│   ├── preprocessing/                 # Stock-specific feature engineering
│   ├── main.py                        # CLI application
│   └── README.md                      # Application documentation
└── intraday_forecasting/              # 📊 High-frequency intraday trading application
    ├── preprocessing/                 # Intraday-specific feature engineering
    ├── main.py                        # CLI application
    └── README.md                      # Application documentation
```

## 💰 Cryptocurrency Support

The framework now supports cryptocurrency data in various formats:

### Supported Data Formats
- **BTC-USD Format**: `Datetime,Open,High,Low,Close,Volume,Dividends,Stock Splits`
- **Yahoo Finance Crypto**: Standard OHLCV with timezone-aware timestamps
- **Generic OHLCV**: Any CSV with timestamp and OHLCV columns

### Features
- **24/7 Trading**: No market hours filtering for cryptocurrency data
- **Auto-Detection**: Automatically detects BTC, ETH, and crypto files
- **Timezone Handling**: Properly handles UTC and timezone-aware timestamps
- **Multiple Timeframes**: 1min, 5min, 15min, 1h cryptocurrency forecasting

### Example Data Format
```csv
Datetime,Open,High,Low,Close,Volume,Dividends,Stock Splits
2025-09-03 18:34:00+00:00,112093.55,112093.55,112093.55,112093.55,0,0.0,0.0
2025-09-03 18:35:00+00:00,112153.06,112153.06,112153.06,112153.06,0,0.0,0.0
```

## ✨ Key Features

### 🧠 Generic TF Predictor Library (`tf_predictor/`)
- **Reusable Base**: Abstract `TimeSeriesPredictor` class for any domain.
- **State-of-the-art Models**: FT-Transformer and CSN-Transformer with attention mechanisms.
- **Configurable Pooling** ⭐ NEW v2.1: Choose from 5 pooling strategies for sequence aggregation.
- **Rich Preprocessing**: Date features, lag features, rolling statistics.
- **Production Ready**: Model persistence, evaluation metrics, data splitting.
- **Group-Based Scaling** ⭐ NEW: Independent scaling per group (e.g., per stock symbol) while training a unified model.

### 📈 Daily Stock Forecasting Application (`daily_stock_forecasting/`)
- **Multi-Target Prediction** ⭐ NEW: Predict multiple variables simultaneously (e.g., close + volume).
- **Multi-Horizon Prediction**: Predict 1, 2, 3+ steps ahead simultaneously.
- **Rich Feature Engineering**: Volume, VWAP (typical price), and cyclical time encodings.
- **Multiple Targets**: Price prediction, returns, percentage changes, volatility.
- **Portfolio Support** ⭐ NEW: Train on multiple stocks with group-based scaling for better accuracy.
- **Complete Workflow**: Data loading, validation, training, visualization.
- **CLI Interface**: Ready-to-use command-line application.

### 📊 Intraday Forecasting Application (`intraday_forecasting/`)
- **Multi-Target Prediction** ⭐ NEW: Predict multiple variables simultaneously (e.g., close + volume).
- **Multi-Market**: US, India stock markets & cryptocurrency (24/7 trading).
- **Intraday Patterns**: Models time-of-day, market sessions, and microstructure.
- **Multi-Timeframe**: Supports 1min, 5min, 15min, and 1h frequencies.
- **Multi-Symbol Support** ⭐ NEW: Train on multiple symbols simultaneously with independent scaling.
- **Crypto Support**: BTC-USD and other cryptocurrency data formats.
- **Live Trading Ready**: Designed for real-time prediction scenarios.
- **Automatic Temporal Ordering** ⭐ NEW: Data automatically sorted to maintain temporal sequences.

## 💻 Programmatic API

### Stock Market Prediction (Single Stock)
```python
from daily_stock_forecasting import StockPredictor, load_stock_data

# Load your data
df = load_stock_data('your_stock_data.csv')

# Initialize model
model = StockPredictor(
    target_column='close',         # or 'pct_change_1d', 'returns', etc.
    sequence_length=10,            # days of history
    prediction_horizon=1,          # steps ahead to predict (1=single, >1=multi-horizon)
    asset_type='stock',            # 'stock' or 'crypto'
    d_token=128,                   # embedding dimension
    n_layers=3,                    # transformer layers
    n_heads=8,                     # attention heads
    dropout=0.1
)

# Train
model.fit(train_df, val_df, epochs=100, batch_size=32)
```

### Portfolio Prediction with Group-Based Scaling ⭐ NEW
```python
from daily_stock_forecasting import StockPredictor

# Load multi-stock portfolio data
# DataFrame should have columns: date, symbol, open, high, low, close, volume
df = pd.read_csv('portfolio_data.csv')

# Initialize model with group-based scaling
model = StockPredictor(
    target_column='close',
    sequence_length=20,
    group_column='symbol',        # ⭐ Enable group-based scaling
    prediction_horizon=1,
    asset_type='stock',           # or 'crypto'
    d_token=192,
    n_layers=4,
    n_heads=8,
    dropout=0.15
)

# Data will be automatically sorted by [symbol, date]
# Each symbol gets its own scaler, but trains in a unified model
model.fit(train_df, val_df, epochs=150, batch_size=64)

# Predictions automatically use correct scaler per symbol
predictions = model.predict(test_df)
```

### Multi-Symbol Intraday Trading ⭐ NEW
```python
from intraday_forecasting import IntradayPredictor

# Load multi-symbol intraday data
# DataFrame: timestamp, symbol, open, high, low, close, volume
df = pd.read_csv('multi_symbol_5min.csv')

# Initialize with group-based scaling
predictor = IntradayPredictor(
    target_column='close',
    timeframe='5min',
    country='US',
    group_column='symbol',        # ⭐ Each symbol gets independent scaling
    sequence_length=20,
    d_token=256,
    n_layers=5,
    n_heads=8,
    dropout=0.2
)

# Automatic temporal ordering ensures sequences are correct
predictor.fit(train_df, val_df, epochs=200, batch_size=128)
```

### Multi-Target Prediction ⭐ NEW
```python
from daily_stock_forecasting import StockPredictor

# Predict multiple targets simultaneously (e.g., price AND volume)
predictor = StockPredictor(
    target_column=['close', 'volume'],  # ⭐ List of targets
    sequence_length=20,
    prediction_horizon=1,
    d_token=192,
    n_layers=4,
    n_heads=8
)

# Train on multiple targets
predictor.fit(train_df, val_df, epochs=150, batch_size=64)

# Predictions returned as dictionary
predictions = predictor.predict(test_df)
# predictions = {'close': array([...]), 'volume': array([...])}

# Evaluate per-target metrics
metrics = predictor.evaluate(test_df)
# metrics = {'close': {'MAE': ..., 'RMSE': ...}, 'volume': {'MAE': ..., 'RMSE': ...}}
```

### Multi-Target with Multi-Horizon and Group Scaling ⭐ NEW
```python
from intraday_forecasting import IntradayPredictor

# The ultimate configuration: multiple targets, horizons, AND groups!
predictor = IntradayPredictor(
    target_column=['close', 'volume'],  # ⭐ Multiple targets
    timeframe='5min',
    country='US',
    group_column='symbol',              # ⭐ Group-based scaling
    prediction_horizon=3,               # ⭐ Multi-horizon (3 steps ahead)
    sequence_length=20,
    d_token=256,
    n_layers=5
)

predictor.fit(train_df, val_df, epochs=200, batch_size=128)

# Returns dict with arrays of shape (n_samples, 3) for each target
predictions = predictor.predict(test_df)
# predictions = {
#   'close': array([[...], [...], [...]]),  # shape: (n_samples, 3)
#   'volume': array([[...], [...], [...]])   # shape: (n_samples, 3)
# }
```

### Custom Time Series Prediction
```python
from tf_predictor.core.predictor import TimeSeriesPredictor
from tf_predictor.preprocessing.time_features import create_date_features, create_lag_features

class EnergyPredictor(TimeSeriesPredictor):
    def create_features(self, df, fit_scaler=False):
        # Custom feature engineering for energy consumption
        df_processed = create_date_features(df, 'date')
        df_processed = create_lag_features(df_processed, 'consumption', [1, 7, 365])
        df_processed['is_working_day'] = (df_processed['dayofweek'] < 5).astype(int)
        return df_processed.fillna(0)

predictor = EnergyPredictor(target_column='consumption', sequence_length=7)
predictor.fit(train_df, val_df, epochs=50)
```

## 📊 What Can You Predict?

### 🏠 Generic Time Series (with `tf_predictor/`)
- **Energy Consumption**: Daily/hourly power usage forecasting.
- **Sales Forecasting**: E-commerce and retail demand prediction.
- **IoT Monitoring**: Sensor data and anomaly detection.
- **Weather Prediction**: Temperature, precipitation, wind forecasting.

### 📈 Stock Market
- **Daily Forecasting** (with `daily_stock_forecasting/`):
    - Price Prediction: Open, High, Low, Close prices
    - Return Forecasting: Daily, weekly, monthly returns
    - Volatility Modeling: Risk assessment and portfolio optimization
- **Intraday Forecasting** (with `intraday_forecasting/`):
    - Minute-level price and volume moves
    - Order flow and market microstructure analysis
    - Multi-market (US, India, Crypto) and multi-timeframe (1min, 5min, 15min, 1h) prediction
    - Cryptocurrency trading (24/7 BTC-USD, ETH-USD, etc.)

## 📚 Documentation

### Core Documentation
- **🧠 Generic Library**: See `tf_predictor/README.md` for API reference.
- **📈 Daily Stock Forecasting**: See `daily_stock_forecasting/README.md` for detailed usage.
- **📊 Intraday Forecasting**: See `intraday_forecasting/README.md` for high-frequency trading details.
- **🎓 Examples**: Working code examples in all packages.

### Quick Reference Guides
- **⚡ Quick Start**: See `QUICK_START.md` for 30-second setup
- **📖 Command Examples**: See `COMMAND_EXAMPLES.md` for comprehensive CLI reference
- **🔧 Pipeline Guide**: See `PIPELINE_QUICK_REFERENCE.md` for v2.0.0 pipeline details
- **📝 Changelog**: See `CHANGELOG.md` for v2.0.0 migration guide
- **📋 Refactoring Summary**: See `REFACTORING_COMPLETE_v2.md` for complete v2.0.0 changes

## 🧪 Testing

All packages include comprehensive test suites:
```bash
# Test generic library
python tf_predictor/tests/test_core.py

# Test daily stock application
python daily_stock_forecasting/tests/test_stock.py

# Test intraday application
python intraday_forecasting/tests/test_intraday.py
```

## 🛠️ Development

The modular design makes it easy to:
- **Extend**: Create new domain-specific applications.
- **Contribute**: Add preprocessing utilities or model improvements.
- **Scale**: Build production applications on the generic foundation.

Start with the stock forecasting examples, then explore building your own domain-specific predictor!
