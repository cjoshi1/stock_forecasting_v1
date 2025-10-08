# 🚀 TF Predictor: Modular Time Series Forecasting Framework

A comprehensive, modular framework for time series forecasting using the TF Predictor (Feature Tokenizer Transformer) architecture. This project provides a generic reusable library and specialized applications for daily and intraday stock market forecasting.

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
cd stock_forecasting/

# Train with sample data - single step prediction (default)
python main.py --use_sample_data --target close --epochs 50

# Multi-horizon prediction - predict 3 steps ahead simultaneously
python main.py --use_sample_data --target close --prediction_horizon 3 --epochs 50

# Train with your data
python main.py --data_path data/raw/AAPL.csv --target close --epochs 100

# Fast training with essential features
python main.py --data_path data/raw/AAPL.csv --target close --use_essential_only --epochs 50
```

### For Custom Time Series (Generic Library)
```python
from tf_predictor.core.predictor import TimeSeriesPredictor
from tf_predictor.preprocessing.time_features import create_date_features

# Create your domain-specific predictor
class MyPredictor(TimeSeriesPredictor):
    def create_features(self, df, fit_scaler=False):
        df_processed = create_date_features(df, 'date')
        # Add your domain-specific features...
        return df_processed.fillna(0)

predictor = MyPredictor(target_column='value', sequence_length=10, prediction_horizon=1)
predictor.fit(train_df, val_df, epochs=100)
```

## 🏗️ Project Structure

```
tf_future/
├── tf_predictor/                    # 🧠 Generic reusable time series library
│   ├── core/                          # Core transformer models and predictor base class
│   ├── preprocessing/                 # Generic time series feature engineering
│   ├── tests/                         # Comprehensive test suite
│   └── README.md                      # Library documentation
├── stock_forecasting/                 # 📈 Daily stock market prediction application
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
- **State-of-the-art Model**: FT-Transformer with attention mechanisms.
- **Rich Preprocessing**: Date features, lag features, rolling statistics.
- **Production Ready**: Model persistence, evaluation metrics, data splitting.

### 📈 Daily Stock Forecasting Application (`stock_forecasting/`)
- **Multi-Horizon Prediction**: Predict 1, 2, 3+ steps ahead simultaneously.
- **Essential vs Full Features**: Fast training with 7 essential features or comprehensive 30+ indicators.
- **Multiple Targets**: Price prediction, returns, percentage changes, volatility.
- **Complete Workflow**: Data loading, validation, training, visualization.
- **CLI Interface**: Ready-to-use command-line application.

### 📊 Intraday Forecasting Application (`intraday_forecasting/`)
- **Multi-Market**: US, India stock markets & cryptocurrency (24/7 trading).
- **Intraday Patterns**: Models time-of-day, market sessions, and microstructure.
- **Multi-Timeframe**: Supports 1min, 5min, 15min, and 1h frequencies.
- **Crypto Support**: BTC-USD and other cryptocurrency data formats.
- **Live Trading Ready**: Designed for real-time prediction scenarios.

## 💻 Programmatic API

### Stock Market Prediction
```python
from stock_forecasting import StockPredictor, load_stock_data

# Load your data
df = load_stock_data('your_stock_data.csv')

# Initialize model
model = StockPredictor(
    target_column='close',         # or 'pct_change_1d', 'returns', etc.
    sequence_length=10,            # days of history
    prediction_horizon=1,          # steps ahead to predict (1=single, >1=multi-horizon)
    use_essential_only=False,      # False=30+ features, True=7 essential features
    d_token=128,                   # embedding dimension
    n_layers=3,                    # transformer layers
    n_heads=8,                     # attention heads
    dropout=0.1
)

# Train
model.fit(train_df, val_df, epochs=100, batch_size=32)
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
- **Daily Forecasting** (with `stock_forecasting/`):
    - Price Prediction: Open, High, Low, Close prices
    - Return Forecasting: Daily, weekly, monthly returns
    - Volatility Modeling: Risk assessment and portfolio optimization
- **Intraday Forecasting** (with `intraday_forecasting/`):
    - Minute-level price and volume moves
    - Order flow and market microstructure analysis
    - Multi-market (US, India, Crypto) and multi-timeframe (1min, 5min, 15min, 1h) prediction
    - Cryptocurrency trading (24/7 BTC-USD, ETH-USD, etc.)

## 📚 Documentation

- **🧠 Generic Library**: See `tf_predictor/README.md` for API reference.
- **📈 Daily Stock Forecasting**: See `stock_forecasting/README.md` for detailed usage.
- **📊 Intraday Forecasting**: See `intraday_forecasting/README.md` for high-frequency trading details.
- **🎓 Examples**: Working code examples in all packages.

## 🧪 Testing

All packages include comprehensive test suites:
```bash
# Test generic library
python tf_predictor/tests/test_core.py

# Test daily stock application
python stock_forecasting/tests/test_stock.py

# Test intraday application
python intraday_forecasting/tests/test_intraday.py
```

## 🛠️ Development

The modular design makes it easy to:
- **Extend**: Create new domain-specific applications.
- **Contribute**: Add preprocessing utilities or model improvements.
- **Scale**: Build production applications on the generic foundation.

Start with the stock forecasting examples, then explore building your own domain-specific predictor!
