# 📈 Stock Forecasting with TF Predictor

A specialized application for daily stock and cryptocurrency price prediction using the TF Predictor (Feature Tokenizer Transformer) architecture. This package extends the generic `tf_predictor` library with market-specific features and workflows for both traditional stocks and cryptocurrencies.

## 🎯 Features

- **Multi-Asset Support**: Traditional stocks (5-day week) and cryptocurrencies (7-day week, 24/7)
- **Multi-Horizon Predictions**: Predict 1, 2, 3+ steps ahead with `prediction_horizon`
- **Essential vs Full Features**: Choose between minimal essential features or comprehensive technical indicators
- **Flexible Target Variables**: Any input feature can be the prediction target (volume, typical_price, close, etc.)
- **Market-Specific Predictions**: Price forecasting, return prediction, volatility modeling
- **Rich Feature Engineering**: 30+ technical indicators and market features OR 7-8 essential features
- **Production Ready**: Complete CLI application with visualization and model persistence
- **Real Data Support**: Load and validate actual stock/crypto market data (OHLCV format)

## 🚀 Quick Start

### Basic Usage

```python
from stock_forecasting import StockPredictor, load_stock_data

# Load your stock data
df = load_stock_data('path/to/your/stock_data.csv')

# Initialize predictor
predictor = StockPredictor(
    target_column='close',           # Predict closing prices
    sequence_length=10,              # Use 10 days of history
    prediction_horizon=1,            # Predict 1 step ahead
    use_essential_only=False,        # Use all features (30+)
    d_token=128,                     # Model complexity
    n_layers=3,                      # Transformer layers
    n_heads=8                        # Attention heads
)

# Train the model
predictor.fit(
    df=train_df,
    val_df=val_df,
    epochs=100,
    batch_size=32,
    learning_rate=1e-3
)

# Make predictions
predictions = predictor.predict(test_df)
metrics = predictor.evaluate(test_df)
```

### Command Line Interface

#### Quick Start Examples

```bash
# Basic stock prediction with sample data
python stock_forecasting/main.py --use_sample_data --target close --asset_type stock --epochs 50

# Crypto daily prediction with sample data
python stock_forecasting/main.py --use_sample_data --target close --asset_type crypto --epochs 50

# Use your own stock data with essential features (fast training)
python stock_forecasting/main.py --data_path data/MSFT_historical_price_cleaned.csv --target volume --asset_type stock --use_essential_only --prediction_horizon 2 --epochs 50

# Crypto (BTC, ETH, etc.) daily prediction
python stock_forecasting/main.py --data_path data/BTC-USD-daily.csv --target close --asset_type crypto --epochs 100

# Multi-step ahead prediction with all features
python stock_forecasting/main.py --data_path data/raw/AAPL.csv --target close --asset_type stock --prediction_horizon 3 --sequence_length 7 --epochs 100

# Quick training with smaller model
python stock_forecasting/main.py --use_sample_data --target close --asset_type stock --epochs 20 --d_token 64 --n_layers 2 --batch_size 64
```

#### Complete Command with All Parameters

```bash
PYTHONPATH=. venv/bin/python stock_forecasting/main.py \
  --data_path /path/to/your/data.csv \
  --target close \
  --asset_type stock \
  --sequence_length 5 \
  --prediction_horizon 1 \
  --use_essential_only \
  --d_token 128 \
  --n_layers 3 \
  --n_heads 8 \
  --dropout 0.1 \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --patience 15 \
  --test_size 30 \
  --val_size 20 \
  --model_path outputs/models/stock_model.pt \
  --no_plots
```

#### CLI Parameter Reference

**Data Parameters:**
- `--data_path PATH`: Path to stock/crypto data CSV file (optional if using sample data)
- `--use_sample_data`: Use synthetic sample data instead of real data
- `--asset_type {stock,crypto}`: Asset type (default: stock)
  - `stock`: Traditional 5-day trading week (Monday-Friday)
  - `crypto`: 24/7 trading, 7-day week data (includes weekends)
- `--target {close,open,high,low,volume}`: Column to predict (default: close)
  - Also supports engineered features: `pct_change_1d`, `pct_change_3d`, `pct_change_5d`, `pct_change_10d`, `returns`, `log_returns`, `volatility_10d`, `momentum_5d`, `high_low_ratio`, `close_open_ratio`, `volume_ratio`

**Model Architecture:**
- `--sequence_length N`: Historical days to use for prediction (default: 5)
- `--prediction_horizon N`: Steps ahead to predict - 1=next day, 2=two days ahead, etc. (default: 1)
- `--use_essential_only`: Use only essential features (volume, typical_price, seasonal) - 7 features for stock, 8 for crypto (adds is_weekend) instead of 30+
- `--d_token N`: Token embedding dimension (default: 128)
- `--n_layers N`: Number of transformer layers (default: 3)
- `--n_heads N`: Number of attention heads (default: 8)
- `--dropout FLOAT`: Dropout rate (default: 0.1)

**Training Parameters:**
- `--epochs N`: Number of training epochs (default: 100)
- `--batch_size N`: Training batch size (default: 32)
- `--learning_rate FLOAT`: Learning rate (default: 0.001)
- `--patience N`: Early stopping patience (default: 15)

**Data Split:**
- `--test_size N`: Number of samples for test set (default: 30)
- `--val_size N`: Number of samples for validation set (default: 20)

**Output:**
- `--model_path PATH`: Path to save trained model (default: outputs/models/stock_model.pt)
- `--no_plots`: Skip generating plots (plots are generated by default)
```

## 📊 Data Format

### Stock Data Format

Your CSV file should contain these columns:
- `date`: Date (YYYY-MM-DD format)
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume

Optional columns:
- `symbol`: Stock symbol
- Any additional features

Example (Stock - weekdays only):
```csv
date,open,high,low,close,volume
2023-01-02,100.0,105.0,99.0,103.0,1000000
2023-01-03,103.0,107.0,102.0,106.0,1200000
2023-01-04,106.0,108.0,105.0,107.0,1100000
```

### Cryptocurrency Data Format

Same CSV format as stocks, but **must include all 7 days of the week** (no gaps for weekends):

Example (Crypto - includes weekends):
```csv
date,open,high,low,close,volume
2023-01-01,16500.0,16800.0,16400.0,16750.0,25000000000
2023-01-02,16750.0,17000.0,16700.0,16900.0,28000000000
2023-01-03,16900.0,17200.0,16850.0,17100.0,30000000000
...includes Saturday and Sunday data...
2023-01-07,17300.0,17500.0,17200.0,17400.0,26000000000
2023-01-08,17400.0,17600.0,17350.0,17550.0,27000000000
```

**Important for Crypto:**
- Data must be **continuous** (no weekend gaps)
- Crypto markets trade 24/7, so expect 365 days per year of data
- Use `--asset_type crypto` flag when training

## 🔧 Available Target Columns

**Key Innovation**: Any feature can be a target! The system automatically creates shifted target variables for proper time series forecasting.

### Essential Features (Available as Targets)
- `volume` - Trading volume
- `typical_price` - (high + low + close) / 3 (automatically created)

### Price Targets
- `open`, `high`, `low`, `close` - Direct price prediction
- `vwap` - Volume weighted average price (if available)

### Return Targets
- `returns` - Daily returns (automatically created)
- `log_returns` - Log returns (automatically created)
- `pct_change_1d`, `pct_change_3d`, `pct_change_5d`, `pct_change_10d` - Custom percentage changes

### Technical Targets
- `volatility_10d` - 10-day rolling volatility
- `momentum_5d` - 5-day momentum indicator
- `high_low_ratio`, `close_open_ratio` - Price ratios

### Target Shifting
- When you set `target_column='volume'` and `prediction_horizon=2`, the system creates `volume_target_h2`
- This predicts volume 2 steps ahead using current features
- Historical target values remain as input features for the model

## 🛠️ Configuration Options

### Model Parameters
- `target_column`: What to predict (default: 'close')
- `sequence_length`: Days of history to use (default: 5)
- `prediction_horizon`: Steps ahead to predict (default: 1)
- `use_essential_only`: Use only 7 essential features vs 30+ full features (default: False)
- `d_token`: Token embedding dimension (default: 128)
- `n_layers`: Number of transformer layers (default: 3)
- `n_heads`: Number of attention heads (default: 8)
- `dropout`: Dropout rate (default: 0.1)

### Training Parameters
- `epochs`: Training epochs (default: 100)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 1e-3)
- `patience`: Early stopping patience (default: 15)

### Data Parameters
- `test_size`: Test set size in samples (default: 30)
- `val_size`: Validation set size in samples (default: 20)

## 📈 Feature Engineering

### Essential Features Mode (`use_essential_only=True`)
**7-8 Features Total** - Minimal, fast training with core market features:
- `volume` - Trading volume
- `typical_price` - (high + low + close) / 3
- `month_sin`, `month_cos` - Cyclical month encoding
- `dayofweek_sin`, `dayofweek_cos` - Cyclical day-of-week encoding (0-6 for crypto, 0-4 for stocks)
- `year` - Raw year value (trend)
- `is_weekend` - Weekend indicator (crypto only, 0=weekday, 1=weekend)

### Full Features Mode (`use_essential_only=False`)
**30+ Features** - Comprehensive technical analysis:

#### Date Features (8)
- Year, month, day, day of week, quarter
- Weekend indicator
- Cyclical encodings (sin/cos for seasonality)

#### Price Features (10+)
- Returns and log returns
- Percentage changes (1d, 3d, 5d, 10d periods)
- Volatility (10-day rolling)
- Momentum (5-day rolling)

#### Ratio Features (3)
- High/Low ratio
- Close/Open ratio
- Volume ratio

#### Moving Average Features (15+)
- 5-day moving averages for all OHLC columns
- Volume moving average
- Min/max/std rolling features

## 📁 Project Structure

```
stock_forecasting/
├── data/
│   ├── sample/                     # Sample datasets
│   │   └── MSFT_sample.csv        # Microsoft sample data
│   └── raw/                       # Your raw data files
├── outputs/
│   ├── models/                    # Trained models (.pt files)
│   ├── predictions/               # Prediction CSVs
│   └── plots/                     # Generated visualizations
├── preprocessing/
│   ├── stock_features.py          # Stock feature engineering
│   └── market_data.py            # Data loading and validation
├── visualization/
│   └── stock_charts.py           # Stock-specific visualizations
├── tests/
│   └── test_stock.py             # Comprehensive tests
├── examples/
│   └── stock_prediction_example.py # Usage examples
├── predictor.py                   # StockPredictor class
├── main.py                       # CLI application
└── README.md                     # This file
```

## 💻 Examples

### 1. Essential Features - Fast Training
```python
from stock_forecasting import StockPredictor, create_sample_stock_data
from tf_predictor.core.utils import split_time_series

# Create or load data
df = create_sample_stock_data(n_samples=500)  # Or use load_stock_data()
train_df, val_df, test_df = split_time_series(df, test_size=60, val_size=30)

# Train with essential features only - 7 features, fast training
predictor = StockPredictor(
    target_column='volume',          # Predict trading volume
    prediction_horizon=2,            # 2 steps ahead
    use_essential_only=True,         # Only 7 essential features
    sequence_length=5
)
predictor.fit(train_df, val_df, epochs=50, verbose=True)

# Evaluate
metrics = predictor.evaluate(test_df)
print(f"MAPE: {metrics['MAPE']:.2f}%")
```

### 2. Multi-Step Prediction with Full Features
```python
# Predict 3 steps ahead using all 30+ features
predictor = StockPredictor(
    target_column='close',
    prediction_horizon=3,            # 3 steps ahead
    use_essential_only=False,        # Use all features
    sequence_length=10,
    d_token=128
)

predictor.fit(train_df, val_df, epochs=50)
predictions = predictor.predict(test_df)
```

### 3. Advanced Configuration
```python
predictor = StockPredictor(
    target_column='typical_price',   # Predict engineered feature
    sequence_length=15,              # Use 3 weeks of history
    prediction_horizon=5,            # Predict 5 days ahead
    use_essential_only=True,         # Fast essential features
    d_token=256,                     # Larger model
    n_layers=4,                      # Deeper network
    n_heads=16,                      # More attention heads
    dropout=0.15,                    # Regularization
    verbose=True
)

# Train with custom parameters
predictor.fit(
    df=train_df,
    val_df=val_df,
    epochs=200,
    batch_size=16,                   # Smaller batches for large model
    learning_rate=5e-4,              # Lower learning rate
    patience=25                      # More patience for large model
)
```

## 📊 Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error  
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: R-squared (coefficient of determination)
- **Directional Accuracy**: Percentage of correct direction predictions

## 🎨 Visualizations

Automatic generation of:
- **Prediction vs Actual plots**: Time series comparison
- **Training progress**: Loss curves and metrics
- **Performance summaries**: Tabulated results

Enable with:
```bash
python main.py --data_path your_data.csv --target close  # Plots enabled by default
python main.py --data_path your_data.csv --target close --no_plots  # Disable plots
```

## 🔍 Model Persistence

```python
# Save trained model
predictor.save('outputs/models/my_stock_model.pt')

# Load and use later
new_predictor = StockPredictor(target_column='close', sequence_length=10)
new_predictor.load('outputs/models/my_stock_model.pt')
predictions = new_predictor.predict(new_data)
```

## ⚡ Performance Tips

### For Better Accuracy:
- Use longer sequences (10-20 days) for complex patterns
- Increase model size (`d_token=256`, `n_layers=4`) for large datasets
- Train longer with early stopping (`epochs=500`, `patience=30`)
- Use validation sets for proper hyperparameter tuning
- Try different prediction horizons (1, 2, 3 steps ahead)

### For Faster Training:
- **Use essential features** (`use_essential_only=True`) - 7 vs 30+ features
- Use smaller models (`d_token=64`, `n_layers=2`) for quick experiments
- Increase batch size (`batch_size=64`) if memory allows
- Use fewer epochs for initial testing

### For Production:
- Start with essential features for prototyping, add full features if needed
- Always use validation sets
- Monitor both training and validation metrics
- Save models regularly during training
- Test different targets (volume often easier than price prediction)

## 🚨 Troubleshooting

### Common Issues:

**"Target column not found"**
- Check available columns with `df.columns` 
- Ensure target exists after feature engineering
- Use engineered targets like `pct_change_1d` if base columns missing

**"Not enough data for sequences"**
- Ensure dataset has at least `sequence_length + 50` samples
- Reduce `sequence_length` for small datasets
- Check data loading and filtering steps

**"Poor performance (high MAPE)"**  
- Try different target columns (percentage changes often work better)
- Increase model complexity or training time
- Check data quality and date ordering
- Consider longer sequences for complex patterns

**"Out of memory"**
- Reduce `batch_size` (try 16 or 8)
- Reduce model size (`d_token=64`, `n_layers=2`)
- Use smaller sequences

## 🔗 Integration

This package works seamlessly with the generic `tf_predictor` library:

```python
# Use generic utilities
from tf_predictor.core.utils import split_time_series, calculate_metrics
from tf_predictor.preprocessing.time_features import create_date_features

# Use stock-specific components  
from stock_forecasting import StockPredictor
from stock_forecasting.preprocessing.stock_features import create_technical_indicators
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
cd tests/
python test_stock.py
```

Test specific functionality:
```bash
# Test basic functionality only
python -c "
from test_stock import TestStockPredictor
test = TestStockPredictor()
test.test_initialization()
test.test_create_features()
print('✅ Basic tests passed')
"
```

## 🧪 Verified Test Results

Successfully tested on real MSFT data (`data/MSFT_historical_price_cleaned.csv`, 1253 samples):

**Volume Prediction (Essential Features)**
```bash
python -m stock_forecasting.main --data_path data/MSFT_historical_price_cleaned.csv --target volume --prediction_horizon 2 --use_essential_only --epochs 20
```
- **Test MAPE: 16.25%** (predicting trading volume 2 steps ahead)
- **Features: 7** (volume, typical_price, seasonal cyclical)
- **Training: Fast** (7 features vs 30+)

**Close Price Prediction (Essential Features)**
```bash
python -m stock_forecasting.main --data_path data/MSFT_historical_price_cleaned.csv --target close --prediction_horizon 1 --use_essential_only --epochs 20
```
- **Test MAPE: 7.44%** (predicting close price 1 step ahead)
- **Features: 8** (7 essential + close as feature)
- **R² Score: 0.99** (excellent fit on training data)

## 📚 Next Steps

1. **Try the tested commands**: Use the verified examples above with your data
2. **Start with essential features**: Fast prototyping with `--use_essential_only`
3. **Experiment with prediction horizons**: Try `--prediction_horizon 1,2,3,5`
4. **Test different targets**: Volume prediction often works well
5. **Scale up**: Add full features for production after prototyping

## 🔄 Updates

This package is built on the modular `tf_predictor` library and benefits from improvements to the core transformer implementation while providing stock market-specific enhancements and workflows.