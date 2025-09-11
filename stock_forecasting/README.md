# 📈 Stock Forecasting with TF Predictor

A specialized application for stock price prediction using the TF Predictor (Feature Tokenizer Transformer) architecture. This package extends the generic `tf_predictor` library with stock-specific features and workflows.

## 🎯 Features

- **Stock-Specific Predictions**: Price forecasting, return prediction, volatility modeling
- **Rich Feature Engineering**: 30+ technical indicators and market features
- **Multiple Target Types**: Close prices, percentage changes, returns, volatility
- **Production Ready**: Complete CLI application with visualization and model persistence
- **Real Data Support**: Load and validate actual stock market data (OHLCV format)

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

```bash
# Basic stock prediction with sample data
python main.py --use_sample_data --target close --epochs 50

# Use your own data
python main.py --data_path data/raw/AAPL.csv --target close --sequence_length 7 --epochs 100

# Predict percentage changes
python main.py --data_path data/raw/MSFT.csv --target pct_change_1d --epochs 50

# Quick training with smaller model
python main.py --use_sample_data --target close --epochs 20 --d_token 64 --n_layers 2 --batch_size 64
```

## 📊 Data Format

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

Example:
```csv
date,open,high,low,close,volume
2023-01-01,100.0,105.0,99.0,103.0,1000000
2023-01-02,103.0,107.0,102.0,106.0,1200000
```

## 🔧 Available Target Columns

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

## 🛠️ Configuration Options

### Model Parameters
- `target_column`: What to predict (default: 'close')
- `sequence_length`: Days of history to use (default: 5)
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

The system automatically creates 30+ features from your OHLCV data:

### Date Features (10)
- Year, month, day, day of week, quarter
- Weekend indicator
- Cyclical encodings (sin/cos for seasonality)

### Price Features (8+)
- Returns and log returns
- Percentage changes (1d, 3d, 5d, 10d periods)  
- Volatility (10-day rolling)
- Momentum (5-day rolling)

### Ratio Features (3)
- High/Low ratio
- Close/Open ratio  
- Volume ratio

### Moving Average Features (5+)
- 5-day moving averages for all OHLC columns
- Volume moving average

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

### 1. Basic Price Prediction
```python
from stock_forecasting import StockPredictor, create_sample_stock_data
from tf_predictor.core.utils import split_time_series

# Create or load data
df = create_sample_stock_data(n_samples=500)  # Or use load_stock_data()
train_df, val_df, test_df = split_time_series(df, test_size=60, val_size=30)

# Train model
predictor = StockPredictor(target_column='close', sequence_length=10)
predictor.fit(train_df, val_df, epochs=50, verbose=True)

# Evaluate
metrics = predictor.evaluate(test_df)
print(f"MAPE: {metrics['MAPE']:.2f}%")
```

### 2. Percentage Change Prediction
```python
# Predict 1-day percentage changes instead of prices
predictor = StockPredictor(
    target_column='pct_change_1d',  
    sequence_length=7,
    d_token=64,
    n_layers=2
)

predictor.fit(train_df, val_df, epochs=30)
predictions = predictor.predict(test_df)
```

### 3. Advanced Configuration
```python
predictor = StockPredictor(
    target_column='close',
    sequence_length=15,              # Use 3 weeks of history
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

### For Faster Training:
- Use smaller models (`d_token=64`, `n_layers=2`) for quick experiments  
- Increase batch size (`batch_size=64`) if memory allows
- Use fewer epochs for initial testing

### For Production:
- Always use validation sets
- Monitor both training and validation metrics
- Save models regularly during training
- Use percentage change targets for normalized predictions

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

## 📚 Next Steps

1. **Try the examples**: Run `python examples/stock_prediction_example.py`
2. **Use your own data**: Place CSV files in `data/raw/` and load with `load_stock_data()`
3. **Experiment with targets**: Try different prediction targets like `pct_change_3d`
4. **Tune hyperparameters**: Adjust model size and training parameters for your data
5. **Add technical indicators**: Use `create_technical_indicators()` for advanced features

## 🔄 Updates

This package is built on the modular `tf_predictor` library and benefits from improvements to the core transformer implementation while providing stock market-specific enhancements and workflows.