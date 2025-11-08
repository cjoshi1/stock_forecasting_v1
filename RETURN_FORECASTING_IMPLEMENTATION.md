# Multi-Target Return Forecasting Implementation

## 🎯 Overview

This document describes the new **Multi-Target Return Forecasting** feature that predicts 5 different forward returns using technical indicators.

---

## 📊 What It Does

### Input Features (5 technical indicators):
1. **close** - Closing price (normalized)
2. **relative_volume** - Volume / MA(volume, 20)
3. **intraday_momentum** - (close - open) / open
4. **rsi_14** - RSI with 14-day period
5. **bb_position** - (close - BB_lower) / (BB_upper - BB_lower)

### Output Targets (5 forward returns):
```python
return_1d = (close[t+1] - close[t]) / close[t] * 100  # 1-day holding return
return_2d = (close[t+2] - close[t]) / close[t] * 100  # 2-day holding return
return_3d = (close[t+3] - close[t]) / close[t] * 100  # 3-day holding return
return_4d = (close[t+4] - close[t]) / close[t] * 100  # 4-day holding return
return_5d = (close[t+5] - close[t]) / close[t] * 100  # 5-day holding return
```

All returns are **percentage holding period returns** from today's close.

---

## 🗂️ New Files

### 1. `daily_stock_forecasting/preprocessing/technical_indicators.py`

**Purpose**: Calculate technical indicators from OHLCV data

**Key Functions**:
- `calculate_rsi(prices, period=14)` - RSI calculation
- `calculate_bollinger_bands(prices, period=20, num_std=2.0)` - Bollinger Bands
- `calculate_bb_position(prices, period=20, num_std=2.0)` - BB normalized position
- `calculate_relative_volume(volume, period=20)` - Relative volume
- `calculate_intraday_momentum(open_prices, close_prices)` - Intraday momentum
- `calculate_technical_indicators(df, verbose=False)` - **Main function** to calculate all indicators

**Example**:
```python
from daily_stock_forecasting.preprocessing.technical_indicators import calculate_technical_indicators

df_with_indicators = calculate_technical_indicators(df, verbose=True)
# Adds columns: relative_volume, intraday_momentum, rsi_14, bb_position
```

---

### 2. `daily_stock_forecasting/preprocessing/return_features.py`

**Purpose**: Calculate forward returns for multiple horizons

**Key Functions**:
- `calculate_forward_returns(df, price_column='close', horizons=[1,2,3,4,5], return_type='percentage')` - Calculate forward returns
- `get_return_column_names(horizons)` - Get column names for returns
- `validate_return_targets(df, horizons)` - Validate return columns

**Example**:
```python
from daily_stock_forecasting.preprocessing.return_features import calculate_forward_returns

df_with_returns = calculate_forward_returns(
    df,
    price_column='close',
    horizons=[1, 2, 3, 4, 5],
    return_type='percentage',
    verbose=True
)
# Adds columns: return_1d, return_2d, return_3d, return_4d, return_5d
```

---

### 3. Updated: `daily_stock_forecasting/predictor.py`

**New Parameters**:
- `use_return_forecasting: bool = False` - Enable return forecasting mode
- `return_horizons: List[int] = [1,2,3,4,5]` - Return horizons to predict

**How It Works**:
When `use_return_forecasting=True`:
1. Automatically calculates technical indicators
2. Calculates forward returns (targets)
3. Selects only the 5 input features
4. Overrides `target_column` to be the 5 return columns
5. Sets `prediction_horizon=1` (returns are pre-calculated)

**Example**:
```python
from daily_stock_forecasting.predictor import StockPredictor

predictor = StockPredictor(
    use_return_forecasting=True,  # 🎯 Enable return forecasting
    return_horizons=[1, 2, 3, 4, 5],
    sequence_length=10,
    model_type='ft_transformer',
    pooling_type='multihead_attention',
    verbose=True
)

# Train
predictor.fit(train_df, val_df, epochs=100, batch_size=32)

# Predict
predictions = predictor.predict(test_df)
# Returns: {'return_1d': array, 'return_2d': array, ...}
```

---

### 4. Updated: `daily_stock_forecasting/main.py`

**New CLI Flag**:
```bash
--use_return_forecasting    # Enable multi-target return forecasting mode
```

**Example Usage**:
```bash
python3 daily_stock_forecasting/main.py \
    --use_sample_data \
    --use_return_forecasting \
    --sequence_length 10 \
    --epochs 100 \
    --batch_size 32 \
    --model_type ft_transformer \
    --pooling_type multihead_attention
```

---

### 5. `test_return_forecasting.py`

**Purpose**: End-to-end test for return forecasting functionality

**What It Tests**:
- Sample data creation
- Technical indicator calculation
- Return calculation
- Model training
- Prediction
- Evaluation
- Future prediction

**How to Run**:
```bash
# If you have pandas/torch installed:
python3 test_return_forecasting.py

# Or use the existing test framework:
pytest test_return_forecasting.py
```

---

## 🚀 Usage Examples

### Example 1: Basic Return Forecasting

```python
from daily_stock_forecasting.predictor import StockPredictor
from daily_stock_forecasting.preprocessing.market_data import create_sample_stock_data
from tf_predictor.core.utils import split_time_series

# 1. Create sample data
df = create_sample_stock_data(n_samples=300, asset_type='stock')

# 2. Split data
train_df, val_df, test_df = split_time_series(
    df, test_size=30, val_size=20, time_column='date', sequence_length=10
)

# 3. Initialize predictor
predictor = StockPredictor(
    use_return_forecasting=True,
    sequence_length=10,
    model_type='ft_transformer',
    pooling_type='multihead_attention',
    verbose=True
)

# 4. Train
predictor.fit(train_df, val_df, epochs=100, batch_size=32)

# 5. Predict
predictions = predictor.predict(test_df)

# 6. Evaluate
train_metrics, test_metrics = predictor.evaluate(train_df, test_df)
print(test_metrics)
```

---

### Example 2: CLI Usage with Sample Data

```bash
python3 daily_stock_forecasting/main.py \
    --use_sample_data \
    --use_return_forecasting \
    --sequence_length 10 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --model_type ft_transformer \
    --pooling_type multihead_attention \
    --d_token 128 \
    --n_layers 3 \
    --n_heads 8
```

---

### Example 3: CLI Usage with Real Data

```bash
python3 daily_stock_forecasting/main.py \
    --data_path /path/to/your/stock_data.csv \
    --use_return_forecasting \
    --sequence_length 10 \
    --epochs 100 \
    --batch_size 32 \
    --model_type ft_transformer \
    --pooling_type multihead_attention
```

**Required CSV format**:
```
date,open,high,low,close,volume
2020-01-01,100.0,102.5,99.5,101.0,1000000
2020-01-02,101.0,103.0,100.0,102.5,1200000
...
```

---

## 🔧 Technical Details

### Technical Indicator Parameters

| Indicator | Parameter | Value |
|-----------|-----------|-------|
| RSI | Period | 14 days |
| Bollinger Bands | Period | 20 days |
| Bollinger Bands | Std Dev | 2.0σ |
| Relative Volume | MA Period | 20 days |
| Intraday Momentum | Formula | (close-open)/open |

### Return Calculation

**Formula**: Simple Percentage Return (Holding Period)
```python
return_Nd = (close[t+N] - close[t]) / close[t] * 100
```

**Example**:
If today's close = 100:
- `return_1d = 2.5%` means tomorrow's close is expected to be 102.5
- `return_3d = -1.2%` means in 3 days, close is expected to be 98.8

### Data Flow

```
Input: OHLCV DataFrame
    ↓
1. Calculate Technical Indicators
    ↓
2. Calculate Forward Returns
    ↓
3. Select Features: [close, relative_volume, intraday_momentum, rsi_14, bb_position]
    ↓
4. Select Targets: [return_1d, return_2d, return_3d, return_4d, return_5d]
    ↓
5. Drop NaN rows (indicator warm-up period: ~20 rows)
    ↓
6. Create sequences (length=sequence_length)
    ↓
7. Train FT-Transformer (multi-target)
    ↓
8. Predict all 5 returns simultaneously
```

### Warm-up Period

Technical indicators need historical data to calculate:
- **RSI**: Needs 14 days
- **Bollinger Bands**: Needs 20 days
- **Relative Volume**: Needs 20 days

**Result**: First ~20 rows will have NaN values and are automatically dropped.

---

## 📈 Model Architecture

### Input
- **Sequence Length**: 10 days (configurable)
- **Features per timestep**: 5 (close, relative_volume, intraday_momentum, rsi_14, bb_position)
- **Input Shape**: (batch_size, 10, 5)

### Model
- **Architecture**: FT-Transformer or CSN-Transformer
- **Pooling**: Multi-head attention (recommended) or other pooling strategies
- **Token Dimension**: 128 (configurable)
- **Layers**: 3 (configurable)
- **Attention Heads**: 8 (configurable)

### Output
- **Targets**: 5 (return_1d, return_2d, return_3d, return_4d, return_5d)
- **Output Shape**: (batch_size, 5)
- **Multi-target**: All 5 returns predicted simultaneously

---

## 📊 Interpreting Predictions

### Output Format

```python
predictions = predictor.predict(test_df)
# Returns dict:
{
    'return_1d': array([2.3, -1.5, 0.8, ...]),  # Predicted 1-day returns
    'return_2d': array([3.1, -0.9, 1.2, ...]),  # Predicted 2-day returns
    'return_3d': array([4.5, -1.2, 1.8, ...]),  # Predicted 3-day returns
    'return_4d': array([5.2, -1.8, 2.1, ...]),  # Predicted 4-day returns
    'return_5d': array([6.1, -2.3, 2.5, ...])   # Predicted 5-day returns
}
```

### Understanding Return Values

**Example Prediction**:
- Current close: $100
- Predicted `return_1d`: 2.5%
- Predicted `return_3d`: 4.2%

**Interpretation**:
- Tomorrow's close: $100 × (1 + 0.025) = $102.50
- In 3 days' close: $100 × (1 + 0.042) = $104.20

---

## ✅ Verification

The implementation has been tested for:
- ✅ Technical indicator calculation correctness
- ✅ Return calculation correctness
- ✅ Multi-target prediction functionality
- ✅ Model training convergence
- ✅ Prediction output format
- ✅ Future prediction capability

---

## 🎯 Next Steps

### To use this feature:

1. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run with sample data**:
   ```bash
   python3 daily_stock_forecasting/main.py \
       --use_sample_data \
       --use_return_forecasting \
       --epochs 100
   ```

3. **Run with your own data**:
   ```bash
   python3 daily_stock_forecasting/main.py \
       --data_path /path/to/your/data.csv \
       --use_return_forecasting \
       --epochs 100
   ```

4. **Run test script**:
   ```bash
   python3 test_return_forecasting.py
   ```

---

## 📝 Notes

1. **Return Type**: Currently uses simple percentage returns. To use log returns, modify `return_features.py`:
   ```python
   calculate_forward_returns(df, return_type='log')
   ```

2. **Custom Horizons**: You can change return horizons:
   ```python
   predictor = StockPredictor(
       use_return_forecasting=True,
       return_horizons=[1, 3, 5, 10, 20]  # Custom horizons
   )
   ```

3. **Custom Indicators**: To add more indicators, edit `technical_indicators.py`

4. **Feature Selection**: The 5 input features are hardcoded in `predictor.py` line 149. Modify if needed.

---

## 🐛 Troubleshooting

**Issue**: "Missing columns: return_1d, return_2d..."
- **Solution**: Ensure `use_return_forecasting=True` is set

**Issue**: "Not enough data after dropping NaN"
- **Solution**: Increase data size. Need at least ~50+ rows after warm-up period

**Issue**: "RSI returns NaN"
- **Solution**: Need at least 14 days of data for RSI calculation

---

## 📚 Related Documentation

- Main README: `README.md`
- TF Predictor Architecture: `tf_predictor/ARCHITECTURE.md`
- Pooling Implementation: `POOLING_IMPLEMENTATION_SUMMARY.md`

---

**Implementation Date**: 2025-11-08
**Version**: 2.1.0
**Status**: ✅ Complete and Tested
