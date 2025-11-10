# Command Reference

**Version:** 2.1.0 (Updated: 2025-11-07)

> **📢 Version 2.1.0 Recent Updates:**
> - **🎯 Configurable Pooling Strategies** - Choose from 5 pooling methods for sequence aggregation
> - **🚀 Multi-Head Attention Pooling** - New default pooling strategy for better performance
> - **Parameter Naming**: d_token, n_heads, n_layers (standardized)
> - **Model Types**: ft_transformer, csn_transformer (removed _cls suffix)
> - **Evaluation alignment fixes** - Correct actuals extraction from shifted target columns
> - **Sequence creation optimization** - 20% more training data utilized
> - Per-horizon target scaling (each horizon gets its own scaler)
> - Automatic cyclical encoding for temporal features
>
> See `POOLING_VERIFICATION_RESULTS.md` for pooling documentation.

---

## Quick Start

### Daily Stock Forecasting

```bash
# Single-target, single-horizon (simplest)
python daily_stock_forecasting/main.py --use_sample_data --target close --epochs 50

# Multi-target (predict close AND volume)
python daily_stock_forecasting/main.py --use_sample_data --target "close,volume" --epochs 50

# Multi-horizon (predict 3 days ahead)
python daily_stock_forecasting/main.py --use_sample_data --target close --prediction_horizon 3 --epochs 50

# Multi-target + Multi-horizon
python daily_stock_forecasting/main.py --use_sample_data --target "close,volume" --prediction_horizon 3 --epochs 50

# Portfolio with group-based scaling
python daily_stock_forecasting/main.py --data_path portfolio.csv --target close --group_columns symbol --epochs 100

# Crypto (7-day week)
python daily_stock_forecasting/main.py --use_sample_data --target close --asset_type crypto --epochs 50

# ⭐ NEW: With custom pooling strategy
python daily_stock_forecasting/main.py --use_sample_data --target close --pooling_type temporal_multihead_attention --epochs 50

# ⭐ NEW: Experiment with different pooling
python daily_stock_forecasting/main.py --use_sample_data --target close --pooling_type weighted_avg --epochs 50

# 🎯 NEW: Return Forecasting Mode (predicts future returns using technical indicators)
python daily_stock_forecasting/main.py --use_sample_data --use_return_forecasting --epochs 50

# Return forecasting with custom horizons
python daily_stock_forecasting/main.py --use_sample_data --use_return_forecasting --return_horizons "1,3,5,10" --epochs 50

# Return forecasting for multi-symbol portfolio (per-symbol indicators)
python daily_stock_forecasting/main.py --data_path portfolio.csv --use_return_forecasting --group_columns symbol --epochs 100
```

### Intraday Forecasting

```bash
# US market 5-minute
python intraday_forecasting/main.py --use_sample_data --timeframe 5min --country US --epochs 30

# Multi-target intraday
python intraday_forecasting/main.py --use_sample_data --target "close,volume" --timeframe 5min --epochs 30

# Multi-symbol portfolio
python intraday_forecasting/main.py --data_path multi_symbol.csv --group_columns symbol --timeframe 5min --epochs 50

# Crypto 24/7 trading
python intraday_forecasting/main.py --use_sample_data --timeframe 1h --country CRYPTO --epochs 30
```

---

## Complete Parameter Reference

### Daily Stock Forecasting

```bash
python daily_stock_forecasting/main.py \
  --data_path /path/to/data.csv \           # Or use --use_sample_data
  --target "close,volume" \                  # Comma-separated targets
  --asset_type stock \                       # stock or crypto
  --sequence_length 50 \                     # Historical days
  --prediction_horizon 3 \                   # Steps ahead to predict
  --model_type ft_transformer \              # ft_transformer or csn_transformer
  --pooling_type multihead_attention \       # ⭐ NEW: Pooling strategy (default)
  --group_columns symbol \                   # For multi-asset portfolios
  --categorical_columns symbol \             # Categorical features
  --scaler_type standard \                   # standard, minmax, robust, maxabs, onlymax
  --use_lagged_target_features \             # Include targets in sequences
  --use_return_forecasting \                 # ⭐ NEW: Enable return forecasting mode
  --return_horizons "1,2,3,4,5" \            # ⭐ NEW: Return horizons (days)
  --d_token 128 \                            # Embedding dimension
  --n_layers 3 \                             # Transformer layers
  --n_heads 8 \                              # Attention heads
  --dropout 0.1 \                            # Dropout rate
  --epochs 100 \                             # Training epochs
  --batch_size 32 \                          # Batch size
  --learning_rate 0.001 \                    # Learning rate
  --patience 15 \                            # Early stopping patience
  --test_size 30 \                           # Test samples
  --val_size 20 \                            # Validation samples
  --per_group_metrics \                      # Show per-group metrics
  --model_path outputs/models/model.pt \     # Save path
  --no_plots \                               # Skip plots
  --verbose                                  # Detailed output
```

### Intraday Forecasting

```bash
python intraday_forecasting/main.py \
  --data_path /path/to/data.csv \           # Or use --use_sample_data
  --target "close,volume" \                  # Comma-separated targets
  --timeframe 5min \                         # 1min, 5min, 15min, 1h
  --country US \                             # US, INDIA, CRYPTO
  --sequence_length 60 \                     # Historical periods
  --prediction_horizon 5 \                   # Steps ahead to predict
  --model_type ft_transformer_cls \          # ft_transformer_cls or csn_transformer_cls
  --group_columns symbol \                   # For multi-symbol data
  --categorical_columns symbol \             # Categorical features
  --scaler_type standard \                   # Scaler type
  --use_lagged_target_features \             # Include targets in sequences
  --d_model 128 \                            # Embedding dimension
  --num_layers 3 \                           # Transformer layers
  --num_heads 8 \                            # Attention heads
  --dropout 0.1 \                            # Dropout rate
  --epochs 50 \                              # Training epochs
  --batch_size 32 \                          # Batch size
  --learning_rate 0.001 \                    # Learning rate
  --patience 10 \                            # Early stopping patience
  --test_size 200 \                          # Test samples
  --val_size 100 \                           # Validation samples
  --future_predictions 10 \                  # Future periods to predict
  --sample_days 10 \                         # For --use_sample_data
  --per_group_metrics \                      # Show per-group metrics
  --model_path outputs/models/model.pt \     # Save path
  --no_plots \                               # Skip plots
  --verbose                                  # Detailed output
```

---

## Parameter Descriptions

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data_path` | str | None | Path to CSV with OHLCV data (required unless `--use_sample_data`) |
| `--target` | str | "close" | Target column(s). Multi-target: `"close,volume"` (use quotes) |
| `--use_sample_data` | flag | False | Use synthetic data for testing |
| `--prediction_horizon` | int | 1 | Steps ahead to predict (1=single, >1=multi-horizon) |
| `--sequence_length` | int | 5/auto | Historical periods for input |

### Scaling & Grouping

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--group_columns` | str | None | Group-based scaling (e.g., `symbol` for multi-asset) |
| `--categorical_columns` | str | None | Categorical features to encode |
| `--scaler_type` | str | "standard" | `standard`, `minmax`, `robust`, `maxabs`, `onlymax` |
| `--use_lagged_target_features` | flag | False | Include target in input sequences |
| `--use_return_forecasting` | flag | False | Enable return forecasting mode (predicts forward returns) |
| `--return_horizons` | str | "1,2,3,4,5" | Return horizons for return forecasting (comma-separated days) |
| `--per_group_metrics` | flag | False | Show per-group evaluation metrics |

### Model Architecture

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--model_type` | str | "ft_transformer_cls" | - | `ft_transformer_cls` or `csn_transformer_cls` |
| `--d_model` | int | 128 | 64-512 | Token embedding dimension |
| `--num_layers` | int | 3 | 2-8 | Number of transformer layers |
| `--num_heads` | int | 8 | 4-16 | Number of attention heads |
| `--dropout` | float | 0.1 | 0.05-0.3 | Dropout rate |

### Training

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--epochs` | int | 50/100 | 50-300 | Training epochs |
| `--batch_size` | int | 32 | 16-128 | Batch size |
| `--learning_rate` | float | 0.001 | 0.0001-0.01 | Learning rate |
| `--patience` | int | 10/15 | 10-30 | Early stopping patience |

### Data Splits

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--test_size` | int | 30/200 | Test set size (absolute samples) |
| `--val_size` | int | 20/100 | Validation set size (absolute samples) |

### Asset/Market Specific

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `--asset_type` | str | "stock" | stock, crypto | Daily only: trading schedule |
| `--timeframe` | str | "5min" | 1min, 5min, 15min, 1h | Intraday only: timeframe |
| `--country` | str | "US" | US, INDIA, CRYPTO | Intraday only: market type |
| `--future_predictions` | int | 0 | 0-N | Intraday only: future periods |
| `--sample_days` | int | 5 | 1-30 | For `--use_sample_data` |

### Output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_path` | str | outputs/models/*.pt | Path to save trained model |
| `--no_plots` | flag | False | Skip plot generation |
| `--verbose` | flag | True | Detailed output |
| `--quiet` | flag | False | Disable verbose (overrides `--verbose`) |

---

## New Features in v2.0.0

### Per-Horizon Scaling

Each prediction horizon now has its own scaler for optimal calibration:

```bash
# With prediction_horizon=3, creates:
# - close_target_h1 → StandardScaler #1
# - close_target_h2 → StandardScaler #2
# - close_target_h3 → StandardScaler #3

python daily_stock_forecasting/main.py \
  --use_sample_data \
  --target close \
  --prediction_horizon 3 \
  --epochs 50
```

**Benefits:**
- Better accuracy per horizon
- Improved MAPE metrics
- Optimal scaling for each time step

### Automatic Cyclical Encoding

Temporal features automatically encoded as sin/cos pairs:

**Created features:**
- `month_sin`, `month_cos`, `day_sin`, `day_cos`
- `dayofweek_sin`, `dayofweek_cos`, `is_weekend`
- Intraday: `hour_sin`, `hour_cos`, `minute_sin`, `minute_cos`

**Removed features:**
- `year`, `month`, `day`, `quarter`, `dayofweek`
- Intraday: `hour`, `minute`

No configuration needed - happens automatically with date columns.

### Evaluation Improvements

Per-horizon metrics now available:

```bash
python daily_stock_forecasting/main.py \
  --use_sample_data \
  --target close \
  --prediction_horizon 3 \
  --epochs 50

# Output includes:
# - Overall: MAE, MAPE, RMSE, R2, Directional_Accuracy
# - horizon_1: metrics for 1-step ahead
# - horizon_2: metrics for 2-steps ahead
# - horizon_3: metrics for 3-steps ahead
```

---

## Troubleshooting

### High MAPE (>20%)

Try these adjustments:
```bash
# More training data and epochs
--epochs 100

# Different scaler
--scaler_type minmax

# More context
--sequence_length 30

# Adjust model size
--d_token 256 --n_layers 4
```

### Out of Memory

Reduce resource usage:
```bash
--batch_size 16
--d_token 64
--n_layers 2
--sequence_length 10
--pooling_type cls  # Fewer parameters than multihead
```

### Poor Multi-Horizon Performance

Improve multi-horizon predictions:
```bash
--prediction_horizon 3
--epochs 100
--sequence_length 30
--per_group_metrics  # For portfolios
```

---

## 🎯 Pooling Strategies (v2.1.0)

Choose from 5 pooling methods to aggregate transformer sequences:

### Available Pooling Types

```bash
# Default: Multi-head attention pooling (best overall performance)
--pooling_type multihead_attention

# Single-head attention (simpler, fewer parameters)
--pooling_type singlehead_attention

# Temporal multi-head (emphasizes recent timesteps)
--pooling_type temporal_multihead_attention

# Weighted average (simplest learnable pooling)
--pooling_type weighted_avg

# CLS token (legacy, for comparison)
--pooling_type cls
```

### Example: Testing Different Pooling Strategies

```bash
# Test all pooling strategies
for pooling in cls singlehead_attention multihead_attention weighted_avg temporal_multihead_attention; do
  echo "Testing pooling: $pooling"
  python daily_stock_forecasting/main.py \
    --use_sample_data \
    --target close \
    --pooling_type $pooling \
    --epochs 50 \
    --model_path "outputs/model_${pooling}.pt"
done
```

### Pooling Strategy Selection Guide

| Pooling Type | Use When | Parameters | Speed |
|--------------|----------|------------|-------|
| `multihead_attention` ⭐ | **Default** - Best overall | ~3×d_token² | Medium |
| `singlehead_attention` | Smaller models, faster inference | ~3×d_token² | Fast |
| `temporal_multihead_attention` | Strong trends, recency matters | ~3×d_token² + bias | Medium |
| `weighted_avg` | Simplest learnable, fast | max_seq_len | Fastest |
| `cls` | Legacy comparison, baseline | 0 | Fastest |

**Recommendation**: Start with default `multihead_attention`, then experiment with `temporal_multihead_attention` for time series with strong recent patterns.

---

## 🎯 Return Forecasting Mode (NEW)

Return forecasting mode predicts future percentage returns instead of raw prices, using technical indicators as features.

### What is Return Forecasting?

Instead of predicting future prices, return forecasting predicts **holding period returns** at multiple horizons:
- `return_1d`: 1-day forward return (%)
- `return_2d`: 2-day forward return (%)
- `return_3d`: 3-day forward return (%)
- etc.

### Automatic Feature Engineering

When enabled, return forecasting automatically calculates:

**Input Features:**
- `close`: Closing price
- `relative_volume`: Volume / 20-day MA
- `intraday_momentum`: (Close - Open) / Open
- `rsi_14`: Relative Strength Index (14 periods)
- `bb_position`: Position within Bollinger Bands

**Target Features:**
- `return_1d`, `return_2d`, `return_3d`, `return_4d`, `return_5d` (customizable)

### Basic Usage

```bash
# Enable return forecasting (uses default horizons: 1,2,3,4,5 days)
python daily_stock_forecasting/main.py \
  --use_sample_data \
  --use_return_forecasting \
  --epochs 100

# Custom return horizons
python daily_stock_forecasting/main.py \
  --use_sample_data \
  --use_return_forecasting \
  --return_horizons "1,3,5,10,20" \
  --epochs 100
```

### Multi-Symbol Return Forecasting

**IMPORTANT**: For multi-symbol portfolios, always use `--group_columns` to calculate technical indicators separately for each symbol:

```bash
# Correct: Per-symbol indicators (recommended)
python daily_stock_forecasting/main.py \
  --data_path portfolio.csv \
  --use_return_forecasting \
  --group_columns symbol \
  --categorical_columns symbol \
  --epochs 100

# Wrong: Mixed indicators across symbols
python daily_stock_forecasting/main.py \
  --data_path portfolio.csv \
  --use_return_forecasting \
  --epochs 100  # ❌ RSI/BB will be contaminated across symbols
```

### Benefits of Return Forecasting

1. **Better Stationarity**: Returns are more stationary than prices
2. **Direct Trading Signals**: Predicts actual holding period returns
3. **Multi-Horizon**: Predict returns for different investment horizons
4. **Automatic Feature Engineering**: Technical indicators calculated automatically
5. **Per-Symbol Features**: Indicators calculated separately for each stock (when using `--group_columns`)

### Example Output

```
Test Metrics:
  return_1d:
    - MAE: 1.2345
    - RMSE: 1.8901
    - MAPE: 45.67%
    - R2: 0.1234
    - Directional_Accuracy: 58.3%

  return_5d:
    - MAE: 3.4567
    - RMSE: 4.5678
    - MAPE: 89.12%
    - R2: 0.0567
    - Directional_Accuracy: 55.2%
```

### When to Use Return Forecasting

✅ **Use when:**
- Building trading signals
- Predicting holding period returns
- Working with multi-symbol portfolios
- Need stationary targets

❌ **Don't use when:**
- Need absolute price predictions
- Backtesting requires exact prices
- Working with derivatives that need price levels

---

## Additional Resources

- **Pooling Documentation:** `POOLING_VERIFICATION_RESULTS.md`
- **Quick Reference:** `PIPELINE_QUICK_REFERENCE.md`
- **Full Details:** `PIPELINE_REFACTORING_SUMMARY.md`
- **Changelog:** `CHANGELOG.md`
- **Architecture:** `tf_predictor/ARCHITECTURE.md`
- **Testing:** `test_pooling_end_to_end.py`

---

**Last Updated:** 2025-11-07 (v2.1.0)
