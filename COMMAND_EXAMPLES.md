# Command Reference

**Version:** 2.0.0 (Updated: 2025-11-01)

> **📢 Version 2.0.0 Recent Updates:**
> - **Evaluation alignment fixes** - Correct actuals extraction from shifted target columns
> - **Sequence creation optimization** - 20% more training data utilized
> - **verbose parameter support** - Control verbosity at initialization
> - Per-horizon target scaling (each horizon gets its own scaler)
> - Automatic cyclical encoding for temporal features
> - Fixed 100% MAPE evaluation bug
>
> See `IMPLEMENTATION_SUMMARY.md` and `PIPELINE_REFACTORING_SUMMARY.md` for technical details.

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
  --model_type ft_transformer_cls \          # ft_transformer_cls or csn_transformer_cls
  --group_columns symbol \                   # For multi-asset portfolios
  --categorical_columns symbol \             # Categorical features
  --scaler_type standard \                   # standard, minmax, robust, maxabs, onlymax
  --use_lagged_target_features \             # Include targets in sequences
  --d_model 128 \                            # Embedding dimension
  --num_layers 3 \                           # Transformer layers
  --num_heads 8 \                            # Attention heads
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
--d_model 256 --num_layers 4
```

### Out of Memory

Reduce resource usage:
```bash
--batch_size 16
--d_model 64
--num_layers 2
--sequence_length 10
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

## Additional Resources

- **Quick Reference:** `PIPELINE_QUICK_REFERENCE.md`
- **Full Details:** `PIPELINE_REFACTORING_SUMMARY.md`
- **Changelog:** `CHANGELOG.md`
- **Architecture:** `tf_predictor/ARCHITECTURE.md`
- **Testing:** `test_pipeline_stages.py`

---

**Last Updated:** 2025-11-01 (v2.0.0)
