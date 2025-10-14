# Complete Command Reference

## Comprehensive Commands

### Intraday Forecasting - Full Command

```bash
python intraday_forecasting/main.py \
  --data_path /path/to/intraday_data.csv \
  --target close \
  --timeframe 5min \
  --model_type ft \
  --country US \
  --group_column symbol \
  --sequence_length 20 \
  --d_token 192 \
  --n_layers 4 \
  --n_heads 8 \
  --dropout 0.15 \
  --epochs 150 \
  --batch_size 128 \
  --learning_rate 0.0008 \
  --patience 20 \
  --test_size 500 \
  --val_size 200 \
  --future_predictions 12 \
  --model_path outputs/models/intraday_model.pt \
  --verbose
```

### Daily Stock Forecasting - Full Command

```bash
python daily_stock_forecasting/main.py \
  --data_path /path/to/stock_data.csv \
  --target close \
  --asset_type stock \
  --group_column symbol \
  --sequence_length 20 \
  --prediction_horizon 1 \
  --use_essential_only \
  --d_token 192 \
  --n_layers 4 \
  --n_heads 8 \
  --dropout 0.15 \
  --epochs 150 \
  --batch_size 64 \
  --learning_rate 0.0008 \
  --patience 20 \
  --test_size 200 \
  --val_size 100 \
  --model_path outputs/models/stock_model.pt
```

---

## Parameter Reference

### Data Parameters

| Parameter | Type | Description | Default | Notes |
|-----------|------|-------------|---------|-------|
| `--data_path` | string | Path to CSV file with OHLCV data | None | Required unless using `--use_sample_data` |
| `--target` | string | Column to predict | close | Options: close, open, high, low, volume |
| `--group_column` | string | Column for group-based scaling | None | Use 'symbol' for multi-asset datasets |
| `--use_sample_data` | flag | Use synthetic sample data | False | For testing without real data |

### Market-Specific Parameters (Intraday Only)

| Parameter | Type | Description | Default | Options |
|-----------|------|-------------|---------|---------|
| `--timeframe` | string | Trading timeframe | 5min | 1min, 5min, 15min, 1h |
| `--country` | string | Market type | US | US, INDIA, CRYPTO |
| `--model_type` | string | Model architecture | ft | ft (FT-Transformer), csn (CSN-Transformer) |
| `--sample_days` | int | Days for sample data | 5 | Used with `--use_sample_data` |
| `--future_predictions` | int | Future periods to predict | 0 | 0 = no future predictions |

### Asset-Specific Parameters (Daily Only)

| Parameter | Type | Description | Default | Options |
|-----------|------|-------------|---------|---------|
| `--asset_type` | string | Asset type | stock | stock (5-day week), crypto (7-day week) |
| `--prediction_horizon` | int | Steps ahead to predict | 1 | 1=single step, >1=multi-horizon |
| `--use_essential_only` | flag | Use minimal features | False | True=7 features, False=30+ features |

### Model Architecture Parameters

| Parameter | Type | Description | Default | Recommended Range |
|-----------|------|-------------|---------|-------------------|
| `--sequence_length` | int | Historical periods for input | Auto/5 | 5-60 |
| `--d_token` | int | Token embedding dimension | 128 | 64-512 |
| `--n_layers` | int | Number of transformer layers | 3 | 2-8 |
| `--n_heads` | int | Number of attention heads | 8 | 4-16 |
| `--dropout` | float | Dropout rate | 0.1 | 0.05-0.3 |

### Training Parameters

| Parameter | Type | Description | Default | Recommended Range |
|-----------|------|-------------|---------|-------------------|
| `--epochs` | int | Training epochs | 50 | 50-300 |
| `--batch_size` | int | Training batch size | 32 | 16-128 |
| `--learning_rate` | float | Learning rate | 0.001 | 0.0001-0.01 |
| `--patience` | int | Early stopping patience | 10 | 10-30 |

### Data Split Parameters

| Parameter | Type | Description | Default | Notes |
|-----------|------|-------------|---------|-------|
| `--test_size` | int | Test set size (samples) | 200/60 | Absolute number of samples |
| `--val_size` | int | Validation set size | 100/30 | Absolute number of samples |

### Output Parameters

| Parameter | Type | Description | Default | Notes |
|-----------|------|-------------|---------|-------|
| `--model_path` | string | Path to save trained model | outputs/models/*.pt | Creates directory if needed |
| `--no_plots` | flag | Skip plot generation | False | Useful for batch training |
| `--verbose` | flag | Enable verbose output | True | Shows detailed progress |
| `--quiet` | flag | Disable verbose output | False | Overrides `--verbose` |

---

## Parameter Explanations

### `--group_column` (⭐ New Feature)

Enables group-based scaling where each unique value in the specified column gets its own scaler for features and targets.

**Use when:**
- Training on multiple stocks/symbols in one dataset
- Different entities have different value ranges (e.g., AAPL ~$150 vs GOOGL ~$2800)
- You want a unified model that learns from all entities

**Benefits:**
- Better normalization per entity
- Unified model learns patterns across all entities
- Automatic temporal ordering within each group
- Prevents scale bias in predictions

**Example:**
```bash
--group_column symbol  # For multi-stock datasets with 'symbol' column
```

### `--sequence_length`

Number of historical time steps used as input for prediction.

**Guidelines:**
- **Intraday (1min)**: 30-60 bars = 30-60 minutes of history
- **Intraday (5min)**: 12-24 bars = 1-2 hours of history
- **Intraday (15min)**: 8-16 bars = 2-4 hours of history
- **Daily stocks**: 5-20 days of history
- **Crypto**: 7-30 days (24/7 trading)

**Trade-offs:**
- Longer = More context but slower training
- Shorter = Faster training but less context

### `--prediction_horizon` (Daily Only)

Number of steps ahead to predict simultaneously.

**Examples:**
- `1` = Predict tomorrow only (single-horizon)
- `3` = Predict next 3 days simultaneously (multi-horizon)
- `5` = Predict next week (multi-horizon)

**Note:** Multi-horizon uses separate scalers per horizon (daily) but same scaler per group (when using `--group_column`).

### `--d_token`, `--n_layers`, `--n_heads`

Control model capacity and complexity.

**Model Size Guidelines:**

| Use Case | d_token | n_layers | n_heads | Approx Parameters |
|----------|---------|----------|---------|-------------------|
| Small dataset (<1K samples) | 64-128 | 2-3 | 4-8 | ~50K-200K |
| Medium dataset (1K-10K) | 128-192 | 3-4 | 8 | ~200K-500K |
| Large dataset (>10K) | 256-512 | 4-8 | 8-16 | ~1M-8M |

### `--dropout`

Regularization to prevent overfitting.

**Guidelines:**
- Small dataset: 0.05-0.1 (less dropout)
- Large dataset: 0.2-0.3 (more dropout)
- Overfitting observed: Increase dropout
- Underfitting observed: Decrease dropout

### `--learning_rate`

Controls how fast the model learns.

**Common values to try:**
- `0.0005` - Conservative, stable training
- `0.001` - Default, balanced
- `0.002` - Aggressive, faster convergence

**Signs:**
- Loss exploding → Reduce learning rate
- Loss not improving → Try different learning rate
- Training too slow → Increase learning rate

### `--batch_size`

Number of samples processed together.

**Guidelines:**
- GPU available: 64-128 (utilize GPU memory)
- CPU only: 16-32 (avoid memory issues)
- Large model: Reduce batch_size
- Out of memory: Reduce batch_size

**Trade-offs:**
- Larger = Faster training, more memory
- Smaller = Slower training, less memory, sometimes better generalization

### `--use_essential_only` (Daily Only)

Choose between feature sets:

**Full features (default):**
- 30+ technical indicators
- Moving averages, RSI, MACD, Bollinger Bands
- Rolling statistics, momentum indicators
- Better accuracy but slower training

**Essential features:**
- Only 7-8 core features
- Volume, typical_price, basic seasonality
- Fast training, good for quick experiments

---

## Data Format Requirements

### CSV Format

Both intraday and daily forecasting expect CSV files with these columns:

**Intraday:**
```csv
timestamp,symbol,open,high,low,close,volume
2024-01-01 09:30:00,AAPL,150.0,151.0,149.5,150.5,1000000
2024-01-01 09:35:00,AAPL,150.5,151.5,150.0,151.0,1100000
2024-01-01 09:30:00,GOOGL,2800.0,2810.0,2795.0,2805.0,500000
```

**Daily:**
```csv
date,symbol,open,high,low,close,volume
2024-01-01,AAPL,150.0,151.0,149.5,150.5,50000000
2024-01-02,AAPL,150.5,152.0,150.0,151.5,52000000
2024-01-01,GOOGL,2800.0,2810.0,2795.0,2805.0,2000000
```

**Required columns:**
- Time: `timestamp` (intraday) or `date` (daily)
- OHLCV: `open`, `high`, `low`, `close`, `volume`
- Group: `symbol` (optional, required for `--group_column`)

**Important:** Data does NOT need to be pre-sorted! The system automatically sorts by `[symbol, timestamp/date]`.

---

## Quick Reference

### Minimal Command (Single Asset)
```bash
python intraday_forecasting/main.py --data_path data.csv --epochs 100
```

### Recommended Command (Multi-Asset Portfolio)
```bash
python intraday_forecasting/main.py \
  --data_path portfolio.csv \
  --group_column symbol \
  --timeframe 5min \
  --epochs 150 \
  --batch_size 128
```

### Fast Training (Testing)
```bash
python intraday_forecasting/main.py \
  --use_sample_data \
  --epochs 20 \
  --d_token 64 \
  --n_layers 2
```

### High Accuracy (Production)
```bash
python intraday_forecasting/main.py \
  --data_path production_data.csv \
  --group_column symbol \
  --sequence_length 30 \
  --d_token 256 \
  --n_layers 5 \
  --epochs 200 \
  --patience 25
```

---

## Troubleshooting

| Error/Issue | Solution |
|-------------|----------|
| "CUDA out of memory" | Reduce `--batch_size`, `--d_token`, or `--n_layers` |
| "No groups had sufficient data" | Reduce `--sequence_length` or filter dataset |
| Model not improving | Adjust `--learning_rate` (try 0.0005, 0.002) |
| Training too slow | Increase `--batch_size`, reduce `--sequence_length` |
| Overfitting (val loss increasing) | Increase `--dropout`, reduce model size |
| Underfitting (both losses high) | Increase model capacity or `--epochs` |

---

For more details:
- **Quick Start**: See `QUICK_START.md`
- **Group Scaling**: See `GROUP_SCALING_SUMMARY.md`
- **Main Documentation**: See `README.md`
