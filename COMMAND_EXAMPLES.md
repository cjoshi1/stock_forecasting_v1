# Complete Command Reference

## Comprehensive Commands

This section provides a complete command example for both intraday and daily forecasting, including every possible parameter. Explanations for each parameter are provided in the tables below.

### Intraday Forecasting - Full Command Example

```bash
python intraday_forecasting/main.py \
  --data_path /path/to/intraday_data.csv \
  --target "close,volume" \
  --timeframe 5min \
  --model_type ft \
  --country US \
  --use_sample_data \
  --sample_days 10 \
  --group_column symbol \
  --sequence_length 60 \
  --prediction_horizon 5 \
  --d_token 128 \
  --n_layers 3 \
  --n_heads 8 \
  --dropout 0.1 \
  --per_group_metrics \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --patience 10 \
  --test_size 200 \
  --val_size 100 \
  --future_predictions 10 \
  --model_path outputs/models/intraday_model.pt \
  --no_plots \
  --verbose \
  --quiet
```

### Daily Stock Forecasting - Full Command Example

```bash
python daily_stock_forecasting/main.py \
  --data_path /path/to/stock_data.csv \
  --target "close,volume" \
  --use_sample_data \
  --asset_type stock \
  --group_column symbol \
  --sequence_length 10 \
  --prediction_horizon 3 \
  --use_essential_only \
  --d_token 128 \
  --n_layers 3 \
  --n_heads 8 \
  --dropout 0.1 \
  --per_group_metrics \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --patience 15 \
  --test_size 30 \
  --val_size 20 \
  --model_path outputs/models/stock_model.pt \
  --no_plots
```

--- 

## Parameter Explanations

### Data Parameters

| Parameter | Type | Description | Default | Notes |
|-----------|------|-------------|---------|-------|
| `--data_path` | string | Path to CSV file with OHLCV data | None | Required unless using `--use_sample_data` |
| `--target` | string | Column(s) to predict | close | Single: `close` or Multi: `close,volume` (comma-separated) |
| `--group_column` | string | Column for group-based scaling | None | Use 'symbol' for multi-asset datasets |
| `--use_sample_data` | flag | Use synthetic sample data | False | For testing without real data |
| `--per_group_metrics` | flag | Show per-group evaluation metrics | False | Only with `--group_column` |

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
| `--use_essential_only` | flag | Use minimal features | False | True=7 features, False=30+ features |

### Model Architecture Parameters

| Parameter | Type | Description | Default | Recommended Range |
|-----------|------|-------------|---------|-------------------|
| `--sequence_length` | int | Historical periods for input | Auto/5 | 5-60 |
| `--prediction_horizon` | int | Steps ahead to predict | 1 | 1=single step, >1=multi-horizon |
| `--d_token` | int | Token embedding dimension | 128 | 64-512 |
| `--n_layers` | int | Number of transformer layers | 3 | 2-8 |
| `--n_heads` | int | Number of attention heads | 8 | 4-16 |
| `--dropout` | float | Dropout rate | 0.1 | 0.05-0.3 |

### Training Parameters

| Parameter | Type | Description | Default | Recommended Range |
|-----------|------|-------------|---------|-------------------|
| `--epochs` | int | Training epochs | 50/100 | 50-300 |
| `--batch_size` | int | Training batch size | 32 | 16-128 |
| `--learning_rate` | float | Learning rate | 0.001 | 0.0001-0.01 |
| `--patience` | int | Early stopping patience | 10/15 | 10-30 |

### Data Split Parameters

| Parameter | Type | Description | Default | Notes |
|-----------|------|-------------|---------|-------|
| `--test_size` | int | Test set size (samples) | 200/30 | Absolute number of samples |
| `--val_size` | int | Validation set size | 100/20 | Absolute number of samples |

### Output Parameters

| Parameter | Type | Description | Default | Notes |
|-----------|------|-------------|---------|-------|
| `--model_path` | string | Path to save trained model | outputs/models/*.pt | Creates directory if needed |
| `--no_plots` | flag | Skip plot generation | False | Useful for batch training |
| `--verbose` | flag | Enable verbose output | True | Shows detailed progress |
| `--quiet` | flag | Disable verbose output | False | Overrides `--verbose` |
