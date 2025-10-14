# Quick Start Guide

## 🚀 Get Started in 30 Seconds

### Intraday Forecasting (5-minute bars)

**Single symbol:**
```bash
python intraday_forecasting/main.py \
  --data_path your_data.csv \
  --target close \
  --timeframe 5min \
  --epochs 100
```

**Multiple symbols with group-based scaling:**
```bash
python intraday_forecasting/main.py \
  --data_path multi_symbol_data.csv \
  --target close \
  --timeframe 5min \
  --group_column symbol \
  --epochs 100
```

### Daily Stock Forecasting

**Single stock:**
```bash
python daily_stock_forecasting/main.py \
  --data_path your_data.csv \
  --target close \
  --epochs 100
```

**Stock portfolio with group-based scaling:**
```bash
python daily_stock_forecasting/main.py \
  --data_path portfolio_data.csv \
  --target close \
  --group_column symbol \
  --epochs 100
```

---

## 📋 Data Format

### CSV Structure
```csv
timestamp/date,symbol,open,high,low,close,volume
2024-01-01 09:30:00,AAPL,150.0,151.0,149.5,150.5,1000000
2024-01-01 09:35:00,AAPL,150.5,151.5,150.0,151.0,1100000
```

**Required columns:**
- Time column: `timestamp` (intraday) or `date` (daily)
- OHLCV: `open`, `high`, `low`, `close`, `volume`
- Group column: `symbol` (for multi-asset datasets)

**✅ Data can be in ANY order** - it will be auto-sorted!

---

## ⚙️ Key Parameters

| Parameter | What it does | Typical values |
|-----------|--------------|----------------|
| `--data_path` | Path to your CSV file | `data/stocks.csv` |
| `--target` | What to predict | `close` (default) |
| `--group_column` | Enable group-based scaling | `symbol` |
| `--timeframe` | Intraday only | `1min`, `5min`, `15min`, `1h` |
| `--sequence_length` | Historical bars to use | 5-60 |
| `--epochs` | Training iterations | 50-200 |
| `--batch_size` | Samples per batch | 32-128 |

---

## 🎯 Common Use Cases

### 1. Test with Sample Data
```bash
python intraday_forecasting/main.py --use_sample_data --epochs 20
```

### 2. Quick Training (Minimal Settings)
```bash
python daily_stock_forecasting/main.py \
  --data_path data.csv \
  --epochs 50 \
  --no_plots
```

### 3. Production Training (Full Settings)
```bash
python intraday_forecasting/main.py \
  --data_path production_data.csv \
  --group_column symbol \
  --timeframe 5min \
  --sequence_length 30 \
  --d_token 256 \
  --n_layers 5 \
  --epochs 200 \
  --batch_size 128 \
  --model_path models/production.pt
```

### 4. Multi-Day Predictions
```bash
python daily_stock_forecasting/main.py \
  --data_path data.csv \
  --prediction_horizon 5 \
  --sequence_length 20 \
  --epochs 150
```

### 5. Cryptocurrency (24/7 Trading)
```bash
# Intraday
python intraday_forecasting/main.py \
  --data_path crypto_1min.csv \
  --country CRYPTO \
  --timeframe 1min \
  --group_column symbol

# Daily
python daily_stock_forecasting/main.py \
  --data_path crypto_daily.csv \
  --asset_type crypto \
  --group_column symbol
```

---

## 🔧 Performance Tips

**For faster training:**
- ⬇️ Reduce `--sequence_length` (e.g., 5-10)
- ⬇️ Reduce `--d_token` (e.g., 64-128)
- ⬆️ Increase `--batch_size` (e.g., 64-128)

**For better accuracy:**
- ⬆️ Increase `--sequence_length` (e.g., 20-60)
- ⬆️ Increase `--d_token` (e.g., 256-512)
- ⬆️ Increase `--epochs` (e.g., 150-300)
- 🎯 Always use `--group_column symbol` for multi-asset data

---

## 📊 What Gets Created

After training, you'll get:

1. **Trained model**: `outputs/models/*.pt`
2. **Plots** (unless `--no_plots`):
   - Training/validation loss curves
   - Predictions vs actuals
   - Feature importance
3. **Console output**:
   - Training progress
   - Performance metrics (RMSE, MAE, R²)
   - Model configuration

---

## 🆘 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| "CUDA out of memory" | Reduce `--batch_size` or `--d_token` |
| "No groups had sufficient data" | Reduce `--sequence_length` |
| "Model not improving" | Adjust `--learning_rate` (try 0.0005) |
| Training too slow | Increase `--batch_size`, reduce model size |

---

## 📚 More Information

- **Full command examples**: See `COMMAND_EXAMPLES.md`
- **Group scaling details**: See `GROUP_SCALING_SUMMARY.md`
- **API documentation**: See module READMEs

---

## ⚡ Ready to Go!

**Simplest possible command:**
```bash
python intraday_forecasting/main.py --use_sample_data
```

**Most common real-world usage:**
```bash
python intraday_forecasting/main.py \
  --data_path your_data.csv \
  --group_column symbol \
  --epochs 100
```

That's it! Your model will train and you'll see results. 🎉
