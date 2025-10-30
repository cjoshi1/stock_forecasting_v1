# Group-Based + Multi-Horizon Implementation Summary

## Overview

This document summarizes the implementation of **per-symbol, per-horizon metrics** and **CSV exports** for the stock forecasting system. This enhancement allows users to get detailed performance breakdowns when using multiple symbols (group-based scaling) combined with multi-horizon predictions.

## Key Features Implemented

### 1. **Predict with Group Information** (`predictor.py`)

**Function**: `predict(df, return_group_info=False)`

- Added `return_group_info` parameter to `predict()` method
- When `return_group_info=True` and `group_column` is set, returns tuple: `(predictions, group_indices)`
- `group_indices` is a list mapping each prediction to its symbol/group
- Backward compatible - defaults to returning only predictions

**Example**:
```python
predictions, groups = model.predict(test_df, return_group_info=True)
# predictions shape: (n_samples, horizons) for multi-horizon
# groups: ['AAPL', 'AAPL', 'GOOGL', 'GOOGL', ...]
```

### 2. **Per-Group, Per-Horizon Evaluation** (`predictor.py`)

**Function**: `evaluate(df, per_group=False)`

- Added `per_group` parameter to `evaluate()` method
- When `per_group=True` and `group_column` is set, returns nested metrics dictionary
- Provides separate metrics for each symbol and each horizon
- Also includes overall (combined) metrics

**Return Structure**:
```python
{
    'overall': {
        'overall': {'MAE': ..., 'RMSE': ..., 'MAPE': ..., 'R2': ...},
        'horizon_1': {'MAE': ..., 'RMSE': ..., ...},
        'horizon_2': {'MAE': ..., 'RMSE': ..., ...},
        ...
    },
    'AAPL': {
        'overall': {'MAE': ..., 'RMSE': ..., 'MAPE': ..., 'R2': ...},
        'horizon_1': {'MAE': ..., 'RMSE': ..., ...},
        'horizon_2': {'MAE': ..., 'RMSE': ..., ...},
        ...
    },
    'GOOGL': {
        ...
    },
    ...
}
```

**Example**:
```python
test_metrics = model.evaluate(test_df, per_group=True)

# Access overall metrics
overall_mae = test_metrics['overall']['overall']['MAE']

# Access per-symbol metrics
aapl_h1_mape = test_metrics['AAPL']['horizon_1']['MAPE']
googl_h2_rmse = test_metrics['GOOGL']['horizon_2']['RMSE']
```

### 3. **CSV Export with Symbol Column** (`stock_charts.py`)

**Function**: `export_predictions_csv(model, train_df, test_df, output_dir)`

- Updated to include `symbol` column (or whatever `group_column` is named)
- Automatically detects if model uses groups
- Exports predictions with symbol mapping preserved

**CSV Format** (Single-Horizon):
```csv
date,symbol,actual,predicted,dataset,error_abs,error_pct
2024-01-15,AAPL,150.25,150.30,train,0.05,0.03
2024-01-15,GOOGL,2805.50,2810.20,train,4.70,0.17
...
```

**CSV Format** (Multi-Horizon):
```csv
date,symbol,actual,pred_h1,pred_h2,pred_h3,error_h1,mape_h1,error_h2,mape_h2,...,dataset
2024-01-15,AAPL,150.25,150.30,150.45,150.60,0.05,0.03,0.20,0.13,...,train
2024-01-15,GOOGL,2805.50,2810.20,2815.00,2820.50,4.70,0.17,9.50,0.34,...,train
...
```

### 4. **Per-Symbol Metrics Display** (`stock_charts.py`)

**Function**: `print_performance_summary(model, train_metrics, test_metrics, saved_plots)`

- Enhanced to display per-symbol metrics tables
- Automatically detects if metrics have per-group breakdown
- Shows overall metrics plus per-symbol breakdown

**Display Format** (Multi-Symbol, Multi-Horizon):
```
📊 Per-Symbol Performance Summary:
Found 3 symbols: AAPL, GOOGL, MSFT

📈 Overall Training Metrics (All Symbols Combined):
┌────────────┬───────────┬────────────┬───────────┬──────────┐
│ Horizon    │ MAE ($)   │ RMSE ($)   │ MAPE (%)  │ R²       │
├────────────┼───────────┼────────────┼───────────┼──────────┤
│ h1 (t+1)   │ $2.45     │ $3.20      │ 1.25%     │ 0.950    │
│ h2 (t+2)   │ $3.10     │ $4.15      │ 1.60%     │ 0.935    │
│ h3 (t+3)   │ $3.75     │ $4.90      │ 1.95%     │ 0.920    │
├────────────┼───────────┼────────────┼───────────┼──────────┤
│ Overall    │ $3.10     │ $4.08      │ 1.60%     │ 0.935    │
└────────────┴───────────┴────────────┴───────────┴──────────┘

📈 Training MAPE by Symbol:
┌────────────┬──────────────────────────────────────────────────┐
│ Symbol     │ MAPE (%) per Horizon                             │
├────────────┼──────────────────────────────────────────────────┤
│ AAPL       │ h1:1.2% h2:1.5% h3:1.8%                          │
│ GOOGL      │ h1:1.3% h2:1.7% h3:2.1%                          │
│ MSFT       │ h1:1.2% h2:1.6% h3:1.9%                          │
└────────────┴──────────────────────────────────────────────────┘
```

## Code Changes

### Files Modified

1. **`tf_predictor/core/predictor.py`**
   - Added `Union` to imports
   - Modified `predict()` signature to support `return_group_info`
   - Modified `evaluate()` to support `per_group` parameter
   - Added `_evaluate_standard()` helper method
   - Added `_evaluate_per_group()` helper method for per-symbol evaluation

2. **`daily_stock_forecasting/visualization/stock_charts.py`**
   - Modified `export_predictions_csv()` to detect and include group column
   - Added `_print_per_group_metrics()` helper function
   - Modified `print_performance_summary()` to handle per-group metrics

### Backward Compatibility

All changes are **fully backward compatible**:

- `predict()` defaults to returning only predictions (existing behavior)
- `evaluate()` defaults to `per_group=False` (existing behavior)
- CSV export automatically adapts based on whether groups are present
- Metrics display automatically detects structure and formats accordingly

## Usage Examples

### Example 1: Multi-Symbol, Single-Horizon

```python
from daily_stock_forecasting import StockPredictor

# Initialize with group column
model = StockPredictor(
    target_column='close',
    sequence_length=20,
    group_column='symbol',  # Enable group-based scaling
    prediction_horizon=1,    # Single horizon
    d_token=192,
    n_layers=4
)

# Train
model.fit(train_df, val_df, epochs=150)

# Evaluate with per-symbol breakdown
test_metrics = model.evaluate(test_df, per_group=True)

print(f"AAPL MAPE: {test_metrics['AAPL']['MAPE']:.2f}%")
print(f"GOOGL MAPE: {test_metrics['GOOGL']['MAPE']:.2f}%")
```

### Example 2: Multi-Symbol, Multi-Horizon

```python
# Initialize with groups and multi-horizon
model = StockPredictor(
    target_column='close',
    sequence_length=20,
    group_column='symbol',
    prediction_horizon=3,    # 3 horizons
    d_token=192,
    n_layers=4
)

# Train
model.fit(train_df, val_df, epochs=150)

# Get predictions with symbol mapping
predictions, symbols = model.predict(test_df, return_group_info=True)

# Evaluate with full breakdown
test_metrics = model.evaluate(test_df, per_group=True)

# Access specific metrics
aapl_h1_mape = test_metrics['AAPL']['horizon_1']['MAPE']
aapl_h2_mape = test_metrics['AAPL']['horizon_2']['MAPE']
overall_mape = test_metrics['overall']['overall']['MAPE']
```

### Example 3: CSV Export with Symbols

```python
from daily_stock_forecasting.visualization.stock_charts import (
    create_comprehensive_plots,
    export_predictions_csv
)

# Export predictions to CSV (automatically includes symbol column)
csv_path = export_predictions_csv(model, train_df, test_df, 'outputs/data')

# Read and analyze
import pandas as pd
csv_df = pd.read_csv(csv_path)

# Filter by symbol
aapl_df = csv_df[csv_df['symbol'] == 'AAPL']
print(f"AAPL predictions: {len(aapl_df)} rows")

# Analyze by horizon
print(f"AAPL H1 MAPE: {aapl_df['mape_h1'].mean():.2f}%")
print(f"AAPL H2 MAPE: {aapl_df['mape_h2'].mean():.2f}%")
```

## Testing

### Test Files Created

1. **`test_group_multi_horizon.py`** - Comprehensive test with full training
2. **`test_group_multi_horizon_quick.py`** - Quick API verification test

### Test Coverage

The tests verify:
- ✅ `predict()` returns group indices when requested
- ✅ `evaluate(per_group=True)` returns correct nested structure
- ✅ Per-symbol metrics are calculated separately
- ✅ Per-horizon metrics are calculated separately
- ✅ CSV export includes symbol column
- ✅ CSV export includes all horizon prediction columns
- ✅ Symbol-to-prediction mapping is preserved throughout pipeline

## Known Limitations

1. **Performance**: Per-group evaluation with many symbols and horizons can be slow due to repeated metric calculations

2. **Pandas Warnings**: There are FutureWarnings about dtype assignment in `_scale_features_grouped()`. These are warnings (not errors) and will be fixed in a future update.

3. **Large Datasets**: With many symbols (>50) and horizons (>5), the metrics display becomes very large. Consider aggregating or filtering for display purposes.

## Future Enhancements

Potential improvements for future versions:

1. **Optimized per-group evaluation** - Cache predictions to avoid redundant calls
2. **Metrics aggregation options** - Average, best, worst symbols
3. **Symbol filtering in evaluate()** - Evaluate specific symbols only
4. **Parallel evaluation** - Evaluate groups in parallel for speed
5. **Export to multiple formats** - JSON, Excel with multiple sheets per symbol

## Command-Line Usage

### Daily Stock Forecasting

```bash
# Multi-symbol portfolio with multi-horizon
python daily_stock_forecasting/main.py \
  --data_path portfolio_data.csv \
  --group_column symbol \
  --target close \
  --prediction_horizon 3 \
  --sequence_length 20 \
  --epochs 150 \
  --batch_size 64 \
  --verbose
```

### Intraday Forecasting

```bash
# Multi-symbol intraday with single horizon
python intraday_forecasting/main.py \
  --data_path intraday_portfolio.csv \
  --group_column symbol \
  --target close \
  --timeframe 5min \
  --country US \
  --sequence_length 20 \
  --epochs 150 \
  --verbose
```

## Summary

This implementation provides comprehensive per-symbol, per-horizon metrics and organized CSV exports for portfolio forecasting scenarios. The nested metrics structure allows users to:

1. Understand overall portfolio performance
2. Identify which symbols are easier/harder to predict
3. See how prediction accuracy degrades across horizons
4. Export organized data for further analysis in Excel or other tools

All functionality is backward compatible and automatically adapts based on whether groups and/or multiple horizons are used.
