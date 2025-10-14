# Group-Based Scaling Implementation Summary

## Overview
Implemented group-based scaling functionality for the stock forecasting system, allowing independent scaling of features and targets for each group (e.g., per stock symbol) while training a unified model.

## Key Features

### 1. **Group-Based Scaling**
- Each unique value in the group column gets its own StandardScaler
- Features and targets scaled independently per group
- Sequences created separately per group, then concatenated for unified model training
- Backward compatible: when `group_column=None`, uses original single-scaler behavior

### 2. **Automatic Temporal Ordering**
- **Automatically sorts data by group and time** before processing
- Detects time columns: `timestamp`, `date`, `datetime`, `time` (case-insensitive)
- Ensures temporal order is maintained within each group
- Works even with shuffled/unsorted input data
- Falls back gracefully when no time column is detected

### 3. **Multi-Horizon Support**
- Uses **ONE scaler per group** (not separate scalers per horizon)
- All horizons within a group share the same target scaler
- Simplifies inverse transforms during prediction

### 4. **Prediction with Group Awareness**
- Tracks which group each sequence belongs to
- Applies correct group-specific inverse transform during prediction
- Handles both single-horizon and multi-horizon predictions

### 5. **Persistence**
- `save()` method persists group scalers
- `load()` method restores group scalers
- Models can be saved and loaded without losing group information

## Implementation Details

### Modified Files

#### **Core Implementation**
- **`tf_predictor/core/predictor.py`**:
  - Added `group_column` parameter to `__init__`
  - Added `group_feature_scalers` and `group_target_scalers` dictionaries
  - Implemented `_scale_features_single()` - original single-scaler logic
  - Implemented `_scale_features_grouped()` - group-based feature scaling
  - Implemented `_prepare_data_grouped()` - group-based sequence creation and target scaling
  - Modified `prepare_features()` to automatically sort by group+time and route to appropriate scaler
  - Modified `prepare_data()` to route between single/grouped methods
  - Modified `predict()` to handle group-based inverse transforms
  - Updated `save()` and `load()` methods

#### **Domain Predictors**
- **`intraday_forecasting/predictor.py`**:
  - Added `group_column` parameter to `__init__`
  - Passes to both FT-Transformer and CSN-Transformer base classes

- **`daily_stock_forecasting/predictor.py`**:
  - Added `group_column` parameter to `__init__`
  - Passes to TimeSeriesPredictor base class

#### **Main Scripts**
- **`intraday_forecasting/main.py`**:
  - Added `--group_column` command-line argument
  - Displays group-based scaling status in model configuration

- **`daily_stock_forecasting/main.py`**:
  - Added `--group_column` command-line argument
  - Displays group-based scaling status in model configuration

### Test Coverage

#### **test_group_scaling.py**
- ✅ Single-group scaling (no group_column)
- ✅ Multi-group scaling with group_column
- ✅ Grouped data preparation (sequences + targets)
- ✅ Scaling consistency (fit vs transform)
- ✅ Single-horizon and multi-horizon support

#### **test_temporal_order.py**
- ✅ Temporal order maintained with interleaved groups
- ✅ Boolean masking preserves DataFrame order

#### **test_auto_sort.py**
- ✅ Automatic sorting fixes unsorted input data
- ✅ Works correctly without timestamp column

## Usage

### Command-Line Usage

**Intraday forecasting with group-based scaling:**
```bash
python intraday_forecasting/main.py \
  --data_path data/multi_symbol_5min.csv \
  --group_column symbol \
  --timeframe 5min \
  --epochs 50
```

**Stock forecasting with group-based scaling:**
```bash
python daily_stock_forecasting/main.py \
  --data_path data/multi_stock_daily.csv \
  --group_column symbol \
  --sequence_length 10 \
  --epochs 50
```

**Without group-based scaling (default):**
```bash
python intraday_forecasting/main.py --data_path data/single_symbol.csv
# No --group_column = single-group scaling (backward compatible)
```

### Programmatic Usage

```python
from intraday_forecasting import IntradayPredictor

# Multi-stock dataset with group-based scaling
predictor = IntradayPredictor(
    target_column='close',
    timeframe='5min',
    group_column='symbol',  # Each symbol gets its own scaler
    sequence_length=20,
    prediction_horizon=3
)

# Data will be automatically sorted by symbol and timestamp
# No need to pre-sort!
predictor.fit(train_df)  # train_df has 'symbol' and 'timestamp' columns

# Predict on new data
predictions = predictor.predict(test_df)

# Save/load with group scalers
predictor.save('model.pt')
loaded_predictor = IntradayPredictor.load('model.pt')
```

## Technical Details

### Automatic Sorting Logic

When `group_column` is specified:
1. **Before feature creation**: Detects time column (`timestamp`, `date`, etc.)
2. **Sorts by**: `[group_column, time_column]`
3. **Ensures**: Temporal order maintained within each group
4. **Handles**: Unsorted/shuffled input data automatically

### Scaling Flow

```
Input Data (unsorted)
    ↓
[1] prepare_features() → Automatic sort by group+time
    ↓
[2] create_features() → Domain-specific feature engineering
    ↓
[3] _scale_features_grouped() → Group-based feature scaling
    ↓
[4] prepare_data() → Routes to _prepare_data_grouped()
    ↓
[5] _prepare_data_grouped() → Creates sequences per group
    ↓                          Scales targets per group
    ↓                          Concatenates all groups
    ↓
Output: (X, y) tensors ready for training
```

### Memory Efficiency

- Features processed and cached per dataset hash
- Group scalers stored as dictionary (minimal overhead)
- Sequences concatenated efficiently with `np.vstack()`
- No data duplication during group processing

## Backward Compatibility

✅ **Fully backward compatible**:
- Default `group_column=None` uses original single-scaler behavior
- All existing code continues to work without modifications
- Tests confirm original functionality preserved

## Performance Characteristics

- **Training**: Slightly slower due to per-group processing
- **Prediction**: Minimal overhead (group index lookup)
- **Memory**: ~O(n_groups × feature_dim) for scalers
- **Accuracy**: Expected improvement for multi-stock/symbol datasets

## Requirements for Users

### Data Format Requirements

1. **Must have a group column** (e.g., 'symbol', 'stock_id')
2. **Must have a time column** (e.g., 'timestamp', 'date')
3. **Data will be auto-sorted** - no pre-sorting required!

### Example Data Structure

```python
# Intraday data
df = pd.DataFrame({
    'symbol': ['AAPL', 'GOOGL', 'AAPL', 'GOOGL', ...],
    'timestamp': ['2024-01-01 09:30', '2024-01-01 09:30', '2024-01-01 09:35', ...],
    'open': [150.0, 2800.0, 150.5, ...],
    'high': [151.0, 2810.0, 151.5, ...],
    'low': [149.5, 2795.0, 150.0, ...],
    'close': [150.5, 2805.0, 151.0, ...],
    'volume': [1000000, 500000, 1100000, ...]
})

# Data can be in ANY order - will be auto-sorted!
```

## Future Enhancements

Potential improvements:
- [ ] Add validation warnings for insufficient group data
- [ ] Support for hierarchical groups (e.g., sector → symbol)
- [ ] Configurable time column name
- [ ] Group-aware train/test splitting
- [ ] Per-group model performance metrics

## Testing

All tests passing:
```bash
python test_group_scaling.py      # Core group scaling tests
python test_temporal_order.py     # Temporal order verification
python test_auto_sort.py           # Automatic sorting tests
```

## Summary

This implementation provides a robust, production-ready solution for multi-stock/symbol forecasting with:
- ✅ Automatic temporal ordering
- ✅ Group-based feature and target scaling
- ✅ Unified model training across all groups
- ✅ Group-aware predictions with correct inverse transforms
- ✅ Full backward compatibility
- ✅ Comprehensive test coverage

The system now handles real-world multi-asset datasets where each asset (stock, cryptocurrency, etc.) may have different price ranges and volatility characteristics, while still benefiting from a unified model trained on all assets simultaneously.
