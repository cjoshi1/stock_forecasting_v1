# Array Alignment Fix for Visualization

## Problem
The visualization code was throwing an error:
```
Error creating visualizations: x and y must have same first dimension, but have shapes (7935,) and (7934,)
```

This was an off-by-one error (or off-by-N for multi-horizon) between predictions and actual values arrays.

## Root Cause

The issue was caused by the shifted target column creation in feature engineering:

### In `intraday_forecasting/preprocessing/intraday_features.py` (lines 93-107):
```python
if prediction_horizon == 1:
    # Single horizon: create one target column
    shifted_target_name = f"{target_column}_target_h1"
    df_processed[shifted_target_name] = df_processed[target_column].shift(-1)
    # Drop rows where target is NaN (last row)
    df_processed = df_processed.dropna(subset=[shifted_target_name])  # ← Removes 1 row
else:
    # Multi-horizon: create multiple target columns
    for h in range(1, prediction_horizon + 1):
        col_name = f"{target_column}_target_h{h}"
        df_processed[col_name] = df_processed[target_column].shift(-h)
        target_columns.append(col_name)
    # Drop rows where ANY target is NaN (last prediction_horizon rows)
    df_processed = df_processed.dropna(subset=target_columns)  # ← Removes N rows
```

The same pattern exists in `stock_forecasting/preprocessing/stock_features.py`.

### Data Flow:
1. **Original DataFrame**: N rows
2. **After `create_features()`**:
   - Single-horizon: N - 1 rows (drops last 1 row due to `shift(-1)` + `dropna()`)
   - Multi-horizon: N - horizon rows (drops last `horizon` rows)
3. **After `create_sequences()`**: Removes another `sequence_length` rows from the beginning
4. **Final predictions**: N - horizon - sequence_length samples

### But the visualization code was doing:
```python
# INCORRECT - only accounts for sequence_length, not the dropna()
train_actual = train_df[target].iloc[sequence_length:]  # N - sequence_length values
```

This created a mismatch:
- **Predictions**: N - horizon - sequence_length samples
- **Actuals**: N - sequence_length samples
- **Difference**: horizon samples (1 for single-horizon, N for multi-horizon)

## Solution

### For `intraday_forecasting/main.py` (lines 102-126):
```python
# Get timestamps for plotting - align with predictions
# create_features() does shift(-h) and dropna() which removes last prediction_horizon rows
# Then create_sequences() removes first sequence_length rows from the processed data
# So we need: iloc[sequence_length : -prediction_horizon] to match
horizon = predictor.prediction_horizon
if horizon == 1:
    # Single-horizon: drops last 1 row
    train_timestamps = train_df[predictor.timestamp_col].iloc[predictor.sequence_length:-1].values
    train_actual = train_df[predictor.target_column].iloc[predictor.sequence_length:-1].values
    # ... same for val and test
else:
    # Multi-horizon: drops last prediction_horizon rows
    train_timestamps = train_df[predictor.timestamp_col].iloc[predictor.sequence_length:-horizon].values
    train_actual = train_df[predictor.target_column].iloc[predictor.sequence_length:-horizon].values
    # ... same for val and test
```

### For `stock_forecasting/visualization/stock_charts.py` (lines 272-282):
```python
# Get corresponding actual values with proper alignment
# create_features() does shift(-h) and dropna() which removes last prediction_horizon rows
# Then create_sequences() removes first sequence_length rows
# So we need: [sequence_length : -prediction_horizon]
horizon = model.prediction_horizon
if horizon == 1:
    train_actual_base = train_df['close'].values[model.sequence_length:-1]
    test_actual_base = test_df['close'].values[model.sequence_length:-1]
else:
    train_actual_base = train_df['close'].values[model.sequence_length:-horizon]
    test_actual_base = test_df['close'].values[model.sequence_length:-horizon]
```

## Files Modified
1. `/Users/chinmay/code/stock_forecasting_v1/intraday_forecasting/main.py`
   - Lines 102-126: Fixed timestamp and actual value extraction
   - Lines 131-141: Removed unnecessary `min()` safeguards for single-horizon
   - Lines 196-223: Removed unnecessary `min()` safeguards for CSV generation

2. `/Users/chinmay/code/stock_forecasting_v1/stock_forecasting/visualization/stock_charts.py`
   - Lines 272-282: Fixed actual value extraction to account for dropped rows
   - Lines 289-294: Removed unnecessary `min()` safeguards for single-horizon

## Result
Arrays are now perfectly aligned:
- **Predictions**: N - horizon - sequence_length samples
- **Actuals**: N - horizon - sequence_length samples
- **Timestamps**: N - horizon - sequence_length samples

No more dimension mismatch errors! ✅
