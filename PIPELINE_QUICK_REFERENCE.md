# Pipeline Quick Reference Guide

**Version:** 2.0.0
**Last Updated:** 2025-10-31

## 📋 Quick Reference Card

### Pipeline Stages

```
Stage 1: _create_base_features()
├─ Domain-specific features (overridable)
├─ Time-series features (automatic)
└─ Output: df with features, unsorted values

Stage 2: create_shifted_targets()
├─ Creates target_h1, target_h2, ..., target_hN
├─ Removes NaN rows
└─ Output: df with shifted targets

Stage 3: Storage (if store_for_evaluation=True)
├─ Stores UNENCODED categorical values
├─ Stores UNSCALED numerical values
└─ Used for evaluation alignment

Stage 4: _encode_categorical_features()
├─ Label encoding
└─ Output: df with encoded categoricals

Stage 5: _determine_numerical_columns()
├─ Auto-detect features
└─ Exclude: targets, categoricals, shifted targets

Stage 6: _scale_features_single/grouped()
├─ Feature scaling (one scaler)
├─ Target scaling (per-horizon)
└─ Output: df with scaled values

Stage 7: _create_sequences()
├─ Sliding window sequences
└─ Output: (X_tensor, y_tensor)
```

## 🎯 Common Tasks

### Extending TimeSeriesPredictor

```python
from tf_predictor.core.predictor import TimeSeriesPredictor

class MyPredictor(TimeSeriesPredictor):
    def _create_base_features(self, df):
        """Add domain-specific features only."""
        df = df.copy()

        # Add your features
        df['my_feature'] = df['value'] * 2

        # Fill NaN
        df = df.fillna(0)

        # Call parent for time-series features
        return super()._create_base_features(df)
```

### Training & Prediction

```python
# Initialize
predictor = MyPredictor(
    target_column='close',
    sequence_length=10,
    prediction_horizon=3,  # Multi-horizon
    model_type='ft_transformer_cls'
)

# Train
predictor.fit(train_df, epochs=50)

# Predict
predictions = predictor.predict(test_df)

# Evaluate
metrics = predictor.evaluate(test_df)
```

### Multi-Target Prediction

```python
predictor = MyPredictor(
    target_column=['close', 'volume'],  # List of targets
    sequence_length=10,
    prediction_horizon=3
)

# Predictions is a dict: {'close': array, 'volume': array}
predictions = predictor.predict(test_df)
```

### Group-Based Scaling

```python
predictor = MyPredictor(
    target_column='close',
    group_columns='symbol',  # Scale per symbol
    sequence_length=10
)

# Each symbol gets its own scaler
predictor.fit(train_df)
```

## 🔍 Understanding Scalers

### Per-Horizon Scalers

```python
# After fit(), check scalers:
predictor.target_scalers_dict
# Output:
# {
#   'close_target_h1': StandardScaler(),
#   'close_target_h2': StandardScaler(),
#   'close_target_h3': StandardScaler(),
# }
```

### Group + Per-Horizon Scalers

```python
predictor.group_target_scalers
# Output:
# {
#   'AAPL': {
#     'close_target_h1': StandardScaler(),
#     'close_target_h2': StandardScaler(),
#   },
#   'GOOGL': {
#     'close_target_h1': StandardScaler(),
#     'close_target_h2': StandardScaler(),
#   }
# }
```

## 📊 Cyclical Encoding

### Automatically Created Features

**For date columns:**
- `month_sin`, `month_cos`
- `day_sin`, `day_cos`
- `dayofweek_sin`, `dayofweek_cos`
- `is_weekend`

**For datetime columns (with time):**
- All date features above, plus:
- `hour_sin`, `hour_cos`
- `minute_sin`, `minute_cos`

**Automatically removed:**
- `year`, `month`, `day`, `quarter`, `dayofweek`, `hour`, `minute`

## 🐛 Debugging

### Check Stored Dataframe

```python
# After predict()
stored_df = predictor._last_processed_df

print(f"Rows: {len(stored_df)}")
print(f"Columns: {list(stored_df.columns)}")

# Should contain:
# - Unscaled numerical values
# - Unencoded categorical values
# - Shifted target columns (target_h1, target_h2, ...)
```

### Check Pipeline Flow

```python
# Stage 1: Base features
df_stage1 = predictor._create_base_features(df.copy())
print("After stage 1:", df_stage1.columns)

# Stage 2: Shifted targets
from tf_predictor.preprocessing.time_features import create_shifted_targets
df_stage2 = create_shifted_targets(
    df_stage1,
    target_column=['close'],
    prediction_horizon=3
)
print("After stage 2:", df_stage2.columns)
```

### Verify Evaluation Alignment

```python
# Make predictions
predictions = predictor.predict(test_df)

# Check alignment
stored_df = predictor._last_processed_df
print(f"Predictions shape: {predictions.shape}")
print(f"Stored df rows: {len(stored_df)}")
print(f"Shifted targets in stored df: {[col for col in stored_df.columns if '_target_h' in col]}")

# Should match!
```

## ⚠️ Common Pitfalls

### 1. Don't Use prepare_features()
```python
# ❌ WRONG (removed in v2.0)
df_features = predictor.prepare_features(df)

# ✅ CORRECT
df_features = predictor._create_base_features(df)
```

### 2. Don't Scale in _create_base_features()
```python
# ❌ WRONG
def _create_base_features(self, df):
    df['feature'] = df['value'] * 2
    df = self.scaler.fit_transform(df)  # DON'T SCALE HERE!
    return df

# ✅ CORRECT
def _create_base_features(self, df):
    df['feature'] = df['value'] * 2
    return df  # Scaling happens in stage 6
```

### 3. Don't Shift Targets in _create_base_features()
```python
# ❌ WRONG
def _create_base_features(self, df):
    df['feature'] = df['value'] * 2
    df['target_h1'] = df['target'].shift(-1)  # DON'T SHIFT HERE!
    return df

# ✅ CORRECT
def _create_base_features(self, df):
    df['feature'] = df['value'] * 2
    return df  # Target shifting happens in stage 2
```

### 4. Always Call Parent _create_base_features()
```python
# ❌ WRONG
def _create_base_features(self, df):
    df['feature'] = df['value'] * 2
    return df  # Missing time-series features!

# ✅ CORRECT
def _create_base_features(self, df):
    df['feature'] = df['value'] * 2
    return super()._create_base_features(df)  # Adds time features
```

## 🎓 Advanced Usage

### Custom Scaler Per Horizon

```python
# Currently all horizons use same scaler type
# Future enhancement: different scaler per horizon

# Current behavior:
predictor = MyPredictor(scaler_type='standard')
# All horizons use StandardScaler

# Future potential:
# predictor = MyPredictor(
#     scaler_config={
#         'h1': 'standard',
#         'h2': 'minmax',
#         'h3': 'robust'
#     }
# )
```

### Accessing Individual Scalers

```python
# Get scaler for specific horizon
h1_scaler = predictor.target_scalers_dict['close_target_h1']

# Manually inverse transform
predictions_h1_scaled = predictions[:, 0]
predictions_h1_original = h1_scaler.inverse_transform(
    predictions_h1_scaled.reshape(-1, 1)
).flatten()
```

## 📚 Related Documentation

- **Full Refactoring Details:** `PIPELINE_REFACTORING_SUMMARY.md`
- **Architecture Guide:** `tf_predictor/ARCHITECTURE.md`
- **Changelog:** `CHANGELOG.md`
- **API Reference:** `tf_predictor/README.md`

## 🔗 Key Files

- **Base Predictor:** `tf_predictor/core/predictor.py`
- **Time Features:** `tf_predictor/preprocessing/time_features.py`
- **Stock Predictor:** `daily_stock_forecasting/predictor.py`
- **Stock Features:** `daily_stock_forecasting/preprocessing/stock_features.py`
