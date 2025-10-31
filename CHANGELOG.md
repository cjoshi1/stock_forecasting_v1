# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2025-10-31

### 🎉 Major Release: Pipeline Refactoring & Per-Horizon Scaling

This release includes a complete refactoring of the preprocessing pipeline with breaking changes but significant improvements in accuracy and maintainability.

### 🐛 Critical Bug Fixes

- **FIXED: 100% MAPE Bug** - Resolved dataframe misalignment issue during evaluation
  - Root cause: Predictions and actuals were extracted from different dataframes
  - Solution: Store processed dataframe at correct pipeline stage (after target shifting, before encoding/scaling)
  - Result: MAPE dropped from 100% to ~1.5% on test data

### ✨ New Features

- **Per-Horizon Target Scaling** - Each prediction horizon now gets its own scaler
  - `close_target_h1` → StandardScaler #1
  - `close_target_h2` → StandardScaler #2
  - `close_target_h3` → StandardScaler #3
  - Improves per-horizon prediction accuracy

- **Automatic Cyclical Encoding** - Temporal features now automatically encoded as sin/cos
  - Auto-detects datetime components (date vs datetime with time)
  - Creates: `month_sin/cos`, `day_sin/cos`, `dayofweek_sin/cos`, `is_weekend`
  - For datetime: `hour_sin/cos`, `minute_sin/cos`
  - Automatically drops original non-cyclical features

- **Storage Flag in prepare_data()** - New `store_for_evaluation` parameter
  - `fit()`: `prepare_data(df, fit_scaler=True, store_for_evaluation=False)`
  - `predict()`: `prepare_data(df, fit_scaler=False, store_for_evaluation=True)`
  - Enables proper evaluation alignment

### 🔄 Refactoring

- **Removed `prepare_features()` method** - Replaced with `_create_base_features()`
  - Clean separation: feature engineering vs transformations
  - More extensible for domain-specific predictors

- **New Pipeline Flow** (7 stages):
  1. `_create_base_features()` - Domain + time-series features
  2. `create_shifted_targets()` - Target shifting
  3. **Storage point** - Store unencoded/unscaled dataframe
  4. `_encode_categorical_features()` - Categorical encoding
  5. `_determine_numerical_columns()` - Feature detection
  6. `_scale_features_single/grouped()` - Per-horizon scaling
  7. `_create_sequences()` - Sequence creation

- **Simplified StockPredictor** - Now only adds vwap
  - Time-series features handled by base class
  - Overrides `_create_base_features()` instead of `prepare_features()`

### 💥 Breaking Changes

#### API Changes

1. **Removed method:** `prepare_features()`
   - **Before:** `df_features = predictor.prepare_features(df, fit_scaler=True)`
   - **After:** Override `_create_base_features(df)` instead (protected method)

2. **New parameter in prepare_data():**
   - **Before:** `prepare_data(df, fit_scaler)`
   - **After:** `prepare_data(df, fit_scaler, store_for_evaluation=False)`

3. **Scaler structure change:**
   - **Before:** `self.target_scaler` (single-target) or `self.target_scalers_dict` (multi-target)
   - **After:** `self.target_scalers_dict` (both modes, per-horizon)

4. **create_stock_features() signature:**
   - **Before:** `create_stock_features(df, target_column, prediction_horizon, asset_type, group_column)`
   - **After:** `create_stock_features(df, verbose=False)` - Only adds vwap

#### Behavior Changes

1. **Temporal features:** Always cyclically encoded with originals dropped
2. **Target scaling:** Per-horizon instead of single scaler
3. **Evaluation:** Uses stored processed dataframe instead of re-processing

### 🔧 Migration Guide

For basic users (no code changes needed):
```python
# This still works the same
predictor = StockPredictor(target_column='close', sequence_length=5)
predictor.fit(train_df, epochs=50)
predictions = predictor.predict(test_df)
metrics = predictor.evaluate(test_df)
```

For developers extending TimeSeriesPredictor:
```python
# OLD
class CustomPredictor(TimeSeriesPredictor):
    def prepare_features(self, df, fit_scaler):
        df = add_custom_features(df)
        return super().prepare_features(df, fit_scaler)

# NEW
class CustomPredictor(TimeSeriesPredictor):
    def _create_base_features(self, df):
        df = add_custom_features(df)
        return super()._create_base_features(df)
```

### 📈 Performance

- **Accuracy:** Improved per-horizon predictions due to independent scaling
- **Memory:** No significant change
- **Speed:** Negligible overhead from per-horizon scaling
- **MAPE:** Dropped from 100% to ~1.5% (bug fix)

### 📝 Documentation

- Added `PIPELINE_REFACTORING_SUMMARY.md` - Detailed refactoring documentation
- Updated `tf_predictor/README.md` - New features and usage
- Updated `tf_predictor/ARCHITECTURE.md` - New pipeline flow diagrams
- Updated `README.md` - Main project readme with new examples

### 🧪 Testing

- Updated `daily_stock_forecasting/tests/test_stock.py` - Tests for new API
- Added `test_pipeline_stages.py` - Comprehensive pipeline verification
- All core tests passing

### 📦 Files Modified

**Core:**
- `tf_predictor/core/predictor.py`
- `tf_predictor/preprocessing/time_features.py`

**Domain-Specific:**
- `daily_stock_forecasting/predictor.py`
- `daily_stock_forecasting/preprocessing/stock_features.py`

**Tests:**
- `daily_stock_forecasting/tests/test_stock.py`
- `test_pipeline_stages.py` (new)

**Documentation:**
- `PIPELINE_REFACTORING_SUMMARY.md` (new)
- `CHANGELOG.md` (new)
- `tf_predictor/README.md`
- `tf_predictor/ARCHITECTURE.md`
- `README.md`

---

## [1.x.x] - Previous Versions

See historical documentation in `docs/historical/` for previous changes:
- Multi-target implementation
- Multi-horizon enhancements
- Group-based scaling
- Memory optimizations
- Alignment fixes
- Dimension fixes
- Data leakage verification

---

## Version Numbering

This project follows [Semantic Versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions
- PATCH version for backwards-compatible bug fixes
