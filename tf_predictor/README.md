# 🧠 TF Predictor: Generic Time Series Forecasting Library

**Version:** 2.1.0 (Updated: 2025-11-05)

A reusable Python library for time series forecasting using the FT-Transformer (Feature Tokenizer Transformer) architecture. This library provides a clean, extensible foundation for building domain-specific time series prediction applications.

## 🎯 Features

### Version 2.1.0 Highlights ⭐ NEW
- **Multi-Column Grouping Support**: Full support for composite grouping (e.g., `['symbol', 'sector']`) with correct evaluation
- **Improved Cache Hashing**: Enhanced DataFrame hashing to prevent collisions
- **Explicit Group Ordering**: Deterministic group processing order for reproducible results
- **Bug Fixes**: Critical fix for per-group evaluation with multi-column grouping
- **Comprehensive Testing**: Added test suite for multi-column grouping scenarios

### Version 2.0.0 Highlights
- **Per-Horizon Target Scaling**: Each prediction horizon gets its own scaler for optimal accuracy
- **Automatic Cyclical Encoding**: Temporal features automatically encoded as sin/cos with original features dropped
- **Evaluation Alignment Fixed**: Proper dataframe storage resolves 100% MAPE bug
- **Clean Architecture**: 7-stage preprocessing pipeline with clear separation of concerns
- **Simplified API**: `_create_base_features()` replaces `prepare_features()` for cleaner inheritance

### Core Features
- **Multi-Target Prediction**: Predict multiple variables simultaneously with one unified model
- **Multi-Horizon Forecasting**: Predict 1 to N steps ahead with per-horizon scaling
- **Multi-Column Grouping**: Support for composite group keys (e.g., symbol + sector) ⭐ NEW
- **Generic Time Series Predictor**: Abstract base class for any time series domain
- **State-of-the-art Architecture**: FT-Transformer with attention mechanisms
- **Group-Based Scaling**: Independent scaling per group (e.g., per entity/symbol) while training unified models
- **Automatic Temporal Ordering**: Data automatically sorted to maintain correct time sequences
- **Flexible Feature Engineering**: Extensible preprocessing pipeline with clean separation
- **Sequence Modeling**: Built-in support for multi-step temporal sequences
- **Production Ready**: Model persistence, evaluation metrics, and utilities

## 🏗️ Architecture

The library follows a clean architectural pattern:
- **Abstract Predictor**: Base class that you extend for your domain
- **Core Models**: FT-Transformer implementations for tabular time series
- **Preprocessing Tools**: Generic time series feature engineering utilities
- **Evaluation Utilities**: Comprehensive metrics and data splitting functions

## 🚀 Quick Start

### 1. Create Your Custom Predictor (v2.1.0 API)

```python
from tf_predictor.core.predictor import TimeSeriesPredictor

class MyPredictor(TimeSeriesPredictor):
    def _create_base_features(self, df):
        """
        Override to add domain-specific features.
        Time-series features (cyclical encoding) are added automatically by parent.

        IMPORTANT (v2.0.0+):
        - Only add domain-specific features here (no scaling, no encoding)
        - Call super()._create_base_features() to add time-series features
        - Target shifting happens automatically in pipeline stage 2
        - Scaling happens automatically in pipeline stage 6
        """
        df_processed = df.copy()

        # Add your domain-specific features
        df_processed['custom_feature'] = df_processed['value'] * 2

        # Fill NaN values
        df_processed = df_processed.fillna(0)

        # Call parent to add time-series features (automatic cyclical encoding)
        return super()._create_base_features(df_processed)
```

### 2. Use Your Predictor

```python
from tf_predictor.core.utils import split_time_series

# Initialize your custom predictor
predictor = MyPredictor(
    target_column='value',
    sequence_length=7,
    prediction_horizon=1,  # Number of steps ahead (default: 1)
    model_type='ft_transformer_cls',  # 'ft_transformer_cls' or 'csn_transformer_cls'
    d_token=128,
    n_layers=3,
    n_heads=8
)

# Split your time series data
train_df, val_df, test_df = split_time_series(df, test_size=30, val_size=15)

# Train the model
predictor.fit(
    df=train_df,
    val_df=val_df,
    epochs=100,
    batch_size=32,
    learning_rate=1e-3
)

# Make predictions
predictions = predictor.predict(test_df)
metrics = predictor.evaluate(test_df)
```

### 3. Multi-Horizon Predictions

```python
# Predict multiple steps ahead simultaneously
multi_predictor = MyPredictor(
    target_column='value',
    sequence_length=7,
    prediction_horizon=3  # Predict 3 steps ahead
)

# Train the model
multi_predictor.fit(train_df, val_df, epochs=100)

# Get predictions for all 3 horizons
predictions = multi_predictor.predict(test_df)
# predictions shape: (n_samples, 3) for 3 horizons
print(f"Predictions shape: {predictions.shape}")
```

### 4. Single-Column Group-Based Scaling ⭐

```python
from tf_predictor.core.predictor import TimeSeriesPredictor

class MultiEntityPredictor(TimeSeriesPredictor):
    def _create_base_features(self, df):
        df_processed = df.copy()
        df_processed = df_processed.fillna(0)
        return super()._create_base_features(df_processed)

# Load data with multiple entities
# DataFrame should have: date, entity_id, value, ...
df = pd.read_csv('multi_entity_data.csv')

# Initialize with single-column group-based scaling
predictor = MultiEntityPredictor(
    target_column='value',
    sequence_length=10,
    group_columns='entity_id',  # ⭐ Single column grouping
    d_token=128,
    n_layers=3,
    n_heads=8
)

# Data will be automatically sorted by [entity_id, date]
# Each entity gets its own scaler, but trains in a unified model
predictor.fit(train_df, val_df, epochs=100)

# Predictions automatically use correct scaler per entity
predictions = predictor.predict(test_df)

# Evaluate with per-group metrics
metrics = predictor.evaluate(test_df, per_group=True)
```

### 5. Multi-Column Group-Based Scaling ⭐ NEW in v2.1.0

```python
from tf_predictor.core.predictor import TimeSeriesPredictor

class MultiDimensionPredictor(TimeSeriesPredictor):
    def _create_base_features(self, df):
        df_processed = df.copy()
        df_processed = df_processed.fillna(0)
        return super()._create_base_features(df_processed)

# Load data with multiple grouping dimensions
# DataFrame should have: date, symbol, sector, value, ...
df = pd.read_csv('stock_data.csv')

# Initialize with multi-column group-based scaling
predictor = MultiDimensionPredictor(
    target_column='close',
    sequence_length=10,
    group_columns=['symbol', 'sector'],  # ⭐ Multi-column grouping (NEW!)
    categorical_columns=['symbol', 'sector'],  # Also encode as categorical
    d_token=128,
    n_layers=3,
    n_heads=8
)

# Data will be automatically sorted by [symbol, sector, date]
# Each (symbol, sector) combination gets its own scaler
predictor.fit(train_df, val_df, epochs=100)

# Predictions automatically use correct scaler per group combination
predictions = predictor.predict(test_df)

# Evaluate with per-group metrics (now works correctly for multi-column!)
metrics = predictor.evaluate(test_df, per_group=True)
# Returns metrics for each (symbol, sector) combination
```

**Benefits of Multi-Column Group-Based Scaling (v2.1.0):**
- ✅ Support for composite group keys (e.g., symbol + sector, region + product)
- ✅ Correct per-group evaluation and filtering
- ✅ Better predictions when entities have hierarchical groupings
- ✅ Single unified model learns patterns across all group combinations
- ✅ Automatic temporal ordering within each composite group
- ✅ Handles missing group combinations gracefully during prediction

### 6. Multi-Target Prediction ⭐

```python
from tf_predictor.core.predictor import TimeSeriesPredictor

class MultiTargetPredictor(TimeSeriesPredictor):
    def _create_base_features(self, df):
        df_processed = df.copy()
        df_processed = df_processed.fillna(0)
        return super()._create_base_features(df_processed)

# Initialize with multiple target columns
predictor = MultiTargetPredictor(
    target_column=['temperature', 'humidity', 'pressure'],  # ⭐ Multiple targets
    sequence_length=10,
    prediction_horizon=1,  # Or use >1 for multi-horizon per target
    d_token=128,
    n_layers=3,
    n_heads=8
)

# Train on multiple targets simultaneously
predictor.fit(train_df, val_df, epochs=100)

# Predictions returned as dictionary
predictions = predictor.predict(test_df)
# predictions = {
#   'temperature': array([...]),
#   'humidity': array([...]),
#   'pressure': array([...])
# }

# Evaluate with per-target metrics
metrics = predictor.evaluate(test_df)
# metrics = {
#   'temperature': {'MAE': 1.2, 'RMSE': 2.3, ...},
#   'humidity': {'MAE': 5.4, 'RMSE': 8.1, ...},
#   'pressure': {'MAE': 0.8, 'RMSE': 1.5, ...}
# }
```

**Benefits of Multi-Target Prediction:**
- ✅ Train one model for multiple predictions
- ✅ Capture correlations between target variables
- ✅ More efficient than training separate models
- ✅ Works with multi-horizon and group-based scaling

### 7. Combining All Features ⭐ ULTIMATE

```python
# Ultimate configuration: Multi-target + Multi-horizon + Multi-column grouping
predictor = MultiTargetPredictor(
    target_column=['value1', 'value2'],        # Multiple targets
    sequence_length=10,
    prediction_horizon=3,                      # 3 steps ahead per target
    group_columns=['symbol', 'sector'],        # Multi-column grouping ⭐ NEW
    categorical_columns=['symbol', 'sector'],  # Categorical encoding
    d_token=192,
    n_layers=4,
    n_heads=8
)

predictor.fit(train_df, val_df, epochs=150, batch_size=64)

# Returns dict with arrays of shape (n_samples, 3) for each target
predictions = predictor.predict(test_df)
# predictions = {
#   'value1': array([[...], [...], [...]]),  # shape: (n_samples, 3)
#   'value2': array([[...], [...], [...]])   # shape: (n_samples, 3)
# }

# Per-group evaluation with multi-column grouping
metrics = predictor.evaluate(test_df, per_group=True)
# Returns nested structure with metrics for each (symbol, sector) combination
```

## 🎨 Complete Example with All Arguments

Here's a comprehensive example showing **all available parameters** for initializing a TimeSeriesPredictor:

```python
from tf_predictor.core.predictor import TimeSeriesPredictor
from tf_predictor.core.utils import split_time_series
import pandas as pd

# Define your custom predictor
class MyPredictor(TimeSeriesPredictor):
    def _create_base_features(self, df):
        df_processed = df.copy()
        # Add domain-specific features here
        df_processed = df_processed.fillna(0)
        return super()._create_base_features(df_processed)

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize with ALL available parameters
predictor = MyPredictor(
    # === Target Configuration ===
    target_column='close',                    # or ['col1', 'col2'] for multi-target

    # === Sequence & Horizon ===
    sequence_length=10,                       # Number of historical timesteps
    prediction_horizon=3,                     # Number of steps ahead (1=single, >1=multi-horizon)

    # === Model Architecture ===
    model_type='ft_transformer_cls',          # 'ft_transformer_cls' or 'csn_transformer_cls'
    d_token=128,                              # Token embedding dimension
    n_layers=3,                             # Number of transformer layers
    n_heads=8,                              # Number of attention heads
    dropout=0.1,                              # Dropout rate for regularization

    # === Scaling & Normalization ===
    scaler_type='standard',                   # 'standard', 'minmax', 'robust', 'maxabs', 'onlymax'
    group_columns=['symbol', 'sector'],       # Single column or list for multi-column grouping ⭐ NEW

    # === Feature Configuration ===
    categorical_columns=['symbol', 'sector'], # Single column or list for categorical encoding
    use_lagged_target_features=False,         # Include target in input sequences (autoregressive)

    # === Misc ===
    verbose=True                              # Print detailed processing information
)

# Split data chronologically
train_df, val_df, test_df = split_time_series(
    df,
    test_size=30,
    val_size=20,
    group_column='symbol',      # Optional: ensure splits respect groups
    time_column='date',          # Optional: specify time column name
    sequence_length=10           # Optional: ensure minimum samples per split
)

# Train with all available parameters
predictor.fit(
    df=train_df,
    val_df=val_df,
    epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    patience=15,                # Early stopping patience
    verbose=True
)

# Make predictions
predictions = predictor.predict(test_df)
# Single-target, single-horizon: returns array of shape (n_samples,)
# Single-target, multi-horizon: returns array of shape (n_samples, n_horizons)
# Multi-target: returns dict {'target1': array, 'target2': array, ...}

# Evaluate performance
metrics = predictor.evaluate(test_df, per_group=False)
# If per_group=True and group_columns is set, returns per-group metrics

# Save model
predictor.save('my_model.pt')

# Load model later
loaded_predictor = MyPredictor(
    target_column='close',
    sequence_length=10,
    prediction_horizon=3
)
loaded_predictor.load('my_model.pt')
```

### Parameter Groups Explained:

| Category | Parameters | Description |
|----------|-----------|-------------|
| **Target** | `target_column` | What to predict (str or list) |
| **Sequence** | `sequence_length`, `prediction_horizon` | Input length and output horizon |
| **Architecture** | `model_type`, `d_token`, `n_layers`, `n_heads`, `dropout` | Model structure |
| **Scaling** | `scaler_type`, `group_columns` | Normalization strategy (supports multi-column) ⭐ |
| **Features** | `categorical_columns`, `use_lagged_target_features` | Feature handling |

## 📚 Core Components

### TimeSeriesPredictor (Abstract Base Class)

The heart of the library - extend this for your domain:

```python
class TimeSeriesPredictor(ABC):
    def _create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Override to add domain-specific features (v2.0.0+ API).

        Only add feature engineering here - no scaling or encoding.
        Call super()._create_base_features() to add time-series features.

        DEPRECATED: prepare_features() method removed in v2.0.0
        """
        # Add automatic cyclical encoding for date/datetime columns
        # ... (handled by base class)
        pass
```

Key methods:
- `fit()`: Train the model with your data
- `predict()`: Generate predictions (handles multi-column grouping correctly)
- `evaluate()`: Calculate comprehensive metrics with per-horizon and per-group breakdown
- `save()` / `load()`: Model persistence

### v2.1.0 Pipeline Stages

The preprocessing pipeline runs in 7 stages with improved robustness:

1. **_create_base_features()**: Domain features + cyclical encoding
2. **create_shifted_targets()**: Create target_h1, target_h2, ... columns
3. **Storage** (if store_for_evaluation=True): Store unscaled/unencoded dataframe
4. **_encode_categorical_features()**: Label encoding for categoricals
5. **_determine_numerical_columns()**: Auto-detect feature columns
6. **_scale_features()**: Per-horizon target scaling + feature scaling (with proper multi-column grouping) ⭐
7. **_create_sequences()**: Create sliding window sequences (with deterministic group ordering) ⭐

### Model Architecture

**FT-Transformer**: Feature Tokenizer Transformer designed for tabular data
- Transforms each feature into learned embeddings (tokens)
- Applies multi-head self-attention across features and time steps
- Handles both numerical and categorical features seamlessly

**Sequence Models**: Built-in support for temporal sequences
- `FTTransformerCLSModel`: CLS token-based aggregation for sequences
- `CSNTransformerCLSModel`: Column-wise split network variant

### Preprocessing Utilities

**Time Features** (`preprocessing/time_features.py`):
```python
from tf_predictor.preprocessing.time_features import (
    create_date_features,         # Extract cyclical temporal features
    create_shifted_targets,       # Create future target columns
    create_input_variable_sequence, # Create sliding window sequences
    create_rolling_features,      # Rolling statistics
)
```

**Core Utilities** (`core/utils.py`):
```python
from tf_predictor.core.utils import (
    split_time_series,         # Chronological data splitting
    calculate_metrics,         # Comprehensive evaluation metrics
)
```

## 🛠️ Configuration

### Model Parameters (v2.1.0)
- `target_column`: Column(s) to predict - str for single-target or list for multi-target (e.g., 'value' or ['value1', 'value2'])
- `sequence_length`: Number of time steps to use as input (default: 5)
- `prediction_horizon`: Number of steps ahead to predict (default: 1, >1 for multi-horizon with per-horizon scaling)
- `model_type`: Model architecture ('ft_transformer_cls' or 'csn_transformer_cls')
- `group_columns`: Column(s) for group-based scaling - str or list (supports multi-column!) ⭐ (default: None)
- `categorical_columns`: Categorical features to encode - str or list (default: None)
- `scaler_type`: Scaler type ('standard', 'minmax', 'robust', 'maxabs', 'onlymax')
- `use_lagged_target_features`: Include target in input sequences (bool, default: False)
- `d_token`: Token embedding dimension (default: 128)
- `n_layers`: Number of transformer layers (default: 3)
- `n_heads`: Number of attention heads (default: 8)
- `dropout`: Dropout rate (default: 0.1)
- `verbose`: Print detailed processing information (bool, default: False)

### Training Parameters
- `epochs`: Training epochs (default: 100)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 1e-3)
- `patience`: Early stopping patience (default: 15)

## 🔧 Feature Engineering Guide

### Date Features (v2.0.0+)
```python
# Extract cyclical date features (automatic in _create_base_features)
df_processed = create_date_features(df, 'date')
# Adds cyclical encodings: month_sin, month_cos, day_sin, day_cos,
#                          dayofweek_sin, dayofweek_cos, is_weekend
# For datetime with time: hour_sin, hour_cos, minute_sin, minute_cos
# Drops: year, month, day, quarter, dayofweek, hour, minute (non-cyclical)
```

### Rolling Features
```python
# Add rolling statistics
df_processed = create_rolling_features(df, 'value', windows=[3, 7, 14])
# Adds: value_rolling_mean_3, value_rolling_std_3, value_rolling_min_3, etc.
```

## 📊 Evaluation Metrics

Comprehensive evaluation with `calculate_metrics()`:
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination
- **Directional Accuracy**: Percentage of correct trend predictions

Multi-horizon evaluation returns metrics per horizon plus overall metrics.
Per-group evaluation (with `per_group=True`) returns metrics for each group or group combination.

## 💾 Model Persistence

```python
# Save trained model
predictor.save('my_model.pt')

# Load model later
new_predictor = MyPredictor(target_column='value', sequence_length=7, prediction_horizon=1)
new_predictor.load('my_model.pt')
predictions = new_predictor.predict(new_data)
```

## 🎯 Example Use Cases

### 1. Multi-Region Product Forecasting (Multi-Column Grouping)
```python
class ProductPredictor(TimeSeriesPredictor):
    def _create_base_features(self, df):
        df_processed = df.copy()
        # Add product-specific features
        df_processed['is_promotion'] = (df_processed['discount'] > 0).astype(int)
        df_processed = df_processed.fillna(0)
        return super()._create_base_features(df_processed)

# Use multi-column grouping for region + product combinations
predictor = ProductPredictor(
    target_column='sales',
    group_columns=['region', 'product_id'],  # ⭐ Composite grouping
    sequence_length=14,
    prediction_horizon=7
)
```

### 2. Stock Market Forecasting (Sector + Symbol)
```python
class StockPredictor(TimeSeriesPredictor):
    def _create_base_features(self, df):
        df_processed = df.copy()
        # Calculate technical indicators
        df_processed['vwap'] = (df_processed['volume'] * df_processed['close']).sum() / df_processed['volume'].sum()
        df_processed = df_processed.fillna(method='bfill').fillna(0)
        return super()._create_base_features(df_processed)

predictor = StockPredictor(
    target_column='close',
    group_columns=['sector', 'symbol'],      # ⭐ Hierarchical grouping
    categorical_columns=['sector', 'symbol'],
    sequence_length=20,
    prediction_horizon=5
)
```

### 3. IoT Sensor Monitoring (Multi-Location)
```python
class SensorPredictor(TimeSeriesPredictor):
    def _create_base_features(self, df):
        df_processed = df.copy()
        # Add sensor-specific features
        df_processed['hour'] = df_processed['timestamp'].dt.hour
        df_processed['is_business_hours'] = df_processed['hour'].between(9, 17).astype(int)
        df_processed = df_processed.fillna(0)
        return super()._create_base_features(df_processed)

predictor = SensorPredictor(
    target_column=['temperature', 'humidity'],  # Multi-target
    group_columns=['building', 'floor'],        # ⭐ Location hierarchy
    sequence_length=60,
    prediction_horizon=10
)
```

## 📁 Library Structure

```
tf_predictor/
├── core/
│   ├── predictor.py          # Abstract TimeSeriesPredictor base class
│   ├── base/
│   │   ├── model_factory.py  # Model registry and factory
│   │   └── model_interface.py # Abstract model interface
│   ├── ft_model.py           # FT-Transformer implementations
│   ├── csn_model.py          # CSN-Transformer implementations
│   ├── feature_detector.py   # Automatic feature detection
│   └── utils.py              # Utilities (metrics, data splitting)
├── preprocessing/
│   ├── time_features.py      # Generic time series preprocessing
│   └── scaler_factory.py     # Scaler factory pattern
├── tests/
│   └── test_no_domain_imports.py  # Test suite
└── README.md                 # This file
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
cd tf_predictor/tests/
python test_no_domain_imports.py
```

Test multi-column grouping (v2.1.0):
```bash
python test_multi_column_grouping.py
```

## 🐛 Bug Fixes (v2.1.0)

### Critical Fixes:
1. **Multi-Column Group Evaluation** ⭐ FIXED
   - Fixed per-group evaluation for multi-column grouping scenarios
   - Added `_filter_dataframe_by_group()` helper method
   - Now correctly handles composite group keys (e.g., `['symbol', 'sector']`)

### Improvements:
2. **Enhanced Cache Hashing** ⭐ IMPROVED
   - Improved DataFrame hash function to sample first, middle, and last rows
   - Significantly reduced collision risk for similar DataFrames

3. **Deterministic Group Ordering** ⭐ IMPROVED
   - Added explicit sorting for group ordering across all operations
   - Ensures consistent behavior independent of implementation details
   - Guarantees reproducible results

For detailed information, see:
- `BUG_REPORT_COMPREHENSIVE.md` - Original bug report
- `BUG_FIXES_SUMMARY.md` - Detailed fix documentation

## ⚡ Performance Tips

### For Better Accuracy:
- Use longer sequences (10-50 time steps) for complex temporal patterns
- Increase model capacity (`d_token=256`, `n_layers=6`) for large datasets
- Add domain-specific features in your `_create_base_features()` method
- Use multi-column grouping for hierarchical data structures
- Leverage multi-target prediction to capture variable correlations

### For Faster Training:
- Use smaller models (`d_token=64`, `n_layers=2`) for quick experiments
- Increase batch size if memory allows
- Use fewer features for initial prototyping

### For Production:
- Always use validation sets for hyperparameter tuning
- Implement proper feature engineering in your `_create_base_features()` method
- Save models regularly and implement checkpointing
- Monitor both training and validation metrics
- Use per-group evaluation to identify problematic entities

## 🚨 Troubleshooting

### Common Issues:

**"Need at least X samples for sequence_length"**
- Reduce `sequence_length` for small datasets
- Ensure you have enough data after feature engineering
- Check that each group has sufficient samples when using grouping

**"No numeric feature columns found"**
- Check that your `_create_base_features()` method returns numeric columns
- Ensure date columns are excluded from feature scaling

**"Poor performance"**
- Add more domain-specific features
- Try different sequence lengths
- Increase model capacity or training time
- Check data quality and preprocessing
- Use group-based scaling if entities have different value ranges

**"Memory errors"**
- Reduce `batch_size` and `d_token`
- Process data in smaller chunks
- Clear feature cache periodically

**"Per-group evaluation errors" (Fixed in v2.1.0)**
- Update to v2.1.0 for multi-column grouping support
- Ensure group columns exist in your DataFrame

## 🔗 Extensions

This library serves as the foundation for domain-specific applications:

- **Stock Forecasting**: `daily_stock_forecasting` package (financial markets)
- **Intraday Forecasting**: `intraday_forecasting` package (high-frequency trading)
- **Energy Forecasting**: Extend for power grid and consumption modeling
- **Sales Forecasting**: E-commerce and retail demand prediction
- **IoT Monitoring**: Sensor data and anomaly detection
- **Weather Forecasting**: Meteorological time series prediction

## 🎓 Research Background

The FT-Transformer architecture is based on:
- **Feature Tokenization**: Each feature becomes a learnable token
- **Multi-Head Self-Attention**: Captures complex feature interactions
- **Positional Encoding**: Preserves temporal ordering information
- **Layer Normalization**: Stable training for deep networks

Key advantages over traditional approaches:
- Handles mixed data types naturally
- Learns complex feature interactions automatically
- Scales well with data size and feature count
- Provides interpretable attention patterns
- Supports hierarchical grouping structures

## 📖 API Reference

### Core Classes

**TimeSeriesPredictor**
- `__init__(target_column, sequence_length, group_columns, ...)` - Initialize predictor
- `_create_base_features(df)` - Abstract method to implement for domain features
- `fit(df, val_df, epochs, batch_size, ...)` - Train the model
- `predict(df, return_group_info)` - Generate predictions
- `evaluate(df, per_group, export_csv)` - Calculate metrics
- `save(path)` / `load(path)` - Model persistence

### Preprocessing Functions

**create_date_features(df, date_column, group_column)**
- Extracts temporal features with cyclical encoding
- Returns DataFrame with sin/cos encodings

**create_shifted_targets(df, target_column, prediction_horizon, group_column)**
- Creates future target columns for multi-horizon prediction
- Supports single and multi-column grouping

**create_rolling_features(df, column, windows)**
- Adds rolling statistics (mean, std, min, max)
- Returns DataFrame with rolling_stat_N columns

**split_time_series(df, test_size, val_size, group_column)**
- Chronological splitting for time series
- Returns (train_df, val_df, test_df) tuple

## 🚀 Getting Started

1. **Install dependencies**: `torch`, `pandas`, `numpy`, `scikit-learn`
2. **Define your predictor**: Extend `TimeSeriesPredictor`
3. **Implement feature engineering**: Override `_create_base_features()`
4. **Load and split your data**: Use `split_time_series()`
5. **Train and evaluate**: Call `fit()`, `predict()`, `evaluate()`
6. **Use advanced features**: Try multi-column grouping, multi-target, multi-horizon

See the stock forecasting packages for complete working examples!

## 🔄 Version History

### v2.1.0 (2025-11-05)
- ✅ Fixed critical multi-column group evaluation bug
- ✅ Improved cache hashing with multi-sample approach
- ✅ Added explicit group ordering for deterministic behavior
- ✅ Added comprehensive test suite for multi-column grouping
- ✅ Enhanced documentation with multi-column examples

### v2.0.0 (2025-10-31)
- ✅ Per-horizon target scaling
- ✅ Automatic cyclical encoding
- ✅ Evaluation alignment fixes
- ✅ Clean 7-stage pipeline architecture
- ✅ Simplified API with `_create_base_features()`

## 📄 License

This is a proprietary library for internal use in time series forecasting applications.

---

**For support, bug reports, or feature requests**, please refer to:
- `BUG_REPORT_COMPREHENSIVE.md` - Comprehensive bug analysis
- `BUG_FIXES_SUMMARY.md` - Recent bug fixes documentation
- `test_multi_column_grouping.py` - Test examples
