# 🧠 TF Predictor: Generic Time Series Forecasting Library

**Version:** 2.0.0 (Updated: 2025-10-31)

A reusable Python library for time series forecasting using the FT-Transformer (Feature Tokenizer Transformer) architecture. This library provides a clean, extensible foundation for building domain-specific time series prediction applications.

## 🎯 Features

### Version 2.0.0 Highlights ⭐ NEW
- **Per-Horizon Target Scaling**: Each prediction horizon gets its own scaler for optimal accuracy
- **Automatic Cyclical Encoding**: Temporal features automatically encoded as sin/cos with original features dropped
- **Evaluation Alignment Fixed**: Proper dataframe storage resolves 100% MAPE bug
- **Clean Architecture**: 7-stage preprocessing pipeline with clear separation of concerns
- **Simplified API**: `_create_base_features()` replaces `prepare_features()` for cleaner inheritance

### Core Features
- **Multi-Target Prediction**: Predict multiple variables simultaneously with one unified model
- **Multi-Horizon Forecasting**: Predict 1 to N steps ahead with per-horizon scaling
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

### 1. Create Your Custom Predictor (v2.0.0 API)

```python
from tf_predictor.core.predictor import TimeSeriesPredictor

class MyPredictor(TimeSeriesPredictor):
    def _create_base_features(self, df):
        """
        Override to add domain-specific features.
        Time-series features (cyclical encoding) are added automatically by parent.

        IMPORTANT (v2.0.0):
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

# Initialize your custom predictor (v2.0.0 parameter names)
predictor = MyPredictor(
    target_column='value',
    sequence_length=7,
    prediction_horizon=1,  # Number of steps ahead (default: 1)
    model_type='ft_transformer_cls',  # 'ft_transformer_cls' or 'csn_transformer_cls'
    d_model=128,  # Renamed from d_token in v2.0.0
    num_layers=3,  # Renamed from n_layers in v2.0.0
    num_heads=8    # Renamed from n_heads in v2.0.0
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

### 4. Group-Based Scaling for Multi-Entity Predictions ⭐ NEW

```python
from tf_predictor.core.predictor import TimeSeriesPredictor
from tf_predictor.preprocessing.time_features import create_date_features

class MultiEntityPredictor(TimeSeriesPredictor):
    def create_features(self, df, fit_scaler=False):
        df_processed = df.copy()
        if 'date' in df_processed.columns:
            df_processed = create_date_features(df_processed, 'date')
        return df_processed.fillna(0)

# Load data with multiple entities
# DataFrame should have: date, entity_id, value, ...
df = pd.read_csv('multi_entity_data.csv')

# Initialize with group-based scaling
predictor = MultiEntityPredictor(
    target_column='value',
    sequence_length=10,
    group_column='entity_id',  # ⭐ Each entity gets independent scaling
    d_token=128,
    n_layers=3,
    n_heads=8
)

# Data will be automatically sorted by [entity_id, date]
# Each entity gets its own scaler, but trains in a unified model
predictor.fit(train_df, val_df, epochs=100)

# Predictions automatically use correct scaler per entity
predictions = predictor.predict(test_df)
```

**Benefits of Group-Based Scaling:**
- ✅ Better predictions when entities have different value ranges
- ✅ Single unified model learns patterns across all entities
- ✅ Automatic temporal ordering within each group
- ✅ Handles missing groups gracefully during prediction

### 5. Multi-Target Prediction ⭐ NEW

```python
from tf_predictor.core.predictor import TimeSeriesPredictor
from tf_predictor.preprocessing.time_features import create_date_features

class MultiTargetPredictor(TimeSeriesPredictor):
    def create_features(self, df, fit_scaler=False):
        df_processed = df.copy()
        if 'date' in df_processed.columns:
            df_processed = create_date_features(df_processed, 'date')
        return df_processed.fillna(0)

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

### 6. Combining All Features ⭐ NEW

```python
# Ultimate configuration: Multi-target + Multi-horizon + Group-based scaling
predictor = MultiTargetPredictor(
    target_column=['value1', 'value2'],  # Multiple targets
    sequence_length=10,
    prediction_horizon=3,                 # 3 steps ahead per target
    group_column='entity_id',             # Group-based scaling
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
```
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
    d_model=128,                              # Token embedding dimension (renamed from d_token)
    num_layers=3,                             # Number of transformer layers (renamed from n_layers)
    num_heads=8,                              # Number of attention heads (renamed from n_heads)
    dropout=0.1,                              # Dropout rate for regularization

    # === Scaling & Normalization ===
    scaler_type='standard',                   # 'standard', 'minmax', 'robust', 'maxabs', 'onlymax'
    group_columns='symbol',                   # or ['col1', 'col2'] for group-based scaling

    # === Feature Configuration ===
    categorical_columns='sector',             # or ['col1', 'col2'] for categorical encoding
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
| **Architecture** | `model_type`, `d_model`, `num_layers`, `num_heads`, `dropout` | Model structure |
| **Scaling** | `scaler_type`, `group_columns` | Normalization strategy |
| **Features** | `categorical_columns`, `use_lagged_target_features` | Feature handling |

## 📚 Core Components

### TimeSeriesPredictor (Abstract Base Class)

The heart of the library - extend this for your domain:

```python
class TimeSeriesPredictor(ABC):
    def _create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Override to add domain-specific features (v2.0.0 API).

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
- `predict()`: Generate predictions
- `evaluate()`: Calculate comprehensive metrics with per-horizon breakdown
- `save()` / `load()`: Model persistence

### v2.0.0 Pipeline Stages

The preprocessing pipeline runs in 7 stages:

1. **_create_base_features()**: Domain features + cyclical encoding
2. **create_shifted_targets()**: Create target_h1, target_h2, ... columns
3. **Storage** (if store_for_evaluation=True): Store unscaled/unencoded dataframe
4. **_encode_categorical_features()**: Label encoding for categoricals
5. **_determine_numerical_columns()**: Auto-detect feature columns
6. **_scale_features()**: Per-horizon target scaling + feature scaling
7. **_create_sequences()**: Create sliding window sequences

### Model Architecture

**FT-Transformer**: Feature Tokenizer Transformer designed for tabular data
- Transforms each feature into learned embeddings (tokens)
- Applies multi-head self-attention across features and time steps
- Handles both numerical and categorical features seamlessly

**Sequence Models**: Built-in support for temporal sequences
- `SequenceFTTransformer`: Multi-step temporal modeling
- `FTTransformerPredictor`: Single-step predictions

### Preprocessing Utilities

**Time Features** (`preprocessing/time_features.py`):
```python
from tf_predictor.preprocessing.time_features import (
    create_date_features,      # Extract year, month, day, etc.
    create_lag_features,       # Add lagged values  
    create_rolling_features,   # Rolling statistics
    create_sequences,          # Convert to sequences
    create_percentage_changes  # Percentage change features
)
```

**Core Utilities** (`core/utils.py`):
```python  
from tf_predictor.core.utils import (
    split_time_series,         # Chronological data splitting
    calculate_metrics,         # Comprehensive evaluation metrics
    load_time_series_data      # Generic data loading
)
```

## 🛠️ Configuration

### Model Parameters (v2.0.0)
- `target_column`: Column(s) to predict - str for single-target or list for multi-target (e.g., 'value' or ['value1', 'value2'])
- `sequence_length`: Number of time steps to use as input (default: 5)
- `prediction_horizon`: Number of steps ahead to predict (default: 1, >1 for multi-horizon with per-horizon scaling)
- `model_type`: Model architecture ('ft_transformer_cls' or 'csn_transformer_cls')
- `group_columns`: Column(s) for group-based scaling - str or list (default: None)
- `categorical_columns`: Categorical features to encode - str or list (default: None)
- `scaler_type`: Scaler type ('standard', 'minmax', 'robust', 'maxabs', 'onlymax')
- `use_lagged_target_features`: Include target in input sequences (bool, default: False)
- `d_model`: Token embedding dimension (default: 128) - **renamed from `d_token` in v2.0.0**
- `num_layers`: Number of transformer layers (default: 3) - **renamed from `n_layers` in v2.0.0**
- `num_heads`: Number of attention heads (default: 8) - **renamed from `n_heads` in v2.0.0**
- `dropout`: Dropout rate (default: 0.1)

### Training Parameters  
- `epochs`: Training epochs (default: 100)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 1e-3)
- `patience`: Early stopping patience (default: 15)

## 🔧 Feature Engineering Guide

### Date Features (v2.0.0)
```python
# Extract cyclical date features (automatic in _create_base_features)
df_processed = create_date_features(df, 'date')
# Adds cyclical encodings: month_sin, month_cos, day_sin, day_cos,
#                          dayofweek_sin, dayofweek_cos, is_weekend
# For datetime with time: hour_sin, hour_cos, minute_sin, minute_cos
# Drops: year, month, day, quarter, dayofweek, hour, minute (non-cyclical)
```

### Lag Features
```python
# Add historical values
df_processed = create_lag_features(df, 'value', lags=[1, 2, 7, 30])
# Adds: value_lag_1, value_lag_2, value_lag_7, value_lag_30
```

### Rolling Features
```python  
# Add rolling statistics
df_processed = create_rolling_features(df, 'value', windows=[3, 7, 14])
# Adds: value_rolling_mean_3, value_rolling_std_3, value_rolling_min_3, etc.
```

### Percentage Changes
```python
# Add percentage change features
df_processed = create_percentage_changes(df, 'value', periods=[1, 7, 30])
# Adds: value_pct_change_1d, value_pct_change_7d, value_pct_change_30d
```

## 📊 Evaluation Metrics

Comprehensive evaluation with `calculate_metrics()`:
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error  
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination
- **Directional Accuracy**: Percentage of correct trend predictions

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

### 1. Energy Consumption Forecasting
```python
class EnergyPredictor(TimeSeriesPredictor):
    def create_features(self, df, fit_scaler=False):
        df_processed = create_date_features(df, 'date')
        df_processed = create_lag_features(df_processed, 'consumption', [1, 7, 365])
        df_processed = create_rolling_features(df_processed, 'temperature', [3, 7])
        # Add energy-specific features
        df_processed['is_working_day'] = (df_processed['dayofweek'] < 5).astype(int)
        return df_processed.fillna(0)
```

### 2. Sales Forecasting  
```python
class SalesPredictor(TimeSeriesPredictor):
    def create_features(self, df, fit_scaler=False):
        df_processed = create_date_features(df, 'date')
        df_processed = create_lag_features(df_processed, 'sales', [1, 7, 14, 28])
        # Add sales-specific features
        df_processed['is_holiday'] = df_processed['date'].dt.date.isin(holidays)
        df_processed['month_end'] = (df_processed['day'] >= 28).astype(int)
        return df_processed.fillna(method='bfill').fillna(0)
```

### 3. IoT Sensor Monitoring
```python
class SensorPredictor(TimeSeriesPredictor):
    def create_features(self, df, fit_scaler=False):
        df_processed = create_date_features(df, 'timestamp') 
        df_processed = create_rolling_features(df_processed, 'sensor_value', [5, 15, 60])
        # Add sensor-specific features
        df_processed['hour'] = df_processed['timestamp'].dt.hour
        df_processed['is_business_hours'] = df_processed['hour'].between(9, 17).astype(int)
        return df_processed.fillna(0)
```

## 📁 Library Structure

```
tf_predictor/
├── core/
│   ├── predictor.py          # Abstract TimeSeriesPredictor base class
│   ├── model.py             # FT-Transformer model implementations  
│   └── utils.py             # Utilities (metrics, data splitting)
├── preprocessing/
│   └── time_features.py     # Generic time series preprocessing
├── tests/
│   └── test_core.py         # Comprehensive test suite
├── examples/
│   └── basic_timeseries_example.py  # Energy consumption example
└── README.md                # This file
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
cd tf_predictor/tests/
python test_core.py
```

Or use pytest if available:
```bash
pytest tf_predictor/tests/
```

## ⚡ Performance Tips

### For Better Accuracy:
- Use longer sequences (10-50 time steps) for complex temporal patterns
- Increase model capacity (`d_token=256`, `n_layers=6`) for large datasets
- Add domain-specific features in your `create_features()` method
- Use rolling statistics and lag features to capture trends

### For Faster Training:
- Use smaller models (`d_token=64`, `n_layers=2`) for quick experiments
- Increase batch size if memory allows
- Use fewer features for initial prototyping

### For Production:  
- Always use validation sets for hyperparameter tuning
- Implement proper feature scaling in your `create_features()` method
- Save models regularly and implement checkpointing
- Monitor both training and validation metrics

## 🚨 Troubleshooting

### Common Issues:

**"Need at least X samples for sequence_length"**
- Reduce `sequence_length` for small datasets
- Ensure you have enough data after feature engineering

**"No numeric feature columns found"**
- Check that your `create_features()` method returns numeric columns
- Ensure date columns are excluded from feature scaling

**"Poor performance"**
- Add more domain-specific features
- Try different sequence lengths
- Increase model capacity or training time
- Check data quality and preprocessing

**"Memory errors"**
- Reduce `batch_size` and `d_token`
- Use gradient checkpointing for very deep models
- Process data in smaller chunks

## 🔗 Extensions

This library serves as the foundation for domain-specific applications:

- **Stock Forecasting**: `daily_stock_forecasting` package (financial markets)
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

## 📖 API Reference

### Core Classes

**TimeSeriesPredictor**
- `__init__(target_column, sequence_length, **ft_kwargs)`
- `create_features(df, fit_scaler)` - Abstract method to implement
- `fit(df, val_df, epochs, batch_size, ...)` - Train the model
- `predict(df)` - Generate predictions
- `evaluate(df)` - Calculate metrics
- `save(path)` / `load(path)` - Model persistence

### Preprocessing Functions  

**create_date_features(df, date_column)**
- Extracts temporal features from date column
- Returns DataFrame with year, month, day, cyclical encodings

**create_lag_features(df, column, lags)**  
- Adds lagged versions of specified column
- Returns DataFrame with lag_N columns

**create_rolling_features(df, column, windows)**
- Adds rolling statistics (mean, std, min, max)
- Returns DataFrame with rolling_stat_N columns

**split_time_series(df, test_size, val_size)**
- Chronological splitting for time series
- Returns (train_df, val_df, test_df) tuple

## 🚀 Getting Started

1. **Install dependencies**: `torch`, `pandas`, `numpy`, `scikit-learn`
2. **Define your predictor**: Extend `TimeSeriesPredictor` 
3. **Implement feature engineering**: Override `create_features()`
4. **Load and split your data**: Use `split_time_series()`
5. **Train and evaluate**: Call `fit()`, `predict()`, `evaluate()`

See `examples/basic_timeseries_example.py` for a complete working example!

## 🔄 Updates

This is a core library designed for stability and extensibility. Updates focus on:
- Performance improvements to the transformer architecture
- Additional preprocessing utilities
- Better memory efficiency and scalability  
- Enhanced evaluation metrics and diagnostics