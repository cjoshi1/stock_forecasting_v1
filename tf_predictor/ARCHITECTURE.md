# 🧠 TF_Predictor Architecture Guide

## Overview
`TimeSeriesPredictor` is the core abstract base class for all time series prediction in tf_predictor. It handles data preprocessing, model training, prediction, and evaluation with support for multi-target, multi-horizon, and group-based operations.

---

## 📊 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TimeSeriesPredictor                           │
│                    (Abstract Base Class)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐ │
│  │ Data Pipeline  │  │ Model Training │  │ Prediction &     │ │
│  │                │  │                │  │ Evaluation       │ │
│  │ • Feature Eng  │  │ • fit()        │  │ • predict()      │ │
│  │ • Scaling      │  │ • Validation   │  │ • evaluate()     │ │
│  │ • Sequencing   │  │ • Early Stop   │  │ • Metrics        │ │
│  └────────────────┘  └────────────────┘  └──────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Complete Data Flow (REFACTORED - Oct 2025)

### Training Flow (fit)

```
User Data (Raw DataFrame)
    │
    ▼
┌───────────────────────────────────────────────────────┐
│ 1. prepare_data(fit_scaler=True, store=False)       │
└───────────────────────────────────────────────────────┘
    │
    ├─► Step 1: _create_base_features()
    │   • Domain-specific features (overridable)
    │   • Time-series features (automatic cyclical encoding)
    │   • Sorts by group+time if group_column exists
    │
    ├─► Step 2: create_shifted_targets()
    │   • Creates target_h1, target_h2, ..., target_hN
    │   • Removes rows with NaN
    │
    ├─► Step 3: SKIP STORAGE (store_for_evaluation=False)
    │
    ├─► Step 4: _encode_categorical_features()
    │   • Label encoding for categorical columns
    │
    ├─► Step 5: _determine_numerical_columns()
    │   • Auto-detect feature columns
    │   • Exclude targets, categoricals, shifted targets
    │
    ├─► Step 6: _scale_features_single/grouped()
    │   • Feature scaling: One scaler for all features
    │   • Target scaling: PER-HORIZON ⭐
    │     - close_target_h1 → StandardScaler #1
    │     - close_target_h2 → StandardScaler #2
    │     - close_target_h3 → StandardScaler #3
    │
    └─► Step 7: _create_sequences()
        • Sliding window sequences
        • Separate numerical (3D) and categorical (2D) tensors
    │
    ▼
┌───────────────────────────────────────────────────────┐
│ 2. Training Loop in fit()                            │
│    • Mini-batch training                              │
│    • Forward pass → Loss → Backward → Update          │
│    • Validation after each epoch                      │
│    • Early stopping monitoring                        │
│    • Per-horizon inverse transform for metrics        │
└───────────────────────────────────────────────────────┘
    │
    ▼
  Model Trained ✓
```

### Prediction Flow (predict)

```
New Data (Raw DataFrame)
    │
    ▼
┌───────────────────────────────────────────────────────┐
│ 1. prepare_data(fit_scaler=False, store=True) ⭐     │
└───────────────────────────────────────────────────────┘
    │
    ├─► Step 1: _create_base_features()
    │   • Same feature engineering as training
    │
    ├─► Step 2: create_shifted_targets()
    │   • Creates shifted targets
    │
    ├─► Step 3: STORE FOR EVALUATION ⭐⭐⭐
    │   • Store df with UNENCODED categorical values
    │   • Store df with UNSCALED numerical values
    │   • Store df with shifted target columns
    │   • This is used for extracting actuals during evaluation
    │   • Fixes 100% MAPE bug (dataframe alignment issue)
    │
    ├─► Step 4: _encode_categorical_features()
    │   • Uses EXISTING encoders (no fitting)
    │
    ├─► Step 5: _determine_numerical_columns()
    │   • Uses cached column lists
    │
    ├─► Step 6: _scale_features_single/grouped()
    │   • Uses EXISTING scalers (no fitting)
    │   • Per-horizon target scalers
    │
    └─► Step 7: _create_sequences()
        • Same sequence structure as training
    │
    ▼
    ▼
┌───────────────────────────────────────────────────────┐
│ 3. Model Forward Pass                                │
│    • model(X) → predictions_scaled                    │
│    • Predictions in scaled space                      │
└───────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────┐
│ 4. Inverse Transform                                 │
│    • Unscale predictions back to original space       │
│    • Uses appropriate scaler(s) per target/group      │
└───────────────────────────────────────────────────────┘
    │
    ▼
  Predictions (Original Scale) ✓
```

---

## 🔧 Key Methods Deep Dive

### 1. **`__init__()` - Initialization**

```
Purpose: Set up the predictor with configuration
Input:
  - target_column: str or List[str]  (what to predict)
  - sequence_length: int             (lookback window)
  - prediction_horizon: int          (steps ahead)
  - group_column: Optional[str]      (for multi-entity data)
  - **ft_kwargs                      (model hyperparameters)

Creates:
  - Scalers (feature + target)
  - Model architecture
  - Training history tracking
```

### 2. **`create_features()` - Abstract Method**

```
Purpose: Domain-specific feature engineering (USER IMPLEMENTS THIS)
Input: Raw DataFrame
Output: DataFrame with engineered features

Example Implementation:
  - Add technical indicators (for stocks)
  - Add lag features
  - Add rolling statistics
  - Create shifted targets using create_shifted_targets()
```

### 3. **`prepare_features()` - Feature Pipeline**

```
┌──────────────────────────────────────────────────────┐
│                  prepare_features()                   │
├──────────────────────────────────────────────────────┤
│                                                       │
│  1. Sort by group + time (if group_column exists)    │
│     ↓                                                 │
│  2. Call user's create_features()                    │
│     ↓                                                 │
│  3. Identify feature columns                         │
│     • Exclude target columns                         │
│     • Exclude shifted targets (_target_h*)           │
│     • Only numeric columns                           │
│     ↓                                                 │
│  4. Scale features                                   │
│     • Single-group: _scale_features_single()         │
│     • Multi-group: _scale_features_grouped()         │
│     ↓                                                 │
│  5. Cache result (optional)                          │
│                                                       │
└──────────────────────────────────────────────────────┘
```

### 4. **Scaling Methods**

#### `_scale_features_single()` - No Grouping

```
Input: DataFrame with features
Process:
  if fit_scaler:
      self.scaler.fit(features)
  features_scaled = self.scaler.transform(features)
Output: DataFrame with scaled features
```

#### `_scale_features_grouped()` - With Grouping

```
For each group in data:
    if fit_scaler:
        self.group_feature_scalers[group] = StandardScaler()
        self.group_feature_scalers[group].fit(group_data)
    group_data_scaled = self.group_feature_scalers[group].transform(group_data)

Concatenate all groups back together
```

### 5. **`prepare_data()` - Sequence & Target Preparation**

```
┌──────────────────────────────────────────────────────────┐
│                     prepare_data()                        │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Route based on configuration:                           │
│                                                           │
│  ┌────────────────────┐  ┌──────────────────────────┐   │
│  │ group_column       │  │ No group_column          │   │
│  │ specified?         │  │                          │   │
│  └────────────────────┘  └──────────────────────────┘   │
│           │                          │                   │
│           ▼                          ▼                   │
│  _prepare_data_grouped()   _prepare_data_single()       │
│                                                           │
└──────────────────────────────────────────────────────────┘

Both methods do:
  1. Create sequences: create_input_variable_sequence()
  2. Extract targets from DataFrame (after sequence_length offset)
  3. Scale targets using appropriate scaler(s)
  4. Return (X_tensor, Y_tensor)
```

#### Target Scaling Logic

```
┌─────────────────────────────────────────────────┐
│        Target Scaling Architecture              │
├─────────────────────────────────────────────────┤
│                                                  │
│  Single-Target                                   │
│  ├─ Single-Horizon: self.target_scaler          │
│  └─ Multi-Horizon:  self.target_scaler (shared) │
│                                                  │
│  Multi-Target                                    │
│  ├─ Single-Horizon: self.target_scalers_dict    │
│  │                   {'close': scaler1,         │
│  │                    'volume': scaler2}        │
│  └─ Multi-Horizon:  self.target_scalers_dict    │
│                     {'close': scaler1,  ← shared│
│                      'volume': scaler2} ← shared│
│                                                  │
│  Group-Based                                     │
│  └─ self.group_target_scalers                   │
│     {'AAPL': {'close': scaler, 'volume': scaler}│
│      'MSFT': {'close': scaler, 'volume': scaler}│
│                                                  │
│  KEY: One scaler per variable (not per horizon) │
└─────────────────────────────────────────────────┘
```

### 6. **`_prepare_data_grouped()` - Group-based Preparation**

```
For each group (e.g., AAPL, MSFT, GOOGL):
    1. Extract group data
    2. Check if group has enough data (> sequence_length)
    3. Create sequences for this group
    4. Extract targets for this group
    5. Scale targets using group-specific scaler
    6. Track group membership for each sequence

Concatenate all groups:
    - All sequences stacked vertically
    - All targets stacked vertically
    - Group indices tracked for later inverse scaling
```

### 7. **`fit()` - Training Loop**

```
┌───────────────────────────────────────────────────────┐
│                    Training Loop                       │
├───────────────────────────────────────────────────────┤
│                                                        │
│  Prepare Training Data:                               │
│    X_train, y_train = prepare_data(train_df, fit=True)│
│                                                        │
│  Prepare Validation Data (if provided):               │
│    X_val, y_val = prepare_data(val_df, fit=False)     │
│                                                        │
│  Initialize Model:                                     │
│    FT-Transformer or CSN-Transformer                   │
│    • Input size = number of features                   │
│    • Output size = num_targets * prediction_horizon    │
│                                                        │
│  For each epoch:                                       │
│    ┌─────────────────────────────────────┐           │
│    │  For each batch:                    │           │
│    │    1. Forward pass                  │           │
│    │    2. Compute loss                  │           │
│    │    3. Backward pass                 │           │
│    │    4. Update weights                │           │
│    └─────────────────────────────────────┘           │
│                                                        │
│    Validation (every epoch):                          │
│      - Compute val_loss                               │
│      - Inverse transform predictions                  │
│      - Calculate MAE & MAPE metrics                   │
│      - Print progress (verbose mode)                  │
│                                                        │
│    Early Stopping:                                    │
│      - Track best validation loss                     │
│      - Stop if no improvement for 'patience' epochs   │
│                                                        │
└───────────────────────────────────────────────────────┘
```

### 8. **`predict()` - Make Predictions**

```
┌───────────────────────────────────────────────────────┐
│                    Prediction Flow                     │
├───────────────────────────────────────────────────────┤
│                                                        │
│  1. prepare_features(df, fit_scaler=False)            │
│     • Use EXISTING scalers                            │
│                                                        │
│  2. Create sequences (no targets needed)              │
│     • create_input_variable_sequence()                │
│                                                        │
│  3. Forward pass through model                        │
│     • predictions_scaled = model(X)                   │
│                                                        │
│  4. Inverse transform                                 │
│     ┌────────────────────────────────┐               │
│     │  No grouping:                  │               │
│     │    Use self.target_scaler      │               │
│     │                                 │               │
│     │  With grouping:                │               │
│     │    For each group:             │               │
│     │      Use group's scaler        │               │
│     │                                 │               │
│     │  Multi-target:                 │               │
│     │    For each target:            │               │
│     │      Use target's scaler       │               │
│     └────────────────────────────────┘               │
│                                                        │
│  5. Return predictions in original scale              │
│                                                        │
└───────────────────────────────────────────────────────┘

Output Shapes:
  Single-target, single-horizon: (n_samples,)
  Single-target, multi-horizon:  (n_samples, horizons)
  Multi-target:                  Dict[target_name, array]
```

### 9. **`evaluate()` - Performance Metrics**

```
┌───────────────────────────────────────────────────────┐
│                   Evaluation Flow                      │
├───────────────────────────────────────────────────────┤
│                                                        │
│  1. Get predictions: predict(df)                      │
│                                                        │
│  2. Get actual values                                 │
│     • Extract from original DataFrame                 │
│     • Align indices properly                          │
│       (account for sequence_length offset)            │
│                                                        │
│  3. Calculate metrics:                                │
│     ┌────────────────────────────────────┐           │
│     │  Standard Mode:                    │           │
│     │    • MSE (Mean Squared Error)      │           │
│     │    • RMSE (Root MSE)               │           │
│     │    • MAE (Mean Absolute Error)     │           │
│     │    • MAPE (Mean Abs % Error)       │           │
│     │    • R² Score                      │           │
│     └────────────────────────────────────┘           │
│                                                        │
│     ┌────────────────────────────────────┐           │
│     │  Per-Group Mode (per_group=True):  │           │
│     │    • Same metrics but per group    │           │
│     │    • Useful for multi-stock data   │           │
│     └────────────────────────────────────┘           │
│                                                        │
│  4. Return metrics dictionary                         │
│                                                        │
└───────────────────────────────────────────────────────┘
```

---

## 🎯 Usage Patterns

### Pattern 1: Single-Target, Single-Horizon

```python
# Simplest case: predict one variable, one step ahead
predictor = MyPredictor(
    target_column='close',
    sequence_length=10,
    prediction_horizon=1
)

predictor.fit(train_df, val_df, epochs=100)
predictions = predictor.predict(test_df)  # Shape: (n_samples,)
```

### Pattern 2: Single-Target, Multi-Horizon

```python
# Predict one variable, multiple steps ahead
predictor = MyPredictor(
    target_column='close',
    sequence_length=10,
    prediction_horizon=3  # Predict 1, 2, 3 steps ahead
)

predictor.fit(train_df, val_df, epochs=100)
predictions = predictor.predict(test_df)  # Shape: (n_samples, 3)
```

### Pattern 3: Multi-Target, Single-Horizon

```python
# Predict multiple variables, one step ahead
predictor = MyPredictor(
    target_column=['close', 'volume'],
    sequence_length=10,
    prediction_horizon=1
)

predictor.fit(train_df, val_df, epochs=100)
predictions = predictor.predict(test_df)
# Returns: {'close': array, 'volume': array}
```

### Pattern 4: Multi-Target, Multi-Horizon

```python
# Predict multiple variables, multiple steps ahead
predictor = MyPredictor(
    target_column=['close', 'volume'],
    sequence_length=10,
    prediction_horizon=3
)

predictor.fit(train_df, val_df, epochs=100)
predictions = predictor.predict(test_df)
# Returns: {
#   'close': array(shape=(n_samples, 3)),
#   'volume': array(shape=(n_samples, 3))
# }
```

### Pattern 5: Group-Based (Multi-Entity)

```python
# Predict across multiple stocks/entities
predictor = MyPredictor(
    target_column='close',
    sequence_length=10,
    prediction_horizon=1,
    group_column='symbol'  # AAPL, MSFT, GOOGL, etc.
)

# Training data has multiple symbols
predictor.fit(train_df, val_df, epochs=100)

# Predictions maintain group-specific scaling
predictions = predictor.predict(test_df)
```

---

## 🔑 Key Design Principles

### 1. **Separation of Concerns**

```
User's Responsibility:
  - create_features() implementation
  - Domain-specific feature engineering

tf_predictor's Responsibility:
  - Scaling
  - Sequencing
  - Training loop
  - Prediction
  - Evaluation
```

### 2. **Scaling Architecture**

```
Rule: One scaler per variable (shared across all horizons)

Why?
  - Consistent scale across time horizons
  - Easier to compare h1 vs h2 vs h3 predictions
  - More stable training
  - Correct inverse transformation

Example:
  close_target_h1, close_target_h2, close_target_h3
         ↓              ↓              ↓
       All use the same close_scaler
```

### 3. **Group-Based Operations**

```
When group_column is specified:
  - Each group gets its own scaler(s)
  - Prevents data leakage across groups
  - Sorting by [group, time] ensures temporal order
  - Predictions use group-specific scalers

Use cases:
  - Multi-stock prediction (group by symbol)
  - Multi-sensor data (group by sensor_id)
  - Multi-customer forecasting (group by customer_id)
```

### 4. **Caching Strategy**

```
Feature Cache:
  - Avoid recomputing features for same data
  - Hash based on: shape + columns + first row
  - Can be disabled: disable_feature_cache()
  - Cleared automatically or manually: clear_feature_cache()

Benefits:
  - Faster repeated calls with same data
  - Useful during experimentation
```

---

## 📈 Complete Example Walkthrough

### Scenario: Multi-Stock, Multi-Horizon Prediction

```python
# Setup
predictor = StockPredictor(
    target_column='close',
    sequence_length=10,
    prediction_horizon=3,
    group_column='symbol'
)

# Data structure
train_df:
  symbol  date        close   volume   ...
  AAPL    2023-01-01  150     1000000
  AAPL    2023-01-02  151     1100000
  ...
  MSFT    2023-01-01  250     800000
  MSFT    2023-01-02  251     850000
  ...
```

### Training Flow

```
1. prepare_features(train_df, fit_scaler=True)
   ├─ Sort by [symbol, date]
   ├─ create_features() adds indicators
   ├─ create_shifted_targets() adds:
   │    close_target_h1, close_target_h2, close_target_h3
   └─ Scale features per group:
        AAPL_scaler.fit(AAPL_data)
        MSFT_scaler.fit(MSFT_data)

2. prepare_data(train_df_processed, fit_scaler=True)
   For AAPL:
     ├─ create_input_variable_sequence() → AAPL_sequences
     ├─ Extract targets: close_target_h1, h2, h3
     └─ Scale with AAPL_target_scaler.fit()

   For MSFT:
     ├─ create_input_variable_sequence() → MSFT_sequences
     ├─ Extract targets: close_target_h1, h2, h3
     └─ Scale with MSFT_target_scaler.fit()

   Stack all: X_train, y_train

3. fit() training loop
   ├─ For each epoch:
   │    ├─ Mini-batch training
   │    └─ Validation with metrics
   └─ Early stopping if no improvement
```

### Prediction Flow

```
1. prepare_features(test_df, fit_scaler=False)
   └─ Use EXISTING scalers (no fitting)

2. Create sequences (no targets needed)

3. model.forward(X_test) → predictions_scaled

4. Inverse transform per group:
   For AAPL predictions:
     └─ AAPL_target_scaler.inverse_transform()
   For MSFT predictions:
     └─ MSFT_target_scaler.inverse_transform()

5. Return predictions in original scale
```

---

## 🛠️ Helper Methods

### Model Persistence

```python
# Save trained model
predictor.save('model.pkl')

# Load trained model
predictor = TimeSeriesPredictor.load('model.pkl')
```

### Advanced Prediction

```python
# Predict with group information
predictions, group_indices = predictor.predict(
    test_df,
    return_group_info=True
)
# group_indices tells you which group each prediction belongs to
```

### Evaluation Options

```python
# Standard evaluation
metrics = predictor.evaluate(test_df)
# Returns: {'mse': ..., 'rmse': ..., 'mae': ..., 'mape': ..., 'r2': ...}

# Per-group evaluation
metrics_per_group = predictor.evaluate(test_df, per_group=True)
# Returns: {
#   'AAPL': {'mse': ..., 'mae': ..., ...},
#   'MSFT': {'mse': ..., 'mae': ..., ...}
# }
```

---

## 🎓 Summary

### Core Responsibilities

1. **Data Pipeline**: Feature engineering → Scaling → Sequencing
2. **Training**: Model training with validation and early stopping
3. **Prediction**: Forward pass → Inverse scaling → Return predictions
4. **Evaluation**: Calculate comprehensive performance metrics

### Supported Configurations

| Feature | Support |
|---------|---------|
| Single-target | ✅ |
| Multi-target | ✅ |
| Single-horizon | ✅ |
| Multi-horizon | ✅ |
| Group-based operations | ✅ |
| Feature caching | ✅ |
| Early stopping | ✅ |
| Model persistence | ✅ |

### Extension Points

- **`create_features()`**: Implement domain-specific feature engineering
- **Model architecture**: Can use FT-Transformer or CSN-Transformer
- **Hyperparameters**: Fully configurable via `**ft_kwargs`

