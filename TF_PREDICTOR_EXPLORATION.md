# TF_Predictor Module Exploration - Key Findings

## Repository Overview
- **Location**: `/home/user/stock_forecasting_v1/tf_predictor/`
- **Language**: Python with PyTorch backend
- **Primary Purpose**: Generic time series prediction using FT-Transformer or CSN-Transformer models

---

## 1. PREDICT() METHOD IMPLEMENTATION
**File**: `/home/user/stock_forecasting_v1/tf_predictor/core/predictor.py` (Line 1389)

### Location and Structure
```python
def predict(self, df: pd.DataFrame, return_group_info: bool = False) 
    -> Union[np.ndarray, Tuple[np.ndarray, list]]
```

### Key Functionality
1. **Input Preparation**: Calls `prepare_data(df, fit_scaler=False, store_for_evaluation=True)`
   - Stores unscaled/unencoded dataframe for evaluation alignment
   - Processes data through the entire pipeline (features → encoding → scaling → sequences)

2. **Batched Inference** (Lines 1416-1449)
   - Processes predictions in batches of 256 samples to avoid OOM errors
   - Handles both categorical models (tuple input: X_num, X_cat) and standard models
   - Moves results to CPU immediately after each batch to free GPU memory

3. **Target Variable Handling** (Lines 1454-1543)
   - **Single-Target Mode**:
     - Returns numpy array of predictions in original scale
     - Uses group-based inverse transform if `group_columns` is set
     - Per-horizon scalers for multi-horizon predictions (e.g., h1, h2, h3)
   
   - **Multi-Target Mode** (Lines 1460-1511):
     - Returns dictionary: `{target_name: predictions_array}`
     - Handles multi-target + multi-horizon combinations
     - Layout for multi-horizon: [close_h1, close_h2, ..., volume_h1, volume_h2, ...]

4. **Inverse Transformation** (Lines 1517-1543)
   - **Single-Target, Single-Horizon**: Reshape and inverse transform
   - **Single-Target, Multi-Horizon**: Each horizon has separate scaler
     - Uses `group_target_scalers[group_value][f"{target_col}_target_h{h}"]`
   - **Multi-Target**: Similar per-target, per-horizon logic

5. **Group-Based Prediction** (Lines 1455-1510)
   - If `group_columns` set: Inverse transform each prediction using its group's scaler
   - Stores `_last_group_indices` to track which group each prediction belongs to
   - Returns `(predictions_dict, group_indices)` if `return_group_info=True`

---

## 2. TIME SERIES SPLIT LOGIC
**File**: `/home/user/stock_forecasting_v1/tf_predictor/core/utils.py` (Line 114)

### Function Signature
```python
def split_time_series(df: pd.DataFrame, test_size: int = 30, val_size: int = None,
                      group_column: str = None, time_column: str = None,
                      sequence_length: int = 1) -> tuple
```

### Non-Grouped Split (Line 135-153)
```
Total data: [--------- TRAIN --------|--- VAL ---|-- TEST --]
                                      |           |
                              (len-test_size-val_size) to (len-test_size)
                                                    to len
```
- Train: `df.iloc[:-test_size]` or `df.iloc[:-(test_size+val_size)]` if val_size set
- Val: `remaining.iloc[-val_size:]` (if val_size provided)
- Test: `df.iloc[-test_size:]` (most recent data)

### Grouped Split (Lines 155-245)
1. **Per-Group Splitting**: Each group gets its own proportional split
   - Maintains temporal order within each group
   - Auto-detects time column if not provided (checks: timestamp, date, datetime)
   
2. **Minimum Data Requirements**:
   ```python
   min_train_samples = sequence_length * 2 + 10
   min_required = test_size + (val_size if val_size else 0) + min_train_samples
   ```
   - Groups with insufficient data are skipped with warning

3. **Split Order**:
   - Most recent data → Test
   - Middle data → Validation
   - Earliest data → Training

4. **Output**: Concatenated splits maintaining temporal order across all groups

---

## 3. SEQUENCE CREATION AND SEQUENCE_LENGTH EFFECTS
**File**: `/home/user/stock_forecasting_v1/tf_predictor/preprocessing/time_features.py` (Line 296)

### Function Signature
```python
def create_input_variable_sequence(
    df: pd.DataFrame,
    sequence_length: int,
    feature_columns: list = None,
    exclude_columns: list = None
) -> np.ndarray
```

### Sliding Window Mechanism (Lines 370-379)
```python
for i in range(len(df) - sequence_length + 1):
    seq = features[i:i+sequence_length]
    sequences.append(seq)
```

### Output Shape
- **Input**: DataFrame with shape `(n_rows, n_features)`
- **Output**: NumPy array with shape `(n_samples, sequence_length, n_features)`
  - Where: `n_samples = len(df) - sequence_length + 1`

### Sequence_Length Impact Examples
```
DataFrame: 100 rows
sequence_length = 5:
  Sequences created: 100 - 5 + 1 = 96
  - Sequence 0: rows [0:5]
  - Sequence 1: rows [1:6]
  - ...
  - Sequence 95: rows [95:100]

sequence_length = 10:
  Sequences created: 100 - 10 + 1 = 91
  - Sequence 0: rows [0:10]
  - ...
  - Sequence 90: rows [90:100]
```

### Target Extraction Offset
```python
# Line 1008 in predictor.py - Non-grouped mode
target_values = df_scaled[shifted_col].values[self.sequence_length - 1:]

# Line 852 in predictor.py - Grouped mode
target_values = group_df[shifted_col].values[self.sequence_length - 1:]
```
- Takes values starting at index `sequence_length - 1`
- This aligns target values with sequences
- Example: If sequence_length=5, targets start at index 4

---

## 4. DATA PREPROCESSING AND FEATURE CALCULATION FLOW

### Overall Pipeline (`prepare_data()` at Line 912)

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Raw DataFrame with features and target columns      │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │ Step 1: Create Base Features │
        │ (_create_base_features)      │
        └──────────────┬──────────────┘
                       │
          - Sort by group columns and time
          - Create date-based cyclical features
          - Can be overridden by subclasses (e.g., StockPredictor adds vwap)
          
        ┌──────────────┴────────────────────┐
        │ Step 2: Create Shifted Targets     │
        │ (create_shifted_targets)          │
        └──────────────┬────────────────────┘
                       │
          - Single-target or multi-target
          - Single-horizon or multi-horizon (e.g., h1, h2, h3)
          - Supports group-based shifting to prevent data leakage
          - Creates columns: '{target}_target_h{horizon}'
          - Removes rows with NaN targets

        ┌──────────────┴────────────────────────────────────┐
        │ Step 3: Store Dataframe for Evaluation (Optional) │
        │ (store_for_evaluation=True in predict())          │
        └──────────────┬────────────────────────────────────┘
                       │
          - Saves unscaled/unencoded dataframe
          - Used for evaluation alignment

        ┌──────────────┴──────────────────────┐
        │ Step 4: Encode Categorical Features  │
        │ (_encode_categorical_features)       │
        └──────────────┬──────────────────────┘
                       │
          - Uses LabelEncoder for each categorical column
          - Stores encoder in self.cat_encoders dict
          - Categorical columns added to feature set

        ┌──────────────┴──────────────────────┐
        │ Step 5: Determine Numerical Columns  │
        │ (_determine_numerical_columns)       │
        └──────────────┬──────────────────────┘
                       │
          - Identifies all numeric columns (excluding categoricals)
          - Sets self.feature_columns attribute

        ┌──────────────┴─────────────────────────────────────┐
        │ Step 6: Scale Numerical Features and Targets        │
        │ (_scale_features_grouped or _scale_features_single) │
        └──────────────┬─────────────────────────────────────┘
                       │
          GROUP-BASED SCALING (_scale_features_grouped):
          ├─ For each unique group:
          │  ├─ Fit/apply scaler to group's features
          │  └─ Fit/apply per-horizon scalers to shifted targets
          │     (e.g., close_target_h1, close_target_h2)
          └─ Per-horizon target scalers enable accurate inverse transform
          
          SINGLE-GROUP SCALING (_scale_features_single):
          ├─ Fit/apply single scaler to all features
          └─ Fit/apply per-horizon scalers to shifted targets

        ┌──────────────┴─────────────────────────────────────┐
        │ Step 7: Create Sequences and Extract Targets        │
        │ (_prepare_data_grouped or direct sequence creation) │
        └──────────────┬─────────────────────────────────────┘
                       │
          GROUPED MODE:
          ├─ For each group:
          │  ├─ Create sequences using _create_sequences_with_categoricals
          │  └─ Extract already-scaled targets with offset
          └─ Concatenate all groups
          
          NON-GROUPED MODE:
          ├─ Create sequences using _create_sequences_with_categoricals
          └─ Extract already-scaled targets with offset

        ┌──────────────┴───────────────────────────────┐
        │  OUTPUT: (X, y) as PyTorch Tensors           │
        │  X: (X_num, X_cat) tuple or single tensor    │
        │  y: Target tensor (already scaled)           │
        └───────────────────────────────────────────────┘
```

### Sequence Creation Details (`_create_sequences_with_categoricals()` at Line 401)

**Input**: 
- Encoded and scaled DataFrame
- `sequence_length`: Number of historical timesteps
- `numerical_columns`: Features for 3D sequences
- `categorical_columns`: Static features per sequence

**Process**:
1. **Numerical Sequences (3D)**:
   - Creates sliding windows of shape `(n_sequences, sequence_length, n_numerical_features)`
   - Uses `create_input_variable_sequence()` internally

2. **Categorical Features (2D)**:
   - Extracts values from **LAST timestep** of each sequence
   - For sequence `i`: uses categorical from df index `sequence_length - 1 + i`
   - Shape: `(n_sequences, n_categorical_features)`
   - These are static per sequence (shared across all timesteps in a sequence)

**Example** (sequence_length=10):
```
DataFrame (100 rows):
Row 0    Row 1    ...  Row 9    [← First sequence's last row (categorical extracted from here)]
Row 1    Row 2    ...  Row 10
...
Row 90   Row 91   ...  Row 99   [← Last sequence's last row]

Output shapes:
X_num: (91, 10, n_features)     [91 sequences × 10 timesteps × n features]
X_cat: (91, n_categorical)      [91 sequences × categorical features from rows 9, 10, ..., 99]
```

### Feature Column Determination

**Method**: `_determine_numerical_columns()` (implicit in Step 5)

**Logic**:
- Starts with all numeric columns
- Excludes:
  - Categorical columns (already encoded)
  - Shifted target columns (`{target}_target_h{h}`)
  - Group columns (used for filtering, not features)
  - Date/datetime columns (converted to cyclical features)

**Result**: `self.feature_columns` contains only input features (no targets or group info)

### Scaler Structure

**Single-Group Scalers**:
```python
self.scaler                    # Features: StandardScaler or variant
self.target_scalers_dict       # Dict[{target}_target_h{h}, Scaler]
```

**Multi-Group Scalers**:
```python
self.group_feature_scalers     # Dict[group_key, Scaler]
self.group_target_scalers      # Dict[group_key, Dict[{target}_target_h{h}, Scaler]]
```

---

## 5. KEY ATTRIBUTES TRACKING

### Initialized in __init__
```python
target_columns              # List[str] - columns to predict
sequence_length            # int - lookback window size
prediction_horizon         # int - steps ahead to predict
group_columns              # List[str] - for group-based scaling
categorical_columns        # List[str] - for encoding
is_multi_target            # bool - derived from target_columns
num_targets                # int - len(target_columns)
model_type                 # str - 'ft_transformer', 'csn_transformer'
scaler_type                # str - 'standard', 'minmax', 'robust', etc.
use_lagged_target_features # bool - include targets in sequences
```

### Set During Training
```python
model                      # PyTorch model instance
feature_columns            # List[str] - input feature columns
numerical_columns          # List[str] - numeric feature columns
cat_encoders               # Dict[col, LabelEncoder]
cat_cardinalities          # List[int] - vocab size per category
scaler                     # StandardScaler for features
target_scalers_dict        # Dict for single-group targets
group_feature_scalers      # Dict for group-based feature scaling
group_target_scalers       # Dict for group-based target scaling
```

### Set During Predict
```python
_last_processed_df         # Unscaled/unencoded df (for evaluation)
_last_group_indices        # List tracking group of each prediction
```

---

## 6. IMPORTANT IMPLEMENTATION DETAILS

### Shifted Target Creation
- Uses `df.groupby(group_column)[target_col].shift(-h)` for group-based shifting
- Forward shift by `-h` steps: targets are from future rows
- Automatic NaN removal after shifting

### Multi-Horizon Prediction
- Creates separate shifted columns: `close_target_h1`, `close_target_h2`, `close_target_h3`
- Each horizon has its own scaler for accurate inverse transform
- Multi-horizon output layout: `[h1, h2, h3, ...]` per target

### Memory Optimization
- Batched prediction (batch_size=256) to avoid OOM
- GPU cache clearing after each batch
- Feature cache with DataFrame hashing
- Immediate CPU movement of predictions

### Temporal Integrity
- Sorting by group + time before shifting ensures no data leakage
- Group-wise splitting ensures test sets don't contain training groups' future data
- Sequence offset calculation (`sequence_length - 1`) aligns inputs with targets

---

## 7. MAIN FILES SUMMARY

| File | Purpose | Key Methods |
|------|---------|------------|
| `core/predictor.py` | Core prediction class | `fit()`, `predict()`, `prepare_data()`, `_create_sequences_with_categoricals()` |
| `preprocessing/time_features.py` | Feature engineering | `create_shifted_targets()`, `create_input_variable_sequence()`, `create_date_features()` |
| `core/utils.py` | Utilities | `split_time_series()`, `calculate_metrics()` |
| `preprocessing/scaler_factory.py` | Scaler creation | `ScalerFactory.create_scaler()` |
| `core/base/model_factory.py` | Model instantiation | `ModelFactory.create_model()` |

