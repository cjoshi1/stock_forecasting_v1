# TF_Predictor - Code Reference and Implementation Examples

## Quick Reference: Core Method Locations

### Prediction Methods
- **`TimeSeriesPredictor.predict()`**: Line 1389 in `core/predictor.py`
  - Handles single/multi-target and single/multi-horizon predictions
  - Manages group-based inverse transformation
  
- **`TimeSeriesPredictor.fit()`**: Line 1037 in `core/predictor.py`
  - Training orchestration
  - Model initialization and training loop

### Data Preparation Methods
- **`TimeSeriesPredictor.prepare_data()`**: Line 912 in `core/predictor.py`
  - 7-step preprocessing pipeline
  - Returns (X, y) PyTorch tensors

- **`TimeSeriesPredictor._prepare_data_grouped()`**: Line 758 in `core/predictor.py`
  - Group-aware sequence creation
  - Per-group scaling already applied

- **`TimeSeriesPredictor._create_sequences_with_categoricals()`**: Line 401 in `core/predictor.py`
  - Creates 3D numerical sequences
  - Extracts 2D categorical features from last timestep

### Feature Engineering Methods
- **`create_shifted_targets()`**: Line 182 in `preprocessing/time_features.py`
  - Forward-shifts target values for autoregressive training
  - Supports group-based shifting to prevent leakage

- **`create_input_variable_sequence()`**: Line 296 in `preprocessing/time_features.py`
  - Sliding window sequence creation
  - Core X data generation

- **`create_date_features()`**: Line 81 in `preprocessing/time_features.py`
  - Cyclical encoding of temporal features
  - Automatic component detection

### Utility Methods
- **`split_time_series()`**: Line 114 in `core/utils.py`
  - Temporal data splitting
  - Group-aware train/val/test partitioning

---

## Code Examples and Patterns

### Example 1: Basic Predict Method Flow

```python
# In TimeSeriesPredictor.predict() - Line 1389+
def predict(self, df: pd.DataFrame, return_group_info: bool = False):
    # Step 1: Prepare data (7-step pipeline)
    X, _ = self.prepare_data(df, fit_scaler=False, store_for_evaluation=True)
    
    # Step 2: Batched inference
    self.model.eval()
    with torch.no_grad():
        batch_size = 256
        all_predictions = []
        
        for i in range(0, num_samples, batch_size):
            # Handle both categorical (tuple) and standard (tensor) inputs
            if isinstance(X, tuple):
                X_num_batch = X_num[i:batch_end].to(self.device)
                X_cat_batch = X_cat[i:batch_end].to(self.device)
                batch_preds = self.model(X_num_batch, X_cat_batch)
            else:
                X_batch = X[i:batch_end]
                batch_preds = self.model(X_batch.to(self.device), None)
            
            all_predictions.append(batch_preds.cpu())
            torch.cuda.empty_cache()  # Memory optimization
    
    # Step 3: Inverse transform predictions
    predictions_scaled = torch.cat(all_predictions, dim=0).numpy()
    
    # For group-based prediction:
    for group_value in self.group_target_scalers.keys():
        group_mask = [g == group_value for g in self._last_group_indices]
        for h in range(self.prediction_horizon):
            shifted_col = f"{target_col}_target_h{h+1}"
            scaler = self.group_target_scalers[group_value][shifted_col]
            predictions[group_mask] = scaler.inverse_transform(...)
    
    return predictions
```

### Example 2: Create Shifted Targets

```python
# In preprocessing/time_features.py - Line 182+
def create_shifted_targets(df, target_column, prediction_horizon=1, 
                          group_column=None, verbose=False):
    
    # Normalize to list
    target_columns_list = [target_column] if isinstance(target_column, str) else list(target_column)
    
    for target_col in target_columns_list:
        if prediction_horizon == 1:
            # Single horizon: create one shifted column
            shifted_name = f"{target_col}_target_h1"
            if group_column and group_column in df.columns:
                # Group-based shifting prevents cross-group leakage
                df[shifted_name] = df.groupby(group_column)[target_col].shift(-1)
            else:
                df[shifted_name] = df[target_col].shift(-1)
        else:
            # Multi-horizon: create multiple shifted columns
            for h in range(1, prediction_horizon + 1):
                col_name = f"{target_col}_target_h{h}"
                if group_column:
                    df[col_name] = df.groupby(group_column)[target_col].shift(-h)
                else:
                    df[col_name] = df[target_col].shift(-h)
    
    # Remove rows with any NaN targets
    all_shifted = [f"{t}_target_h{h}" for t in target_columns_list 
                   for h in range(1, prediction_horizon + 1)]
    df = df.dropna(subset=all_shifted)
    
    return df
```

### Example 3: Sliding Window Sequence Creation

```python
# In preprocessing/time_features.py - Line 296+
def create_input_variable_sequence(df, sequence_length, feature_columns=None):
    
    # Determine features to use
    if feature_columns is None:
        exclude_set = set(exclude_columns or [])
        feature_columns = [col for col in df.columns 
                          if col not in exclude_set and pd.api.types.is_numeric_dtype(df[col])]
    
    features = df[feature_columns].values  # (n_rows, n_features)
    sequences = []
    
    # Sliding window: creates (len(df) - sequence_length + 1) sequences
    for i in range(len(df) - sequence_length + 1):
        seq = features[i:i+sequence_length]  # (sequence_length, n_features)
        sequences.append(seq)
    
    return np.array(sequences)  # (n_samples, sequence_length, n_features)
```

**Important**: If df has 100 rows and sequence_length=10:
- Creates 91 sequences (not 90!)
- Sequence 0: rows [0:10]
- Sequence 90: rows [90:100]

### Example 4: Grouped Data Preparation

```python
# In TimeSeriesPredictor._prepare_data_grouped() - Line 758+
def _prepare_data_grouped(self, df_processed, fit_scaler):
    
    # Create group keys
    df_processed['_group_key'] = self._create_group_key(df_processed)
    unique_groups = sorted(df_processed['_group_key'].unique())
    
    all_sequences_num = []
    all_sequences_cat = []
    all_targets = []
    group_indices = []
    
    for group_value in unique_groups:
        group_mask = df_processed['_group_key'] == group_value
        group_df = df_processed[group_mask].copy()
        
        # Skip groups with insufficient data
        if len(group_df) <= self.sequence_length:
            continue
        
        # Create sequences for this group
        # Features are already scaled by group
        sequences_num, sequences_cat = self._create_sequences_with_categoricals(
            group_df,
            self.sequence_length,
            numerical_feature_cols,
            self.categorical_columns
        )
        
        # Extract targets (already scaled per-group)
        y_list = []
        for target_col in self.target_columns:
            for h in range(1, self.prediction_horizon + 1):
                shifted_col = f"{target_col}_target_h{h}"
                # Offset: sequence_length - 1 aligns targets with sequences
                target_values = group_df[shifted_col].values[self.sequence_length - 1:]
                y_list.append(target_values)
        
        # Combine targets (shape: (n_sequences,) or (n_sequences, n_targets))
        y_combined = y_list[0] if len(y_list) == 1 else np.column_stack(y_list)
        
        all_sequences_num.append(sequences_num)
        if sequences_cat is not None:
            all_sequences_cat.append(sequences_cat)
        all_targets.append(y_combined)
        group_indices.extend([group_value] * len(sequences_num))
    
    # Concatenate all groups
    X_num = np.vstack(all_sequences_num)  # (total_sequences, seq_len, n_features)
    X_cat = np.vstack(all_sequences_cat) if all_sequences_cat else None
    y = np.vstack(all_targets) if self.prediction_horizon > 1 else np.concatenate(all_targets)
    
    self._last_group_indices = group_indices  # Track for inverse transform
    
    return X, y
```

### Example 5: Group-Based Feature Scaling

```python
# In TimeSeriesPredictor._scale_features_grouped() - Line 648+
def _scale_features_grouped(self, df_processed, fit_scaler, shifted_target_columns=None):
    
    df_scaled = df_processed.copy()
    df_scaled['_group_key'] = self._create_group_key(df_scaled)
    unique_groups = sorted(df_scaled['_group_key'].unique())
    
    # Scale features per group
    for group_key in unique_groups:
        group_mask = df_scaled['_group_key'] == group_key
        group_data = df_scaled.loc[group_mask, self.numerical_columns]
        
        if fit_scaler:
            scaler = ScalerFactory.create_scaler(self.scaler_type)
            scaled_data = scaler.fit_transform(group_data)
            self.group_feature_scalers[group_key] = scaler
        else:
            scaler = self.group_feature_scalers[group_key]
            scaled_data = scaler.transform(group_data)
        
        df_scaled.loc[group_mask, self.numerical_columns] = scaled_data
    
    # Scale targets per group, per horizon
    if shifted_target_columns:
        for group_key in unique_groups:
            group_mask = df_scaled['_group_key'] == group_key
            
            for shifted_col in shifted_target_columns:
                values = df_scaled.loc[group_mask, shifted_col].values.reshape(-1, 1)
                
                if fit_scaler:
                    scaler = ScalerFactory.create_scaler(self.scaler_type)
                    df_scaled.loc[group_mask, shifted_col] = scaler.fit_transform(values).flatten()
                    
                    if group_key not in self.group_target_scalers:
                        self.group_target_scalers[group_key] = {}
                    self.group_target_scalers[group_key][shifted_col] = scaler
                else:
                    scaler = self.group_target_scalers[group_key][shifted_col]
                    df_scaled.loc[group_mask, shifted_col] = scaler.transform(values).flatten()
    
    return df_scaled
```

### Example 6: Time Series Splitting (Group-Aware)

```python
# In core/utils.py - Line 114+
def split_time_series(df, test_size=30, val_size=None, 
                     group_column=None, time_column=None, sequence_length=1):
    
    if group_column is None:
        # Simple temporal split
        test_df = df.iloc[-test_size:].copy()
        remaining_df = df.iloc[:-test_size].copy()
        
        if val_size is not None and len(remaining_df) > val_size:
            val_df = remaining_df.iloc[-val_size:].copy()
            train_df = remaining_df.iloc[:-val_size].copy()
        else:
            val_df = None
            train_df = remaining_df.copy()
        
        return train_df, val_df, test_df
    else:
        # Group-wise splitting
        # Sort by group and time
        if time_column and time_column in df.columns:
            df_sorted = df.sort_values([group_column, time_column]).reset_index(drop=True)
        else:
            df_sorted = df.copy()
        
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for group_value in df_sorted[group_column].unique():
            group_df = df_sorted[df_sorted[group_column] == group_value].copy()
            
            # Check minimum data requirement
            min_train_samples = sequence_length * 2 + 10
            min_required = test_size + (val_size if val_size else 0) + min_train_samples
            
            if len(group_df) < min_required:
                print(f"Skipping group '{group_value}' - insufficient data")
                continue
            
            # Split: test from end, validation from middle, train from beginning
            group_test = group_df.iloc[-test_size:].copy()
            group_remaining = group_df.iloc[:-test_size].copy()
            
            if val_size is not None and len(group_remaining) > val_size:
                group_val = group_remaining.iloc[-val_size:].copy()
                group_train = group_remaining.iloc[:-val_size].copy()
            else:
                group_val = None
                group_train = group_remaining.copy()
            
            train_dfs.append(group_train)
            if group_val is not None:
                val_dfs.append(group_val)
            test_dfs.append(group_test)
        
        # Concatenate all groups
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else None
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        return train_df, val_df, test_df
```

### Example 7: Categorical Feature Extraction

```python
# In TimeSeriesPredictor._create_sequences_with_categoricals() - Line 401+
def _create_sequences_with_categoricals(self, df, sequence_length, 
                                       numerical_columns, categorical_columns):
    
    # Step 1: Create 3D numerical sequences
    X_num = create_input_variable_sequence(
        df,
        sequence_length,
        feature_columns=numerical_columns
    )  # Shape: (n_sequences, sequence_length, n_numerical)
    
    # Step 2: Extract categorical from LAST timestep of each sequence
    if categorical_columns:
        num_sequences = len(X_num)
        
        # For sequence i starting at row i in the sliding window:
        # - It uses rows [i:i+sequence_length]
        # - Last timestep is at row i+sequence_length-1
        # So categorical indices are: sequence_length-1, sequence_length, ..., sequence_length-1+num_sequences
        cat_indices = np.arange(sequence_length - 1, sequence_length - 1 + num_sequences)
        X_cat = df[categorical_columns].values[cat_indices]  # Shape: (n_sequences, n_categorical)
        
        # Verify shape consistency
        assert X_cat.shape[0] == X_num.shape[0]
    else:
        X_cat = None
    
    return X_num, X_cat
```

**Key Point**: Categorical features are extracted from the **LAST timestep** of each sequence because they are typically static (e.g., symbol, sector) and don't change within the lookback window.

---

## Data Flow Diagram: From Raw Data to Model Output

```
Raw Data
   │
   ├─ target_column: 'close'
   ├─ sequence_length: 10
   ├─ prediction_horizon: 1
   └─ group_columns: ['symbol']
   
   ▼
[Step 1] _create_base_features()
   │
   ├─ Sort by symbol, date
   ├─ Create cyclical date features (month_sin, month_cos, etc.)
   └─ Output: Base features + cyclical encodings
   
   ▼
[Step 2] create_shifted_targets()
   │
   ├─ Group by symbol
   ├─ Shift 'close' by -1 rows → creates 'close_target_h1'
   ├─ Drop NaN rows
   └─ Output: +1 column (close_target_h1)
   
   ▼
[Step 3] Store unscaled dataframe (for evaluation)
   
   ▼
[Step 4] _encode_categorical_features()
   │
   ├─ LabelEncode 'symbol': AAPL→0, MSFT→1, GOOGL→2
   └─ Output: 'symbol' now contains [0, 1, 2, ...]
   
   ▼
[Step 5] _determine_numerical_columns()
   │
   ├─ Identify numeric columns
   ├─ Exclude: symbol (categorical), close (target), close_target_h1 (shifted target)
   └─ Output: feature_columns = ['open', 'high', 'low', 'volume', 'month_sin', ...]
   
   ▼
[Step 6] _scale_features_grouped()
   │
   ├─ For symbol AAPL:
   │  ├─ Fit StandardScaler on AAPL's features → group_feature_scalers['AAPL']
   │  ├─ Scale AAPL's close_target_h1 → group_target_scalers['AAPL']['close_target_h1']
   │  └─ Apply to AAPL's rows
   │
   ├─ For symbol MSFT: (repeat)
   │
   └─ For symbol GOOGL: (repeat)
   
   ▼
[Step 7] _prepare_data_grouped()
   │
   ├─ For symbol AAPL:
   │  ├─ Create sequences:
   │  │  Sequence 0: rows [0:10] → X_num[0].shape = (10, n_features)
   │  │  Sequence 1: rows [1:11] → X_num[1].shape = (10, n_features)
   │  │  ...
   │  │  N sequences total
   │  │
   │  ├─ Extract categorical (from row 9, 10, ..., 9+N):
   │  │  X_cat[0] = symbol encoded value for row 9 = [1]  (assuming 1 = AAPL)
   │  │
   │  └─ Extract scaled targets (offset by seq_len-1):
   │     y[0] = scaled close_target_h1[9]  ← target for sequence 0
   │
   ├─ For symbol MSFT: (repeat)
   │
   └─ For symbol GOOGL: (repeat)
   
   ▼
Output: X_num (batch, 10, n_features), X_cat (batch, 1), y (batch,)
```

---

## Key Implementation Details

### Target Offset Calculation
```python
# When creating sequences with offset calculation:
sequence_length = 10
num_rows = 100

# Sequences created:
# - Sequence 0: rows [0:10]
# - Sequence 1: rows [1:11]
# - ...
# - Sequence 90: rows [90:100]
# Total: 91 sequences

# Targets must align:
# - Target for sequence 0: shifted_targets[9]  (sequence_length - 1)
# - Target for sequence 1: shifted_targets[10]
# - ...
# - Target for sequence 90: shifted_targets[99]

# Therefore: targets = shifted_targets[sequence_length - 1:]
#            = shifted_targets[9:]
#            = 91 values (matching 91 sequences)
```

### Multi-Horizon Output Layout
```python
# Single target, multi-horizon (horizon=3)
# predict() returns: (batch, 3)
# Layout: [h1, h2, h3]
# Example: [[1.05, 1.10, 1.15],   # h1=1.05, h2=1.10, h3=1.15
#           [2.05, 2.10, 2.15],
#           ...]

# Multi-target, multi-horizon (2 targets, horizon=2)
# predict() returns: dict {'close': (batch, 2), 'volume': (batch, 2)}
# OR as array layout: [close_h1, close_h2, volume_h1, volume_h2]
```

### Group Key Handling
```python
# Single group column:
group_columns = ['symbol']
_create_group_key(df) returns: Series(['AAPL', 'MSFT', 'AAPL', ...])
                              Scalar per row: 'AAPL'

# Multiple group columns:
group_columns = ['symbol', 'sector']
_create_group_key(df) returns: Series([('AAPL', 'Tech'), ('MSFT', 'Tech'), ...])
                              Tuple per row: ('AAPL', 'Tech')
```

