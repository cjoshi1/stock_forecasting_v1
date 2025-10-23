# Logical Checks for Time Series Forecasting Code

## Purpose
This document outlines all logical checks that should be performed on the stock forecasting codebase to ensure correctness, especially for grouped time series data with multiple prediction horizons.

## 1. Temporal Order Checks

### 1.1 Data Loading and Sorting
- [ ] **Check 1.1.1**: Data is sorted by group (symbol) first, then by date
- [ ] **Check 1.1.2**: Sorting is maintained throughout the entire pipeline
- [ ] **Check 1.1.3**: No shuffling occurs that breaks temporal order within groups
- [ ] **Check 1.1.4**: Train/val/test splits respect temporal order (no future data leaking into past)

### 1.2 Sequence Creation
- [ ] **Check 1.2.1**: Sequences are created within each group independently
- [ ] **Check 1.2.2**: No sequences span across different groups
- [ ] **Check 1.2.3**: Sequences maintain chronological order (t, t+1, t+2, ...)
- [ ] **Check 1.2.4**: Target values are always in the future relative to input features

### 1.3 Data Split Timing
- [ ] **Check 1.3.1**: Split is done before any feature engineering that uses future information
- [ ] **Check 1.3.2**: Train split uses only oldest data, test split uses only newest data
- [ ] **Check 1.3.3**: Validation split is temporally between train and test

## 2. Scaling Checks

### 2.1 Group-Based Scaling
- [ ] **Check 2.1.1**: Scalers are fit separately for each group
- [ ] **Check 2.1.2**: Each group uses only its own scaler for transformation
- [ ] **Check 2.1.3**: Group assignment is correct during scaling (no mixing of groups)
- [ ] **Check 2.1.4**: Scalers are stored and retrieved correctly for each group

### 2.2 Input Feature Scaling
- [ ] **Check 2.2.1**: All numeric input features are scaled
- [ ] **Check 2.2.2**: Categorical/non-numeric columns (like 'symbol') are excluded from scaling
- [ ] **Check 2.2.3**: Date columns are excluded from scaling
- [ ] **Check 2.2.4**: Scaler is fit ONLY on training data
- [ ] **Check 2.2.5**: Same scaler (fit on train) is used to transform val/test data

### 2.3 Target Variable Scaling
- [ ] **Check 2.3.1**: Target variables are scaled separately from features
- [ ] **Check 2.3.2**: Each target variable has its own scaler per group
- [ ] **Check 2.3.3**: For multi-horizon targets (t+1, t+2, t+3), the SAME scaler is used for all horizons of the same variable
- [ ] **Check 2.3.4**: Target scalers are fit ONLY on training targets
- [ ] **Check 2.3.5**: Predictions are inverse-transformed using the correct group's scaler

### 2.4 Scaler Application Order
- [ ] **Check 2.4.1**: Scalers are fit on train data first
- [ ] **Check 2.4.2**: Transform is applied in correct order: train → val → test
- [ ] **Check 2.4.3**: No data leakage from val/test into scaler fitting

## 3. Multi-Horizon Prediction Checks

### 3.1 Target Creation
- [ ] **Check 3.1.1**: Multiple horizons (t+1, t+2, t+3) are created correctly for each target
- [ ] **Check 3.1.2**: Horizon columns are named consistently (e.g., 'close_horizon_1', 'close_horizon_2')
- [ ] **Check 3.1.3**: Correct shift is applied for each horizon
- [ ] **Check 3.1.4**: NaN values from shifting are handled properly

### 3.2 Scaling Consistency Across Horizons
- [ ] **Check 3.2.1**: All horizons of 'close' use the same 'close' scaler
- [ ] **Check 3.2.2**: All horizons of 'volume' use the same 'volume' scaler
- [ ] **Check 3.2.3**: Multi-target predictions maintain scaler mapping correctly
- [ ] **Check 3.2.4**: Inverse transform uses correct scaler for each target type

### 3.3 Prediction Output Structure
- [ ] **Check 3.3.1**: Model outputs correct number of values (num_targets × num_horizons)
- [ ] **Check 3.3.2**: Output dimensions match target dimensions
- [ ] **Check 3.3.3**: Predictions are mapped to correct horizon and target

## 4. Train/Val/Test Split Checks

### 4.1 Split Strategy
- [ ] **Check 4.1.1**: Split is done per group (each group has same split logic)
- [ ] **Check 4.1.2**: Split respects temporal order within each group
- [ ] **Check 4.1.3**: Test size and val size are applied correctly per group
- [ ] **Check 4.1.4**: No overlap between train/val/test within any group

### 4.2 Split Integrity
- [ ] **Check 4.2.1**: All samples are accounted for (train + val + test = total per group)
- [ ] **Check 4.2.2**: No samples are duplicated across splits
- [ ] **Check 4.2.3**: Group assignment is maintained in all splits
- [ ] **Check 4.2.4**: Date ranges don't overlap between splits

## 5. Feature Engineering Checks

### 5.1 Feature Creation Timing
- [ ] **Check 5.1.1**: Features using aggregations (rolling windows) don't use future data
- [ ] **Check 5.1.2**: Lagged features are created correctly (using past data only)
- [ ] **Check 5.1.3**: Feature creation respects group boundaries
- [ ] **Check 5.1.4**: Features are created before splitting (if they don't cause leakage)

### 5.2 Derived Features
- [ ] **Check 5.2.1**: VWAP calculation is correct: (high + low + close) / 3
- [ ] **Check 5.2.2**: Cyclical features (sin/cos) are computed correctly
- [ ] **Check 5.2.3**: Date extraction (month, day of week) uses correct timezone
- [ ] **Check 5.2.4**: Crypto mode uses 7-day week correctly

### 5.3 Feature Consistency
- [ ] **Check 5.3.1**: Same features are created for train/val/test
- [ ] **Check 5.3.2**: Feature order is consistent across all datasets
- [ ] **Check 5.3.3**: Missing features are handled consistently

## 6. Data Leakage Checks

### 6.1 Information Leakage
- [ ] **Check 6.1.1**: No future information is used in input features
- [ ] **Check 6.1.2**: Validation/test data not used in any training-time calculations
- [ ] **Check 6.1.3**: Scalers fit only on training data
- [ ] **Check 6.1.4**: No look-ahead bias in feature engineering

### 6.2 Cross-Group Leakage
- [ ] **Check 6.2.1**: Statistics from one group don't affect another group
- [ ] **Check 6.2.2**: Scaling is group-specific
- [ ] **Check 6.2.3**: No global statistics that mix groups (unless intended)

## 7. Sequence and Batch Creation Checks

### 7.1 Sequence Generation
- [ ] **Check 7.1.1**: Sequences respect group boundaries (no cross-group sequences)
- [ ] **Check 7.1.2**: Sequence length is consistent
- [ ] **Check 7.1.3**: Overlapping sequences are created correctly (sliding window)
- [ ] **Check 7.1.4**: Insufficient data groups are handled properly

### 7.2 Batch Creation
- [ ] **Check 7.2.1**: Batches maintain data integrity
- [ ] **Check 7.2.2**: Shuffle (if used) only happens within training, not val/test
- [ ] **Check 7.2.3**: Group information is preserved in batches
- [ ] **Check 7.2.4**: Padding (if used) is applied correctly

## 8. Model Architecture Checks

### 8.1 Input/Output Dimensions
- [ ] **Check 8.1.1**: Model input dimension matches feature dimension
- [ ] **Check 8.1.2**: Model output dimension matches (num_targets × num_horizons)
- [ ] **Check 8.1.3**: Sequence length matches expected input
- [ ] **Check 8.1.4**: Batch dimension is handled correctly

### 8.2 Model Configuration
- [ ] **Check 8.2.1**: Model configured for multi-target prediction
- [ ] **Check 8.2.2**: Model configured for multi-horizon prediction
- [ ] **Check 8.2.3**: Activation functions are appropriate
- [ ] **Check 8.2.4**: Dropout is applied correctly

## 9. Training Process Checks

### 9.1 Training Loop
- [ ] **Check 9.1.1**: Model is in train mode during training
- [ ] **Check 9.1.2**: Model is in eval mode during validation
- [ ] **Check 9.1.3**: Gradients are computed correctly
- [ ] **Check 9.1.4**: Optimizer updates weights correctly

### 9.2 Loss Calculation
- [ ] **Check 9.2.1**: Loss is calculated on correct targets
- [ ] **Check 9.2.2**: Loss accounts for all targets and horizons
- [ ] **Check 9.2.3**: Loss is averaged/summed correctly
- [ ] **Check 9.2.4**: NaN/Inf losses are handled

### 9.3 Early Stopping
- [ ] **Check 9.3.1**: Validation loss is monitored correctly
- [ ] **Check 9.3.2**: Best model is saved based on validation performance
- [ ] **Check 9.3.3**: Patience mechanism works correctly
- [ ] **Check 9.3.4**: Training stops at right time

## 10. Evaluation Checks

### 10.1 Metric Calculation
- [ ] **Check 10.1.1**: Predictions are inverse-transformed before metric calculation
- [ ] **Check 10.1.2**: Ground truth is inverse-transformed using same scaler
- [ ] **Check 10.1.3**: Metrics are calculated per group correctly
- [ ] **Check 10.1.4**: Metrics are calculated per horizon correctly

### 10.2 Metric Aggregation
- [ ] **Check 10.2.1**: Overall metrics are aggregated correctly from per-group metrics
- [ ] **Check 10.2.2**: Per-horizon metrics are separated correctly
- [ ] **Check 10.2.3**: Multi-target metrics are handled correctly
- [ ] **Check 10.2.4**: Weighted vs unweighted aggregation is correct

### 10.3 Metric Validity
- [ ] **Check 10.3.1**: R² calculation is correct (especially for negative values)
- [ ] **Check 10.3.2**: MAPE handles zero values correctly
- [ ] **Check 10.3.3**: Directional accuracy calculation is correct
- [ ] **Check 10.3.4**: All metrics use same data (aligned predictions and targets)

## 11. Inverse Transform Checks

### 11.1 Transform Mapping
- [ ] **Check 11.1.1**: Predictions are mapped to correct group before inverse transform
- [ ] **Check 11.1.2**: Correct target scaler is used (close vs volume)
- [ ] **Check 11.1.3**: All horizons of same target use same scaler
- [ ] **Check 11.1.4**: Shape is preserved during inverse transform

### 11.2 Transform Correctness
- [ ] **Check 11.2.1**: Inverse transform formula is correct
- [ ] **Check 11.2.2**: Mean and std are applied in correct order
- [ ] **Check 11.2.3**: Transform and inverse transform are truly inverse operations
- [ ] **Check 11.2.4**: Numerical precision is maintained

## 12. Edge Cases

### 12.1 Data Quality
- [ ] **Check 12.1.1**: Missing values are handled correctly
- [ ] **Check 12.1.2**: Outliers don't break scaling
- [ ] **Check 12.1.3**: Zero/negative values in volume are handled
- [ ] **Check 12.1.4**: Duplicate timestamps are handled

### 12.2 Group Edge Cases
- [ ] **Check 12.2.1**: Groups with insufficient data are handled
- [ ] **Check 12.2.2**: Single-sample groups don't break pipeline
- [ ] **Check 12.2.3**: Empty groups after filtering are handled
- [ ] **Check 12.2.4**: New groups in test data are handled

### 12.3 Numerical Stability
- [ ] **Check 12.3.1**: Division by zero is prevented in scaling
- [ ] **Check 12.3.2**: Very large/small values don't cause overflow
- [ ] **Check 12.3.3**: NaN/Inf values are detected and handled
- [ ] **Check 12.3.4**: Numerical precision is sufficient for metrics

## 13. Code Consistency Checks

### 13.1 Parameter Passing
- [ ] **Check 13.1.1**: Group column is passed correctly through all functions
- [ ] **Check 13.1.2**: Target columns are consistently referenced
- [ ] **Check 13.1.3**: Sequence length is used consistently
- [ ] **Check 13.1.4**: Prediction horizons are handled consistently

### 13.2 Data Structure Consistency
- [ ] **Check 13.2.1**: Column names are consistent (no renaming issues)
- [ ] **Check 13.2.2**: Index is maintained correctly through operations
- [ ] **Check 13.2.3**: Data types are preserved or converted correctly
- [ ] **Check 13.2.4**: Group identifiers are preserved

## 14. Specific Issues to Investigate

Based on your observations of negative R² and poor performance:

### 14.1 Priority Checks
- [ ] **Check 14.1.1**: VERIFY temporal order is maintained within each group throughout entire pipeline
- [ ] **Check 14.1.2**: VERIFY scaling is done per-group for both inputs AND outputs
- [ ] **Check 14.1.3**: VERIFY same scaler is used for all horizons of the same variable (close_horizon_1, close_horizon_2, close_horizon_3 all use "close" scaler)
- [ ] **Check 14.1.4**: VERIFY inverse transform uses correct group-specific scaler
- [ ] **Check 14.1.5**: VERIFY train/val/test split doesn't break temporal order per group
- [ ] **Check 14.1.6**: VERIFY no data leakage from future to past
- [ ] **Check 14.1.7**: VERIFY predictions and ground truth are aligned correctly (same timestamps, same groups)
- [ ] **Check 14.1.8**: VERIFY metric calculation uses correctly inverse-transformed values

### 14.2 Red Flags to Look For
- [ ] **Red Flag 1**: R² significantly worse on validation/test than training → potential data leakage or scaling issue
- [ ] **Red Flag 2**: Very high MAPE on certain groups → potential scaling issue for that group
- [ ] **Red Flag 3**: Directional accuracy around 50% → model not learning patterns
- [ ] **Red Flag 4**: Negative R² → model worse than baseline, possible inverse transform issue
- [ ] **Red Flag 5**: Different metrics for same variable across horizons → potential scaler inconsistency

## Checklist Completion

- **Total Checks**: 135+
- **Critical Priority**: Checks marked 14.1.x
- **High Priority**: Checks in sections 1, 2, 3, 10, 11
- **Medium Priority**: Checks in sections 4, 5, 6, 7
- **Low Priority**: Checks in sections 12, 13

## Next Steps

1. Start with Priority Checks (14.1.x)
2. Add instrumentation/logging to verify each check
3. Create unit tests for critical checks
4. Document findings for each failed check
5. Fix issues in order of priority
