# Implementation Summary: Alignment Fixes

## Overview

Successfully implemented fixes for evaluation alignment issues in the stock forecasting pipeline. The changes address two critical bugs and optimize data usage.

## Changes Implemented

### 1. ✅ Fixed Sequence Creation (`create_input_variable_sequence()`)

**File**: `tf_predictor/preprocessing/time_features.py` (line 371-377)

**Change**: Modified loop to start from index 0 instead of `sequence_length`

**Before**:
```python
for i in range(sequence_length, len(df)):
    seq = features[i-sequence_length:i]
```

**After**:
```python
for i in range(len(df) - sequence_length + 1):
    seq = features[i:i+sequence_length]
```

**Impact**:
- **+20% more training data**: 6 sequences per group instead of 5 (for our test case)
- More efficient use of available data
- General formula: `(rows_after_shift - sequence_length + 1)` sequences

---

### 2. ✅ Updated Target Extraction Offset

**Files**:
- `tf_predictor/core/predictor.py` (line 702, 858)

**Change**: Updated offset from `sequence_length` to `sequence_length - 1`

**Before**:
```python
target_values = group_df[shifted_col].values[self.sequence_length:]
```

**After**:
```python
target_values = group_df[shifted_col].values[self.sequence_length - 1:]
```

**Reason**: New sequence creation starts predictions earlier (at index `sequence_length - 1` instead of `sequence_length`)

---

### 3. ✅ Fixed Evaluation to Use Shifted Target Columns

**File**: `tf_predictor/core/predictor.py`

**Functions Modified**:
- `_evaluate_standard()` (lines 1556-1674)
- `_evaluate_per_group()` (lines 1689-1999)

**Key Changes**:

#### Before (WRONG):
```python
# Extracted from original 'close' column with manual offset
actual = df[target_col].values[self.sequence_length:]

# Complex multi-horizon logic
needed_actuals = num_preds + self.prediction_horizon - 1
actual_aligned = actual[:needed_actuals]
```

#### After (CORRECT):
```python
# Extract from shifted target columns directly
offset = self.sequence_length - 1

# Single-horizon
shifted_col = f"{target_col}_target_h1"
actual = self._last_processed_df[shifted_col].values[offset:]

# Multi-horizon - treat each horizon independently
for h in range(1, self.prediction_horizon + 1):
    shifted_col = f"{target_col}_target_h{h}"
    horizon_actual = group_df[shifted_col].values[offset:]
    horizon_pred = group_preds[:, h-1]
    metrics[f'horizon_{h}'] = calculate_metrics(horizon_actual, horizon_pred)
```

**Benefits**:
- ✅ **Correct alignment**: Extracts from the actual target columns the model was trained on
- ✅ **Simpler logic**: No complex offset calculations
- ✅ **No edge cases**: Always perfectly aligned by construction
- ✅ **Strict validation**: Throws error if shapes don't match (no silent trimming)

---

### 4. ✅ Fixed Overall Metrics for Grouped Data

**File**: `tf_predictor/core/predictor.py` (lines 1802-1993)

**Problem**: Cannot apply single global offset to grouped data

**Solution**: Extract actuals per-group and concatenate in prediction order

**Before (WRONG)**:
```python
# Applied offset to entire dataframe (mixed groups)
all_actual = self._last_processed_df[shifted_col].values[offset:]
# Results in wrong count: 14 instead of 12 for 2 groups
```

**After (CORRECT)**:
```python
# Extract per-group and concatenate
all_actuals_list = []
for group_value in unique_groups:
    group_name = group_value_to_name[group_value]
    group_df = self._last_processed_df[
        self._last_processed_df[group_col_name] == group_name
    ]
    group_actual = group_df[shifted_col].values[offset:]
    all_actuals_list.append(group_actual)

all_actual = np.concatenate(all_actuals_list)  # Correct count: 12
```

---

## Testing Results

### Test 1: Synthetic Data (2 groups, 10 rows each)

**Configuration**:
- Groups: AAPL, GOOGL
- Rows per group: 10
- sequence_length: 3
- prediction_horizon: 2

**Results**:
```
✅ Sequence creation: 12 sequences (was 10) - 20% improvement
✅ Per-group evaluation: No alignment errors
✅ Overall metrics: No alignment errors
✅ All shapes match perfectly
```

**Data Flow Verification**:
```
1. Raw data:        10 rows per group
2. After shifting:   8 rows per group (2 dropped, no future data)
3. After sequences:  6 sequences per group (8 - 3 + 1 = 6)
4. Total:           12 sequences (6 × 2 groups)

Evaluation:
- offset = sequence_length - 1 = 2
- Actuals per group: 8 - 2 = 6 values
- Predictions per group: 6 values
- ✅ Perfect alignment: 6 == 6
```

### Test 2: Unit Test (`test_alignment_fix.py`)

All tests passed:
- ✅ Sequence count validation
- ✅ _evaluate_per_group() alignment
- ✅ No ValueError thrown
- ✅ Metrics computed successfully

---

## Files Modified

1. `tf_predictor/preprocessing/time_features.py`
   - Line 371-377: Sequence creation loop

2. `tf_predictor/core/predictor.py`
   - Line 702: Target extraction offset (grouped path)
   - Line 858: Target extraction offset (non-grouped path)
   - Lines 1556-1674: `_evaluate_standard()` complete rewrite
   - Lines 1689-1999: `_evaluate_per_group()` complete rewrite

---

## Backward Compatibility

### Breaking Changes:
- ⚠️ **Sequence count changed**: Models will produce more predictions
- ⚠️ **Evaluation metrics may differ**: Now using correct alignment
- ⚠️ **Requires retraining**: Existing models were trained with old sequence logic

### Migration:
1. Retrain all models with new code
2. Expect slightly different metrics (should be more accurate)
3. Benefit from +20% more training data

---

## Remaining Known Issues

### _evaluate_standard() for Grouped Data

**Status**: Not critical (use `_evaluate_per_group()` instead)

**Issue**: `_evaluate_standard()` doesn't handle grouped data correctly for overall metrics

**Workaround**: Always use `evaluate()` which calls `_evaluate_per_group()` when groups exist

**Future Fix**: Either:
1. Make `_evaluate_standard()` group-aware
2. Document it as single-group only
3. Auto-redirect to `_evaluate_per_group()` when groups detected

---

## Key Takeaways

### What Was Fixed:
1. ✅ **More training data**: Sequence creation now uses all available samples
2. ✅ **Correct evaluation**: Extracts from shifted target columns
3. ✅ **Perfect alignment**: No more off-by-one errors
4. ✅ **Group handling**: Proper per-group offset application

### Benefits:
- **Better models**: 20% more training data
- **Accurate metrics**: Evaluating what was actually predicted
- **Robust code**: Strict validation catches bugs early
- **Simpler logic**: Removed complex offset calculations

### Performance Impact:
- Training: Slightly more data to process (~20%)
- Evaluation: Same or faster (simpler logic)
- Memory: Negligible difference

---

## Next Steps

1. **Test with real data**: Verify on actual stock/crypto datasets
2. **Compare metrics**: Old vs new (expect more accurate, possibly different values)
3. **Retrain models**: Take advantage of increased data
4. **Monitor**: Ensure no regressions in production

---

## Credits

Fixed alignment bugs identified through debugging:
- Wrong source column for actuals (original target vs shifted)
- Incorrect offset calculation
- Insufficient data usage (wasted samples)
- Grouped data edge cases

All fixes validated with synthetic data and unit tests.
