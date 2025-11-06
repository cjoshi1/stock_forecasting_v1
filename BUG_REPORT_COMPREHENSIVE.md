# Comprehensive Bug Report: tf_predictor Module Review

**Date**: 2025-11-05
**Reviewer**: Claude Opus 4.1
**Scope**: Full review of tf_predictor module focusing on grouped operations, scaling, alignment, multi-target, and multi-horizon forecasting

---

## Executive Summary

The tf_predictor module is generally well-architected with proper separation of concerns and comprehensive support for grouped data processing, multi-target forecasting, and multi-horizon predictions. However, **one critical bug** was identified that will cause incorrect results for multi-column grouping scenarios during evaluation.

### Severity Classification:
- **CRITICAL (1)**: Multi-column group evaluation bug
- **MEDIUM (2)**: Potential issues requiring verification
- **LOW (0)**: Minor improvements suggested

---

## CRITICAL BUGS

### 1. Multi-Column Group Evaluation Failure ⚠️ CRITICAL

**Location**: `/home/user/stock_forecasting_v1/tf_predictor/core/predictor.py:1738-1780`

**Description**:
The `_evaluate_per_group()` method only uses the first group column when mapping back to original data during per-group evaluation. This causes incorrect filtering and alignment when using multi-column grouping (e.g., `group_columns=['symbol', 'sector']`).

**Root Cause**:
```python
# Line 1742 - Only takes first column
group_col_name = self.group_columns[0]  # Assuming single group column for now

# Line 1746-1748 - Uses encoder for first column only
encoder = self.cat_encoders[group_col_name]
group_value_to_name = {i: name for i, name in enumerate(encoder.classes_)}

# Line 1778-1780 - Filters using only first column
group_df_processed = self._last_processed_df[
    self._last_processed_df[group_col_name] == group_name
].copy()
```

**The Problem**:
1. During training/prediction, group keys are created as **tuples** for multi-column grouping via `_create_group_key()`:
   - Single column: `'AAPL'`
   - Multi-column: `('AAPL', 'Tech')`

2. These composite keys are stored in `group_indices` and used throughout sequence creation and scaling

3. During evaluation, the code tries to map back using only the **first column**, causing:
   - **Incorrect data filtering**: Gets all rows matching first column value, ignoring other columns
   - **Misalignment**: Predictions grouped by `('AAPL', 'Tech')` get compared to actuals for all `'AAPL'` rows regardless of sector

**Example Failure Scenario**:
```python
# Setup with multi-column grouping
predictor = TimeSeriesPredictor(
    target_column='close',
    group_columns=['symbol', 'sector'],  # Multi-column grouping
    ...
)

# During prediction, group_indices contains:
# [('AAPL', 'Tech'), ('AAPL', 'Tech'), ('GOOGL', 'Tech'), ...]

# During evaluation, code does:
# group_col_name = 'symbol'  # Only first column!
# group_name = 'AAPL'  # Only first value of tuple!
# Filters: _last_processed_df[_last_processed_df['symbol'] == 'AAPL']
# Gets ALL AAPL rows, including other sectors - WRONG!
```

**Impact**:
- **Per-group metrics will be completely wrong** for multi-column grouping
- **Evaluation will not fail** (no error), making this a **silent data corruption bug**
- Affects both multi-target and single-target modes
- Affects both single-horizon and multi-horizon modes

**Files Affected**:
- `/home/user/stock_forecasting_v1/tf_predictor/core/predictor.py:1718-1998` (`_evaluate_per_group` method)

**Recommended Fix**:
```python
# Option 1: Store composite group keys properly
if self.group_columns:
    if len(self.group_columns) == 1:
        # Single column - use existing logic
        group_col_name = self.group_columns[0]
        encoder = self.cat_encoders[group_col_name]
        group_value_to_name = {i: name for i, name in enumerate(encoder.classes_)}

        # Filter using single column
        group_df_processed = self._last_processed_df[
            self._last_processed_df[group_col_name] == group_name
        ].copy()
    else:
        # Multi-column - create composite filter
        # group_value is a tuple like ('AAPL', 'Tech')
        # Need to filter using all columns
        for group_value in unique_groups:
            # group_value is already the tuple from _create_group_key
            filter_conditions = [
                self._last_processed_df[col] == group_value[i]
                for i, col in enumerate(self.group_columns)
            ]
            group_mask = pd.concat(filter_conditions, axis=1).all(axis=1)
            group_df_processed = self._last_processed_df[group_mask].copy()
```

**Testing Required**:
1. Create test with multi-column grouping: `group_columns=['symbol', 'sector']`
2. Verify per-group evaluation returns correct metrics for each (symbol, sector) combination
3. Test with both single-target and multi-target modes
4. Test with both single-horizon and multi-horizon modes

---

## MEDIUM SEVERITY ISSUES

### 2. Potential Memory Efficiency Issue with Feature Cache

**Location**: `/home/user/stock_forecasting_v1/tf_predictor/core/predictor.py:138-147`

**Description**:
The feature cache mechanism uses DataFrame hashing that only samples the first row, which may cause cache collisions for DataFrames with identical first rows but different data.

**Code**:
```python
def _get_dataframe_hash(self, df: pd.DataFrame) -> str:
    """Generate a hash key for DataFrame caching."""
    import hashlib
    key_data = f"{df.shape}_{list(df.columns)}_{df.iloc[0].to_dict() if len(df) > 0 else {}}"
    return hashlib.md5(key_data.encode()).hexdigest()
```

**Issue**:
- Only uses first row for hashing
- Different DataFrames with same shape/columns/first row will collide
- Could return cached features for wrong data

**Impact**: LOW-MEDIUM (cache is cleared frequently, disabled by default for critical operations)

**Recommendation**: Use more robust hashing or disable caching for production use.

---

### 3. Group Order Consistency Assumption

**Location**: Multiple locations in `_prepare_data_grouped` and `_evaluate_per_group`

**Description**:
The code assumes that iterating through `unique_groups` in the same order will maintain alignment between predictions and actuals.

**Analysis**:
- `_prepare_data_grouped` (line 677): Iterates `for group_value in unique_groups`
- Builds `group_indices` by extending with group_value for each sequence (line 717)
- `_evaluate_per_group` (line 1774, 1845): Iterates through same `unique_groups`
- Relies on consistent iteration order

**Current Status**:
- Python 3.7+ dicts maintain insertion order
- `unique()` on pandas Series returns in order of first appearance
- **Should be safe**, but relies on implementation details

**Risk**: If pandas changes `unique()` behavior or dict ordering assumptions break, alignment could fail

**Recommendation**: Add explicit sorting of `unique_groups` or use OrderedDict for clarity

---

## VERIFIED CORRECT IMPLEMENTATIONS

### ✅ Grouped Feature Creation
**Location**: `predictor.py:321-373`, `predictor.py:613-762`

**Verified Correct**:
- Proper sorting by group columns + time (lines 340-363)
- Prevents data leakage across group boundaries
- Sequence creation respects group boundaries (lines 684-687 skip insufficient groups)
- Uses `_create_group_key` consistently for composite keys

### ✅ Grouped Scaling
**Location**: `predictor.py:504-611`

**Verified Correct**:
- Separate scalers per group for features (lines 559-563)
- Separate scalers per group AND per horizon for targets (lines 595-606)
- Structure: `group_target_scalers[group_key][shifted_col]` properly handles both dimensions
- Inverse transform correctly applies group-specific scalers (lines 1324-1390)

### ✅ Sequence-Target Alignment
**Location**: `time_features.py:296-379`, `predictor.py:697-762`

**Verified Correct**:
- `create_input_variable_sequence` creates sequences using `range(len(df) - sequence_length + 1)` (line 374)
- Creates N = len(df) - sequence_length + 1 sequences
- Target extraction uses `values[sequence_length - 1:]` (lines 704, 860)
- This gives N targets matching N sequences
- Offset of `sequence_length - 1` consistently used in evaluation (lines 1602, 1665, 1763)

**Example Trace** (verified correct):
```
df length: 100, sequence_length: 5
Sequences: [0:5], [1:6], ..., [95:100] → 96 sequences
Shifted targets created with shift(-1): close_target_h1[i] = close[i+1]
Extract from index 4: close_target_h1[4:100] → 96 values
Alignment: ✓ 96 sequences, 96 targets
```

### ✅ Multi-Target Support
**Location**: `predictor.py:65-71`, `predictor.py:1310-1361`, `predictor.py:1401-1433`

**Verified Correct**:
- Target columns normalized to list (lines 65-71)
- Predictions returned as dict: `{target_name: predictions_array}`
- Inverse transform correctly handles dictionary structure
- Evaluation extracts actuals per target with proper alignment
- Supports both grouped and non-grouped scenarios

### ✅ Multi-Horizon Support
**Location**: `predictor.py:113-118`, `time_features.py:268-276`

**Verified Correct**:
- Per-horizon scalers: `target_scalers_dict[f"{target}_target_h{h}"]` (lines 113-118)
- Each horizon gets independent scaler for better accuracy
- Shifted targets created for all horizons (lines 270-276 in time_features.py)
- Evaluation correctly extracts each horizon separately (lines 1627-1639, 1687-1714)
- Inverse transform applies per-horizon scalers (lines 1347-1352, 1386-1389)

---

## CODE QUALITY OBSERVATIONS

### Strengths:
1. **Excellent documentation**: Comprehensive docstrings with examples
2. **Modular design**: Clear separation between feature creation, scaling, sequence generation
3. **Defensive programming**: Extensive validation and error messages
4. **Flexible architecture**: Supports multiple targets, horizons, and grouping configurations
5. **Memory management**: Explicit cache clearing and garbage collection

### Areas for Improvement:
1. **Multi-column grouping**: Not fully tested/implemented (see Bug #1)
2. **Code comments**: Some complex sections (like sequence alignment) could use inline comments
3. **Test coverage**: Critical edge cases (multi-column groups) appear untested
4. **Type hints**: Could be more comprehensive for complex return types

---

## TESTING RECOMMENDATIONS

### Critical Tests Needed:

1. **Multi-Column Grouping Test**:
```python
def test_multi_column_grouping_evaluation():
    """Test per-group evaluation with multiple group columns"""
    predictor = TimeSeriesPredictor(
        target_column='close',
        group_columns=['symbol', 'sector'],  # Multi-column
        ...
    )
    # Create data with multiple symbols and sectors
    # Verify per-group metrics are correct
```

2. **Alignment Verification Test**:
```python
def test_sequence_target_alignment():
    """Verify sequences and targets align correctly"""
    # Test with various sequence_lengths and data sizes
    # Explicitly check indices match
```

3. **Grouped Scaling Test**:
```python
def test_grouped_scaling_isolation():
    """Verify groups are scaled independently"""
    # Create data with very different scales per group
    # Verify each group's scaler parameters are different
```

---

## PERFORMANCE NOTES

### Memory Usage:
- Batched inference (line 1266): Good practice
- Explicit GPU cache clearing (line 1298): Good
- Feature cache clearing before operations (lines 1539, 1256): Good
- Potential issue: `_last_processed_df` stored for evaluation could be large for big datasets

### Computational Efficiency:
- Per-group operations use vectorized pandas operations: Good
- Sequence creation uses numpy: Good
- Could optimize: Repeated filtering by group (consider groupby once)

---

## SUMMARY TABLE

| Component | Status | Issues Found | Severity |
|-----------|--------|--------------|----------|
| Grouped Feature Creation | ✅ PASS | 0 | - |
| Grouped Scaling | ✅ PASS | 0 | - |
| Sequence-Target Alignment | ✅ PASS | 0 | - |
| Multi-Target Support | ✅ PASS | 0 | - |
| Multi-Horizon Support | ✅ PASS | 0 | - |
| Single-Column Group Evaluation | ✅ PASS | 0 | - |
| **Multi-Column Group Evaluation** | ❌ **FAIL** | **1** | **CRITICAL** |
| Feature Cache | ⚠️ WARNING | 1 | MEDIUM |
| Group Order Consistency | ⚠️ WARNING | 1 | MEDIUM |

---

## CONCLUSION

The tf_predictor module demonstrates solid engineering with proper handling of complex time series scenarios. **However, multi-column grouping evaluation is broken and will produce incorrect results.** This should be fixed before using the module with multi-column group configurations.

All other components (scaling, alignment, multi-target, multi-horizon) are correctly implemented and thoroughly verified.

### Immediate Actions Required:
1. ✅ **Fix multi-column group evaluation bug** (CRITICAL)
2. ⚠️ **Add tests for multi-column grouping scenarios** (HIGH)
3. ⚠️ **Review feature cache implementation** (MEDIUM)
4. ℹ️ **Add explicit group ordering** (LOW)

---

**End of Report**
