# Multi-Column Grouping Alignment Test - Modifications Summary

**Date:** 2025-11-06
**Purpose:** Modified `debug_alignment_simple.py` to test multi-column grouping bug fix

---

## Changes Made

### 1. Enhanced Synthetic Data (4 Groups with Multi-Column Keys)

**Before:** 2 groups with single column 'symbol'
- AAPL
- GOOGL

**After:** 4 groups with multi-column ['symbol', 'sector']
- ('AAPL', 'Tech')
- ('GOOGL', 'Tech')
- ('MSFT', 'Consumer')
- ('AMZN', 'Consumer')

Each group has 10 rows with incrementing close values:
- AAPL: 100-109
- GOOGL: 200-209
- MSFT: 300-309
- AMZN: 400-409

### 2. Updated Predictor Configuration

**Before:**
```python
group_columns='symbol',
categorical_columns='symbol',
```

**After:**
```python
group_columns=['symbol', 'sector'],  # MULTI-COLUMN GROUPING
categorical_columns=['symbol', 'sector'],
```

Also updated parameter names to use new conventions:
- `d_model` → `d_token`
- `num_heads` → `n_heads`
- `num_layers` → `n_layers`

### 3. Enhanced `print_df_state()` Function

Updated to handle multi-column grouping:

**Before:** Only supported single group column
```python
def print_df_state(df, title, group_col='symbol', ...):
    if group_col in df.columns:
        for symbol in sorted(df[group_col].unique()):
            ...
```

**After:** Supports single or multi-column grouping
```python
def print_df_state(df, title, group_col=None, ...):
    if group_col:
        group_cols = [group_col] if isinstance(group_col, str) else group_col
        grouped = df.groupby(group_cols)
        for group_key, group_df in grouped:
            # Format composite keys: "symbol=AAPL + sector=Tech"
            ...
```

### 4. Updated Expected Results

**Before:**
- 2 groups × 6 sequences per group = 12 total sequences
- Metrics for 2 groups

**After:**
- 4 groups × 6 sequences per group = 24 total sequences
- Metrics for 4 groups with composite keys

### 5. Enhanced Summary Output

Added verification points specific to multi-column grouping:

```
🔍 This demonstrates the NEW alignment (FIXED) with MULTI-COLUMN GROUPING:
   - Multi-column groups use composite keys: ('AAPL', 'Tech'), ('GOOGL', 'Tech'), etc.
   - _filter_dataframe_by_group() properly handles tuple keys
   - Predictions count: (rows_after_shift - sequence_length + 1) per group
   - Actuals extracted from shifted columns with offset = sequence_length - 1
   - Each horizon is independent: close_target_h1, close_target_h2, etc.
   - Actuals count ALWAYS equals predictions count FOR EACH GROUP!

💡 Multi-column grouping verification:
   1. ✓ Composite keys respected (no mixing between groups)
   2. ✓ Sequence offset applied consistently per group
   3. ✓ Group boundaries respected (no data leakage across groups)
   4. ✓ Multi-horizon actual extraction correct for each group
```

---

## What This Test Validates

This modified test specifically validates the **multi-column grouping bug fix** implemented in `tf_predictor/core/predictor.py`:

### The Bug (Fixed)

**Problem:** The old code only used the first group column when filtering:
```python
# OLD CODE (BUGGY)
group_col_name = self.group_columns[0]  # Only takes 'symbol'
group_df = df[df[group_col_name] == group_name]  # Ignores 'sector'
```

**Impact:** When using `group_columns=['symbol', 'sector']`, all sectors for a symbol would be mixed together, causing incorrect evaluation metrics.

### The Fix

**Solution:** New `_filter_dataframe_by_group()` helper method properly handles composite keys:
```python
# NEW CODE (FIXED)
def _filter_dataframe_by_group(self, df: pd.DataFrame, group_key) -> pd.DataFrame:
    """Filter dataframe to rows matching the given group key."""
    if len(self.group_columns) == 1:
        col = self.group_columns[0]
        return df[df[col] == group_key].copy()
    else:
        # Multi-column: group_key is a tuple like ('AAPL', 'Tech')
        if not isinstance(group_key, tuple):
            raise ValueError(f"Expected tuple for multi-column group key")

        mask = pd.Series([True] * len(df), index=df.index)
        for i, col in enumerate(self.group_columns):
            mask &= (df[col] == group_key[i])
        return df[mask].copy()
```

---

## How to Run the Test

### Prerequisites

Ensure your Python environment has the required packages:
```bash
pip install pandas numpy torch scikit-learn
```

### Run the Test

```bash
cd /home/user/stock_forecasting_v1
PYTHONPATH=. python debug_alignment_simple.py
```

### Expected Output

The test will generate `alignment_test_results.md` with:

1. **Test 1: Single-Target (close only)**
   - 4 groups with multi-column keys
   - 24 sequences total (6 per group)
   - Alignment table showing predictions vs actuals for each group
   - Perfect alignment verification: 6 predictions = 6 actuals per group

2. **Test 2: Multi-Target (close + volume)**
   - Same 4 groups
   - 24 sequences total
   - Alignment tables for both targets (close and volume)
   - Separate per-horizon metrics for each target

---

## Expected Test Results

### Data Flow Summary

```
1. Raw data:        10 rows per group × 4 groups = 40 total rows
2. After shifting:   8 rows per group × 4 groups = 32 total rows (2 dropped per group)
3. After sequences:  6 predictions per group × 4 groups = 24 total predictions

4. Evaluation extracts actuals from Step 2 (8 rows per group)
   - Applies sequence offset: 8 - (3-1) = 6 actuals per group
   - For each horizon, extract from shifted columns directly
   - Available: 6 actuals per group, Predictions: 6 per group
   - ✅ PERFECT ALIGNMENT!
```

### Per-Group Alignment Example

For group ('AAPL', 'Tech'):
```
Date       | close_h1_pred | close_h1_actual | close_h2_pred | close_h2_actual
-----------|---------------|-----------------|---------------|----------------
2024-01-03 | 103.xxx       | 103.0           | 104.xxx       | 104.0
2024-01-04 | 104.xxx       | 104.0           | 105.xxx       | 105.0
2024-01-05 | 105.xxx       | 105.0           | 106.xxx       | 106.0
2024-01-06 | 106.xxx       | 106.0           | 107.xxx       | 107.0
2024-01-07 | 107.xxx       | 107.0           | 108.xxx       | 108.0
2024-01-08 | 108.xxx       | 108.0           | 109.xxx       | 109.0
```

Same alignment tables will be generated for all 4 groups:
- ('AAPL', 'Tech')
- ('GOOGL', 'Tech')
- ('MSFT', 'Consumer')
- ('AMZN', 'Consumer')

---

## Files Modified

1. **debug_alignment_simple.py** - Main test script
   - `create_synthetic_data()`: 4 groups instead of 2
   - `test_single_target()`: Multi-column grouping configuration
   - `test_multi_target()`: Multi-column grouping configuration
   - `print_df_state()`: Enhanced for multi-column grouping display
   - `run_test()`: Updated for multi-column group handling
   - `main()`: Updated summary and expectations

2. **alignment_test_results.md** - Will be regenerated with new results

---

## Success Criteria

The test passes if:

1. ✅ **24 sequences created** (6 per group × 4 groups)
2. ✅ **Alignment tables show perfect match** for each group
3. ✅ **No mixing between groups** (e.g., AAPL Tech metrics don't include AAPL Consumer data)
4. ✅ **Evaluation completes successfully** for both single-target and multi-target
5. ✅ **Per-group metrics are calculated correctly** with composite keys

---

## Next Steps

1. **Run the test** in an environment with required dependencies
2. **Review alignment_test_results.md** to verify multi-column grouping works correctly
3. **Compare with previous results** (single-column grouping from Nov 3rd)
4. **Confirm no regressions** in existing functionality

---

**Status:** Test modifications complete, ready to run.
**Dependencies:** pandas, numpy, torch, scikit-learn, tf_predictor module
