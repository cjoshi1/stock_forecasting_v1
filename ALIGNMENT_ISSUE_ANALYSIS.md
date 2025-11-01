# Alignment Issue Analysis

## Problem Identified

**Critical Issue**: For multi-horizon predictions, there are insufficient actual values to evaluate all predictions correctly.

## Root Cause

The alignment problem occurs due to how multi-horizon evaluation extracts actual values:

### Data Flow Example (sequence_length=3, prediction_horizon=2):

1. **Raw data**: 10 rows per group
   - AAPL: dates 2024-01-01 to 2024-01-10, close values: 100-109

2. **After `create_shifted_targets()`**: 8 rows per group
   - Last 2 rows dropped (no future data for h=2)
   - AAPL: dates 2024-01-01 to 2024-01-08, close values: 100-107
   - Target columns created:
     - `close_target_h1` = next day's close
     - `close_target_h2` = 2 days ahead close

3. **After sequence creation**: 5 predictions per group
   - First 3 rows used for sequence window
   - Predictions correspond to rows 3-7 (indices after sequence_length offset)

### The Alignment Problem

**For evaluation**, the code does:
```python
# Extract actuals with sequence offset
group_actual_full = group_df_processed['close'].values[sequence_length:]
# After offset: [103, 104, 105, 106, 107] - 5 values

# For multi-horizon predictions
num_preds = 5
needed_actuals = num_preds + prediction_horizon - 1
# needed_actuals = 5 + 2 - 1 = 6

# ⚠️ PROBLEM: Available=5, Needed=6
```

**Why 6 actuals are needed:**
- Prediction 0 needs actuals at indices [0, 1] for horizons h1, h2
- Prediction 1 needs actuals at indices [1, 2]
- ...
- Prediction 4 needs actuals at indices [4, 5]
- But we only have actuals up to index 4!

## Visual Representation

```
Processed DF (after shifting, 8 rows):
Index:  0    1    2    3    4    5    6    7
Close:  100  101  102  103  104  105  106  107
                       ↑                   ↑
                   seq_len=3           last row

After sequence offset (skip first 3 rows):
Index:  3    4    5    6    7
Close:  103  104  105  106  107
        ↑                   ↑
    pred[0]             pred[4]

Predictions (5 total):
pred[0] predicts h1=104, h2=105  → needs actuals[0]=103, actuals[1]=104, actuals[2]=105
pred[1] predicts h1=105, h2=106  → needs actuals[1]=104, actuals[2]=105, actuals[3]=106
pred[2] predicts h1=106, h2=107  → needs actuals[2]=105, actuals[3]=106, actuals[4]=107
pred[3] predicts h1=107, h2=???  → needs actuals[3]=106, actuals[4]=107, actuals[5]=??? ❌
pred[4] predicts h1=???, h2=???  → needs actuals[4]=107, actuals[5]=???, actuals[6]=??? ❌

Missing: actuals[5] and actuals[6] (values 108 and 109)
```

## Why Actuals Are Missing

The actuals are missing because:
1. `create_shifted_targets()` drops the last `prediction_horizon` rows (can't create targets without future data)
2. This means the processed dataframe ends at row 7 (close=107)
3. But predictions 3 and 4 need to compare against future values (108, 109) that don't exist in the processed dataframe

## Impact on Evaluation

**Current behavior:**
```python
# In _evaluate_per_group() at line 1740
if len(group_actual_full) >= needed_actuals:
    group_actual_aligned = group_actual_full[:needed_actuals]
else:
    # Not enough actuals - trim predictions
    num_preds = max(0, len(group_actual_full) - self.prediction_horizon + 1)
    group_actual_aligned = group_actual_full
    group_preds = group_preds[:num_preds] if num_preds > 0 else group_preds[:0]
```

**Result:**
- For horizon=2, can only evaluate 4 out of 5 predictions (last prediction discarded)
- For horizon=3, can only evaluate 3 out of 5 predictions
- This leads to **incorrect metrics** because:
  - Not all predictions are evaluated
  - Train/val/test sets have different effective sizes than expected

## The Real Issue: Wrong Actual Values Being Used

Looking more carefully at the code, there's a **second, more critical issue**:

### The evaluation extracts actuals from the WRONG column!

In `_evaluate_per_group()` at line 1709:
```python
group_actual_full = group_df_processed[target_col].values[sequence_length:]
```

This extracts from the **original `close` column**, but it should extract from the **shifted target columns** (`close_target_h1`, `close_target_h2`)!

### Why This Is Wrong

The model is trained to predict `close_target_h1` and `close_target_h2`, which are the shifted future values. But evaluation compares predictions against the original `close` column with a manual offset.

**Example:**
```
Row 3 in processed_df:
  - close = 103
  - close_target_h1 = 104 (this is what model learned to predict)
  - close_target_h2 = 105 (this is what model learned to predict)

Current evaluation:
  - Extracts: actuals = [103, 104, 105, 106, 107] from 'close' column
  - For pred[0] (corresponding to row 3):
    - Compares pred[0][h1] vs actuals[1] = 104 ✓
    - Compares pred[0][h2] vs actuals[2] = 105 ✓
  - This happens to work for this simple case...

But it's semantically wrong! The model was trained on:
  - Row 3: X → predict [close_target_h1, close_target_h2] = [104, 105]

Evaluation should extract:
  - actual_h1 from 'close_target_h1' column
  - actual_h2 from 'close_target_h2' column
```

## The Correct Approach

Evaluation should:

1. **Extract actuals from the shifted target columns directly**:
   ```python
   # For single-horizon
   actual_h1 = group_df_processed['close_target_h1'].values[sequence_length:]

   # For multi-horizon
   actual_h1 = group_df_processed['close_target_h1'].values[sequence_length:]
   actual_h2 = group_df_processed['close_target_h2'].values[sequence_length:]
   # Stack into (n_samples, horizons) array
   actuals_multi_horizon = np.column_stack([actual_h1, actual_h2])
   ```

2. **This automatically has the correct length**:
   - No need for complex offset calculations
   - `actuals_multi_horizon.shape[0]` == `predictions.shape[0]` by construction
   - Each row i compares: `predictions[i, h]` vs `actuals[i, h]`

## Summary

**Two critical bugs identified:**

1. **Off-by-one alignment issue**: Not enough actual values for multi-horizon evaluation due to dropping last `prediction_horizon` rows during shifting

2. **Wrong column extraction**: Evaluation extracts from original `close` column instead of shifted target columns (`close_target_h1`, `close_target_h2`), leading to complex manual offset logic that's error-prone

**Solution:** Extract actuals directly from shifted target columns in `_last_processed_df`, which automatically ensures correct alignment.
