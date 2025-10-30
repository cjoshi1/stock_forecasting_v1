# Daily Forecasting 100% MAPE Bug - FINAL FIX

## Root Cause
Evaluation code extracted actual values from **raw dataframe** but predictions came from **processed dataframe** (after feature engineering, NaN drops, target shifting). This caused misalignment where `prediction[i]` was compared to `actual[i+offset]`, resulting in 100% MAPE.

---

## Changes Implemented

### File: `tf_predictor/core/predictor.py`

#### 1. Store Processed DataFrame (Line ~1399)
```python
# In predict() method
self._last_processed_df = self.prepare_features(df.copy(), fit_scaler=False)
```
**Purpose**: Cache the processed dataframe that predictions are actually based on.

#### 2. Replace 4 Buggy Evaluation Locations
All 4 locations now:
- **Check** if `_last_processed_df` exists
- **Raise RuntimeError** if missing (fail-fast instead of returning wrong metrics)
- **Extract actuals** from `_last_processed_df` instead of raw `df`

**Locations fixed**:
- Multi-target per-group evaluation (~line 1811)
- Multi-target overall evaluation (~line 1877)  
- Single-target per-group evaluation (~line 1933)
- Single-target overall evaluation (~line 1987)

**Pattern used everywhere**:
```python
if not hasattr(self, '_last_processed_df') or self._last_processed_df is None:
    raise RuntimeError(
        "Processed dataframe not available for evaluation. "
        "This is a bug - predict() should have set _last_processed_df. "
        "Please report this issue."
    )

# Extract from processed df (guaranteed aligned with predictions)
all_actual = self._last_processed_df[target_col].values[self.sequence_length:]
```

#### 3. Memory Cleanup (Line ~1663)
```python
# In evaluate() method, after metrics calculated
if hasattr(self, '_last_processed_df'):
    del self._last_processed_df
```
**Purpose**: Free memory by deleting cached dataframe after evaluation.

---

## Why Fail-Fast Instead of Fallback?

**Old approach** (removed): 
```python
if _last_processed_df exists:
    use it  # correct
else:
    use raw df  # buggy - produces 100% MAPE
```

**Problem**: Silent failure - returns wrong metrics without warning.

**New approach**:
```python
if _last_processed_df missing:
    raise RuntimeError  # loud failure
```

**Benefits**:
- ✅ **No silent failures**: If something goes wrong, we know immediately
- ✅ **No wrong metrics**: Better to crash than return 100% MAPE that looks like "bad model"
- ✅ **Easier debugging**: Clear error message points to the problem
- ✅ **Prevents misuse**: Forces correct usage pattern (predict → evaluate)

---

## Expected Results

### Before Fix
- MAPE: ~100%
- R²: < -1 (negative)
- Directional Accuracy: Meaningless (~40-60% random)
- Metrics: Completely wrong due to alignment bug

### After Fix
- MAPE: 10-30% (realistic for crypto daily forecasting)
- R²: 0.3-0.7 (positive, meaningful)
- Directional Accuracy: 52-58% (better than random)
- Metrics: Correct alignment, interpretable results

---

## Testing Commands

### Quick Test (Single-horizon)
```bash
python daily_stock_forecasting/main.py \
  --use_sample_data \
  --target close \
  --asset_type crypto \
  --prediction_horizon 1 \
  --sequence_length 10 \
  --test_size 30 \
  --epochs 20
```

### Full Test (Multi-group, Multi-horizon, Multi-target)
```bash
python daily_stock_forecasting/main.py \
  --data_path data/crypto_portfolio.csv \
  --target "close,volume" \
  --asset_type crypto \
  --group_columns symbol \
  --per_group_metrics \
  --prediction_horizon 3 \
  --sequence_length 20 \
  --test_size 60 \
  --scaler_type standard \
  --epochs 100
```

---

## Summary

- **Lines changed**: ~80 lines across 6 locations
- **Files modified**: 1 file (`tf_predictor/core/predictor.py`)
- **Behavior**: Fail-fast on errors, no silent failures
- **Impact**: Fixes 100% MAPE bug completely

The fix ensures predictions and actuals are always aligned by using the same processed dataframe for both.
