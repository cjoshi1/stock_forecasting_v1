# Multi-Symbol Forecasting Debugging Plan

**Issue:** Multi-symbol forecasting with group-based scaling shows poor performance compared to single-symbol baseline.

**Hypothesis:** The group-based scaling implementation may have issues with:
1. Data leakage between groups
2. Incorrect scaler application
3. Sequence creation across group boundaries
4. Model learning from mixed group patterns

---

## 🎯 Debugging Strategy

### Phase 1: Baseline Establishment (Day 1)
**Goal:** Establish single-symbol performance as baseline

### Phase 2: Controlled Comparison (Day 1-2)
**Goal:** Compare single vs multi-symbol with same data

### Phase 3: Root Cause Analysis (Day 2-3)
**Goal:** Identify specific failure points

### Phase 4: Fix & Verify (Day 3-4)
**Goal:** Implement fixes and validate improvements

---

## 📋 Detailed Step-by-Step Plan

### Step 1: Create Diagnostic Test Framework ✅

**Purpose:** Build automated testing infrastructure for comparison

**Tasks:**
1. Create `debug_single_vs_multi.py` script
2. Implement side-by-side comparison tests
3. Add detailed logging and metrics collection
4. Create visualization tools for comparison

**Deliverables:**
- Automated test script
- Metrics comparison report
- Visualization plots

---

### Step 2: Baseline Single-Symbol Tests

**Purpose:** Establish performance baseline with original single-symbol approach

**Test Cases:**

#### Test 2.1: Single Symbol, No Group Column
```python
# Daily forecasting
python daily_stock_forecasting/main.py \
  --data_path data/AAPL_only.csv \
  --target close \
  --group_column None \
  --epochs 100 \
  --verbose

# Expected: Good R², low MAPE (baseline performance)
```

**Metrics to Collect:**
- Train R², MAPE, RMSE
- Validation R², MAPE, RMSE
- Test R², MAPE, RMSE
- Training time
- Number of parameters

**Success Criteria:**
- Test R² > 0.7
- Test MAPE < 5%
- Directional accuracy > 55%

---

### Step 3: Multi-Symbol Without Group Scaling

**Purpose:** Test if issue is with multiple symbols or group scaling specifically

#### Test 3.1: Multiple Symbols, No Group Column (Baseline)
```python
# Concatenate multiple symbols but treat as one dataset
python daily_stock_forecasting/main.py \
  --data_path data/multi_symbols.csv \
  --target close \
  --group_column None \
  --epochs 100

# Expected: Poor performance (symbols have different scales)
```

**What This Tests:**
- Whether model can learn from mixed-scale data
- If symbol differences cause confusion

**Expected Result:**
- Poor performance due to scale differences
- This confirms we NEED group-based scaling

---

### Step 4: Multi-Symbol With Group Scaling

**Purpose:** Test current multi-symbol implementation

#### Test 4.1: Multiple Symbols, With Group Column
```python
python daily_stock_forecasting/main.py \
  --data_path data/multi_symbols.csv \
  --target close \
  --group_column symbol \
  --epochs 100 \
  --per_group_metrics

# Current: Poor performance
# Expected after fixes: Good performance
```

**Deep Dive Checks:**
1. **Scaler Verification:**
   - Print scaler statistics per group
   - Verify each group has independent scalers
   - Check scaler mean/std values are reasonable

2. **Data Split Verification:**
   - Ensure temporal order per group
   - Check no data leakage between groups
   - Verify split sizes per group

3. **Sequence Verification:**
   - Check sequences don't cross group boundaries
   - Verify sequence lengths are correct
   - Inspect first/last sequences per group

---

### Step 5: Controlled Single-Symbol via Group Column

**Purpose:** Isolate group scaling mechanism from multi-symbol complexity

#### Test 5.1: Single Symbol With Group Column
```python
# Use single symbol but enable group_column
python daily_stock_forecasting/main.py \
  --data_path data/AAPL_only.csv \
  --target close \
  --group_column symbol \
  --epochs 100

# Expected: Should match Test 2.1 performance
# If worse: group scaling mechanism itself has issues
# If same: group scaling is fine, issue is with multiple groups
```

**What This Tests:**
- Group scaling overhead
- Whether group column mechanism adds bugs
- Scaler initialization correctness

**Critical Comparison:**
- Compare Test 2.1 (no group) vs Test 5.1 (with group, single symbol)
- Performance should be identical
- If different: **group scaling code has bugs**

---

### Step 6: Incremental Multi-Symbol Testing

**Purpose:** Find the breaking point as we add symbols

#### Test 6.1: Two Symbols Only
```python
# Use just 2 symbols (e.g., AAPL, GOOGL)
python daily_stock_forecasting/main.py \
  --data_path data/two_symbols.csv \
  --group_column symbol \
  --per_group_metrics
```

#### Test 6.2: Three Symbols
```python
# Add one more (AAPL, GOOGL, MSFT)
```

#### Test 6.3: Five Symbols
```python
# Add two more (+ AMZN, META)
```

**What This Tests:**
- Does performance degrade linearly with more symbols?
- Is there a specific number where it breaks?
- Per-group metrics: which symbols perform well/poorly?

---

### Step 7: Data Quality Checks

**Purpose:** Ensure data quality isn't causing issues

**Checks:**

1. **Symbol Distribution:**
   ```python
   df.groupby('symbol').size()
   # Should be roughly balanced
   ```

2. **Date Range Alignment:**
   ```python
   df.groupby('symbol')['date'].agg(['min', 'max'])
   # Should have overlapping periods
   ```

3. **Price Range Comparison:**
   ```python
   df.groupby('symbol')['close'].agg(['mean', 'std', 'min', 'max'])
   # Check for extreme differences
   ```

4. **Missing Data:**
   ```python
   df.groupby('symbol').apply(lambda x: x.isnull().sum())
   # Should be minimal
   ```

---

### Step 8: Scaler Deep Dive

**Purpose:** Verify scaler statistics and application

**Investigations:**

1. **Scaler Statistics:**
   ```python
   for group, scaler in predictor.group_target_scalers.items():
       print(f"{group}: mean={scaler.mean_}, std={scaler.scale_}")
   ```

2. **Inverse Transform Test:**
   ```python
   # For each group:
   # 1. Transform data
   # 2. Inverse transform
   # 3. Compare with original
   # Should be identical (within floating point precision)
   ```

3. **Cross-Group Contamination Test:**
   ```python
   # Transform AAPL data with GOOGL scaler
   # Should give very different (wrong) results
   # If not: scalers aren't actually different
   ```

---

### Step 9: Sequence Creation Verification

**Purpose:** Ensure sequences respect group boundaries

**Tests:**

1. **Boundary Check:**
   ```python
   # For each sequence:
   # Verify all rows belong to same group
   # Verify temporal order within sequence
   ```

2. **Sequence Statistics:**
   ```python
   # Count sequences per group
   # Should be proportional to data size
   ```

3. **Visual Inspection:**
   ```python
   # Print first 3 sequences from each group
   # Manually verify correctness
   ```

---

### Step 10: Model Architecture Analysis

**Purpose:** Check if model architecture is suitable for multi-group

**Tests:**

1. **Capacity Test:**
   ```python
   # Try with more layers/heads
   --n_layers 6 --n_heads 16 --d_token 256
   # Does performance improve?
   ```

2. **Model Comparison:**
   ```python
   # Test both architectures
   --model_type ft  # vs
   --model_type csn
   # Which performs better on multi-symbol?
   ```

3. **Sequence Length Test:**
   ```python
   # Try different sequence lengths
   --sequence_length 10  # vs 20 vs 30
   # Optimal might differ for multi-symbol
   ```

---

## 🔍 Specific Issues to Investigate

Based on the logical checks, here are the most likely culprits:

### Issue 1: Target Scaler Not Per-Group (FIXED ✅)
- **Status:** Already verified via logical checks
- **Result:** Working correctly now

### Issue 2: Sequences Crossing Group Boundaries
**Location:** `tf_predictor/core/predictor.py::_prepare_data_grouped()`

**Check:**
```python
# In _prepare_data_grouped, line ~375
for group_value in unique_groups:
    group_mask = df_processed[self.group_column] == group_value
    group_df = df_processed[group_mask].copy()

    # ADD THIS CHECK:
    print(f"Group {group_value}: {len(group_df)} rows")
    sequences, targets = create_sequences(...)
    print(f"  Created {len(sequences)} sequences")

    # Verify: sequences should never span multiple groups
```

### Issue 3: Model Overfitting to Dominant Symbol
**Symptom:** Model learns well for one symbol, poorly for others

**Check:**
```python
# Run with --per_group_metrics
# Compare per-symbol performance
# If one symbol has R²=0.9 and others have R²=0.1:
# → Model is overfitting to that symbol
```

**Fix:**
- Balance training data per symbol
- Use weighted loss function
- Sample sequences proportionally

### Issue 4: Data Leakage via Feature Engineering
**Location:** Feature creation functions

**Check:**
```python
# In create_stock_features() or create_intraday_features()
# Ensure rolling windows don't cross group boundaries
# Example bad code:
df['ma_20'] = df['close'].rolling(20).mean()  # ❌ Crosses groups!

# Good code:
df['ma_20'] = df.groupby('symbol')['close'].rolling(20).mean()  # ✅
```

---

## 📊 Comparison Metrics Template

For each test, collect:

```python
{
    "test_name": "Test_2.1_Single_Symbol_No_Group",
    "config": {
        "num_symbols": 1,
        "group_column": None,
        "sequence_length": 5,
        "prediction_horizon": 1
    },
    "data": {
        "train_samples": 1000,
        "val_samples": 200,
        "test_samples": 200
    },
    "metrics": {
        "train": {"R2": 0.85, "MAPE": 2.3, "RMSE": 1.2},
        "val": {"R2": 0.78, "MAPE": 3.1, "RMSE": 1.5},
        "test": {"R2": 0.75, "MAPE": 3.5, "RMSE": 1.6}
    },
    "per_group_metrics": {
        "AAPL": {"R2": 0.75, "MAPE": 3.5}
        # ... if applicable
    },
    "training": {
        "time_seconds": 120,
        "final_train_loss": 0.01,
        "final_val_loss": 0.02
    }
}
```

---

## 🎯 Expected Outcomes by Phase

### After Step 2 (Baseline):
- **Success:** Test R² > 0.7, MAPE < 5%
- **Document:** Baseline metrics for comparison

### After Step 5 (Single Symbol + Group):
- **Critical Test:** Should match Step 2 performance
- **If Different:** Group scaling mechanism has bugs → **Fix Priority 1**
- **If Same:** Group mechanism is fine, issue is with multiple groups

### After Step 6 (Incremental Multi-Symbol):
- **Pattern 1:** Performance degrades linearly → Model capacity issue
- **Pattern 2:** Specific symbols perform poorly → Data quality issue
- **Pattern 3:** All symbols perform poorly equally → Systematic bug

---

## 🛠️ Tools & Scripts to Create

### 1. `debug_single_vs_multi.py`
Automated comparison framework (create this first)

### 2. `analyze_scalers.py`
Deep dive into scaler statistics

### 3. `verify_sequences.py`
Verify sequence creation correctness

### 4. `compare_results.py`
Generate comparison reports and plots

### 5. `data_quality_report.py`
Analyze input data quality per symbol

---

## 📝 Next Immediate Steps

1. **Now:** Create `debug_single_vs_multi.py` framework
2. **Next:** Run Step 2 (baseline single-symbol tests)
3. **Then:** Run Step 5 (critical comparison test)
4. **Based on results:** Follow appropriate path

---

## ⚠️ Red Flags to Watch For

1. **Different scalers but same mean/std values** → Scalers not fitted per group
2. **Sequences with mixed symbols** → Boundary violation
3. **One symbol R²=0.9, others R²=0.1** → Overfitting to dominant symbol
4. **Test 5.1 ≠ Test 2.1** → Group scaling mechanism broken
5. **Negative R² on validation** → Severe overfitting or data leakage
6. **Loss increases with more symbols** → Model capacity insufficient

---

## 🎓 Learning Outcomes

By the end of this debugging:
- **Identified:** Exact failure point in multi-symbol pipeline
- **Fixed:** Root cause of performance degradation
- **Validated:** Multi-symbol performance matches single-symbol
- **Documented:** Best practices for multi-symbol forecasting

---

**Ready to start?** Let's begin with creating the debugging framework!
