# Multi-Symbol Debugging - Quick Start Guide

**Problem:** Multi-symbol forecasting performance is worse than single-symbol baseline.

**Solution:** Systematic debugging to identify root cause.

---

## 🚀 Quick Start (5 minutes)

### Step 1: Run Diagnostic Tests

```bash
# Run automated comparison
python debug_single_vs_multi.py
```

**What it does:**
- Test 1: Single symbol without group column (baseline)
- Test 2: Single symbol WITH group column (mechanism test)
- Test 3: Multi-symbol with group column (current implementation)

**Time:** ~5-10 minutes (depending on data size and epochs)

---

### Step 2: Interpret Results

The script will output a comparison table like this:

```
test                        symbols  group_col  test_r2  test_mape
Test1_Single_NoGroup        1        No         0.7500   3.50
Test2_Single_WithGroup      1        Yes        0.7480   3.52
Test3_Multi_WithGroup       3        Yes        0.2100   15.80
```

**Decision Tree:**

```
┌─ Test2 ≈ Test1? (R² difference < 0.05)
│
├─ YES ✅ → Group scaling mechanism is OK
│   │
│   └─ Test3 < Test2? (R² much worse)
│       │
│       ├─ YES 🔍 → Multi-symbol specific issue
│       │           → Go to "Multi-Symbol Issues" section
│       │
│       └─ NO ✅ → Everything works!
│
└─ NO ❌ → Group scaling mechanism has bugs
            → Go to "Group Scaling Issues" section
```

---

## 🔍 Issue Categories

### A. Group Scaling Mechanism Broken (Test2 ≠ Test1)

**Symptom:** Test 2 performs significantly different from Test 1

**What this means:**
- The group scaling code itself has bugs
- Even with one symbol, adding `group_column` changes results
- **This is a critical bug**

**Next Steps:**
1. Check scaler statistics in debug output
2. Verify scaler mean/std values
3. Run: `python analyze_scalers.py` (see DEBUGGING_PLAN.md Step 8)
4. Fix scaler initialization/application code

**Likely Causes:**
- Scalers not fitted per group
- Inverse transform using wrong scaler
- Group assignment incorrect

---

### B. Multi-Symbol Specific Issues (Test2 ≈ Test1, but Test3 << Test2)

**Symptom:** Single-symbol with group column works fine, but multi-symbol doesn't

**What this means:**
- Group scaling mechanism is correct
- Issue is with handling multiple groups simultaneously
- Could be data-related or model-related

**Sub-Issues to Investigate:**

#### B1. Data Imbalance
```python
# Check symbol distribution
df.groupby('symbol').size()

# If severely imbalanced (e.g., 1000 rows for AAPL, 100 for GOOGL):
# → Model overfits to dominant symbol
```

**Solution:** Balance training data or use weighted sampling

#### B2. Sequences Crossing Group Boundaries
```python
# Verify sequences don't mix symbols
# Run: python verify_sequences.py
```

**Solution:** Fix sequence creation logic

#### B3. Model Capacity Insufficient
```python
# Try larger model
python daily_stock_forecasting/main.py \
  --data_path data.csv \
  --group_column symbol \
  --n_layers 6 \
  --n_heads 16 \
  --d_token 256
```

**Solution:** Increase model capacity

#### B4. Feature Engineering Crosses Groups
```python
# Check for rolling windows that span symbols
# Example bad code:
df['ma_20'] = df['close'].rolling(20).mean()  # ❌

# Should be:
df['ma_20'] = df.groupby('symbol')['close'].rolling(20).mean()  # ✅
```

**Solution:** Fix feature engineering to respect groups

---

## 📋 Detailed Investigation Steps

### Investigation 1: Check Scaler Statistics

Look in the debug output for scaler information:

```json
"scalers": {
  "feature_scalers": {
    "AAPL": {"mean": 150.5, "std": 10.2},
    "GOOGL": {"mean": 2800.3, "std": 85.4},
    "MSFT": {"mean": 320.1, "std": 15.6}
  },
  "target_scalers": {
    "AAPL": {"mean": 151.2, "std": 10.5},
    "GOOGL": {"mean": 2805.1, "std": 86.1},
    "MSFT": {"mean": 321.5, "std": 15.8}
  }
}
```

**Check:**
- ✅ Each symbol has different mean/std (indicates independent scalers)
- ❌ All symbols have same mean/std (scalers not independent!)
- ✅ Mean/std values match actual data ranges
- ❌ Mean/std don't make sense (scaler fitted wrong)

---

### Investigation 2: Check Per-Group Metrics

In Test 3 output, look for per-group metrics:

```json
"metrics_test": {
  "AAPL": {"R2": 0.75, "MAPE": 3.2},
  "GOOGL": {"R2": 0.05, "MAPE": 25.8},
  "MSFT": {"R2": 0.68, "MAPE": 4.1}
}
```

**Patterns:**

**Pattern A:** One symbol good, others bad
- **Diagnosis:** Model overfits to one symbol
- **Solution:** Balance data, increase model capacity

**Pattern B:** All symbols equally bad
- **Diagnosis:** Systematic issue (scaling, sequences, etc.)
- **Solution:** Check scalers and sequence creation

**Pattern C:** High-price symbols worse
- **Diagnosis:** Scaling issue (larger numbers harder to predict)
- **Solution:** Verify scaler application

---

## 🛠️ Quick Fixes to Try

### Fix 1: Verify Group Column is Preserved

```python
# In your feature engineering code, ensure:
if 'symbol' in df.columns:
    # Keep symbol column
    df_processed['symbol'] = df['symbol']
```

### Fix 2: Check Data Quality

```bash
# Run data quality report
python -c "
import pandas as pd
df = pd.read_csv('your_data.csv')

print('Symbol distribution:')
print(df.groupby('symbol').size())

print('\nDate ranges per symbol:')
print(df.groupby('symbol')['date'].agg(['min', 'max']))

print('\nPrice ranges per symbol:')
print(df.groupby('symbol')['close'].agg(['mean', 'std', 'min', 'max']))
"
```

### Fix 3: Increase Model Capacity

```bash
# If model is too small for multiple symbols
python daily_stock_forecasting/main.py \
  --group_column symbol \
  --n_layers 6 \
  --n_heads 16 \
  --d_token 256 \
  --epochs 100
```

### Fix 4: Try Different Model Architecture

```bash
# Try CSNTransformer instead of FT-Transformer
python daily_stock_forecasting/main.py \
  --group_column symbol \
  --model_type csn
```

---

## 📊 Success Criteria

After fixes, you should see:

```
Test 1 (Single, No Group):    R² = 0.75  ✅
Test 2 (Single, With Group):  R² = 0.74  ✅ (within 0.05 of Test 1)
Test 3 (Multi, With Group):   R² = 0.72  ✅ (within 0.10 of Test 2)
```

**Acceptable degradation:** Multi-symbol can be 5-10% worse than single-symbol
**Unacceptable degradation:** >20% worse indicates a bug

---

## 🆘 If Nothing Works

If you've tried everything and Test 3 is still much worse:

### Nuclear Option: Disable Group Scaling

Train separate models for each symbol:

```bash
# Extract each symbol
for symbol in AAPL GOOGL MSFT; do
  python daily_stock_forecasting/main.py \
    --data_path data/${symbol}.csv \
    --target close \
    --group_column None \
    --model_path models/${symbol}_model.pt
done
```

This trades convenience for performance, but guarantees baseline performance.

---

## 📞 Getting Help

**Debug Output Location:** `debug_results/`

**Key Files:**
- `comparison_YYYYMMDD_HHMMSS.csv` - Comparison table
- `Test*_YYYYMMDD_HHMMSS.json` - Detailed results per test

**Share these when asking for help!**

---

## 🎯 Expected Timeline

- **Day 1 (2-3 hours):**
  - Run diagnostic tests ✅
  - Identify issue category ✅
  - Try quick fixes ✅

- **Day 2 (4-6 hours):**
  - Deep dive into specific issue
  - Implement proper fixes
  - Re-run tests

- **Day 3 (2-4 hours):**
  - Validate fixes
  - Test on real data
  - Document learnings

**Total:** 1-3 days depending on issue complexity

---

## ✅ Checklist Before Starting

- [ ] Have multi-symbol data ready (CSV with 'symbol' column)
- [ ] Data has at least 100 rows per symbol
- [ ] Data has 'date' or 'timestamp' column
- [ ] Data has OHLCV columns
- [ ] Python environment set up
- [ ] 30-60 minutes available for initial tests
- [ ] Read DEBUGGING_PLAN.md for detailed steps

---

**Ready?** Run: `python debug_single_vs_multi.py`
