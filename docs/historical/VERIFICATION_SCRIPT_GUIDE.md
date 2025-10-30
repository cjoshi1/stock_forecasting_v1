# Pipeline Verification Script Guide

## Overview

The `verify_pipeline_stages.py` script shows you **every transformation** your data goes through from raw CSV to final predictions, with matrix dimensions at each step.

## What It Verifies

### ✅ Stage 1: Raw Data Loading
- Loads CSV and normalizes column names
- Shows number of rows, columns, and groups
- Displays price ranges per symbol
- Verifies date ranges

**Output Example:**
```
✓ Loaded 1098 rows
✓ Groups: ['BTC-USD', 'ETH-USD', 'XRP-USD']
BTC-USD: $66,642 - $124,753 (mean: $100,740)
ETH-USD: $1,473 - $4,831 (mean: $3,084)
XRP-USD: $0.50 - $3.56 (mean: $2.38)
```

---

### ✅ Stage 2: Feature Engineering
- Creates technical features (VWAP, cyclical time features)
- Shows which features were added
- Verifies group column is preserved
- Checks for NaN values

**Output Example:**
```
✓ Original columns: 7
✓ After feature engineering: 16
✓ New features: ['vwap', 'month', 'dayofweek', 'month_sin', 'month_cos', ...]
✓ Group column 'symbol' preserved
```

---

### ✅ Stage 3: Data Splitting (Group-wise Temporal)
- Splits data per group while preserving temporal order
- Verifies no temporal leakage (train dates < test dates)
- Shows exact date ranges per split

**Output Example:**
```
BTC-USD:
  Train: 316 samples (2024-10-24 to 2025-09-04)
  Val: 20 samples
  Test: 30 samples (2025-09-25 to 2025-10-24)
  ✓ No temporal leakage
```

---

### ✅ Stage 4: Feature Scaling (Per-Group)
- Fits scalers independently per group
- Shows scaler statistics (mean, std) for each group
- Verifies scaled data has mean≈0, std≈1

**Output Example:**
```
✓ Group-based scaling active: 3 groups

BTC-USD:
  Mean: 98,477.03
  Std: 12,579.37
  Features: 12

Scaled data verification:
BTC-USD: Mean: 0.0000, Std: 1.0016 ✓
```

---

### ✅ Stage 5: Target Scaling (Per-Group)
- Fits target scalers independently per group
- Shows target scaler statistics
- Verifies scaled targets have mean≈0, std≈1

**Output Example:**
```
✓ Per-group target scaling active: 3 groups

BTC-USD:
  Mean: 98,883.97
  Std: 13,048.65

Scaled targets: Mean: 0.0000, Std: 1.0005 ✓
```

---

### ✅ Stage 6: Sequence Creation
- Creates sequences from time series data
- Shows sequence dimensions (batch, timesteps, features)
- Verifies sequences don't cross group boundaries
- Displays sample sequence

**Output Example:**
```
✓ Sequences shape: torch.Size([931, 5, 12])
  - 931 sequences
  - 5 time steps per sequence
  - 12 features per time step

Sample sequence (first 3 timesteps):
  Time 0: [-2.5298, -2.5266, -2.4029, -0.8593, -2.4641]
  Time 1: [-2.4096, -2.5328, -2.4768, -0.4580, -2.5317]
  Time 2: [-2.5318, -2.6459, -2.4103, -1.3313, -2.5369]
```

---

### ✅ Training
- Trains model for specified epochs
- Shows training progress with verbose output

---

### ✅ Stage 6B: Transformer Architecture Trace
- **Traces data flow through transformer layers**
- **Shows matrix dimensions at each step**
- **Displays layer-by-layer transformations**

**For FT-Transformer:**
```
1️⃣  INPUT TO MODEL
   Shape: torch.Size([4, 5, 12])
   [batch_size=4, seq_len=5, n_features=12]

2️⃣  FEATURE TOKENIZATION
   Reshaped to: torch.Size([20, 12])
   After tokenization: torch.Size([20, 12, 128])
   [batch_size*seq_len=20, n_features=12, d_token=128]

3️⃣  CLS TOKEN
   CLS token shape: torch.Size([20, 1, 128])
   After adding CLS: torch.Size([20, 13, 128])

4️⃣  TRANSFORMER BLOCKS (3 layers)
   Layer 1:
      Input: torch.Size([20, 13, 128])
      → Multi-Head Attention: 8 heads, d_model=128
      → Feed-Forward: 128 → 512 → 128
      Output: torch.Size([20, 13, 128])

   Layer 2:
      Input: torch.Size([20, 13, 128])
      → Multi-Head Attention: 8 heads
      → Feed-Forward: 128 → 512 → 128
      Output: torch.Size([20, 13, 128])

   Layer 3:
      [similar structure]

5️⃣  EXTRACT CLS TOKEN
   CLS token extracted: torch.Size([20, 128])

6️⃣  RESHAPE TO SEQUENCES
   Reshaped: torch.Size([4, 5, 128])

7️⃣  OUTPUT HEAD
   After head: torch.Size([4, 5, 1])

8️⃣  FINAL OUTPUT
   Shape: torch.Size([4, 5, 1])
   Sample predictions: [pred1, pred2, pred3, pred4]
```

---

### ✅ Stage 7: Model Predictions
- Generates predictions on test set
- Shows prediction shape and sample values

---

### ✅ Stage 8: Inverse Transform
- Transforms predictions back to original scale
- Compares predictions vs actuals
- Shows per-group statistics

---

### ✅ Stage 9: Metrics Calculation
- Calculates final metrics (R², MAPE, RMSE, etc.)
- Shows overall and per-group metrics

**Output Example:**
```
Overall Metrics:
  R2: 0.1569
  MAPE: 77.53%
  RMSE: 49021.39

Per-Group Metrics:
BTC-USD:
  R2: -237.86
  MAPE: 80.21%

ETH-USD:
  R2: -241.43
  MAPE: 98.22%

XRP-USD:
  R2: -0.28
  MAPE: 79.96%
```

---

## How to Run

### Basic Usage:
```bash
python verify_pipeline_stages.py \
  --data_path /path/to/your/data.csv \
  --group_column symbol \
  --target close \
  --epochs 10
```

### Full Options:
```bash
python verify_pipeline_stages.py \
  --data_path /path/to/data.csv \
  --group_column symbol \
  --target close \
  --sequence_length 5 \
  --test_size 30 \
  --val_size 20 \
  --epochs 10 \
  --asset_type crypto
```

---

## Output Files

### Console Output
All stages are printed to console with detailed information and verification checks.

### Saved Files (in `verification_output/`):
- `architecture_trace_TIMESTAMP.txt` - Transformer architecture details
- Logs showing all transformations

---

## What This Helps You Verify

### ✅ Data Integrity:
- No temporal leakage between splits
- Group column preserved throughout pipeline
- No NaN values after feature engineering

### ✅ Scaling Correctness:
- Each group has independent scalers
- Scalers have correct statistics (matching actual data ranges)
- Scaled data has mean≈0, std≈1
- Inverse transform works correctly

### ✅ Sequence Validity:
- Sequences don't cross group boundaries
- Correct sequence dimensions
- Proper temporal ordering within sequences

### ✅ Model Architecture:
- Data flows correctly through transformer
- Dimensions match at each layer
- Output shape is correct

### ✅ Predictions:
- Predictions are in correct scale (after inverse transform)
- Metrics are calculated correctly
- Per-group metrics available

---

## Common Issues and What to Look For

### ❌ If Group Scaling Broken:
You'll see:
- Same mean/std for all groups (should be different)
- Scaled data not centered at 0
- Poor per-group performance

### ❌ If Temporal Leakage:
You'll see:
- Train dates overlap with test dates
- Suspiciously good metrics
- "WARNING: Train/test dates overlap!"

### ❌ If Sequences Cross Boundaries:
You'll see:
- "WARNING: Sequence crosses group boundary!"
- Mixed symbols in same sequence
- Corrupted patterns

### ❌ If Inverse Transform Fails:
You'll see:
- Predictions in wrong scale (e.g., scaled values like 0.5 instead of $100k)
- Huge metric errors
- Predictions don't match actual value ranges

---

## Success Criteria

✅ **All stages should show:**
- No warnings or errors
- Reasonable value ranges at each step
- Correct dimensions throughout
- Independent scalers per group
- No temporal leakage
- Valid sequence boundaries

✅ **Final metrics should be:**
- Reasonable for your data (crypto is hard to predict)
- Consistent across stages
- Per-group metrics available if using groups

---

## Example Full Run

See `verification_final.log` for a complete example run with BTC/ETH/XRP data showing all stages in detail.

---

## Tips

1. **Start with 10 epochs** for quick verification
2. **Check each stage output** - don't skip to the end
3. **Verify scaler statistics** match your data ranges
4. **Look for warnings** - they indicate potential issues
5. **Compare predictions vs actuals** in Stage 8
6. **Review architecture trace** to understand model structure

---

**Created:** 2025-10-24
**Script:** `verify_pipeline_stages.py`
**Purpose:** Complete pipeline transparency and verification
