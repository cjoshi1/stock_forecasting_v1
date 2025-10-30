# Repository Cleanup Recommendations

**Date**: 2025-10-30
**Purpose**: Identify files and directories that can be safely deleted

## 📊 Repository Analysis Summary

**Total files analyzed**: 60+ root level files
**Directories**: 10 (including outputs, debug results, etc.)
**Size**: ~23MB in outputs directory alone

---

## 🗑️ SAFE TO DELETE - Debugging & Development Artifacts

### High Priority (Delete Now)

#### 1. Debug/Verification Logs (All can be deleted)
These are temporary debugging outputs with no long-term value:

```bash
# Log files - 220KB total
debug_50epoch.log                    # 114KB - old debug run
training.log                         # 758B - temporary training log
training_output.log                  # 4.4KB - temporary output
verification_complete.log            # 30KB - verification run
verification_final.log               # 36KB - verification run
verification_run.log                 # 502B - verification run
```

**Recommendation**: ✅ DELETE ALL - These are temporary debugging artifacts

#### 2. Debug Result Directories (48KB + 4KB)
```bash
debug_results/                       # 48KB - temporary debug outputs
verification_output/                 # 4KB - temporary verification outputs
check_results/                       # Multiple timestamped JSON/txt files
```

**Recommendation**: ✅ DELETE ALL - These are temporary test results

#### 3. Temporary Test Scripts (22 files)
These were created during development/debugging and are no longer needed:

```bash
# Phase-specific tests (covered by final implementation)
test_phase1_implementation.py        # Phase 1 tests - implementation complete
test_phase3a.py                      # Phase 3A tests - implementation complete
test_phase3a_e2e.py                  # Phase 3A end-to-end - implementation complete
test_phase3b.py                      # Phase 3B tests - implementation complete
test_phase3b_simple.py               # Phase 3B simple test - implementation complete

# Feature verification scripts (one-time use)
test_consistency_fixes.py            # Consistency tests - fixes applied
test_auto_sort.py                    # Auto-sort verification - working
test_temporal_order.py               # Temporal order checks - verified
test_group_wise_split.py             # Group split verification - verified
test_group_scaling.py                # Group scaling tests - verified
test_group_multi_horizon.py          # Multi-horizon tests - verified
test_group_multi_horizon_quick.py    # Quick test version - redundant

# Multi-target tests (functionality now in main tests)
test_multi_target.py                 # Multi-target tests - covered
test_multi_target_evaluation.py      # Evaluation tests - covered
test_save_load_multi_target.py       # Save/load tests - covered
test_stock_multi_target.py           # Stock multi-target - covered
test_stock_grouped_multi_target.py   # Grouped multi-target - covered
test_intraday_group_column.py        # Intraday group tests - covered
test_intraday_grouped_multi_target.py # Intraday grouped - covered

# Debug/profile scripts (one-time use)
debug_single_vs_multi.py             # Debugging script - issue resolved
run_debug_crypto.py                  # Crypto debugging - issue resolved
run_debug_with_sample.py             # Sample debugging - issue resolved
verify_pipeline_stages.py            # Pipeline verification - verified
profile_features.py                  # Feature profiling - profiled
profile_training.py                  # Training profiling - profiled
list_features.py                     # Feature listing - utility

# Logical check runners (can keep logical_checks_*.py, delete runners)
run_all_logical_checks.py           # Runner script - can use individual scripts
```

**Recommendation**: ✅ DELETE ALL EXCEPT:
- Keep: `test_scaler_types.py` (current feature test)
- Keep: `test_onlymax_scaler.py` (current feature test)
- Keep: `logical_checks_daily.py` (useful validation script)
- Keep: `logical_checks_intraday.py` (useful validation script)

---

## 📚 CONDITIONAL DELETE - Documentation

### Historical Documentation (Superseded by Current Docs)

These docs were useful during development but are now superseded:

```bash
ALIGNMENT_FIX.md                     # Old bug fix - issue resolved
CONSISTENCY_FIXES_SUMMARY.md         # Old fixes - incorporated
DEBUGGING_PLAN.md                    # Old debugging plan - completed
DEBUGGING_QUICKSTART.md              # Old debug guide - not needed
DIAGNOSTIC_RESULTS_ANALYSIS.md       # Old analysis - completed
DIMENSION_FIX_SUMMARY.md             # Old bug fix - issue resolved
GROUP_SCALING_SUMMARY.md             # Old feature - now in main docs
GROUP_MULTI_HORIZON_SUMMARY.md       # Old feature - now in main docs
MEMORY_FIX_SUMMARY.md                # Old bug fix - issue resolved
DATA_LEAKAGE_VERIFICATION.md         # Old verification - passed
VERIFICATION_SCRIPT_GUIDE.md         # Old guide - not needed
MULTI_HORIZON_ENHANCEMENT_PLAN.md    # Old plan - implemented
MULTI_TARGET_IMPLEMENTATION_PLAN.md  # Old plan - implemented
```

**Recommendation**: 🟡 MOVE TO `docs/historical/` OR DELETE

**Keep if you want historical record of development process**
**Delete if you only care about current state**

---

## ✅ KEEP - Current & Important Documentation

### Essential Documentation (Keep)

```bash
README.md                            # ✅ Main project README
QUICK_START.md                       # ✅ Quick start guide
COMMAND_EXAMPLES.md                  # ✅ Command reference
LOGICAL_CHECKS.md                    # ✅ Validation documentation
LOGICAL_CHECKS_COMPLETE_REPORT.md    # ✅ Validation results

# Recent implementation docs (keep)
MULTI_HORIZON_IMPLEMENTATION_SUMMARY.md  # ✅ Multi-horizon feature
MODULE_SEPARATION_RECOMMENDATIONS.md     # ✅ Architecture guide
CLEAN_SEPARATION_SUMMARY.md             # ✅ Module separation
ONLYMAX_SCALER_IMPLEMENTATION.md        # ✅ Recent feature

# Architecture documentation
tf_predictor/REFACTORING_COMPLETE.md    # ✅ Complete refactoring guide
tf_predictor/ARCHITECTURE.md            # ✅ Architecture documentation (if exists)
```

---

## 📂 Output Directories

### `outputs/` Directory (23MB)

Contains prediction results and visualizations from past runs.

**Contents**:
- CSV prediction files (with timestamps)
- PNG visualization files (with timestamps)
- `data/` subdirectory

**Recommendation**: 🟡 CONDITIONAL
- ✅ **Delete if**: You don't need old prediction results
- ⚠️ **Keep if**: You want historical prediction records
- 💡 **Alternative**: Archive important results, delete the rest

```bash
# Option 1: Delete all
rm -rf outputs/

# Option 2: Keep structure, delete old files
mkdir -p outputs/archive
mv outputs/*.csv outputs/*.png outputs/archive/
```

---

## 🎯 Recommended Cleanup Commands

### Step 1: Delete Temporary Debug Files (Safe)

```bash
# Delete all log files
rm -f *.log

# Delete debug directories
rm -rf debug_results/ verification_output/ check_results/

# Delete temporary test scripts
rm -f test_phase*.py \
      test_consistency_fixes.py \
      test_auto_sort.py \
      test_temporal_order.py \
      test_group_wise_split.py \
      test_group_scaling.py \
      test_group_multi_horizon*.py \
      test_multi_target*.py \
      test_stock_multi_target.py \
      test_stock_grouped_multi_target.py \
      test_intraday_group_column.py \
      test_intraday_grouped_multi_target.py \
      debug_single_vs_multi.py \
      run_debug_*.py \
      verify_pipeline_stages.py \
      profile_*.py \
      list_features.py \
      run_all_logical_checks.py
```

**Result**: Removes ~22 test files + 6 log files + 3 directories

### Step 2: Archive or Delete Historical Documentation (Optional)

```bash
# Option A: Delete old documentation
rm -f ALIGNMENT_FIX.md \
      CONSISTENCY_FIXES_SUMMARY.md \
      DEBUGGING_PLAN.md \
      DEBUGGING_QUICKSTART.md \
      DIAGNOSTIC_RESULTS_ANALYSIS.md \
      DIMENSION_FIX_SUMMARY.md \
      GROUP_SCALING_SUMMARY.md \
      GROUP_MULTI_HORIZON_SUMMARY.md \
      MEMORY_FIX_SUMMARY.md \
      DATA_LEAKAGE_VERIFICATION.md \
      VERIFICATION_SCRIPT_GUIDE.md \
      MULTI_HORIZON_ENHANCEMENT_PLAN.md \
      MULTI_TARGET_IMPLEMENTATION_PLAN.md

# Option B: Archive old documentation
mkdir -p docs/historical
mv ALIGNMENT_FIX.md \
   CONSISTENCY_FIXES_SUMMARY.md \
   DEBUGGING_PLAN.md \
   DEBUGGING_QUICKSTART.md \
   DIAGNOSTIC_RESULTS_ANALYSIS.md \
   DIMENSION_FIX_SUMMARY.md \
   GROUP_SCALING_SUMMARY.md \
   GROUP_MULTI_HORIZON_SUMMARY.md \
   MEMORY_FIX_SUMMARY.md \
   DATA_LEAKAGE_VERIFICATION.md \
   VERIFICATION_SCRIPT_GUIDE.md \
   MULTI_HORIZON_ENHANCEMENT_PLAN.md \
   MULTI_TARGET_IMPLEMENTATION_PLAN.md \
   docs/historical/
```

**Result**: Removes/archives 13 old documentation files

### Step 3: Clean Output Directory (Optional)

```bash
# Option A: Delete all outputs
rm -rf outputs/

# Option B: Keep latest, delete old
cd outputs/
# Keep files from last month, delete older
find . -name "*.csv" -mtime +30 -delete
find . -name "*.png" -mtime +30 -delete
cd ..

# Option C: Archive then clean
mkdir -p outputs/archive
mv outputs/*.csv outputs/*.png outputs/archive/ 2>/dev/null || true
```

**Result**: Cleans up 23MB of old prediction outputs

---

## 📋 Final Repository Structure (After Cleanup)

```
stock_forecasting_v1/
├── README.md                                    # ✅ Keep
├── QUICK_START.md                               # ✅ Keep
├── COMMAND_EXAMPLES.md                          # ✅ Keep
├── LOGICAL_CHECKS.md                            # ✅ Keep
├── LOGICAL_CHECKS_COMPLETE_REPORT.md            # ✅ Keep
├── MULTI_HORIZON_IMPLEMENTATION_SUMMARY.md      # ✅ Keep
├── MODULE_SEPARATION_RECOMMENDATIONS.md         # ✅ Keep
├── CLEAN_SEPARATION_SUMMARY.md                  # ✅ Keep
├── ONLYMAX_SCALER_IMPLEMENTATION.md            # ✅ Keep
├── CLEANUP_RECOMMENDATIONS.md                   # ✅ Keep (this file)
│
├── logical_checks_daily.py                      # ✅ Keep (validation)
├── logical_checks_intraday.py                   # ✅ Keep (validation)
├── test_scaler_types.py                         # ✅ Keep (current test)
├── test_onlymax_scaler.py                       # ✅ Keep (current test)
│
├── tf_predictor/                                # ✅ Core library
│   ├── setup.py                                 # ✅ Package setup
│   ├── REFACTORING_COMPLETE.md                  # ✅ Architecture doc
│   └── ...
│
├── daily_stock_forecasting/                     # ✅ Stock application
├── intraday_forecasting/                        # ✅ Intraday application
├── data/                                        # ✅ Data directory
├── outputs/                                     # 🟡 Optional (clean periodically)
└── venv/                                        # ✅ Virtual environment
```

---

## 📊 Cleanup Impact Summary

| Category | Files | Size | Recommendation |
|----------|-------|------|----------------|
| **Log files** | 6 | 220KB | ✅ DELETE |
| **Debug directories** | 3 | 52KB | ✅ DELETE |
| **Test scripts** | 22 | ~100KB | ✅ DELETE |
| **Old documentation** | 13 | ~50KB | 🟡 ARCHIVE or DELETE |
| **Output files** | Many | 23MB | 🟡 CONDITIONAL |
| **Total removable** | 44+ | ~23.5MB | Can free significant space |

---

## 🎯 Recommended Action

### Conservative Approach (Recommended)
1. ✅ Delete all temporary files (logs, debug dirs, test scripts)
2. 🟡 Archive old documentation to `docs/historical/`
3. 🟡 Keep `outputs/` but clean periodically

### Aggressive Approach
1. ✅ Delete all temporary files
2. ✅ Delete all old documentation
3. ✅ Delete `outputs/` directory (or keep only recent)

### Create a Cleanup Script

```bash
#!/bin/bash
# cleanup.sh - Repository cleanup script

echo "🧹 Starting repository cleanup..."

# Step 1: Remove temporary files
echo "Removing log files..."
rm -f *.log

echo "Removing debug directories..."
rm -rf debug_results/ verification_output/ check_results/

echo "Removing temporary test scripts..."
rm -f test_phase*.py \
      test_consistency_fixes.py \
      test_auto_sort.py \
      test_temporal_order.py \
      test_group_wise_split.py \
      test_group_scaling.py \
      test_group_multi_horizon*.py \
      test_multi_target*.py \
      test_stock_multi_target.py \
      test_stock_grouped_multi_target.py \
      test_intraday_group_column.py \
      test_intraday_grouped_multi_target.py \
      debug_single_vs_multi.py \
      run_debug_*.py \
      verify_pipeline_stages.py \
      profile_*.py \
      list_features.py \
      run_all_logical_checks.py

# Step 2: Archive old documentation (optional - comment out if you want to delete)
echo "Archiving old documentation..."
mkdir -p docs/historical
mv ALIGNMENT_FIX.md \
   CONSISTENCY_FIXES_SUMMARY.md \
   DEBUGGING_PLAN.md \
   DEBUGGING_QUICKSTART.md \
   DIAGNOSTIC_RESULTS_ANALYSIS.md \
   DIMENSION_FIX_SUMMARY.md \
   GROUP_SCALING_SUMMARY.md \
   GROUP_MULTI_HORIZON_SUMMARY.md \
   MEMORY_FIX_SUMMARY.md \
   DATA_LEAKAGE_VERIFICATION.md \
   VERIFICATION_SCRIPT_GUIDE.md \
   MULTI_HORIZON_ENHANCEMENT_PLAN.md \
   MULTI_TARGET_IMPLEMENTATION_PLAN.md \
   docs/historical/ 2>/dev/null || true

echo "✅ Cleanup complete!"
echo ""
echo "Summary:"
echo "- Removed: temporary logs and debug files"
echo "- Removed: temporary test scripts"
echo "- Archived: old documentation to docs/historical/"
echo ""
echo "To also clean outputs directory, run:"
echo "  rm -rf outputs/  # or clean selectively"
```

Save as `cleanup.sh`, make executable: `chmod +x cleanup.sh`, then run: `./cleanup.sh`

---

## ⚠️ Important Notes

1. **Before deletion**: Consider committing current state to git first
2. **Git history**: Deleted files remain in git history (can recover if needed)
3. **Outputs**: Back up any important prediction results before cleaning
4. **Documentation**: Consider archiving rather than deleting for historical record

---

## ✅ Final Recommendation

**Execute the conservative cleanup**:
1. Delete temporary debugging artifacts (no value)
2. Archive old documentation (historical record preserved)
3. Keep outputs but clean periodically
4. Keep current test files and validation scripts

This will:
- ✅ Remove ~44 unnecessary files
- ✅ Free ~23.5MB of space
- ✅ Clean up repository significantly
- ✅ Preserve important historical documentation
- ✅ Keep all current, useful files
