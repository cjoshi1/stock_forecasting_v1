#!/bin/bash
# Repository Cleanup Script
# Removes temporary debug files and archives old documentation

set -e  # Exit on error

echo "🧹 Starting repository cleanup..."
echo ""

# Step 1: Remove temporary log files
echo "📝 Removing log files..."
rm -f debug_50epoch.log \
      training.log \
      training_output.log \
      verification_complete.log \
      verification_final.log \
      verification_run.log
echo "   ✅ Removed 6 log files"

# Step 2: Remove debug directories
echo ""
echo "🗂️  Removing debug directories..."
rm -rf debug_results/ verification_output/ check_results/
echo "   ✅ Removed 3 debug directories"

# Step 3: Remove temporary test scripts
echo ""
echo "🧪 Removing temporary test scripts..."
rm -f test_phase1_implementation.py \
      test_phase3a.py \
      test_phase3a_e2e.py \
      test_phase3b.py \
      test_phase3b_simple.py \
      test_consistency_fixes.py \
      test_auto_sort.py \
      test_temporal_order.py \
      test_group_wise_split.py \
      test_group_scaling.py \
      test_group_multi_horizon.py \
      test_group_multi_horizon_quick.py \
      test_multi_target.py \
      test_multi_target_evaluation.py \
      test_save_load_multi_target.py \
      test_stock_multi_target.py \
      test_stock_grouped_multi_target.py \
      test_intraday_group_column.py \
      test_intraday_grouped_multi_target.py \
      debug_single_vs_multi.py \
      run_debug_crypto.py \
      run_debug_with_sample.py \
      verify_pipeline_stages.py \
      profile_features.py \
      profile_training.py \
      list_features.py \
      run_all_logical_checks.py
echo "   ✅ Removed 27 temporary test files"

# Step 4: Archive old documentation
echo ""
echo "📚 Archiving old documentation..."
mkdir -p docs/historical

# Move old docs to archive
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
   docs/historical/ 2>/dev/null || echo "   ⚠️  Some docs already moved or not found"

echo "   ✅ Archived 13 old documentation files to docs/historical/"

# Summary
echo ""
echo "=" | awk '{for(i=1;i<=80;i++)printf "="; print ""}'
echo "✅ CLEANUP COMPLETE!"
echo "=" | awk '{for(i=1;i<=80;i++)printf "="; print ""}'
echo ""
echo "Summary of changes:"
echo "  ✅ Removed: 6 log files"
echo "  ✅ Removed: 3 debug directories"
echo "  ✅ Removed: 27 temporary test scripts"
echo "  ✅ Archived: 13 old documentation files → docs/historical/"
echo ""
echo "Files kept:"
echo "  ✅ test_scaler_types.py (current feature test)"
echo "  ✅ test_onlymax_scaler.py (current feature test)"
echo "  ✅ logical_checks_daily.py (validation script)"
echo "  ✅ logical_checks_intraday.py (validation script)"
echo "  ✅ All current documentation files"
echo ""
echo "Optional next steps:"
echo "  💡 To clean outputs directory: rm -rf outputs/"
echo "  💡 To view archived docs: ls docs/historical/"
echo "  💡 Review CLEANUP_RECOMMENDATIONS.md for more options"
echo ""
echo "=" | awk '{for(i=1;i<=80;i++)printf "="; print ""}'
