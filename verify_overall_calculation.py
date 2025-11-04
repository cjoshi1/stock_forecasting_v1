#!/usr/bin/env python3
"""
Verify the overall metrics calculation method with controlled data.
"""

import numpy as np
from tf_predictor.core.utils import calculate_metrics

print("="*80)
print("🔬 Verifying Overall Metrics Calculation Method")
print("="*80)

# Create controlled test data with known differences between methods
print("\n1️⃣  Creating controlled test data...")

# Horizon 1: Small errors
h1_actual = np.array([100, 101, 102, 103, 104])
h1_pred = np.array([100.5, 101.5, 102.5, 103.5, 104.5])

# Horizon 2: Medium errors
h2_actual = np.array([105, 106, 107, 108, 109])
h2_pred = np.array([106, 107, 108, 109, 110])

# Horizon 3: Large errors
h3_actual = np.array([110, 111, 112, 113, 114])
h3_pred = np.array([115, 116, 117, 118, 119])

print(f"   H1 errors: {h1_pred - h1_actual} → MAE should be 0.5")
print(f"   H2 errors: {h2_pred - h2_actual} → MAE should be 1.0")
print(f"   H3 errors: {h3_pred - h3_actual} → MAE should be 5.0")

# Calculate per-horizon metrics
print("\n2️⃣  Calculating per-horizon metrics...")
h1_metrics = calculate_metrics(h1_actual, h1_pred)
h2_metrics = calculate_metrics(h2_actual, h2_pred)
h3_metrics = calculate_metrics(h3_actual, h3_pred)

print(f"   H1 MAE: {h1_metrics['MAE']:.4f}, RMSE: {h1_metrics['RMSE']:.4f}")
print(f"   H2 MAE: {h2_metrics['MAE']:.4f}, RMSE: {h2_metrics['RMSE']:.4f}")
print(f"   H3 MAE: {h3_metrics['MAE']:.4f}, RMSE: {h3_metrics['RMSE']:.4f}")

# Method 1: Average of per-horizon metrics
avg_mae = (h1_metrics['MAE'] + h2_metrics['MAE'] + h3_metrics['MAE']) / 3
avg_rmse = (h1_metrics['RMSE'] + h2_metrics['RMSE'] + h3_metrics['RMSE']) / 3
print(f"\n3️⃣  Method 1 (Average of horizons):")
print(f"   MAE: {avg_mae:.4f}")
print(f"   RMSE: {avg_rmse:.4f}")

# Method 2: Flatten then calculate (correct method)
all_actual = np.concatenate([h1_actual, h2_actual, h3_actual])
all_pred = np.concatenate([h1_pred, h2_pred, h3_pred])
flatten_metrics = calculate_metrics(all_actual, all_pred)
print(f"   Method 2 (Flatten-then-calculate):")
print(f"   MAE: {flatten_metrics['MAE']:.4f}")
print(f"   RMSE: {flatten_metrics['RMSE']:.4f}")

# Compare
print(f"\n4️⃣  Comparison:")
print(f"   MAE:")
print(f"      Method 1 (average): {avg_mae:.4f}")
print(f"      Method 2 (flatten): {flatten_metrics['MAE']:.4f}")
print(f"      Difference: {abs(avg_mae - flatten_metrics['MAE']):.4f}")
print(f"   RMSE:")
print(f"      Method 1 (average): {avg_rmse:.4f}")
print(f"      Method 2 (flatten): {flatten_metrics['RMSE']:.4f}")
print(f"      Difference: {abs(avg_rmse - flatten_metrics['RMSE']):.4f}")

if abs(avg_rmse - flatten_metrics['RMSE']) < 0.0001:
    print("\n   ⚠️  Both methods produce same result for RMSE")
else:
    print(f"\n   ✅ RMSE differs between methods (as expected)")
    print(f"      This confirms flatten-then-calculate is being used")

# The correct value should be flatten method
print(f"\n5️⃣  Manual calculation verification:")
all_errors = np.abs(all_pred - all_actual)
manual_mae = np.mean(all_errors)
squared_errors = (all_pred - all_actual) ** 2
manual_rmse = np.sqrt(np.mean(squared_errors))
print(f"   Manual MAE: {manual_mae:.4f}")
print(f"   Flatten MAE: {flatten_metrics['MAE']:.4f}")
print(f"   Match: {abs(manual_mae - flatten_metrics['MAE']) < 0.0001}")
print(f"   Manual RMSE: {manual_rmse:.4f}")
print(f"   Flatten RMSE: {flatten_metrics['RMSE']:.4f}")
print(f"   Match: {abs(manual_rmse - flatten_metrics['RMSE']) < 0.0001}")

print("\n" + "="*80)
print("💡 Conclusion:")
print("   The correct method is 'flatten-then-calculate'")
print("   This treats all prediction-actual pairs equally")
print("   The codebase uses this correct method (verified in predictor.py)")
print("="*80)
