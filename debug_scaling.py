"""
Debug script to test scaling behavior for multi-group, multi-target predictions.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Simulate the three groups with realistic values
print("="*60)
print("Simulating Group-Based Scaling for Multi-Target Prediction")
print("="*60)

# Create sample data matching actual stats
btc_close = np.array([70000, 75000, 80000])
eth_close = np.array([2000, 2500, 3000])
xrp_close = np.array([1.0, 1.5, 2.0])

btc_volume = np.array([40e9, 50e9, 60e9])
eth_volume = np.array([20e9, 25e9, 30e9])
xrp_volume = np.array([5e9, 6e9, 7e9])

# Create group-specific scalers (simulating what the code does)
group_target_scalers = {}

for group_id, (close_data, vol_data, name) in enumerate([
    (btc_close, btc_volume, 'BTC'),
    (eth_close, eth_volume, 'ETH'),
    (xrp_close, xrp_volume, 'XRP')
]):
    group_target_scalers[group_id] = {}

    # For each target, create a scaler
    for target_name, data in [('close', close_data), ('volume', vol_data)]:
        scaler = StandardScaler()
        # Fit on the group's data (reshaped for sklearn)
        scaler.fit(data.reshape(-1, 1))
        group_target_scalers[group_id][target_name] = scaler

        print(f"\nGroup {group_id} ({name}) - {target_name}:")
        print(f"  Mean: {scaler.mean_[0]:.2f}")
        print(f"  Std:  {scaler.scale_[0]:.2f}")

# Now simulate what happens during prediction
print("\n" + "="*60)
print("Simulating Inverse Transform")
print("="*60)

# Simulate model output (scaled predictions) - let's say model predicts perfectly (all zeros in scaled space)
# For multi-target multi-horizon with prediction_horizon=3 and 2 targets:
# Layout: [close_h1, close_h2, close_h3, volume_h1, volume_h2, volume_h3]
prediction_horizon = 3
num_targets = 2

# Create perfectly scaled predictions (zeros = mean in original space)
# Suppose we have 3 predictions, one from each group
num_preds = 3
predictions_scaled = np.zeros((num_preds, num_targets * prediction_horizon))

# Group indices: [0, 1, 2] (one prediction from each group)
group_indices = [0, 1, 2]

print(f"\nScaled predictions shape: {predictions_scaled.shape}")
print(f"Layout: [close_h1, close_h2, close_h3, volume_h1, volume_h2, volume_h3]")

# Now inverse transform
target_columns = ['close', 'volume']
predictions_dict = {}

for idx, target_col in enumerate(target_columns):
    print(f"\n--- Processing target: {target_col} (index {idx}) ---")

    # Initialize array for this target's predictions
    target_preds = np.zeros((len(predictions_scaled), prediction_horizon))

    # Inverse transform each group separately
    for group_value in [0, 1, 2]:
        group_mask = np.array([g == group_value for g in group_indices])
        print(f"\nGroup {group_value}: mask = {group_mask}")

        if not group_mask.any():
            continue

        # Multi-horizon: extract horizons for this target
        # Layout: [close_h1, close_h2, close_h3, volume_h1, volume_h2, volume_h3]
        start_idx = idx * prediction_horizon
        end_idx = start_idx + prediction_horizon

        print(f"  Extracting columns {start_idx}:{end_idx}")

        # Extract all horizons for this target in this group
        group_horizons_scaled = predictions_scaled[group_mask, start_idx:end_idx]
        print(f"  Scaled values: {group_horizons_scaled}")

        # Inverse transform using this group's scaler for this target
        group_horizons_original = group_target_scalers[group_value][target_col].inverse_transform(group_horizons_scaled)
        print(f"  Unscaled values: {group_horizons_original}")
        print(f"  Expected (mean): {group_target_scalers[group_value][target_col].mean_[0]:.2f}")

        target_preds[group_mask] = group_horizons_original

    predictions_dict[target_col] = target_preds
    print(f"\nFinal {target_col} predictions:\n{target_preds}")

print("\n" + "="*60)
print("Testing WRONG scaler usage (what might be happening)")
print("="*60)

# What if we accidentally use the wrong scaler?
print("\nWhat if XRP prediction uses BTC scaler?")
xrp_scaled = np.zeros((1, 3))  # XRP prediction (scaled)
btc_scaler = group_target_scalers[0]['close']  # BTC scaler
xrp_scaler = group_target_scalers[2]['close']  # XRP scaler

wrong_result = btc_scaler.inverse_transform(xrp_scaled)
correct_result = xrp_scaler.inverse_transform(xrp_scaled)

print(f"XRP actual mean: {xrp_scaler.mean_[0]:.2f}")
print(f"BTC actual mean: {btc_scaler.mean_[0]:.2f}")
print(f"Using BTC scaler on XRP data: {wrong_result[0]}")
print(f"Using XRP scaler on XRP data: {correct_result[0]}")
print(f"Error ratio: {wrong_result[0, 0] / correct_result[0, 0]:.2f}x")
