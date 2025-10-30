"""
Quick test to verify OnlyMaxScaler behavior.
Ensures it divides by max without shifting minimum to 0.
"""

import numpy as np
from tf_predictor import ScalerFactory

print("=" * 80)
print("ONLYMAX SCALER VERIFICATION")
print("=" * 80)

# Test data: [5, 10, 15]
# Expected: [5/15, 10/15, 15/15] = [0.333..., 0.666..., 1.0]
# NOT [0, 0.5, 1.0] which would be MinMaxScaler
test_data = np.array([[5], [10], [15]])

print("\nOriginal data:")
print(test_data.flatten())

# Create OnlyMaxScaler
scaler = ScalerFactory.create_scaler('onlymax')
scaler.fit(test_data)

print(f"\nFitted max value: {scaler.data_max_[0]}")

# Transform
scaled = scaler.transform(test_data)
print(f"\nScaled data (X / X_max):")
print(scaled.flatten())

# Expected values
expected = np.array([5/15, 10/15, 15/15])
print(f"\nExpected values:")
print(expected)

# Verify
assert np.allclose(scaled.flatten(), expected), "Scaling incorrect!"
print("\n✓ OnlyMaxScaler working correctly!")
print("  - Divides by max only")
print("  - Does NOT shift minimum to 0")
print("  - Preserves original distribution shape")

# Inverse transform
original = scaler.inverse_transform(scaled)
print(f"\nInverse transformed:")
print(original.flatten())
assert np.allclose(original, test_data), "Inverse transform incorrect!"
print("✓ Inverse transform working correctly!")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
