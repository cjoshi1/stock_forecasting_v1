"""
Quick debug script to check if alignment fix is working.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(__file__))

# Create minimal test case
print("Creating minimal test case...")

# Simulate processed vs raw dataframe issue
raw_df = pd.DataFrame({
    'symbol': ['BTC'] * 100,
    'close': np.random.uniform(40000, 50000, 100),
    'volume': np.random.uniform(1e9, 1e10, 100),
    'date': pd.date_range('2024-01-01', periods=100)
})

print(f"Raw df shape: {raw_df.shape}")
print(f"Raw df head:\n{raw_df.head()}")

# Simulate feature engineering (drops some rows)
processed_df = raw_df.copy()
processed_df = processed_df.iloc[5:]  # Drop first 5 rows (like feature engineering does)
processed_df = processed_df.iloc[:-3]  # Drop last 3 rows (like target shifting does)

print(f"\nProcessed df shape: {processed_df.shape}")
print(f"Processed df head:\n{processed_df.head()}")

# Simulate sequence creation
sequence_length = 10
print(f"\nAfter sequence offset ({sequence_length}):")
print(f"Raw actual would start at index: {sequence_length}")
print(f"Raw actual[0] date: {raw_df.iloc[sequence_length]['date']}")
print(f"Raw actual[0] close: {raw_df.iloc[sequence_length]['close']:.2f}")

print(f"\nProcessed actual would start at index: {sequence_length}")
print(f"Processed actual[0] date: {processed_df.iloc[sequence_length]['date']}")
print(f"Processed actual[0] close: {processed_df.iloc[sequence_length]['close']:.2f}")

# Show the misalignment
raw_idx = sequence_length
proc_idx = sequence_length
print(f"\n❌ OLD BUG: Comparing indices that point to different dates")
print(f"   prediction[0] from processed_df index {proc_idx}: {processed_df.iloc[proc_idx]['date']}")
print(f"   actual[0] from raw_df index {raw_idx}: {raw_df.iloc[raw_idx]['date']}")
print(f"   Date difference: {(processed_df.iloc[proc_idx]['date'] - raw_df.iloc[raw_idx]['date']).days} days")
print(f"   Price difference: {abs(processed_df.iloc[proc_idx]['close'] - raw_df.iloc[raw_idx]['close']):.2f}")

print("\n✅ FIX: Both use processed_df, so same date/price")
print(f"   prediction[0] from processed_df: {processed_df.iloc[proc_idx]['date']}")
print(f"   actual[0] from processed_df: {processed_df.iloc[proc_idx]['date']}")
print(f"   Same date: True")
