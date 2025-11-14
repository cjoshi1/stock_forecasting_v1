"""
Test the fix for split_time_series with test_size=0
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from tf_predictor.core.utils import split_time_series

# Test 1: Non-grouped with test_size=0
print("Test 1: Non-grouped split with test_size=0")
df = pd.DataFrame({
    'value': range(100),
    'date': pd.date_range('2020-01-01', periods=100)
})

train, val, test = split_time_series(df, test_size=0, val_size=20, sequence_length=5)

print(f"  Train samples: {len(train)}")
print(f"  Val samples: {len(val)}")
print(f"  Test samples: {len(test)}")
assert len(train) > 0, "Train should not be empty!"
assert len(val) == 20, "Val should have 20 samples"
assert len(test) == 0, "Test should be empty when test_size=0"
print("  ✅ PASSED\n")

# Test 2: Grouped with test_size=0
print("Test 2: Grouped split with test_size=0")
df_grouped = pd.DataFrame({
    'Store': np.repeat([1, 2, 3], 100),
    'value': range(300),
    'Date': pd.date_range('2020-01-01', periods=100).tolist() * 3
})

train, val, test = split_time_series(
    df_grouped,
    test_size=0,
    val_size=20,
    group_column='Store',
    time_column='Date',
    sequence_length=5
)

print(f"  Train samples: {len(train)}")
print(f"  Val samples: {len(val)}")
print(f"  Test samples: {len(test) if test is not None else 0}")
assert len(train) > 0, "Train should not be empty!"
assert len(val) > 0, "Val should not be empty!"
assert len(test) == 0, "Test should be empty when test_size=0"

# Check each group has proper split
for store in [1, 2, 3]:
    train_store = train[train['Store'] == store]
    val_store = val[val['Store'] == store]
    print(f"  Store {store}: Train={len(train_store)}, Val={len(val_store)}")
    assert len(train_store) > 0, f"Store {store} train should not be empty"
    assert len(val_store) > 0, f"Store {store} val should not be empty"

print("  ✅ PASSED\n")

# Test 3: Grouped with val_size=0 (edge case)
print("Test 3: Grouped split with val_size=0")
train, val, test = split_time_series(
    df_grouped,
    test_size=20,
    val_size=0,
    group_column='Store',
    time_column='Date',
    sequence_length=5
)

print(f"  Train samples: {len(train)}")
print(f"  Val samples: {len(val) if val is not None else 0}")
print(f"  Test samples: {len(test)}")
assert len(train) > 0, "Train should not be empty!"
assert val is None or len(val) == 0, "Val should be empty when val_size=0"
assert len(test) > 0, "Test should not be empty!"
print("  ✅ PASSED\n")

print("="*60)
print("✅ All tests passed! The fix works correctly.")
print("="*60)
