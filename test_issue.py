# Test the exact issue
import sys

# Simulate the command parsing
target_arg = "close, volume"  # Your command has space after comma

if ',' in target_arg:
    target_columns = [t.strip() for t in target_arg.split(',')]
    print(f"Parsed targets: {target_columns}")
    print(f"Target 1: '{target_columns[0]}' (len={len(target_columns[0])})")
    print(f"Target 2: '{target_columns[1]}' (len={len(target_columns[1])})")
else:
    target_columns = target_arg

# Check for any hidden characters
for i, t in enumerate(target_columns):
    print(f"\nTarget {i}: repr={repr(t)}, bytes={t.encode()}")
