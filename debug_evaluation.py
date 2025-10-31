#!/usr/bin/env python3
"""
Debug evaluation alignment issues by printing dataframe state after each operation.
"""
import pandas as pd
import numpy as np

def debug_print_df_by_group(df, group_col, title, max_rows_per_group=10):
    """
    Print rows from dataframe, showing max_rows_per_group for each group.

    Args:
        df: DataFrame to print
        group_col: Column to group by (e.g., 'symbol')
        title: Description of the current dataframe state
        max_rows_per_group: Max rows to show per group (default 10)
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"Total rows: {len(df)}")

    if group_col not in df.columns:
        print(f"WARNING: Group column '{group_col}' not found in dataframe")
        print(f"Available columns: {list(df.columns)}")
        return

    unique_groups = sorted(df[group_col].unique())
    print(f"Groups: {unique_groups}")

    # Key columns to show
    key_cols = ['date', group_col]

    # Add target columns if they exist
    target_candidates = ['close', 'volume', 'open', 'high', 'low']
    for col in target_candidates:
        if col in df.columns:
            key_cols.append(col)

    # Add shifted target columns if they exist
    shifted_cols = [c for c in df.columns if '_target_h' in c]
    key_cols.extend(shifted_cols[:6])  # Show first 6 shifted columns max

    # Remove duplicates while preserving order
    key_cols = list(dict.fromkeys(key_cols))

    # Filter to columns that exist
    display_cols = [c for c in key_cols if c in df.columns]

    print(f"Showing columns: {display_cols}")
    print()

    for group in unique_groups:
        group_df = df[df[group_col] == group].reset_index(drop=True)
        print(f"\n--- Group: {group} ({len(group_df)} rows) ---")

        # Show first max_rows_per_group rows
        if len(group_df) > max_rows_per_group:
            print(f"First {max_rows_per_group} rows:")
            print(group_df[display_cols].head(max_rows_per_group).to_string(index=True))
            print(f"... ({len(group_df) - max_rows_per_group} more rows)")
        else:
            print(group_df[display_cols].to_string(index=True))


if __name__ == "__main__":
    # Example test
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=30),
        'symbol': ['AAPL'] * 10 + ['GOOGL'] * 10 + ['MSFT'] * 10,
        'close': np.random.randn(30) * 10 + 100,
        'volume': np.random.randint(1000, 10000, 30),
        'close_target_h1': np.random.randn(30) * 10 + 100,
    })

    debug_print_df_by_group(df, 'symbol', "Test DataFrame", max_rows_per_group=5)
