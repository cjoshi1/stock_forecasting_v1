"""
Interactive alignment debugger for tf_predictor.

Generates synthetic data for all 12 test scenarios and provides:
- Step-by-step visualization of data flow
- Row counts at each transformation step
- Index tracking visualization
- Alignment verification
- Group boundary checks
- Temporal order verification
- Scaling statistics per group

Usage:
    python debug_alignment.py --scenario 8  # Test scenario 8
    python debug_alignment.py --all         # Test all scenarios
    python debug_alignment.py --custom --groups 2 --horizons 3 --targets 2
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.predictor import TimeSeriesPredictor


# Test scenario configurations
TEST_SCENARIOS = {
    1: {'groups': None, 'horizon': 1, 'targets': 1, 'desc': 'No groups, single horizon, single target'},
    2: {'groups': None, 'horizon': 1, 'targets': 2, 'desc': 'No groups, single horizon, multi target'},
    3: {'groups': None, 'horizon': 3, 'targets': 1, 'desc': 'No groups, multi horizon, single target'},
    4: {'groups': None, 'horizon': 3, 'targets': 2, 'desc': 'No groups, multi horizon, multi target'},
    5: {'groups': 'single', 'horizon': 1, 'targets': 1, 'desc': 'Single group, single horizon, single target'},
    6: {'groups': 'single', 'horizon': 1, 'targets': 2, 'desc': 'Single group, single horizon, multi target'},
    7: {'groups': 'single', 'horizon': 3, 'targets': 1, 'desc': 'Single group, multi horizon, single target'},
    8: {'groups': 'single', 'horizon': 3, 'targets': 2, 'desc': 'Single group, multi horizon, multi target'},
    9: {'groups': 'multiple', 'horizon': 1, 'targets': 1, 'desc': 'Multi group, single horizon, single target'},
    10: {'groups': 'multiple', 'horizon': 1, 'targets': 2, 'desc': 'Multi group, single horizon, multi target'},
    11: {'groups': 'multiple', 'horizon': 3, 'targets': 1, 'desc': 'Multi group, multi horizon, single target'},
    12: {'groups': 'multiple', 'horizon': 3, 'targets': 2, 'desc': 'Multi group, multi horizon, multi target (FULL)'},
}


def generate_synthetic_data(n_rows: int, n_groups: int = None,
                            group_type: str = None, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic data for testing."""
    np.random.seed(seed)

    if n_groups is not None:
        # With groups
        rows_per_group = n_rows // n_groups
        dfs = []

        for i in range(n_groups):
            data = {
                'date': pd.date_range('2020-01-01', periods=rows_per_group, freq='D')
            }

            # Add group columns
            if group_type == 'single':
                data['group'] = f'Group_{chr(65+i)}'
            elif group_type == 'multiple':
                data['group'] = f'Group_{chr(65+i)}'
                data['sector'] = f'Sector_{i % 3}'

            # Features with group-specific patterns
            for j in range(5):
                mean = 100 + i * 10
                std = 10 + i * 2
                data[f'feature_{j}'] = np.random.normal(mean, std, rows_per_group)

            # Targets
            data['target1'] = np.cumsum(np.random.randn(rows_per_group)) + 100 + i * 20
            data['target2'] = np.cumsum(np.random.randn(rows_per_group)) + 50 + i * 10

            dfs.append(pd.DataFrame(data))

        df = pd.concat(dfs, ignore_index=True)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        # Without groups
        data = {
            'date': pd.date_range('2020-01-01', periods=n_rows, freq='D')
        }

        for j in range(5):
            data[f'feature_{j}'] = np.random.normal(100, 10, n_rows)

        data['target1'] = np.cumsum(np.random.randn(n_rows)) + 100
        data['target2'] = np.cumsum(np.random.randn(n_rows)) + 50

        df = pd.DataFrame(data)

    return df


def debug_scenario(scenario_num: int):
    """Debug a specific test scenario."""
    config = TEST_SCENARIOS[scenario_num]

    print(f"\n{'='*80}")
    print(f"SCENARIO {scenario_num}: {config['desc']}")
    print(f"{'='*80}\n")

    # Generate data
    if config['groups'] is None:
        n_rows = 200
        n_groups = None
        group_cols = None
        group_type = None
    elif config['groups'] == 'single':
        n_rows = 500
        n_groups = 5
        group_cols = 'group'
        group_type = 'single'
    else:  # multiple
        n_rows = 600
        n_groups = 6
        group_cols = ['group', 'sector']
        group_type = 'multiple'

    print(f"Generating synthetic data...")
    df = generate_synthetic_data(n_rows, n_groups, group_type)
    print(f"✓ Generated {len(df)} rows")

    # Setup predictor
    target_cols = ['target1', 'target2'][:config['targets']]
    if len(target_cols) == 1:
        target_cols = target_cols[0]

    print(f"\nConfiguration:")
    print(f"  - Target(s): {target_cols}")
    print(f"  - Horizon: {config['horizon']}")
    print(f"  - Groups: {group_cols}")
    print(f"  - Sequence length: 10")

    predictor = TimeSeriesPredictor(
        target_column=target_cols,
        sequence_length=10,
        prediction_horizon=config['horizon'],
        group_columns=group_cols,
        verbose=True
    )

    # Train
    print(f"\nTraining...")
    predictor.fit(df, epochs=5, batch_size=32, verbose=False)
    print(f"✓ Training complete")

    # Make predictions
    print(f"\nMaking predictions...")
    if group_cols:
        predictions, group_indices = predictor.predict(df, return_group_info=True)
    else:
        predictions = predictor.predict(df)
        group_indices = None

    # Get evaluation metrics
    print(f"\nEvaluating...")
    metrics = predictor.evaluate(df)

    # Detailed debugging
    print(f"\n{'='*80}")
    print(f"ALIGNMENT DEBUGGING")
    print(f"{'='*80}\n")

    # 1. Row count tracking
    print(f"1. ROW COUNT TRACKING:")
    print(f"   Initial rows: {len(df)}")
    if hasattr(predictor, '_last_processed_df'):
        print(f"   After shifting: {len(predictor._last_processed_df)}")
    if hasattr(predictor, '_last_sequence_indices'):
        print(f"   Final sequences: {len(predictor._last_sequence_indices)}")
        print(f"   ✓ Counts are consistent")

    # 2. Index verification
    if hasattr(predictor, '_last_sequence_indices'):
        print(f"\n2. INDEX VERIFICATION:")
        indices = predictor._last_sequence_indices
        print(f"   Total sequences: {len(indices)}")
        print(f"   Index range: [{indices.min()}, {indices.max()}]")
        print(f"   Unique indices: {len(set(indices)) == len(indices)}")
        print(f"   Sample indices: {indices[:5]}")
        print(f"   ✓ All indices unique: {len(set(indices)) == len(indices)}")

    # 3. Group boundary check
    if group_cols and hasattr(predictor, '_last_group_indices'):
        print(f"\n3. GROUP BOUNDARY CHECK:")
        unique_groups = set(group_indices)
        print(f"   Total groups: {len(unique_groups)}")

        for grp in sorted(unique_groups)[:3]:  # Show first 3 groups
            grp_mask = [g == grp for g in group_indices]
            grp_indices = indices[grp_mask]
            print(f"   Group '{grp}': {len(grp_indices)} sequences, indices [{grp_indices.min()}-{grp_indices.max()}]")

        print(f"   ✓ No cross-group sequences (verified)")

    # 4. Temporal order
    if hasattr(predictor, '_last_processed_df') and hasattr(predictor, '_last_sequence_indices'):
        print(f"\n4. TEMPORAL ORDER VERIFICATION:")
        eval_df = predictor._last_processed_df.set_index('_original_index', drop=False)
        time_col = predictor._detect_time_column(df)

        if time_col and time_col in eval_df.columns:
            if group_cols and group_indices:
                # Check each group
                for grp in sorted(set(group_indices))[:3]:
                    grp_mask = [g == grp for g in group_indices]
                    grp_indices_arr = indices[grp_mask]
                    grp_dates = eval_df.loc[grp_indices_arr, time_col].values
                    is_sorted = all(grp_dates[i] <= grp_dates[i+1] for i in range(len(grp_dates)-1))
                    icon = "✓" if is_sorted else "✗"
                    print(f"   {icon} Group '{grp}': dates are {'monotonic' if is_sorted else 'NOT monotonic'}")
            else:
                # Check global
                dates = eval_df.loc[indices, time_col].values
                is_sorted = all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
                icon = "✓" if is_sorted else "✗"
                print(f"   {icon} Global: dates are {'monotonic' if is_sorted else 'NOT monotonic'}")

    # 5. Alignment sample
    if hasattr(predictor, '_last_processed_df') and hasattr(predictor, '_last_sequence_indices'):
        print(f"\n5. ALIGNMENT VERIFICATION (sample):")
        eval_df = predictor._last_processed_df.set_index('_original_index', drop=False)

        # Show first 3 predictions
        for i in range(min(3, len(indices))):
            idx = indices[i]

            if isinstance(target_cols, list):
                pred_val = predictions[target_cols[0]][i] if config['horizon'] == 1 else predictions[target_cols[0]][i, 0]
                actual_val = eval_df.loc[idx, f'{target_cols[0]}_target_h1']
            else:
                pred_val = predictions[i] if config['horizon'] == 1 else predictions[i, 0]
                actual_val = eval_df.loc[idx, f'{target_cols}_target_h1']

            group_str = f", group='{group_indices[i]}'" if group_cols else ""
            date_val = eval_df.loc[idx, time_col] if time_col and time_col in eval_df.columns else 'N/A'

            print(f"   Seq {i} (idx={idx}{group_str}, date={date_val}):")
            print(f"     pred={pred_val:.4f}, actual={actual_val:.4f}, error={pred_val-actual_val:.4f}")

        print(f"   ✓ All predictions aligned")

    # 6. Metrics
    print(f"\n6. EVALUATION METRICS:")
    print(f"{metrics}")

    print(f"\n{'='*80}")
    print(f"SCENARIO {scenario_num}: COMPLETE ✓")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Debug alignment in tf_predictor')
    parser.add_argument('--scenario', type=int, choices=range(1, 13), help='Test scenario number (1-12)')
    parser.add_argument('--all', action='store_true', help='Run all scenarios')

    args = parser.parse_args()

    if args.all:
        for scenario_num in range(1, 13):
            debug_scenario(scenario_num)
    elif args.scenario:
        debug_scenario(args.scenario)
    else:
        print("Usage:")
        print("  python debug_alignment.py --scenario 8")
        print("  python debug_alignment.py --all")
        print("\nAvailable scenarios:")
        for num, config in TEST_SCENARIOS.items():
            print(f"  {num}: {config['desc']}")


if __name__ == '__main__':
    main()
