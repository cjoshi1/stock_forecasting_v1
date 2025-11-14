#!/usr/bin/env python3
"""
Compare and visualize experiment results.

Provides utilities to compare different experiment runs, show best results,
and analyze the experiment tracking CSV.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import pandas as pd
from pathlib import Path
from typing import Optional


def load_experiment_results(csv_path: str = 'rossman_forecasting/experiments/experiment_results.csv') -> pd.DataFrame:
    """
    Load experiment results CSV.

    Args:
        csv_path: Path to experiment results CSV

    Returns:
        DataFrame with all experiments
    """
    csv_file = Path(csv_path)

    if not csv_file.exists():
        print(f"❌ No experiment results found at: {csv_path}")
        print("Run experiments first with: python rossman_forecasting/main.py --experiment_config <config>")
        return pd.DataFrame()

    df = pd.read_csv(csv_file)
    return df


def show_all_experiments(df: pd.DataFrame, sort_by: str = 'val_rmspe') -> None:
    """
    Display all experiments in a readable format.

    Args:
        df: Experiment results DataFrame
        sort_by: Column to sort by
    """
    if df.empty:
        return

    # Sort by specified metric (ascending for error metrics)
    if sort_by in df.columns:
        df_sorted = df.sort_values(sort_by, ascending=True)
    else:
        df_sorted = df

    print("\n" + "="*100)
    print("📊 All Experiments")
    print("="*100)

    # Display key columns
    display_cols = [
        'experiment_id',
        'experiment_name',
        'timestamp',
        'val_rmspe',
        'val_mae',
        'val_rmse',
        'train_rmspe',
        'epochs',
        'd_token',
        'n_layers',
        'sequence_length',
        'preprocessing_version'
    ]

    # Filter to existing columns
    display_cols = [col for col in display_cols if col in df_sorted.columns]

    # Format the dataframe for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)

    print(df_sorted[display_cols].to_string(index=False))
    print("\n" + "="*100)


def show_best_experiments(df: pd.DataFrame, n: int = 5, metric: str = 'val_rmspe') -> None:
    """
    Show top N experiments by specified metric.

    Args:
        df: Experiment results DataFrame
        n: Number of top experiments to show
        metric: Metric to rank by (default: val_rmspe)
    """
    if df.empty:
        return

    if metric not in df.columns:
        print(f"❌ Metric '{metric}' not found in results")
        return

    # Sort by metric (ascending for error metrics)
    df_sorted = df.sort_values(metric, ascending=True).head(n)

    print("\n" + "="*100)
    print(f"🏆 Top {n} Experiments by {metric}")
    print("="*100)

    for idx, row in df_sorted.iterrows():
        rank = df_sorted.index.get_loc(idx) + 1
        print(f"\n{rank}. {row['experiment_name']} ({row['experiment_id']})")
        print(f"   {metric}: {row[metric]:.4f}")
        print(f"   Timestamp: {row['timestamp']}")
        print(f"   Config: {row['config_file']}")
        print(f"   Model: d_token={row.get('d_token', 'N/A')}, "
              f"n_layers={row.get('n_layers', 'N/A')}, "
              f"seq_len={row.get('sequence_length', 'N/A')}")
        print(f"   Training: epochs={row.get('epochs', 'N/A')}, "
              f"batch_size={row.get('batch_size', 'N/A')}, "
              f"lr={row.get('learning_rate', 'N/A')}")
        if 'notes' in row and pd.notna(row['notes']) and row['notes']:
            print(f"   Notes: {row['notes']}")

    print("\n" + "="*100)


def compare_experiments(df: pd.DataFrame, experiment_ids: list) -> None:
    """
    Compare specific experiments side-by-side.

    Args:
        df: Experiment results DataFrame
        experiment_ids: List of experiment IDs to compare
    """
    if df.empty:
        return

    # Filter to requested experiments
    df_compare = df[df['experiment_id'].isin(experiment_ids)]

    if df_compare.empty:
        print(f"❌ No experiments found with IDs: {experiment_ids}")
        return

    print("\n" + "="*100)
    print(f"🔍 Comparing Experiments: {', '.join(experiment_ids)}")
    print("="*100)

    # Key metrics to compare
    metrics = ['val_rmspe', 'val_mae', 'val_rmse', 'train_rmspe']
    config_params = ['d_token', 'n_layers', 'n_heads', 'sequence_length',
                    'epochs', 'batch_size', 'learning_rate', 'preprocessing_version']

    for idx, row in df_compare.iterrows():
        print(f"\n{row['experiment_id']} - {row['experiment_name']}")
        print(f"   Timestamp: {row['timestamp']}")
        print(f"   Config File: {row['config_file']}")

        print(f"\n   Metrics:")
        for metric in metrics:
            if metric in row and pd.notna(row[metric]):
                print(f"      {metric}: {row[metric]:.4f}")

        print(f"\n   Configuration:")
        for param in config_params:
            if param in row and pd.notna(row[param]):
                print(f"      {param}: {row[param]}")

        print(f"\n   Training:")
        if 'training_time_mins' in row and pd.notna(row['training_time_mins']):
            print(f"      Training time: {row['training_time_mins']:.1f} minutes")
        if 'best_epoch' in row and pd.notna(row['best_epoch']):
            print(f"      Best epoch: {row['best_epoch']}")

        print(f"\n   Data:")
        if 'train_samples' in row and pd.notna(row['train_samples']):
            print(f"      Train samples: {int(row['train_samples']):,}")
        if 'val_samples' in row and pd.notna(row['val_samples']):
            print(f"      Val samples: {int(row['val_samples']):,}")

    print("\n" + "="*100)


def show_summary_stats(df: pd.DataFrame) -> None:
    """
    Show summary statistics across all experiments.

    Args:
        df: Experiment results DataFrame
    """
    if df.empty:
        return

    print("\n" + "="*100)
    print("📈 Summary Statistics")
    print("="*100)

    # Count experiments
    print(f"\nTotal experiments: {len(df)}")

    # Best RMSPE
    if 'val_rmspe' in df.columns:
        best_rmspe = df['val_rmspe'].min()
        best_exp = df.loc[df['val_rmspe'].idxmin()]
        print(f"\nBest validation RMSPE: {best_rmspe:.4f}")
        print(f"   Experiment: {best_exp['experiment_id']} - {best_exp['experiment_name']}")
        print(f"   Config: {best_exp['config_file']}")

    # Average metrics
    metric_cols = ['val_rmspe', 'val_mae', 'val_rmse', 'train_rmspe']
    print("\nAverage Metrics:")
    for col in metric_cols:
        if col in df.columns:
            avg = df[col].mean()
            std = df[col].std()
            print(f"   {col}: {avg:.4f} ± {std:.4f}")

    # Training time stats
    if 'training_time_mins' in df.columns:
        avg_time = df['training_time_mins'].mean()
        total_time = df['training_time_mins'].sum()
        print(f"\nTraining Time:")
        print(f"   Average: {avg_time:.1f} minutes")
        print(f"   Total: {total_time:.1f} minutes ({total_time/60:.1f} hours)")

    # Most common configs
    if 'preprocessing_version' in df.columns:
        print(f"\nPreprocessing Versions:")
        for version, count in df['preprocessing_version'].value_counts().head(5).items():
            print(f"   {version}: {count} experiments")

    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(
        description='Compare and analyze Rossmann forecasting experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show all experiments
  python rossman_forecasting/experiments/compare.py --show_all

  # Show top 10 experiments by RMSPE
  python rossman_forecasting/experiments/compare.py --best 10

  # Compare specific experiments
  python rossman_forecasting/experiments/compare.py --compare exp_001 exp_002 exp_003

  # Show summary statistics
  python rossman_forecasting/experiments/compare.py --summary
        """
    )

    parser.add_argument('--csv_path', type=str,
                       default='rossman_forecasting/experiments/experiment_results.csv',
                       help='Path to experiment results CSV')
    parser.add_argument('--show_all', action='store_true',
                       help='Show all experiments')
    parser.add_argument('--best', type=int, metavar='N',
                       help='Show top N experiments by validation RMSPE')
    parser.add_argument('--metric', type=str, default='val_rmspe',
                       help='Metric to rank by (default: val_rmspe)')
    parser.add_argument('--compare', nargs='+', metavar='ID',
                       help='Compare specific experiments by ID')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary statistics')
    parser.add_argument('--sort_by', type=str, default='val_rmspe',
                       help='Column to sort by in --show_all (default: val_rmspe)')

    args = parser.parse_args()

    # Load results
    df = load_experiment_results(args.csv_path)

    if df.empty:
        return

    # Execute requested action
    if args.show_all:
        show_all_experiments(df, sort_by=args.sort_by)
    elif args.best:
        show_best_experiments(df, n=args.best, metric=args.metric)
    elif args.compare:
        compare_experiments(df, args.compare)
    elif args.summary:
        show_summary_stats(df)
    else:
        # Default: show summary and top 5
        show_summary_stats(df)
        show_best_experiments(df, n=5, metric=args.metric)


if __name__ == '__main__':
    main()
