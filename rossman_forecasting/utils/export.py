"""
Export utilities for predictions and Kaggle submissions.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Union


def save_kaggle_submission(
    test_ids: np.ndarray,
    predictions: np.ndarray,
    output_path: str = 'rossman_forecasting/data/predictions/submission.csv'
) -> str:
    """
    Create Kaggle submission file in the required format.

    Kaggle format:
    Id,Sales
    1,5263
    2,6064
    ...

    Args:
        test_ids: Test set IDs from test.csv
        predictions: Predicted sales values
        output_path: Path to save submission file

    Returns:
        Path to saved submission file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create submission dataframe
    submission_df = pd.DataFrame({
        'Id': test_ids.astype(int),
        'Sales': predictions.flatten()
    })

    # Ensure Sales are non-negative (business constraint)
    submission_df['Sales'] = submission_df['Sales'].clip(lower=0)

    # Save to CSV
    submission_df.to_csv(output_file, index=False)

    print(f"✅ Kaggle submission saved to: {output_file}")
    print(f"   Total predictions: {len(submission_df)}")
    print(f"   Sales range: {submission_df['Sales'].min():.2f} - {submission_df['Sales'].max():.2f}")

    return str(output_file)


def save_predictions_with_actuals(
    dates: np.ndarray,
    store_ids: np.ndarray,
    actuals: np.ndarray,
    predictions: np.ndarray,
    dataset_split: str,
    output_path: str,
    metrics: Optional[Dict] = None
) -> str:
    """
    Save predictions alongside actuals for analysis.

    Format:
    Store,Date,Actual_Sales,Predicted_Sales,Error,Abs_Error,Pct_Error,Dataset

    Args:
        dates: Date values
        store_ids: Store IDs
        actuals: Actual sales values
        predictions: Predicted sales values
        dataset_split: 'train', 'val', or 'test'
        output_path: Path to save file
        metrics: Optional metrics dictionary to include in header

    Returns:
        Path to saved file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Calculate errors
    errors = actuals - predictions
    abs_errors = np.abs(errors)
    pct_errors = np.where(actuals != 0, (errors / actuals) * 100, 0)

    # Create dataframe
    df = pd.DataFrame({
        'Store': store_ids,
        'Date': dates,
        'Actual_Sales': actuals,
        'Predicted_Sales': predictions.flatten(),
        'Error': errors,
        'Abs_Error': abs_errors,
        'Pct_Error': pct_errors,
        'Dataset': dataset_split
    })

    # Save with metrics header if provided
    with open(output_file, 'w') as f:
        if metrics:
            f.write(f"# Dataset: {dataset_split}\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Metrics:\n")
            for metric_name, value in metrics.items():
                f.write(f"#   {metric_name}: {value:.4f}\n")
            f.write("#\n")

        df.to_csv(f, index=False)

    print(f"✅ Predictions saved to: {output_file}")
    print(f"   Records: {len(df)}")
    print(f"   Dataset: {dataset_split}")

    return str(output_file)


def combine_predictions(
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None,
    output_path: str = 'rossman_forecasting/data/predictions/all_predictions.csv'
) -> str:
    """
    Combine train, val, and optionally test predictions into single file.

    Args:
        train_path: Path to train predictions CSV
        val_path: Path to val predictions CSV
        test_path: Optional path to test predictions CSV
        output_path: Path to save combined file

    Returns:
        Path to combined file
    """
    dfs = []

    for path, split_name in [(train_path, 'train'), (val_path, 'val'), (test_path, 'test')]:
        if path and Path(path).exists():
            df = pd.read_csv(path, comment='#')
            if 'Dataset' not in df.columns:
                df['Dataset'] = split_name
            dfs.append(df)

    if not dfs:
        raise ValueError("No prediction files found")

    combined_df = pd.concat(dfs, ignore_index=True)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    combined_df.to_csv(output_file, index=False)

    print(f"✅ Combined predictions saved to: {output_file}")
    print(f"   Total records: {len(combined_df)}")
    print(f"   Breakdown:")
    for split in combined_df['Dataset'].unique():
        count = (combined_df['Dataset'] == split).sum()
        print(f"   - {split}: {count}")

    return str(output_file)


if __name__ == '__main__':
    # Test the export functions
    test_ids = np.arange(1, 101)
    predictions = np.random.uniform(1000, 5000, 100)

    print("Test Kaggle Submission Export:")
    print("-" * 60)
    save_kaggle_submission(
        test_ids,
        predictions,
        'rossman_forecasting/data/predictions/test_submission.csv'
    )
