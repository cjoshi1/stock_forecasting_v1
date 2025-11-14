"""
Experiment tracking utilities for Rossmann forecasting.

Handles experiment configuration, tracking, and result logging.
"""
import os
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple


def generate_experiment_id(base_dir: str = 'rossman_forecasting/experiments/runs') -> str:
    """
    Generate next experiment ID (exp_001, exp_002, etc.).

    Args:
        base_dir: Directory containing experiment runs

    Returns:
        Next experiment ID string
    """
    runs_dir = Path(base_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Find existing experiment directories
    existing_runs = [d.name for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('exp_')]

    if not existing_runs:
        return 'exp_001'

    # Extract numbers and find max
    numbers = []
    for run in existing_runs:
        try:
            num = int(run.split('_')[1])
            numbers.append(num)
        except (IndexError, ValueError):
            continue

    next_num = max(numbers) + 1 if numbers else 1
    return f'exp_{next_num:03d}'


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """
    Load complete experiment configuration from YAML.

    Args:
        config_path: Path to experiment config file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)

    # Try different locations
    search_paths = [
        config_file,
        Path('rossman_forecasting/experiments/configs') / config_file.name,
        Path('rossman_forecasting/experiments/configs') / f'{config_path}.yaml',
    ]

    for path in search_paths:
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(f"Experiment config not found: {config_path}")


def create_experiment_directory(
    experiment_id: str,
    experiment_name: str,
    base_dir: str = 'rossman_forecasting/experiments/runs'
) -> Path:
    """
    Create directory for experiment outputs.

    Args:
        experiment_id: Experiment ID (exp_001)
        experiment_name: Experiment name
        base_dir: Base directory for runs

    Returns:
        Path to created directory
    """
    exp_dir = Path(base_dir) / f'{experiment_id}_{experiment_name}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_experiment_config(config: Dict[str, Any], exp_dir: Path) -> None:
    """
    Save copy of config to experiment directory for reproducibility.

    Args:
        config: Configuration dictionary
        exp_dir: Experiment directory
    """
    config_path = exp_dir / 'config_used.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_predictions_with_errors(
    store_ids: np.ndarray,
    dates: np.ndarray,
    actuals: Optional[np.ndarray],
    predictions: np.ndarray,
    output_path: Path,
    dataset_name: str = 'val'
) -> pd.DataFrame:
    """
    Save predictions with detailed error analysis.

    Args:
        store_ids: Store IDs
        dates: Date values
        actuals: Actual sales (None for test set)
        predictions: Predicted sales
        output_path: Path to save CSV
        dataset_name: 'train', 'val', or 'test'

    Returns:
        DataFrame with predictions and errors
    """
    predictions_flat = predictions.flatten() if isinstance(predictions, np.ndarray) else predictions

    if actuals is not None:
        # Train/Val with actuals
        actuals_flat = actuals.flatten() if isinstance(actuals, np.ndarray) else actuals

        errors = actuals_flat - predictions_flat
        abs_errors = np.abs(errors)
        pct_errors = np.where(actuals_flat != 0, (errors / actuals_flat) * 100, 0)
        rmspe_contrib = np.where(actuals_flat != 0, ((errors / actuals_flat) ** 2), 0)

        df = pd.DataFrame({
            'Store': store_ids,
            'Date': dates,
            'Actual_Sales': actuals_flat,
            'Predicted_Sales': predictions_flat,
            'Error': errors,
            'Abs_Error': abs_errors,
            'Pct_Error': pct_errors,
            'RMSPE_Contribution': rmspe_contrib
        })
    else:
        # Test without actuals
        df = pd.DataFrame({
            'Store': store_ids,
            'Date': dates,
            'Predicted_Sales': predictions_flat
        })

    df.to_csv(output_path, index=False)
    return df


def save_detailed_metrics(
    experiment_id: str,
    experiment_name: str,
    config: Dict[str, Any],
    train_metrics: Dict,
    val_metrics: Dict,
    training_info: Dict,
    data_info: Dict,
    exp_dir: Path
) -> None:
    """
    Save detailed metrics to JSON file.

    Args:
        experiment_id: Experiment ID
        experiment_name: Experiment name
        config: Experiment configuration
        train_metrics: Training metrics
        val_metrics: Validation metrics
        training_info: Training information
        data_info: Data information
        exp_dir: Experiment directory
    """
    metrics_data = {
        'experiment_id': experiment_id,
        'experiment_name': experiment_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config_file': config.get('_source_file', 'unknown'),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'training_info': training_info,
        'data_info': data_info,
        'model_config': config.get('model', {}),
        'preprocessing_config': config.get('preprocessing', {})
    }

    metrics_path = exp_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2, default=str)


def log_to_experiment_results(
    experiment_id: str,
    experiment_name: str,
    config: Dict[str, Any],
    train_metrics: Dict,
    val_metrics: Dict,
    training_info: Dict,
    data_info: Dict,
    file_paths: Dict[str, str],
    notes: str = '',
    csv_path: str = 'rossman_forecasting/experiments/experiment_results.csv'
) -> None:
    """
    Append experiment results to master CSV file.

    Args:
        experiment_id: Experiment ID
        experiment_name: Experiment name
        config: Configuration
        train_metrics: Training metrics
        val_metrics: Validation metrics
        training_info: Training info
        data_info: Data info
        file_paths: Paths to output files
        notes: Optional notes
        csv_path: Path to master CSV
    """
    # Extract metrics (handle nested dicts from multi-target/multi-horizon)
    def get_metric(metrics_dict, key, default=None):
        if key in metrics_dict:
            return metrics_dict[key]
        elif 'overall' in metrics_dict:
            return metrics_dict['overall'].get(key, default)
        return default

    model_cfg = config.get('model', {})
    training_cfg = config.get('training', {})
    preproc_cfg = config.get('preprocessing', {})

    row = {
        'experiment_id': experiment_id,
        'experiment_name': experiment_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config_file': config.get('_source_file', 'unknown'),
        'preprocessing_version': preproc_cfg.get('config', 'unknown'),

        # Model config
        'model_type': model_cfg.get('model_type', 'unknown'),
        'd_token': model_cfg.get('d_token', 0),
        'n_layers': model_cfg.get('n_layers', 0),
        'n_heads': model_cfg.get('n_heads', 0),
        'pooling_type': model_cfg.get('pooling_type', 'unknown'),
        'sequence_length': model_cfg.get('sequence_length', 0),
        'prediction_horizon': model_cfg.get('prediction_horizon', 1),
        'scaler_type': model_cfg.get('scaler_type', 'standard'),

        # Training config
        'epochs': training_cfg.get('epochs', 0),
        'batch_size': training_cfg.get('batch_size', 0),
        'learning_rate': training_cfg.get('learning_rate', 0),

        # Data info
        'train_samples': data_info.get('train_samples', 0),
        'val_samples': data_info.get('val_samples', 0),
        'test_samples': data_info.get('test_samples', 0),

        # Train metrics
        'train_mae': get_metric(train_metrics, 'MAE', 0),
        'train_rmse': get_metric(train_metrics, 'RMSE', 0),
        'train_rmspe': get_metric(train_metrics, 'RMSPE', 0),

        # Val metrics
        'val_mae': get_metric(val_metrics, 'MAE', 0),
        'val_rmse': get_metric(val_metrics, 'RMSE', 0),
        'val_rmspe': get_metric(val_metrics, 'RMSPE', 0),

        # Training info
        'training_time_mins': training_info.get('training_time_mins', 0),
        'best_epoch': training_info.get('best_epoch', 0),

        # File paths
        'train_predictions_file': file_paths.get('train_predictions', ''),
        'val_predictions_file': file_paths.get('val_predictions', ''),
        'test_predictions_file': file_paths.get('test_predictions', ''),
        'model_path': file_paths.get('model', ''),

        'notes': notes
    }

    # Create or append to CSV
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([row])
    if csv_file.exists():
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)


def generate_experiment_summary(
    experiment_id: str,
    experiment_name: str,
    config: Dict[str, Any],
    train_metrics: Dict,
    val_metrics: Dict,
    training_info: Dict,
    data_info: Dict,
    exp_dir: Path
) -> None:
    """
    Generate markdown summary for experiment.

    Args:
        experiment_id: Experiment ID
        experiment_name: Experiment name
        config: Configuration
        train_metrics: Training metrics
        val_metrics: Validation metrics
        training_info: Training info
        data_info: Data info
        exp_dir: Experiment directory
    """
    def get_metric(metrics_dict, key, default='N/A'):
        if key in metrics_dict:
            val = metrics_dict[key]
        elif 'overall' in metrics_dict:
            val = metrics_dict['overall'].get(key, default)
        else:
            val = default

        if isinstance(val, (int, float)):
            return f"{val:.4f}"
        return str(val)

    model_cfg = config.get('model', {})
    training_cfg = config.get('training', {})
    preproc_cfg = config.get('preprocessing', {})

    summary = f"""# Experiment: {experiment_name}

**ID:** {experiment_id}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Config:** {config.get('_source_file', 'unknown')}

## Results

### Validation Metrics
- **RMSPE:** {get_metric(val_metrics, 'RMSPE')} ⭐ (Kaggle metric)
- **MAE:** {get_metric(val_metrics, 'MAE')}
- **RMSE:** {get_metric(val_metrics, 'RMSE')}
- **R²:** {get_metric(val_metrics, 'R2')}

### Training Metrics
- **RMSPE:** {get_metric(train_metrics, 'RMSPE')}
- **MAE:** {get_metric(train_metrics, 'MAE')}
- **RMSE:** {get_metric(train_metrics, 'RMSE')}

## Configuration

### Preprocessing
- Config: {preproc_cfg.get('config', 'unknown')}
- Version: {preproc_cfg.get('version', 'unknown')}

### Model
- Architecture: {model_cfg.get('model_type', 'unknown')}
- d_token: {model_cfg.get('d_token', 'N/A')}
- Layers: {model_cfg.get('n_layers', 'N/A')}
- Heads: {model_cfg.get('n_heads', 'N/A')}
- Pooling: {model_cfg.get('pooling_type', 'N/A')}
- Sequence: {model_cfg.get('sequence_length', 'N/A')} days
- Horizon: {model_cfg.get('prediction_horizon', 1)} step(s)

### Training
- Epochs: {training_cfg.get('epochs', 'N/A')} (best: {training_info.get('best_epoch', 'N/A')})
- Batch size: {training_cfg.get('batch_size', 'N/A')}
- Learning rate: {training_cfg.get('learning_rate', 'N/A')}
- Time: {training_info.get('training_time_mins', 'N/A'):.1f} minutes

## Data
- Train samples: {data_info.get('train_samples', 'N/A'):,}
- Val samples: {data_info.get('val_samples', 'N/A'):,}
- Test samples: {data_info.get('test_samples', 'N/A'):,}
- Stores: {data_info.get('num_stores', 'N/A')}

## Files
- Model: `model.pt`
- Train predictions: `train_predictions.csv`
- Val predictions: `val_predictions.csv`
- Test predictions: `test_predictions.csv`
- Kaggle submission: `submission.csv`
- Detailed metrics: `metrics.json`

## Notes
Ready for evaluation. Consider submitting to Kaggle if RMSPE is competitive.
"""

    summary_path = exp_dir / 'SUMMARY.md'
    with open(summary_path, 'w') as f:
        f.write(summary)
