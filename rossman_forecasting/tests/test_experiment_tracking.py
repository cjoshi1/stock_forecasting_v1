"""
Test experiment tracking utilities.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from rossman_forecasting.experiments.experiment_tracking import (
    generate_experiment_id,
    load_experiment_config,
    create_experiment_directory,
    save_experiment_config,
    save_predictions_with_errors,
    save_detailed_metrics,
    log_to_experiment_results,
    generate_experiment_summary
)


def test_generate_experiment_id():
    """Test experiment ID generation."""
    # Use temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # First ID should be exp_001
        exp_id = generate_experiment_id(base_dir=tmp_dir)
        assert exp_id == 'exp_001'

        # Create that directory
        (Path(tmp_dir) / 'exp_001').mkdir()

        # Next should be exp_002
        exp_id = generate_experiment_id(base_dir=tmp_dir)
        assert exp_id == 'exp_002'


def test_load_experiment_config():
    """Test loading experiment config."""
    # Load an existing config
    config = load_experiment_config('quick_test')

    assert 'experiment' in config
    assert 'preprocessing' in config
    assert 'model' in config
    assert 'training' in config
    assert 'output' in config

    assert config['experiment']['name'] == 'quick_test'


def test_create_experiment_directory():
    """Test experiment directory creation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        exp_dir = create_experiment_directory(
            experiment_id='exp_001',
            experiment_name='test_exp',
            base_dir=tmp_dir
        )

        assert exp_dir.exists()
        assert exp_dir.is_dir()
        assert 'exp_001_test_exp' in str(exp_dir)


def test_save_predictions_with_errors():
    """Test saving predictions with error analysis."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create sample data
        n_samples = 100
        store_ids = np.random.randint(1, 10, n_samples)
        dates = pd.date_range('2015-01-01', periods=n_samples)
        actuals = np.random.randint(1000, 8000, n_samples).astype(float)
        predictions = actuals + np.random.randn(n_samples) * 500

        output_path = Path(tmp_dir) / 'predictions.csv'

        # Save with actuals
        df = save_predictions_with_errors(
            store_ids=store_ids,
            dates=dates,
            actuals=actuals,
            predictions=predictions,
            output_path=output_path,
            dataset_name='val'
        )

        # Check file was created
        assert output_path.exists()

        # Check columns
        assert 'Store' in df.columns
        assert 'Date' in df.columns
        assert 'Actual_Sales' in df.columns
        assert 'Predicted_Sales' in df.columns
        assert 'Error' in df.columns
        assert 'Abs_Error' in df.columns
        assert 'Pct_Error' in df.columns
        assert 'RMSPE_Contribution' in df.columns

        # Test without actuals (test set)
        output_path_test = Path(tmp_dir) / 'test_predictions.csv'
        df_test = save_predictions_with_errors(
            store_ids=store_ids,
            dates=dates,
            actuals=None,
            predictions=predictions,
            output_path=output_path_test,
            dataset_name='test'
        )

        assert output_path_test.exists()
        assert 'Actual_Sales' not in df_test.columns
        assert 'Predicted_Sales' in df_test.columns


def test_save_detailed_metrics():
    """Test saving detailed metrics to JSON."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        exp_dir = Path(tmp_dir)

        config = {
            '_source_file': 'test_config.yaml',
            'model': {'d_token': 128, 'n_layers': 3},
            'preprocessing': {'config': 'baseline'}
        }

        train_metrics = {'MAE': 500.0, 'RMSE': 700.0, 'RMSPE': 0.15}
        val_metrics = {'MAE': 550.0, 'RMSE': 750.0, 'RMSPE': 0.17}

        training_info = {
            'training_time_mins': 10.5,
            'best_epoch': 25,
            'total_epochs': 50
        }

        data_info = {
            'train_samples': 10000,
            'val_samples': 2000,
            'test_samples': 5000
        }

        save_detailed_metrics(
            experiment_id='exp_001',
            experiment_name='test_exp',
            config=config,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            training_info=training_info,
            data_info=data_info,
            exp_dir=exp_dir
        )

        metrics_path = exp_dir / 'metrics.json'
        assert metrics_path.exists()

        # Load and verify
        import json
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)

        assert metrics_data['experiment_id'] == 'exp_001'
        assert metrics_data['train_metrics']['MAE'] == 500.0
        assert metrics_data['val_metrics']['RMSPE'] == 0.17


def test_generate_experiment_summary():
    """Test generating experiment summary markdown."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        exp_dir = Path(tmp_dir)

        config = {
            '_source_file': 'test_config.yaml',
            'model': {
                'model_type': 'ft_transformer',
                'd_token': 128,
                'n_layers': 3,
                'n_heads': 8,
                'pooling_type': 'multihead_attention',
                'sequence_length': 14,
                'prediction_horizon': 1
            },
            'training': {
                'epochs': 50,
                'batch_size': 64,
                'learning_rate': 0.001
            },
            'preprocessing': {
                'config': 'baseline',
                'version': 'v1'
            }
        }

        train_metrics = {'MAE': 500.0, 'RMSE': 700.0, 'RMSPE': 0.15}
        val_metrics = {'MAE': 550.0, 'RMSE': 750.0, 'RMSPE': 0.17, 'R2': 0.85}

        training_info = {
            'training_time_mins': 10.5,
            'best_epoch': 25,
            'total_epochs': 50
        }

        data_info = {
            'train_samples': 10000,
            'val_samples': 2000,
            'test_samples': 5000,
            'num_stores': 100
        }

        generate_experiment_summary(
            experiment_id='exp_001',
            experiment_name='test_exp',
            config=config,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            training_info=training_info,
            data_info=data_info,
            exp_dir=exp_dir
        )

        summary_path = exp_dir / 'SUMMARY.md'
        assert summary_path.exists()

        # Read and check content
        with open(summary_path, 'r') as f:
            content = f.read()

        assert 'exp_001' in content
        assert 'test_exp' in content
        assert '0.17' in content  # RMSPE
        assert 'ft_transformer' in content


def test_log_to_experiment_results():
    """Test logging to master CSV."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / 'experiment_results.csv'

        config = {
            '_source_file': 'test_config.yaml',
            'model': {
                'model_type': 'ft_transformer',
                'd_token': 128,
                'n_layers': 3,
                'n_heads': 8,
                'pooling_type': 'multihead_attention',
                'sequence_length': 14,
                'prediction_horizon': 1,
                'scaler_type': 'standard'
            },
            'training': {
                'epochs': 50,
                'batch_size': 64,
                'learning_rate': 0.001
            },
            'preprocessing': {
                'config': 'baseline',
                'version': 'v1'
            }
        }

        train_metrics = {'MAE': 500.0, 'RMSE': 700.0, 'RMSPE': 0.15}
        val_metrics = {'MAE': 550.0, 'RMSE': 750.0, 'RMSPE': 0.17}

        training_info = {
            'training_time_mins': 10.5,
            'best_epoch': 25
        }

        data_info = {
            'train_samples': 10000,
            'val_samples': 2000,
            'test_samples': 5000
        }

        file_paths = {
            'train_predictions': 'train.csv',
            'val_predictions': 'val.csv',
            'test_predictions': 'test.csv',
            'model': 'model.pt'
        }

        # Log first experiment
        log_to_experiment_results(
            experiment_id='exp_001',
            experiment_name='test_exp_1',
            config=config,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            training_info=training_info,
            data_info=data_info,
            file_paths=file_paths,
            notes='Test experiment 1',
            csv_path=str(csv_path)
        )

        assert csv_path.exists()

        # Read and verify
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df.loc[0, 'experiment_id'] == 'exp_001'
        assert df.loc[0, 'val_rmspe'] == 0.17

        # Log second experiment
        log_to_experiment_results(
            experiment_id='exp_002',
            experiment_name='test_exp_2',
            config=config,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            training_info=training_info,
            data_info=data_info,
            file_paths=file_paths,
            notes='Test experiment 2',
            csv_path=str(csv_path)
        )

        # Read and verify both experiments
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert df.loc[1, 'experiment_id'] == 'exp_002'


if __name__ == '__main__':
    # Run tests
    print("Running experiment tracking tests...")
    print("-" * 60)

    test_generate_experiment_id()
    print("✅ test_generate_experiment_id passed")

    test_load_experiment_config()
    print("✅ test_load_experiment_config passed")

    test_create_experiment_directory()
    print("✅ test_create_experiment_directory passed")

    test_save_predictions_with_errors()
    print("✅ test_save_predictions_with_errors passed")

    test_save_detailed_metrics()
    print("✅ test_save_detailed_metrics passed")

    test_generate_experiment_summary()
    print("✅ test_generate_experiment_summary passed")

    test_log_to_experiment_results()
    print("✅ test_log_to_experiment_results passed")

    print("-" * 60)
    print("✅ All experiment tracking tests passed!")
