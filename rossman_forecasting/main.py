#!/usr/bin/env python3
"""
Main CLI for Rossmann Store Sales Forecasting.

This application uses the TF Predictor framework for time series forecasting
specialized for the Rossmann Store Sales Kaggle competition.
"""
import argparse
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from rossman_forecasting.predictor import RossmannPredictor
from rossman_forecasting.preprocessing import load_and_preprocess_data, list_available_configs
from rossman_forecasting.utils import (
    download_rossmann_data,
    check_data_exists,
    save_kaggle_submission,
    rmspe
)
from tf_predictor.core.utils import split_time_series


def main():
    parser = argparse.ArgumentParser(
        description='Rossmann Store Sales Forecasting with TF Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data (first time only)
  python rossman_forecasting/main.py --download_data

  # Train with default settings
  python rossman_forecasting/main.py --epochs 50

  # Train with custom model and preprocessing
  python rossman_forecasting/main.py \\
    --preprocessing_config competition_enhanced \\
    --d_token 192 --n_layers 4 --n_heads 8 \\
    --sequence_length 21 --epochs 100

  # Quick test on few stores
  python rossman_forecasting/main.py --max_stores 10 --epochs 20

  # List available preprocessing configs
  python rossman_forecasting/main.py --list_configs
        """
    )

    # Data arguments
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('--download_data', action='store_true',
                           help='Download Rossmann data from Kaggle')
    data_group.add_argument('--data_dir', type=str, default='rossman_forecasting/data/raw',
                           help='Directory with raw data files')
    data_group.add_argument('--preprocessing_config', type=str, default='baseline',
                           help='Preprocessing configuration name (default: baseline)')
    data_group.add_argument('--use_cached', action='store_true', default=True,
                           help='Use cached preprocessed data if available')
    data_group.add_argument('--force_preprocess', action='store_true',
                           help='Force reprocessing even if cache exists')
    data_group.add_argument('--max_stores', type=int, default=None,
                           help='Limit to N stores for quick testing')
    data_group.add_argument('--list_configs', action='store_true',
                           help='List available preprocessing configs and exit')

    # Model architecture arguments
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--sequence_length', type=int, default=14,
                            help='Number of historical days to use (default: 14)')
    model_group.add_argument('--prediction_horizon', type=int, default=1,
                            help='Number of steps ahead to predict (default: 1)')
    model_group.add_argument('--model_type', type=str, default='ft_transformer',
                            choices=['ft_transformer', 'csn_transformer'],
                            help='Model architecture (default: ft_transformer)')
    model_group.add_argument('--pooling_type', type=str, default='multihead_attention',
                            choices=['cls', 'singlehead_attention', 'multihead_attention',
                                   'weighted_avg', 'temporal_multihead_attention'],
                            help='Pooling strategy (default: multihead_attention)')
    model_group.add_argument('--d_token', type=int, default=128,
                            help='Token embedding dimension (default: 128)')
    model_group.add_argument('--n_layers', type=int, default=3,
                            help='Number of transformer layers (default: 3)')
    model_group.add_argument('--n_heads', type=int, default=8,
                            help='Number of attention heads (default: 8)')
    model_group.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout rate (default: 0.1)')
    model_group.add_argument('--scaler_type', type=str, default='standard',
                            choices=['standard', 'minmax', 'robust', 'maxabs', 'onlymax'],
                            help='Scaler type (default: standard)')

    # Training arguments
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument('--epochs', type=int, default=50,
                            help='Number of training epochs (default: 50)')
    train_group.add_argument('--batch_size', type=int, default=64,
                            help='Training batch size (default: 64)')
    train_group.add_argument('--learning_rate', type=float, default=1e-3,
                            help='Learning rate (default: 0.001)')
    train_group.add_argument('--patience', type=int, default=15,
                            help='Early stopping patience (default: 15)')

    # Validation split arguments
    split_group = parser.add_argument_group('Data Split Options')
    split_group.add_argument('--val_ratio', type=float, default=0.2,
                            help='Validation set ratio from train.csv (default: 0.2)')

    # Output arguments
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--model_path', type=str,
                             default='rossman_forecasting/models/rossmann_model.pt',
                             help='Path to save trained model')
    output_group.add_argument('--export_predictions', action='store_true', default=True,
                             help='Export predictions to CSV (default: True)')
    output_group.add_argument('--create_submission', action='store_true', default=True,
                             help='Create Kaggle submission file (default: True)')
    output_group.add_argument('--no_save_model', action='store_true',
                             help='Do not save the trained model')

    args = parser.parse_args()

    # List configs if requested
    if args.list_configs:
        list_available_configs()
        return

    print("="*70)
    print("🏪 Rossmann Store Sales Forecasting with TF Predictor")
    print("="*70)

    # Download data if requested
    if args.download_data:
        print("\n📥 Downloading Rossmann data from Kaggle...")
        success = download_rossmann_data(data_dir=args.data_dir)
        if not success:
            print("❌ Data download failed. Please check error messages above.")
            return
        print()

    # Check if data exists
    if not check_data_exists(args.data_dir):
        print(f"\n❌ Data not found in {args.data_dir}")
        print("Please run with --download_data to download from Kaggle")
        return

    # Load and preprocess data
    print(f"\n📊 Loading and preprocessing data...")
    print(f"   Config: {args.preprocessing_config}")

    try:
        train_processed, test_processed = load_and_preprocess_data(
            config_name=args.preprocessing_config,
            use_cached=args.use_cached,
            force_preprocess=args.force_preprocess,
            data_dir=args.data_dir,
            verbose=True
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Limit stores if requested (for quick testing)
    if args.max_stores:
        print(f"\n   Limiting to {args.max_stores} stores for quick testing...")
        unique_stores = train_processed['Store'].unique()[:args.max_stores]
        train_processed = train_processed[train_processed['Store'].isin(unique_stores)]
        test_processed = test_processed[test_processed['Store'].isin(unique_stores)]
        print(f"   Train: {len(train_processed)} samples")
        print(f"   Test: {len(test_processed)} samples")

    print(f"\n   Train date range: {train_processed['Date'].min()} to {train_processed['Date'].max()}")
    print(f"   Test date range: {test_processed['Date'].min()} to {test_processed['Date'].max()}")
    print(f"   Number of stores: {train_processed['Store'].nunique()}")

    # Split train into train/val using tf_predictor utility
    print(f"\n🔄 Splitting data (temporal)...")
    print(f"   Validation ratio: {args.val_ratio}")

    # Calculate val_size as number of samples per store
    samples_per_store = train_processed.groupby('Store').size().min()
    val_size_per_store = int(samples_per_store * args.val_ratio)

    train_df, val_df, _ = split_time_series(
        train_processed,
        test_size=0,  # No test split from train.csv
        val_size=val_size_per_store,
        group_column='Store',
        time_column='Date',
        sequence_length=args.sequence_length
    )

    if train_df is None or val_df is None:
        print("   ❌ Insufficient data for splitting")
        return

    print(f"   Train samples: {len(train_df)}")
    print(f"   Validation samples: {len(val_df)}")
    print(f"   Test samples (Kaggle): {len(test_processed)}")

    # Initialize Model
    print(f"\n🧠 Initializing Rossmann Predictor...")

    model = RossmannPredictor(
        target_column='Sales',
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        group_columns='Store',
        model_type=args.model_type,
        pooling_type=args.pooling_type,
        d_token=args.d_token,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        scaler_type=args.scaler_type,
        verbose=True
    )

    print(f"\n   Model Configuration:")
    print(f"   - Model: {args.model_type}")
    print(f"   - Pooling: {args.pooling_type}")
    print(f"   - Sequence length: {args.sequence_length}")
    print(f"   - Prediction horizon: {args.prediction_horizon}")
    print(f"   - Token dimension: {args.d_token}")
    print(f"   - Layers: {args.n_layers}")
    print(f"   - Attention heads: {args.n_heads}")
    print(f"   - Dropout: {args.dropout}")
    print(f"   - Scaler: {args.scaler_type}")
    print(f"   - Per-store scaling: enabled")

    # Train Model
    print(f"\n🏋️  Training model...")

    model.fit(
        df=train_df,
        val_df=val_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        verbose=True
    )

    # Save Model
    if not args.no_save_model:
        print(f"\n💾 Saving model...")
        model_path = Path(args.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        print(f"   ✅ Model saved to: {model_path}")

    # Evaluate Model
    print(f"\n📈 Evaluating model...")

    # Train metrics
    train_metrics = model.evaluate(
        train_df,
        export_csv='rossman_forecasting/data/predictions/train_predictions.csv' if args.export_predictions else None
    )

    # Val metrics
    val_metrics = model.evaluate(
        val_df,
        export_csv='rossman_forecasting/data/predictions/val_predictions.csv' if args.export_predictions else None
    )

    # Print metrics
    print(f"\n   Training Metrics:")
    for metric, value in train_metrics.items():
        if isinstance(value, (int, float)):
            print(f"   - {metric}: {value:.4f}")

    print(f"\n   Validation Metrics:")
    for metric, value in val_metrics.items():
        if isinstance(value, (int, float)):
            print(f"   - {metric}: {value:.4f}")

    # Predict on test set (Kaggle submission)
    if args.create_submission:
        print(f"\n🔮 Generating Kaggle submission...")

        # Use inference mode (no Sales column in test.csv)
        test_predictions = model.predict(test_processed, inference_mode=True)

        # Create submission file
        if 'Id' in test_processed.columns:
            submission_path = save_kaggle_submission(
                test_processed['Id'].values,
                test_predictions,
                'rossman_forecasting/data/predictions/submission.csv'
            )
            print(f"   ✅ Submission file: {submission_path}")
        else:
            print("   ⚠️  Warning: 'Id' column not found in test data")

    # Summary
    print(f"\n🎉 Rossmann forecasting completed successfully!")
    print(f"   Validation RMSPE: {val_metrics.get('RMSPE', 'N/A'):.4f}")
    if not args.no_save_model:
        print(f"   Model saved to: {args.model_path}")
    if args.export_predictions:
        print(f"   Predictions saved to: rossman_forecasting/data/predictions/")
    if args.create_submission:
        print(f"   Kaggle submission: rossman_forecasting/data/predictions/submission.csv")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
