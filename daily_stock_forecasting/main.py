#!/usr/bin/env python3
"""
Main script for stock forecasting using FT-Transformer.

This script demonstrates how to use the FT-Transformer library specifically for stock prediction.
It includes stock data loading, model training, evaluation, and visualization.
"""

import argparse
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Handle both direct execution and module import
if __name__ == "__main__" and __package__ is None:
    # Direct execution - add parent directory to path
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from daily_stock_forecasting.predictor import StockPredictor
    from daily_stock_forecasting.preprocessing.market_data import load_stock_data, create_sample_stock_data
    from daily_stock_forecasting.visualization.stock_charts import create_comprehensive_plots, print_performance_summary
    from tf_predictor.core.utils import split_time_series
else:
    # Module import - use relative imports
    from .predictor import StockPredictor
    from .preprocessing.market_data import load_stock_data, create_sample_stock_data
    from .visualization.stock_charts import create_comprehensive_plots, print_performance_summary
    from tf_predictor.core.utils import split_time_series


def main():
    parser = argparse.ArgumentParser(description='Train FT-Transformer for stock prediction')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to stock data CSV file')
    parser.add_argument('--target', type=str, default='close',
                       help='Target column(s) to predict. Single: "close" or Multiple: "close,volume"')
    parser.add_argument('--use_sample_data', action='store_true',
                       help='Use synthetic sample data instead of real data')
    parser.add_argument('--asset_type', type=str, default='stock',
                       choices=['stock', 'crypto'],
                       help='Asset type: stock (5-day week) or crypto (7-day week, 24/7 trading)')
    parser.add_argument('--group_column', type=str, default=None,
                       help='Column for group-based scaling (e.g., "symbol" for multi-stock datasets). If specified, each group gets separate scalers.')

    # Model arguments
    parser.add_argument('--sequence_length', type=int, default=5,
                       help='Number of historical days to use for prediction')
    parser.add_argument('--prediction_horizon', type=int, default=1,
                       help='Number of steps ahead to predict (1=next step, 2=two steps ahead, etc.)')
    parser.add_argument('--d_token', type=int, default=128,
                       help='Token embedding dimension')
    parser.add_argument('--n_layers', type=int, default=3,
                       help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')

    # Evaluation arguments
    parser.add_argument('--per_group_metrics', action='store_true',
                       help='Calculate and display per-group metrics (only when --group_column is set)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    # Data split arguments
    parser.add_argument('--test_size', type=int, default=30,
                       help='Number of samples for test set')
    parser.add_argument('--val_size', type=int, default=20,
                       help='Number of samples for validation set')
    
    # Output arguments
    parser.add_argument('--model_path', type=str, default='outputs/models/stock_model.pt',
                       help='Path to save trained model')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.model_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🚀 Stock Forecasting with FT-Transformer")
    print(f"   Asset Type: {args.asset_type.upper()}")
    print("="*60)

    # 1. Load Data
    print(f"\n📊 Loading {args.asset_type} data...")

    if args.use_sample_data:
        print(f"   Using synthetic sample {args.asset_type} data")
        df = create_sample_stock_data(n_samples=300, asset_type=args.asset_type)
        print(f"   Generated {len(df)} samples of synthetic {args.asset_type} data")
    else:
        if args.data_path is None:
            # Try to find data file in stock_forecasting/data/
            data_candidates = [
                'data/raw/MSFT_historical_price.csv',
                'data/sample/MSFT_sample.csv',
                'data/raw/stock_data.csv'
            ]
            
            for candidate in data_candidates:
                if os.path.exists(candidate):
                    args.data_path = candidate
                    break
            
            if args.data_path is None:
                print("   ❌ No data file found. Using sample data instead.")
                df = create_sample_stock_data(n_samples=300, asset_type=args.asset_type)
            else:
                print(f"   Found data file: {args.data_path}")
                df = load_stock_data(args.data_path, asset_type=args.asset_type, group_column=args.group_column)
        else:
            df = load_stock_data(args.data_path, asset_type=args.asset_type, group_column=args.group_column)
        
        print(f"   Loaded {len(df)} samples from {args.data_path}")
    
    # Validate target column (include engineered features)
    base_columns = df.columns.tolist()

    # Add engineered percentage change features that will be created
    engineered_pct_features = []
    periods = [1, 3, 5, 10]
    for period in periods:
        engineered_pct_features.append(f'pct_change_{period}d')

    # Add other engineered features
    other_engineered = [
        'returns', 'log_returns', 'volatility_10d', 'momentum_5d',
        'high_low_ratio', 'close_open_ratio', 'volume_ratio'
    ]

    all_possible_targets = base_columns + engineered_pct_features + other_engineered

    # Parse target column(s) for validation
    if ',' in args.target:
        target_list = [t.strip() for t in args.target.split(',')]
        # Validate each target
        for target in target_list:
            if target not in all_possible_targets:
                print(f"   ❌ Target column '{target}' not found.")
                print(f"   📊 Base columns: {base_columns}")
                print(f"   🔧 Engineered % change features: pct_change_[1,3,5,10]d")
                print(f"   📈 Other features: returns, volatility_10d, momentum_5d, ratios")
                return
        print(f"   Target columns: {', '.join(target_list)} (multi-target)")
    else:
        if args.target not in all_possible_targets:
            print(f"   ❌ Target column '{args.target}' not found.")
            print(f"   📊 Base columns: {base_columns}")
            print(f"   🔧 Engineered % change features: pct_change_[1,3,5,10]d")
            print(f"   📈 Other features: returns, volatility_10d, momentum_5d, ratios")
            return
        print(f"   Target column: {args.target}")

    print(f"   Date range: {df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "   No date column found")
    
    # 2. Split Data
    print(f"\n🔄 Splitting data...")
    train_df, val_df, test_df = split_time_series(
        df,
        test_size=args.test_size,
        val_size=args.val_size if len(df) > args.test_size + args.val_size + 50 else None,
        group_column=args.group_column,
        time_column='date',
        sequence_length=args.sequence_length
    )
    
    if train_df is None:
        print("   ❌ Insufficient data for splitting")
        return
    
    print(f"   Train samples: {len(train_df)}")
    if val_df is not None:
        print(f"   Validation samples: {len(val_df)}")
    print(f"   Test samples: {len(test_df) if test_df is not None else 0}")
    
    # 3. Initialize Model
    print(f"\n🧠 Initializing Stock Predictor...")

    # Parse target column(s)
    if ',' in args.target:
        target_columns = [t.strip() for t in args.target.split(',')]
        print(f"   Multi-target prediction: {target_columns}")
    else:
        target_columns = args.target

    model = StockPredictor(
        target_column=target_columns,  # Can be str or list
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        asset_type=args.asset_type,
        group_column=args.group_column,
        d_token=args.d_token,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout
    )

    print(f"   Model configuration:")
    print(f"   - Asset type: {args.asset_type}")
    if isinstance(target_columns, list):
        print(f"   - Targets: {', '.join(target_columns)} (multi-target)")
    else:
        print(f"   - Target: {target_columns}")
    print(f"   - Sequence length: {args.sequence_length}")
    print(f"   - Prediction horizon: {args.prediction_horizon} step(s) ahead")
    print(f"   - Token dimension: {args.d_token}")
    print(f"   - Layers: {args.n_layers}")
    print(f"   - Attention heads: {args.n_heads}")
    print(f"   - Dropout: {args.dropout}")
    if args.group_column:
        print(f"   - Group-based scaling: enabled (group_column='{args.group_column}')")
    else:
        print(f"   - Scaling: single-group (global)")

    # 4. Train Model
    print(f"\n🏋️ Training model...")
    
    model.fit(
        df=train_df,
        val_df=val_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        verbose=True
    )
    
    # 5. Save Model
    print(f"\n💾 Saving model...")
    model.save(args.model_path)
    print(f"   ✅ Model saved to: {args.model_path}")

    # 6. Evaluate Model
    print(f"\n📈 Evaluating model...")

    # Helper function to recursively print metrics
    def print_metrics_recursive(metrics_dict, indent=0, prefix=""):
        """Recursively print nested metrics dictionary."""
        indent_str = "   " * indent

        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                # Nested dict (e.g., per-target, per-horizon, or per-group)
                print(f"{indent_str}{prefix}{key}:")
                print_metrics_recursive(value, indent + 1)
            elif isinstance(value, (int, float)):
                # Leaf metric value
                if not np.isnan(value):
                    # Format differently based on metric type
                    if key in ['MAPE', 'Directional_Accuracy']:
                        print(f"{indent_str}- {key}: {value:.2f}%")
                    else:
                        print(f"{indent_str}- {key}: {value:.4f}")

    # Determine if we should use per-group evaluation
    use_per_group = args.per_group_metrics and args.group_column is not None

    # Train metrics
    if use_per_group:
        train_metrics = model.evaluate(train_df, per_group=True)
        print(f"\n   Training Metrics (per-group):")
        print_metrics_recursive(train_metrics, indent=2)
    else:
        train_metrics = model.evaluate(train_df, per_group=False)
        print(f"\n   Training Metrics:")
        print_metrics_recursive(train_metrics, indent=2)

    # Validation metrics
    val_metrics = None
    if val_df is not None and len(val_df) > 0:
        if use_per_group:
            val_metrics = model.evaluate(val_df, per_group=True)
            print(f"\n   Validation Metrics (per-group):")
            print_metrics_recursive(val_metrics, indent=2)
        else:
            val_metrics = model.evaluate(val_df, per_group=False)
            print(f"\n   Validation Metrics:")
            print_metrics_recursive(val_metrics, indent=2)

    # Test metrics
    test_metrics = None
    if test_df is not None and len(test_df) > 0:
        if use_per_group:
            test_metrics = model.evaluate(test_df, per_group=True)
            print(f"\n   Test Metrics (per-group):")
            print_metrics_recursive(test_metrics, indent=2)
        else:
            test_metrics = model.evaluate(test_df, per_group=False)
            print(f"\n   Test Metrics:")
            print_metrics_recursive(test_metrics, indent=2)

    # Prepare features for visualization (after evaluation to avoid double computation)
    print("\n   Preparing features for visualization...")
    train_features = model.prepare_features(train_df, fit_scaler=False)
    if val_df is not None and len(val_df) > 0:
        val_features = model.prepare_features(val_df, fit_scaler=False)
    else:
        val_features = None
    if test_df is not None and len(test_df) > 0:
        test_features = model.prepare_features(test_df, fit_scaler=False)
    else:
        test_features = None
    
    # 7. Generate Comprehensive Plots
    if not args.no_plots:
        print(f"\n📊 Generating comprehensive visualizations...")

        try:
            # Create comprehensive plots with proper alignment and MAPE annotations
            # Use organized output structure within stock_forecasting
            base_output = "outputs"
            saved_plots = create_comprehensive_plots(model, train_df, test_df, base_output)

            if test_metrics is not None:
                # Print comprehensive performance summary (using cached metrics)
                print_performance_summary(
                    model,
                    train_metrics,
                    test_metrics,
                    saved_plots
                )

        except Exception as e:
            print(f"   ⚠️  Error generating comprehensive plots: {e}")
    
    # 8. Summary
    print(f"\n🎉 Stock forecasting completed successfully!")
    print(f"   Model saved to: {args.model_path}")
    if not args.no_plots:
        print(f"   Plots saved to: outputs/")

    # Show best metrics (handle both single and multi-target)
    if test_df is not None and test_metrics is not None:
        if isinstance(target_columns, list):
            # Multi-target: show summary for each target
            print(f"\n   Test Metrics Summary (multi-target):")
            for target in target_columns:
                if target in test_metrics:
                    target_metrics = test_metrics[target]
                    # Handle nested structure for multi-horizon
                    if 'overall' in target_metrics:
                        overall = target_metrics['overall']
                    else:
                        overall = target_metrics

                    mape = overall.get('MAPE', 0)
                    print(f"   - {target.upper()} MAPE: {mape:.2f}%")

                    if 'Directional_Accuracy' in overall:
                        direction_acc = overall['Directional_Accuracy']
                        print(f"     Directional Accuracy: {direction_acc:.1f}%")
        else:
            # Single-target (handle both flat and nested structure)
            if 'overall' in test_metrics:
                overall_metrics = test_metrics['overall']
            else:
                overall_metrics = test_metrics

            mape_value = overall_metrics.get('MAPE', 0)
            print(f"   Test MAPE: {mape_value:.2f}%")

            if 'Directional_Accuracy' in overall_metrics:
                direction_acc = overall_metrics['Directional_Accuracy']
                print(f"   Directional Accuracy: {direction_acc:.1f}%")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()