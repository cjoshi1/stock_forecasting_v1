#!/usr/bin/env python3
"""
Main script for intraday forecasting using FT-Transformer.

This script demonstrates how to use the FT-Transformer library specifically for intraday prediction.
It includes minute-level data loading, model training, evaluation, and visualization.
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
    
    from intraday_forecasting.predictor import IntradayPredictor
    from intraday_forecasting.preprocessing.market_data import (
        load_intraday_data, create_sample_intraday_data, prepare_intraday_for_training
    )
    from tf_predictor.core.utils import split_time_series
else:
    # Module import - use relative imports
    from .predictor import IntradayPredictor
    from .preprocessing.market_data import (
        load_intraday_data, create_sample_intraday_data, prepare_intraday_for_training
    )
    from tf_predictor.core.utils import split_time_series


def create_intraday_visualizations(predictor, train_df, test_df, output_dir="outputs", future_predictions=None, args=None, train_features=None, test_features=None, val_df=None, val_features=None, train_metrics=None, val_metrics=None, test_metrics=None):
    """
    Create intraday-specific visualizations.

    Args:
        predictor: Trained IntradayPredictor
        train_df: Training data
        test_df: Test data
        output_dir: Output directory for plots
        future_predictions: DataFrame with future predictions (optional)
        args: Command line arguments for crypto detection
        train_features: Preprocessed training features (optional)
        test_features: Preprocessed test features (optional)
        val_df: Validation data (optional)
        val_features: Preprocessed validation features (optional)
        train_metrics: Training metrics dictionary (optional)
        val_metrics: Validation metrics dictionary (optional)
        test_metrics: Test metrics dictionary (optional)

    Returns:
        Dictionary with paths to created files
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Detect multi-target mode
        is_multi_target = predictor.is_multi_target
        target_columns = predictor.target_columns if is_multi_target else [predictor.target_column]

        # Get predictions using optimized method (reuse cached features if available)
        if train_features is not None:
            # Use provided cached features
            train_predictions = predictor.predict_from_features(train_features)
        else:
            # Fall back to regular prediction
            train_predictions = predictor.predict(train_df)

        # Handle validation data
        if val_df is not None:
            if val_features is not None:
                val_predictions = predictor.predict_from_features(val_features)
            else:
                val_predictions = predictor.predict(val_df)
        else:
            val_predictions = None

        if test_df is not None:
            if test_features is not None:
                test_predictions = predictor.predict_from_features(test_features)
            else:
                test_predictions = predictor.predict(test_df)
        else:
            test_predictions = None

        # Get timestamps for plotting - align with predictions
        # create_features() does shift(-h) and dropna() which removes last prediction_horizon rows
        # Then create_input_variable_sequence() removes first sequence_length rows from the processed data
        # So we need: iloc[sequence_length : -prediction_horizon] to match
        horizon = predictor.prediction_horizon

        # Extract timestamps (same for all targets)
        if horizon == 1:
            # Single-horizon: drops last 1 row
            train_timestamps = train_df[predictor.timestamp_col].iloc[predictor.sequence_length:-1].values
            if val_df is not None:
                val_timestamps = val_df[predictor.timestamp_col].iloc[predictor.sequence_length:-1].values
            if test_df is not None:
                test_timestamps = test_df[predictor.timestamp_col].iloc[predictor.sequence_length:-1].values
        else:
            # Multi-horizon: drops last prediction_horizon rows
            train_timestamps = train_df[predictor.timestamp_col].iloc[predictor.sequence_length:-horizon].values
            if val_df is not None:
                val_timestamps = val_df[predictor.timestamp_col].iloc[predictor.sequence_length:-horizon].values
            if test_df is not None:
                test_timestamps = test_df[predictor.timestamp_col].iloc[predictor.sequence_length:-horizon].values

        # Extract group column values if group-based scaling is enabled
        train_groups = None
        val_groups = None
        test_groups = None

        if predictor.group_column is not None:
            if horizon == 1:
                train_groups = train_df[predictor.group_column].iloc[predictor.sequence_length:-1].values
                if val_df is not None:
                    val_groups = val_df[predictor.group_column].iloc[predictor.sequence_length:-1].values
                if test_df is not None:
                    test_groups = test_df[predictor.group_column].iloc[predictor.sequence_length:-1].values
            else:
                train_groups = train_df[predictor.group_column].iloc[predictor.sequence_length:-horizon].values
                if val_df is not None:
                    val_groups = val_df[predictor.group_column].iloc[predictor.sequence_length:-horizon].values
                if test_df is not None:
                    test_groups = test_df[predictor.group_column].iloc[predictor.sequence_length:-horizon].values

        # Extract actual values per target
        train_actual_dict = {}
        val_actual_dict = {} if val_df is not None else None
        test_actual_dict = {} if test_df is not None else None

        for target_col in target_columns:
            if horizon == 1:
                train_actual_dict[target_col] = train_df[target_col].iloc[predictor.sequence_length:-1].values
                if val_df is not None:
                    val_actual_dict[target_col] = val_df[target_col].iloc[predictor.sequence_length:-1].values
                if test_df is not None:
                    test_actual_dict[target_col] = test_df[target_col].iloc[predictor.sequence_length:-1].values
            else:
                train_actual_dict[target_col] = train_df[target_col].iloc[predictor.sequence_length:-horizon].values
                if val_df is not None:
                    val_actual_dict[target_col] = val_df[target_col].iloc[predictor.sequence_length:-horizon].values
                if test_df is not None:
                    test_actual_dict[target_col] = test_df[target_col].iloc[predictor.sequence_length:-horizon].values
        
        # Create comprehensive plot(s)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        plot_paths = []

        # For multi-target, create separate plots for each target
        num_targets = len(target_columns)
        for idx, target_col in enumerate(target_columns):
            # Get predictions for this target
            if is_multi_target:
                train_preds = train_predictions[target_col]
                test_preds = test_predictions[target_col] if test_predictions is not None else None
            else:
                train_preds = train_predictions
                test_preds = test_predictions

            # Get actual values for this target
            train_act = train_actual_dict[target_col]
            test_act = test_actual_dict[target_col] if test_df is not None else None

            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))

            # Plot 1: Predictions vs Actual
            axes[0].plot(train_timestamps, train_act,
                        label='Actual (Train)', alpha=0.7, linewidth=1)
            axes[0].plot(train_timestamps, train_preds,
                        label='Predicted (Train)', alpha=0.8, linewidth=1)

            if test_preds is not None and len(test_preds) > 0:
                axes[0].plot(test_timestamps, test_act,
                            label='Actual (Test)', alpha=0.7, linewidth=1)
                axes[0].plot(test_timestamps, test_preds,
                            label='Predicted (Test)', alpha=0.8, linewidth=1)

            # Add future predictions if available (only for single-target for now)
            if not is_multi_target and future_predictions is not None and len(future_predictions) > 0:
                future_timestamps = future_predictions[predictor.timestamp_col]
                future_col = f'predicted_{predictor.original_target_column}'
                future_values = future_predictions[future_col]
                axes[0].plot(future_timestamps, future_values,
                            label='Future Predictions', alpha=0.9, linewidth=2,
                            linestyle='--', color='red', marker='o', markersize=3)

            axes[0].set_title(f'Intraday {target_col.title()} Predictions - {predictor.timeframe}')
            axes[0].set_ylabel(target_col.title())
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Format x-axis for intraday data (crypto-aware)
            if len(train_timestamps) > 0:
                is_crypto = args and args.country == 'CRYPTO'
                if is_crypto:
                    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                    axes[0].xaxis.set_major_locator(mdates.HourLocator(interval=4))
                else:
                    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    axes[0].xaxis.set_major_locator(mdates.HourLocator(interval=1))

            # Plot 2: Training Progress (same for all targets)
            if hasattr(predictor, 'history') and predictor.history['train_loss']:
                epochs = range(1, len(predictor.history['train_loss']) + 1)
                axes[1].plot(epochs, predictor.history['train_loss'], label='Training Loss', linewidth=2)
                if predictor.history['val_loss']:
                    axes[1].plot(epochs, predictor.history['val_loss'], label='Validation Loss', linewidth=2)
                axes[1].set_title('Training Progress')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Loss')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, 'No training history available',
                            transform=axes[1].transAxes, ha='center', va='center')
                axes[1].set_title('Training Progress')

            plt.tight_layout()

            # Save plot
            if is_multi_target:
                plot_path = os.path.join(output_dir, f'intraday_predictions_{target_col}_{timestamp}.png')
            else:
                plot_path = os.path.join(output_dir, f'intraday_predictions_{timestamp}.png')

            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Plot saved to: {plot_path}")
            plt.close()
            plot_paths.append(plot_path)
        
        # Save enhanced predictions to CSV with data split indicators and metrics
        data_frames = []

        # Build CSV data - handle both single and multi-target
        csv_dict = {predictor.timestamp_col: train_timestamps}

        # Add group column if group-based scaling is enabled
        if predictor.group_column is not None and train_groups is not None:
            csv_dict[predictor.group_column] = train_groups

        # Add actual and predicted values for each target
        for target_col in target_columns:
            csv_dict[f'actual_{target_col}'] = train_actual_dict[target_col]
            if is_multi_target:
                csv_dict[f'predicted_{target_col}'] = train_predictions[target_col]
            else:
                csv_dict[f'predicted_{target_col}'] = train_predictions

        csv_dict['data_split'] = 'train'
        train_csv_data = pd.DataFrame(csv_dict)
        data_frames.append(train_csv_data)

        # Add validation data if available
        if val_predictions is not None and len(val_predictions) > 0:
            val_csv_dict = {predictor.timestamp_col: val_timestamps}
            if predictor.group_column is not None and val_groups is not None:
                val_csv_dict[predictor.group_column] = val_groups
            for target_col in target_columns:
                val_csv_dict[f'actual_{target_col}'] = val_actual_dict[target_col]
                if is_multi_target:
                    val_csv_dict[f'predicted_{target_col}'] = val_predictions[target_col]
                else:
                    val_csv_dict[f'predicted_{target_col}'] = val_predictions
            val_csv_dict['data_split'] = 'validation'
            val_csv_data = pd.DataFrame(val_csv_dict)
            data_frames.append(val_csv_data)

        # Add test data if available
        if test_predictions is not None and len(test_predictions) > 0:
            test_csv_dict = {predictor.timestamp_col: test_timestamps}
            if predictor.group_column is not None and test_groups is not None:
                test_csv_dict[predictor.group_column] = test_groups
            for target_col in target_columns:
                test_csv_dict[f'actual_{target_col}'] = test_actual_dict[target_col]
                if is_multi_target:
                    test_csv_dict[f'predicted_{target_col}'] = test_predictions[target_col]
                else:
                    test_csv_dict[f'predicted_{target_col}'] = test_predictions
            test_csv_dict['data_split'] = 'test'
            test_csv_data = pd.DataFrame(test_csv_dict)
            data_frames.append(test_csv_data)

        # Add future predictions if available (only for single-target for now)
        if not is_multi_target and future_predictions is not None and len(future_predictions) > 0:
            future_csv_data = pd.DataFrame({
                predictor.timestamp_col: future_predictions[predictor.timestamp_col],
                f'actual_{predictor.target_column}': None,
                f'predicted_{predictor.target_column}': future_predictions[f'predicted_{predictor.target_column}'],
                'data_split': 'future'
            })
            data_frames.append(future_csv_data)

        # Combine all data
        combined_data = pd.concat(data_frames, ignore_index=True)

        # Create enhanced CSV with metrics header
        csv_path = os.path.join(output_dir, f'intraday_predictions_{timestamp}.csv')

        # Write metrics header and then the data
        with open(csv_path, 'w') as f:
            # Write metrics summary as comments
            f.write("# METRICS SUMMARY\n")
            f.write("# Dataset,MAE,MSE,RMSE,MAPE,R2,Directional_Accuracy\n")

            # Helper function to format metric value
            def format_metric(value):
                if value is None:
                    return 'N/A'
                try:
                    if np.isnan(value) or np.isinf(value):
                        return 'N/A'
                    return f"{value:.4f}"
                except (TypeError, ValueError):
                    return 'N/A'

            # Helper to write metrics for a dataset
            def write_metrics(dataset_name, metrics):
                if not metrics:
                    return
                if is_multi_target:
                    # Multi-target: write metrics for each target
                    for target_col in target_columns:
                        if target_col in metrics:
                            m = metrics[target_col]
                            if 'overall' in m:
                                m = m['overall']
                            f.write(f"# {dataset_name}_{target_col},{format_metric(m.get('MAE'))},{format_metric(m.get('MSE'))},{format_metric(m.get('RMSE'))},{format_metric(m.get('MAPE'))},{format_metric(m.get('R2'))},{format_metric(m.get('Directional_Accuracy'))}\n")
                else:
                    # Single-target
                    m = metrics.get('overall', metrics)
                    f.write(f"# {dataset_name},{format_metric(m.get('MAE'))},{format_metric(m.get('MSE'))},{format_metric(m.get('RMSE'))},{format_metric(m.get('MAPE'))},{format_metric(m.get('R2'))},{format_metric(m.get('Directional_Accuracy'))}\n")

            # Write metrics for each dataset
            write_metrics("Train", train_metrics)
            write_metrics("Validation", val_metrics)
            write_metrics("Test", test_metrics)

            f.write("#\n")  # Separator line

        # Append the actual data
        combined_data.to_csv(csv_path, mode='a', index=False)
        print(f"   ✅ Predictions CSV saved to: {csv_path}")

        return {
            'plots': plot_paths,  # List of plot paths (one per target for multi-target)
            'csv': csv_path
        }
        
    except ImportError:
        print("Matplotlib not available, skipping visualizations")
        return {}
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description='Train FT-Transformer for intraday forecasting')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to intraday data CSV file')
    parser.add_argument('--target', type=str, default='close',
                       help='Target column(s) to predict. Single: "close" or Multiple: "close,volume"')
    parser.add_argument('--timeframe', type=str, default='5min',
                       choices=['1min', '5min', '15min', '1h'],
                       help='Trading timeframe for prediction')
    parser.add_argument('--model_type', type=str, default='ft_transformer',
                       choices=['ft_transformer', 'csn_transformer'],
                       help='Model architecture (ft_transformer=FT-Transformer, csn_transformer=CSN-Transformer)')
    parser.add_argument('--pooling_type', type=str, default='multihead_attention',
                       choices=['cls', 'singlehead_attention', 'multihead_attention', 'weighted_avg', 'temporal_multihead_attention'],
                       help='Pooling strategy for sequence aggregation (default: multihead_attention)')
    parser.add_argument('--country', type=str, default='US',
                       choices=['US', 'INDIA', 'CRYPTO'],
                       help='Country market (US, INDIA, or CRYPTO for 24/7 cryptocurrency trading)')
    parser.add_argument('--use_sample_data', action='store_true',
                       help='Use synthetic sample data instead of real data')
    parser.add_argument('--sample_days', type=int, default=5,
                       help='Number of days for sample data generation')
    parser.add_argument('--group_columns', type=str, default=None,
                       help='Column(s) for group-based scaling (e.g., "symbol" for multi-stock datasets). If specified, each group gets separate scalers. Multiple: "symbol,sector"')
    parser.add_argument('--categorical_columns', type=str, default=None,
                       help='Column(s) to encode and pass as categorical features (e.g., "symbol,sector"). Multiple: "symbol,sector"')
    parser.add_argument('--scaler_type', type=str, default='standard',
                       choices=['standard', 'minmax', 'robust', 'maxabs', 'onlymax'],
                       help='Type of scaler for normalization (default: standard)')
    parser.add_argument('--use_lagged_target_features', action='store_true',
                       help='Include target columns in input sequences for autoregressive modeling')

    # Model arguments
    parser.add_argument('--sequence_length', type=int, default=None,
                       help='Number of historical periods (auto-selected if not specified)')
    parser.add_argument('--prediction_horizon', type=int, default=1,
                       help='Number of future steps to predict (1=next step, 3=three steps ahead, etc.)')
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
                       help='Calculate and display per-group metrics (only when --group_columns is set)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Data split arguments
    parser.add_argument('--test_size', type=int, default=200,
                       help='Number of samples for test set')
    parser.add_argument('--val_size', type=int, default=100,
                       help='Number of samples for validation set')
    
    # Prediction arguments
    parser.add_argument('--future_predictions', type=int, default=0,
                       help='Number of future periods to predict (0 = no future predictions)')

    # Output arguments
    parser.add_argument('--model_path', type=str, default='outputs/models/intraday_model.pt',
                       help='Path to save trained model')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose output (default: True)')
    parser.add_argument('--quiet', action='store_true',
                       help='Disable verbose output')
    
    args = parser.parse_args()

    # Handle quiet flag (overrides verbose)
    if args.quiet:
        args.verbose = False

    # Create output directory
    output_dir = Path(args.model_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🚀 Intraday Forecasting with FT-Transformer")
    print(f"   Market: {args.country} | Timeframe: {args.timeframe}")
    print("="*60)
    
    # 1. Load Data
    print(f"\n📊 Loading intraday data...")
    
    if args.use_sample_data:
        print(f"   Generating {args.sample_days} days of sample intraday data")
        df = create_sample_intraday_data(n_days=args.sample_days)
        print(f"   Generated {len(df)} minute-level samples")
    else:
        if args.data_path is None:
            print("   ❌ No data file specified. Use --data_path or --use_sample_data")
            print("   Using sample data instead...")
            df = create_sample_intraday_data(n_days=args.sample_days)
        else:
            # Auto-detect cryptocurrency data and set country to CRYPTO
            file_name = os.path.basename(args.data_path).upper()
            if any(crypto in file_name for crypto in ['BTC', 'ETH', 'CRYPTO']) and args.country == 'US':
                print(f"   📊 Detected cryptocurrency data, automatically setting country to 'CRYPTO' for 24/7 trading")
                args.country = 'CRYPTO'

            print(f"   Loading from: {args.data_path}")
            # Parse group_columns if provided (use first one for loading)
            group_col_parsed = None
            if args.group_columns:
                group_col_parsed = args.group_columns.split(',')[0].strip() if ',' in args.group_columns else args.group_columns
            df = load_intraday_data(args.data_path, group_column=group_col_parsed)

        print(f"   Loaded {len(df)} samples")
    
    # 2. Prepare Data for Training
    print(f"\n🔄 Preparing data for {args.timeframe} forecasting ({args.country} market)...")

    # Parse target column(s) early for preparation
    if ',' in args.target:
        target_for_prep = [t.strip() for t in args.target.split(',')]
    else:
        target_for_prep = args.target

    # Parse group_columns for preparation (use first one)
    group_col_for_prep = None
    if args.group_columns:
        group_col_for_prep = args.group_columns.split(',')[0].strip() if ',' in args.group_columns else args.group_columns

    preparation_result = prepare_intraday_for_training(
        df,
        target_column=target_for_prep,
        timeframe=args.timeframe,
        country=args.country,
        group_column=group_col_for_prep,
        verbose=args.verbose
    )
    
    df_processed = preparation_result['data']
    timeframe_config = preparation_result['config']
    
    print(f"   Processed {preparation_result['processed_length']} {args.timeframe} bars")
    print(f"   Date range: {df_processed['timestamp'].min()} to {df_processed['timestamp'].max()}")
    
    # 3. Split Data
    print(f"\n🔄 Splitting data...")
    # Use timeframe-specific sequence length for split validation
    split_sequence_length = args.sequence_length or timeframe_config['sequence_length']
    # Parse group_columns for split (use first one)
    group_col_for_split = None
    if args.group_columns:
        group_col_for_split = args.group_columns.split(',')[0].strip() if ',' in args.group_columns else args.group_columns
    train_df, val_df, test_df = split_time_series(
        df_processed,
        test_size=args.test_size,
        val_size=args.val_size if len(df_processed) > args.test_size + args.val_size + 100 else None,
        group_column=group_col_for_split,
        time_column='timestamp',
        sequence_length=split_sequence_length
    )
    
    if train_df is None:
        print("   ❌ Insufficient data for splitting")
        return
    
    print(f"   Train samples: {len(train_df)}")
    if val_df is not None:
        print(f"   Validation samples: {len(val_df)}")
    print(f"   Test samples: {len(test_df) if test_df is not None else 0}")
    
    # 4. Initialize Model
    print(f"\n🧠 Initializing Intraday Predictor...")

    # Parse target column(s)
    if ',' in args.target:
        target_columns = [t.strip() for t in args.target.split(',')]
        print(f"   Multi-target prediction: {target_columns}")
    else:
        target_columns = args.target

    # Use timeframe-specific sequence length if not provided
    sequence_length = args.sequence_length or timeframe_config['sequence_length']

    # Parse group_columns
    group_cols_for_model = None
    if args.group_columns:
        if ',' in args.group_columns:
            group_cols_for_model = [g.strip() for g in args.group_columns.split(',')]
        else:
            group_cols_for_model = args.group_columns

    # Parse categorical_columns
    cat_cols_for_model = None
    if args.categorical_columns:
        if ',' in args.categorical_columns:
            cat_cols_for_model = [c.strip() for c in args.categorical_columns.split(',')]
        else:
            cat_cols_for_model = args.categorical_columns

    model = IntradayPredictor(
        target_column=target_columns,  # Can be str or list
        timeframe=args.timeframe,
        model_type=args.model_type,
        country=args.country,
        sequence_length=sequence_length,
        prediction_horizon=args.prediction_horizon,
        group_columns=group_cols_for_model,
        categorical_columns=cat_cols_for_model,
        scaler_type=args.scaler_type,
        use_lagged_target_features=args.use_lagged_target_features,
        d_token=args.d_token,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        verbose=args.verbose
    )

    print(f"   Model configuration:")
    print(f"   - Market: {args.country}")
    print(f"   - Timeframe: {args.timeframe} ({timeframe_config['description']})")
    print(f"   - Sequence length: {sequence_length} {args.timeframe} bars")
    print(f"   - Prediction horizon: {args.prediction_horizon} step(s) ahead")
    if isinstance(target_columns, list):
        print(f"   - Targets: {', '.join(target_columns)} (multi-target)")
    else:
        print(f"   - Target: {target_columns}")
    print(f"   - Token dimension: {args.d_token}")
    print(f"   - Layers: {args.n_layers}")
    print(f"   - Attention heads: {args.n_heads}")
    print(f"   - Scaler type: {args.scaler_type}")
    if args.use_lagged_target_features:
        print(f"   - Lagged target features: enabled")
    if group_cols_for_model:
        print(f"   - Group-based scaling: enabled (group_columns='{args.group_columns}')")
    else:
        print(f"   - Scaling: single-group (global)")
    if cat_cols_for_model:
        print(f"   - Categorical features: {args.categorical_columns}")

    # 5. Train Model
    print(f"\n🏋️ Training model...")
    
    model.fit(
        df=train_df,
        val_df=val_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        verbose=args.verbose
    )
    
    # 6. Save Model
    print(f"\n💾 Saving model...")
    model.save(args.model_path)
    print(f"   ✅ Model saved to: {args.model_path}")
    
    # 7. Evaluate Model
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
    use_per_group = args.per_group_metrics and args.group_columns is not None

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

    # 8. Generate Future Predictions
    future_predictions = None
    if args.future_predictions > 0:
        print(f"\n🔮 Predicting next {args.future_predictions} {args.timeframe} periods...")

        # Note: predict_next_bars may not work correctly with group-based predictions for future bars
        # This is a limitation when using group_columns since we don't know which group to predict for
        if args.group_columns is not None:
            print(f"   ⚠️  Future predictions not supported with group-based scaling (--group_columns)")
            print(f"   Skipping future predictions...")
        else:
            # Use the full processed dataset for future predictions
            try:
                future_predictions = model.predict_next_bars(df_processed, n_predictions=args.future_predictions)

                if isinstance(future_predictions, pd.DataFrame) and len(future_predictions) > 0:
                    print(f"   Future predictions:")
                    for _, row in future_predictions.iterrows():
                        timestamp = row[model.timestamp_col]
                        predicted_value = row[f'predicted_{args.target}']
                        if args.target == 'volume':
                            print(f"   {timestamp.strftime('%Y-%m-%d %H:%M')}: {predicted_value:.0f}")
                        else:
                            print(f"   {timestamp.strftime('%Y-%m-%d %H:%M')}: {predicted_value:.2f}")
                else:
                    print(f"   ⚠️  No future predictions generated")
            except Exception as e:
                print(f"   ⚠️  Error generating future predictions: {e}")

    # 9. Generate Visualizations
    if not args.no_plots:
        print(f"\n📊 Generating intraday visualizations...")

        try:
            # Pass cached features and metrics to visualization function
            cached_train_features = train_features if 'train_features' in locals() else None
            cached_val_features = val_features if 'val_features' in locals() else None
            cached_test_features = test_features if 'test_features' in locals() else None
            saved_files = create_intraday_visualizations(
                model, train_df, test_df, "outputs", future_predictions, args,
                cached_train_features, cached_test_features, val_df, cached_val_features,
                train_metrics, val_metrics, test_metrics
            )

        except Exception as e:
            print(f"   ⚠️  Error generating visualizations: {e}")
    
    # 10. Summary
    print(f"\n🎉 Intraday forecasting completed successfully!")
    print(f"   Model saved to: {args.model_path}")
    if not args.no_plots:
        print(f"   Outputs saved to: outputs/")

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

            # Intraday-specific insights
            if 'Directional_Accuracy' in overall_metrics:
                direction_acc = overall_metrics['Directional_Accuracy']
                print(f"   Directional Accuracy: {direction_acc:.1f}%")
    
    print(f"\n📝 Model Summary:")
    timeframe_info = model.get_timeframe_info()
    print(f"   - Timeframe: {timeframe_info['timeframe']}")
    print(f"   - Description: {timeframe_info['description']}")
    print(f"   - Sequence Length: {timeframe_info['sequence_length']} bars")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()