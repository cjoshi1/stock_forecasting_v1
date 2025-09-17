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


def create_intraday_visualizations(predictor, train_df, test_df, output_dir="outputs", future_predictions=None, args=None, train_features=None, test_features=None):
    """
    Create intraday-specific visualizations.

    Args:
        predictor: Trained IntradayPredictor
        train_df: Training data
        test_df: Test data
        output_dir: Output directory for plots
        future_predictions: DataFrame with future predictions (optional)
        args: Command line arguments for crypto detection

    Returns:
        Dictionary with paths to created files
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get predictions using optimized method (reuse cached features if available)
        if train_features is not None:
            # Use provided cached features
            train_predictions = predictor.predict_from_features(train_features)
            train_processed = train_features
        else:
            # Fall back to regular prediction
            train_predictions = predictor.predict(train_df)
            train_processed = predictor.prepare_features(train_df, fit_scaler=False)

        if test_df is not None:
            if test_features is not None:
                test_predictions = predictor.predict_from_features(test_features)
                test_processed = test_features
            else:
                test_predictions = predictor.predict(test_df)
                test_processed = predictor.prepare_features(test_df, fit_scaler=False)
        else:
            test_predictions = None
            test_processed = None

        # Get timestamps for plotting
        train_timestamps = train_df[predictor.timestamp_col].iloc[predictor.sequence_length:]
        if test_df is not None:
            test_timestamps = test_df[predictor.timestamp_col].iloc[predictor.sequence_length:]

        # Get actual values
        train_actual = train_processed[predictor.target_column].iloc[predictor.sequence_length:]

        if test_df is not None and test_processed is not None:
            test_actual = test_processed[predictor.target_column].iloc[predictor.sequence_length:]
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Predictions vs Actual
        axes[0].plot(train_timestamps, train_actual[:len(train_predictions)], 
                    label='Actual (Train)', alpha=0.7, linewidth=1)
        axes[0].plot(train_timestamps, train_predictions[:len(train_actual)], 
                    label='Predicted (Train)', alpha=0.8, linewidth=1)
        
        if test_predictions is not None and len(test_predictions) > 0:
            axes[0].plot(test_timestamps, test_actual[:len(test_predictions)],
                        label='Actual (Test)', alpha=0.7, linewidth=1)
            axes[0].plot(test_timestamps, test_predictions[:len(test_actual)],
                        label='Predicted (Test)', alpha=0.8, linewidth=1)

        # Add future predictions to plot
        if future_predictions is not None and len(future_predictions) > 0:
            future_timestamps = future_predictions[predictor.timestamp_col]
            future_values = future_predictions[f'predicted_{predictor.target_column}']
            axes[0].plot(future_timestamps, future_values,
                        label='Future Predictions', alpha=0.9, linewidth=2,
                        linestyle='--', color='red', marker='o', markersize=3)
        
        axes[0].set_title(f'Intraday {predictor.target_column.title()} Predictions - {predictor.timeframe}')
        axes[0].set_ylabel(predictor.target_column.title())
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Format x-axis for intraday data (crypto-aware)
        if len(train_timestamps) > 0:
            is_crypto = args and args.country == 'CRYPTO'
            if is_crypto:
                # 24/7 crypto trading - show date and time
                axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                axes[0].xaxis.set_major_locator(mdates.HourLocator(interval=4))  # Every 4 hours
            else:
                # Traditional market hours - show time only
                axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                axes[0].xaxis.set_major_locator(mdates.HourLocator(interval=1))
        
        # Plot 2: Training Progress
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        plot_path = os.path.join(output_dir, f'intraday_predictions_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save predictions to CSV
        csv_data = {
            predictor.timestamp_col: train_timestamps,
            f'actual_{predictor.target_column}': train_actual[:len(train_predictions)],
            f'predicted_{predictor.target_column}': train_predictions[:len(train_actual)]
        }
        
        data_frames = [pd.DataFrame(csv_data)]

        if test_predictions is not None and len(test_predictions) > 0:
            # Add test data
            test_csv_data = pd.DataFrame({
                predictor.timestamp_col: test_timestamps,
                f'actual_{predictor.target_column}': test_actual[:len(test_predictions)],
                f'predicted_{predictor.target_column}': test_predictions[:len(test_actual)]
            })
            data_frames.append(test_csv_data)

        # Add future predictions to the same CSV
        if future_predictions is not None and len(future_predictions) > 0:
            future_csv_data = pd.DataFrame({
                predictor.timestamp_col: future_predictions[predictor.timestamp_col],
                f'actual_{predictor.target_column}': None,  # No actual values for future
                f'predicted_{predictor.target_column}': future_predictions[f'predicted_{predictor.target_column}']
            })
            data_frames.append(future_csv_data)

        combined_data = pd.concat(data_frames, ignore_index=True)
        
        csv_path = os.path.join(output_dir, f'intraday_predictions_{timestamp}.csv')
        combined_data.to_csv(csv_path, index=False)
        
        return {
            'plot': plot_path,
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
                       help='Target column to predict (close, open, high, low)')
    parser.add_argument('--timeframe', type=str, default='5min',
                       choices=['1min', '5min', '15min', '1h'],
                       help='Trading timeframe for prediction')
    parser.add_argument('--country', type=str, default='US',
                       choices=['US', 'INDIA', 'CRYPTO'],
                       help='Country market (US, INDIA, or CRYPTO for 24/7 cryptocurrency trading)')
    parser.add_argument('--use_sample_data', action='store_true',
                       help='Use synthetic sample data instead of real data')
    parser.add_argument('--sample_days', type=int, default=5,
                       help='Number of days for sample data generation')
    
    # Model arguments
    parser.add_argument('--sequence_length', type=int, default=None,
                       help='Number of historical periods (auto-selected if not specified)')
    parser.add_argument('--d_token', type=int, default=128,
                       help='Token embedding dimension')
    parser.add_argument('--n_layers', type=int, default=3,
                       help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
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
            df = load_intraday_data(args.data_path)

        print(f"   Loaded {len(df)} samples")
    
    # 2. Prepare Data for Training
    print(f"\n🔄 Preparing data for {args.timeframe} forecasting ({args.country} market)...")
    
    preparation_result = prepare_intraday_for_training(
        df, 
        target_column=args.target,
        timeframe=args.timeframe,
        country=args.country,
        verbose=args.verbose
    )
    
    df_processed = preparation_result['data']
    timeframe_config = preparation_result['config']
    
    print(f"   Processed {preparation_result['processed_length']} {args.timeframe} bars")
    print(f"   Date range: {df_processed['timestamp'].min()} to {df_processed['timestamp'].max()}")
    
    # 3. Split Data
    print(f"\n🔄 Splitting data...")
    train_df, val_df, test_df = split_time_series(
        df_processed,
        test_size=args.test_size,
        val_size=args.val_size if len(df_processed) > args.test_size + args.val_size + 100 else None
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
    
    # Use timeframe-specific sequence length if not provided
    sequence_length = args.sequence_length or timeframe_config['sequence_length']
    
    model = IntradayPredictor(
        target_column=args.target,
        timeframe=args.timeframe,
        country=args.country,
        sequence_length=sequence_length,
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
    print(f"   - Token dimension: {args.d_token}")
    print(f"   - Layers: {args.n_layers}")
    print(f"   - Attention heads: {args.n_heads}")
    
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
    
    # 7. Evaluate Model (optimized with cached features)
    print(f"\n📈 Evaluating model...")

    # Prepare features once and cache them
    print("   Preparing features for evaluation...")
    train_features = model.prepare_features(train_df, fit_scaler=False)
    if test_df is not None and len(test_df) > 0:
        test_features = model.prepare_features(test_df, fit_scaler=False)

    # Train metrics using cached features
    train_metrics = model.evaluate_from_features(train_features)
    print(f"\n   Training Metrics:")
    for metric, value in train_metrics.items():
        if not np.isnan(value):
            print(f"   - {metric}: {value:.4f}")

    # Test metrics using cached features
    if test_df is not None and len(test_df) > 0:
        test_metrics = model.evaluate_from_features(test_features)
        print(f"\n   Test Metrics:")
        for metric, value in test_metrics.items():
            if not np.isnan(value):
                print(f"   - {metric}: {value:.4f}")

    # 8. Generate Future Predictions
    future_predictions = None
    if args.future_predictions > 0:
        print(f"\n🔮 Predicting next {args.future_predictions} {args.timeframe} periods...")

        # Use the full processed dataset for future predictions
        future_predictions = model.predict_next_bars(df_processed, n_predictions=args.future_predictions)

        if len(future_predictions) > 0:
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

    # 9. Generate Visualizations
    if not args.no_plots:
        print(f"\n📊 Generating intraday visualizations...")
        
        try:
            # Pass cached features to visualization function
            cached_train_features = train_features if 'train_features' in locals() else None
            cached_test_features = test_features if 'test_features' in locals() else None
            saved_files = create_intraday_visualizations(model, train_df, test_df, "outputs", future_predictions, args, cached_train_features, cached_test_features)
            
            if saved_files:
                if 'plot' in saved_files:
                    print(f"   ✅ Intraday plot: {saved_files['plot']}")
                if 'csv' in saved_files:
                    print(f"   ✅ Predictions CSV: {saved_files['csv']}")
            
        except Exception as e:
            print(f"   ⚠️  Error generating visualizations: {e}")
    
    # 10. Summary
    print(f"\n🎉 Intraday forecasting completed successfully!")
    print(f"   Model saved to: {args.model_path}")
    if not args.no_plots:
        print(f"   Outputs saved to: outputs/")
    
    # Show best metrics
    if test_df is not None:
        mape_value = test_metrics.get('MAPE', 0)
        print(f"   Test MAPE: {mape_value:.2f}%")
        
        # Intraday-specific insights
        if 'Directional_Accuracy' in test_metrics:
            direction_acc = test_metrics['Directional_Accuracy']
            print(f"   Directional Accuracy: {direction_acc:.1f}%")
    
    print(f"\n📝 Model Summary:")
    timeframe_info = model.get_timeframe_info()
    print(f"   - Timeframe: {timeframe_info['timeframe']}")
    print(f"   - Description: {timeframe_info['description']}")
    print(f"   - Sequence Length: {timeframe_info['sequence_length']} bars")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()