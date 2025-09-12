#!/usr/bin/env python3
"""
Intraday Forecasting Runner Script

This script allows you to easily run intraday (minute-level) forecasting with configurable parameters.
You can specify the CSV path, timeframe, target column, and other model parameters.

Usage:
    python run_intraday_forecasting.py --csv_path data/AAPL_intraday.csv --timeframe 5min --target_column close --epochs 50
    
Example CSV format (required columns):
    timestamp,open,high,low,close,volume
    2023-01-01 09:30:00,150.0,155.0,149.0,154.0,1000000
    2023-01-01 09:35:00,154.0,158.0,153.0,157.0,1200000
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intraday_forecasting.predictor import IntradayPredictor
from intraday_forecasting.preprocessing.intraday_data import create_sample_intraday_data, prepare_intraday_for_training, load_intraday_data
from tf_predictor.core.utils import split_time_series


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Intraday Forecasting Model')
    
    # Data arguments
    parser.add_argument('--csv_path', type=str, default=None,
                       help='Path to CSV file with intraday data (timestamp, OHLCV format)')
    parser.add_argument('--use_sample_data', action='store_true',
                       help='Use generated sample intraday data instead of real data')
    parser.add_argument('--sample_days', type=int, default=7,
                       help='Number of days for sample data generation')
    
    # Model configuration
    parser.add_argument('--target_column', type=str, default='close',
                       help='Target column to predict (close, open, high, low, volume)')
    parser.add_argument('--timeframe', type=str, default='5min',
                       choices=['1min', '5min', '15min', '1h'],
                       help='Trading timeframe for predictions')
    parser.add_argument('--timestamp_col', type=str, default='timestamp',
                       help='Name of timestamp column in CSV')
    parser.add_argument('--country', type=str, default='US',
                       choices=['US', 'INDIA'],
                       help='Country market (affects market hours and features)')
    
    # Model architecture
    parser.add_argument('--d_token', type=int, default=128,
                       help='Token embedding dimension')
    parser.add_argument('--n_layers', type=int, default=3,
                       help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--sequence_length', type=int, default=None,
                       help='Override default sequence length (will use timeframe default if not set)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    
    # Data split
    parser.add_argument('--test_size', type=int, default=100,
                       help='Number of samples for test set')
    parser.add_argument('--val_size', type=int, default=50,
                       help='Number of samples for validation set')
    
    # Prediction options
    parser.add_argument('--future_predictions', type=int, default=10,
                       help='Number of future periods to predict')
    
    # Other options
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--save_model', type=str, default=None,
                       help='Path to save trained model')
    parser.add_argument('--predictions_output', type=str, default=None,
                       help='Path to save predictions CSV')
    
    return parser.parse_args()


def load_data(args):
    """Load and validate intraday data."""
    if args.use_sample_data:
        print(f"📊 Generating sample intraday data for {args.sample_days} days...")
        df = create_sample_intraday_data(n_days=args.sample_days)
        print(f"   Generated {len(df)} minute-level samples")
    else:
        if not args.csv_path:
            raise ValueError("Please provide --csv_path or use --use_sample_data")
        
        if not os.path.exists(args.csv_path):
            raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
        
        print(f"📊 Loading intraday data from: {args.csv_path}")
        df = load_intraday_data(args.csv_path, timestamp_col=args.timestamp_col)
        print(f"   Loaded {len(df)} intraday samples")
    
    # Validate target column
    if args.target_column not in df.columns:
        print(f"❌ Target column '{args.target_column}' not found in data")
        print(f"   Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Validate timestamp column
    if args.timestamp_col not in df.columns:
        print(f"❌ Timestamp column '{args.timestamp_col}' not found in data")
        print(f"   Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Display data info
    print(f"   Time range: {df[args.timestamp_col].min()} to {df[args.timestamp_col].max()}")
    print(f"   Target column: {args.target_column}")
    if args.target_column != 'volume':
        print(f"   Price range: {df[args.target_column].min():.2f} - {df[args.target_column].max():.2f}")
    else:
        print(f"   Volume range: {df[args.target_column].min():.0f} - {df[args.target_column].max():.0f}")
    
    return df


def prepare_data(df, args):
    """Prepare intraday data for training."""
    print(f"\n🔄 Preparing data for {args.timeframe} forecasting...")
    
    preparation_result = prepare_intraday_for_training(
        df=df,
        target_column=args.target_column,
        timeframe=args.timeframe,
        timestamp_col=args.timestamp_col,
        country=args.country,
        verbose=args.verbose
    )
    
    df_processed = preparation_result['data']
    print(f"   Processed {len(df_processed)} {args.timeframe} bars")
    print(f"   Features created: {len(df_processed.columns)} columns")
    
    if args.verbose:
        print(f"   Market configuration: {preparation_result.get('market_config', {})}")
        print(f"   Timeframe configuration: {preparation_result.get('timeframe_config', {})}")
    
    return df_processed


def main():
    """Main execution function."""
    print("🚀 Intraday Forecasting Runner")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Load raw data
        df = load_data(args)
        
        # Prepare data for the specified timeframe
        df_processed = prepare_data(df, args)
        
        # Split data
        print(f"\n🔄 Splitting data...")
        train_df, val_df, test_df = split_time_series(
            df_processed, 
            test_size=args.test_size, 
            val_size=args.val_size
        )
        print(f"   Train: {len(train_df)} samples")
        print(f"   Validation: {len(val_df) if val_df is not None else 0} samples")
        print(f"   Test: {len(test_df) if test_df is not None else 0} samples")
        
        # Check minimum data requirements
        if len(train_df) < 100:
            print(f"   ⚠️  Warning: Limited training data ({len(train_df)} samples)")
        
        # Initialize model
        print(f"\n🧠 Initializing IntradayPredictor...")
        
        # Use sequence_length from args or let the model use timeframe default
        predictor_kwargs = {
            'target_column': args.target_column,
            'timeframe': args.timeframe,
            'timestamp_col': args.timestamp_col,
            'country': args.country,
            'd_token': args.d_token,
            'n_layers': args.n_layers,
            'n_heads': args.n_heads,
            'dropout': args.dropout,
            'verbose': args.verbose
        }
        
        if args.sequence_length is not None:
            predictor_kwargs['sequence_length'] = args.sequence_length
        
        predictor = IntradayPredictor(**predictor_kwargs)
        
        # Display model info
        timeframe_info = predictor.get_timeframe_info()
        print(f"   Target: {args.target_column}")
        print(f"   Timeframe: {args.timeframe} ({timeframe_info['description']})")
        print(f"   Country: {args.country} ({timeframe_info['market_hours']})")
        print(f"   Architecture: {args.d_token}d x {args.n_layers}L x {args.n_heads}H")
        print(f"   Sequence length: {predictor.sequence_length} {args.timeframe} bars")
        
        # Train model
        print(f"\n🏋️ Training model...")
        predictor.fit(
            df=train_df,
            val_df=val_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            verbose=args.verbose
        )
        
        # Evaluate on test set
        print(f"\n📊 Evaluating model...")
        if test_df is not None and len(test_df) > predictor.sequence_length:
            test_metrics = predictor.evaluate(test_df)
            print(f"   Test Results:")
            for metric, value in test_metrics.items():
                if not np.isnan(value):
                    if args.target_column == 'volume':
                        unit = "" if metric == 'MAPE' else ""
                    else:
                        unit = "$" if metric in ['MAE', 'RMSE'] else ("%" if 'MAPE' in metric else "")
                    print(f"   - {metric}: {value:.4f}{unit}")
        else:
            print("   ⚠️  Insufficient test data for evaluation")
        
        # Generate predictions on test set
        if test_df is not None and len(test_df) > predictor.sequence_length:
            print(f"\n🔮 Generating predictions on test set...")
            predictions = predictor.predict(test_df)
            print(f"   Generated {len(predictions)} predictions")
            
            # Show sample predictions
            if len(predictions) >= 5:
                print(f"\n   Sample predictions (first 5):")
                test_processed = predictor.prepare_features(test_df, fit_scaler=False)
                actual_values = test_processed[args.target_column].iloc[predictor.sequence_length:predictor.sequence_length+5]
                test_timestamps = test_df[args.timestamp_col].iloc[predictor.sequence_length:predictor.sequence_length+5]
                
                for timestamp, actual, pred in zip(test_timestamps, actual_values, predictions[:5]):
                    error = abs(actual - pred)
                    error_pct = (error / actual * 100) if actual != 0 else 0
                    if args.target_column == 'volume':
                        print(f"   {timestamp.strftime('%Y-%m-%d %H:%M')}: Actual={actual:.0f}, Predicted={pred:.0f}, Error={error:.0f} ({error_pct:.1f}%)")
                    else:
                        print(f"   {timestamp.strftime('%Y-%m-%d %H:%M')}: Actual={actual:.2f}, Predicted={pred:.2f}, Error={error:.2f} ({error_pct:.1f}%)")
        
        # Generate future predictions
        if args.future_predictions > 0:
            print(f"\n🔮 Predicting next {args.future_predictions} {args.timeframe} periods...")
            
            # Use the full processed dataset for future predictions
            future_predictions = predictor.predict_next_bars(df_processed, n_predictions=args.future_predictions)
            
            if len(future_predictions) > 0:
                print(f"   Future predictions:")
                for _, row in future_predictions.iterrows():
                    timestamp = row[args.timestamp_col]
                    predicted_value = row[f'predicted_{args.target_column}']
                    if args.target_column == 'volume':
                        print(f"   {timestamp.strftime('%Y-%m-%d %H:%M')}: {predicted_value:.0f}")
                    else:
                        print(f"   {timestamp.strftime('%Y-%m-%d %H:%M')}: {predicted_value:.2f}")
                
                # Save future predictions if requested
                if args.predictions_output:
                    future_output = args.predictions_output.replace('.csv', '_future.csv')
                    future_predictions.to_csv(future_output, index=False)
                    print(f"   💾 Saved future predictions to: {future_output}")
            else:
                print(f"   ⚠️  No future predictions generated")
        
        # Save test predictions if requested
        if args.predictions_output and test_df is not None and len(test_df) > predictor.sequence_length:
            print(f"\n💾 Saving test predictions to: {args.predictions_output}")
            test_processed = predictor.prepare_features(test_df, fit_scaler=False)
            predictions = predictor.predict(test_df)
            
            # Create predictions DataFrame
            pred_df = pd.DataFrame({
                args.timestamp_col: test_df[args.timestamp_col].iloc[predictor.sequence_length:predictor.sequence_length+len(predictions)],
                f'actual_{args.target_column}': test_processed[args.target_column].iloc[predictor.sequence_length:predictor.sequence_length+len(predictions)],
                f'predicted_{args.target_column}': predictions,
            })
            pred_df['error'] = pred_df[f'actual_{args.target_column}'] - pred_df[f'predicted_{args.target_column}']
            pred_df['error_pct'] = (pred_df['error'] / pred_df[f'actual_{args.target_column}'] * 100).round(2)
            
            pred_df.to_csv(args.predictions_output, index=False)
            print(f"   Saved {len(pred_df)} test predictions")
        
        # Save model if requested
        if args.save_model:
            print(f"\n💾 Saving model to: {args.save_model}")
            # Note: Model saving functionality would need to be implemented in the base predictor
            print(f"   ⚠️  Model saving not yet implemented in base predictor")
        
        print(f"\n✅ Intraday forecasting completed successfully!")
        
        # Show summary
        print(f"\n📋 Summary:")
        print(f"   - Timeframe: {args.timeframe}")
        print(f"   - Target: {args.target_column}")
        print(f"   - Country: {args.country}")
        print(f"   - Training samples: {len(train_df)}")
        print(f"   - Model: {args.d_token}d x {args.n_layers}L x {args.n_heads}H")
        if test_df is not None and len(test_df) > predictor.sequence_length:
            test_metrics = predictor.evaluate(test_df)
            mape = test_metrics.get('MAPE', np.nan)
            if not np.isnan(mape):
                print(f"   - Test MAPE: {mape:.2f}%")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()