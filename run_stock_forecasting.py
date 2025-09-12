#!/usr/bin/env python3
"""
Stock Forecasting Runner Script

This script allows you to easily run stock price prediction with configurable parameters.
You can specify the CSV path, target column, and other model parameters.

Usage:
    python run_stock_forecasting.py --csv_path data/AAPL.csv --target_column close --epochs 50
    
Example CSV format (required columns):
    date,open,high,low,close,volume
    2023-01-01,150.0,155.0,149.0,154.0,1000000
    2023-01-02,154.0,158.0,153.0,157.0,1200000
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stock_forecasting.predictor import StockPredictor
from stock_forecasting.preprocessing.market_data import load_stock_data, create_sample_stock_data
from tf_predictor.core.utils import split_time_series


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Stock Forecasting Model')
    
    # Data arguments
    parser.add_argument('--csv_path', type=str, default=None,
                       help='Path to CSV file with stock data (OHLCV format)')
    parser.add_argument('--use_sample_data', action='store_true',
                       help='Use generated sample data instead of real data')
    
    # Model configuration
    parser.add_argument('--target_column', type=str, default='close',
                       help='Target column to predict (close, open, high, low, pct_change_1d, etc.)')
    parser.add_argument('--sequence_length', type=int, default=10,
                       help='Number of historical days to use for prediction')
    
    # Model architecture
    parser.add_argument('--d_token', type=int, default=128,
                       help='Token embedding dimension')
    parser.add_argument('--n_layers', type=int, default=3,
                       help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    
    # Data split
    parser.add_argument('--test_size', type=int, default=60,
                       help='Number of samples for test set')
    parser.add_argument('--val_size', type=int, default=30,
                       help='Number of samples for validation set')
    
    # Other options
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--save_model', type=str, default=None,
                       help='Path to save trained model')
    parser.add_argument('--predictions_output', type=str, default=None,
                       help='Path to save predictions CSV')
    
    return parser.parse_args()


def load_data(args):
    """Load and validate stock data."""
    if args.use_sample_data:
        print("📊 Generating sample stock data...")
        df = create_sample_stock_data(n_samples=500)
        print(f"   Generated {len(df)} trading days")
    else:
        if not args.csv_path:
            raise ValueError("Please provide --csv_path or use --use_sample_data")
        
        if not os.path.exists(args.csv_path):
            raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
        
        print(f"📊 Loading data from: {args.csv_path}")
        df = load_stock_data(args.csv_path)
        print(f"   Loaded {len(df)} trading days")
    
    # Validate target column
    if args.target_column not in df.columns:
        print(f"❌ Target column '{args.target_column}' not found in data")
        print(f"   Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Display data info
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Target column: {args.target_column}")
    print(f"   Price range: {df[args.target_column].min():.2f} - {df[args.target_column].max():.2f}")
    
    return df


def main():
    """Main execution function."""
    print("🚀 Stock Forecasting Runner")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Load data
        df = load_data(args)
        
        # Split data
        print(f"\n🔄 Splitting data...")
        train_df, val_df, test_df = split_time_series(
            df, 
            test_size=args.test_size, 
            val_size=args.val_size
        )
        print(f"   Train: {len(train_df)} samples")
        print(f"   Validation: {len(val_df) if val_df is not None else 0} samples")
        print(f"   Test: {len(test_df) if test_df is not None else 0} samples")
        
        # Initialize model
        print(f"\n🧠 Initializing StockPredictor...")
        predictor = StockPredictor(
            target_column=args.target_column,
            sequence_length=args.sequence_length,
            d_token=args.d_token,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            verbose=args.verbose
        )
        
        print(f"   Target: {args.target_column}")
        print(f"   Architecture: {args.d_token}d x {args.n_layers}L x {args.n_heads}H")
        print(f"   Sequence length: {args.sequence_length} days")
        
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
        if test_df is not None and len(test_df) > args.sequence_length:
            test_metrics = predictor.evaluate(test_df)
            print(f"   Test Results:")
            for metric, value in test_metrics.items():
                if not np.isnan(value):
                    unit = "$" if metric in ['MAE', 'RMSE'] else ("%" if 'MAPE' in metric else "")
                    print(f"   - {metric}: {value:.4f}{unit}")
        else:
            print("   ⚠️  Insufficient test data for evaluation")
        
        # Generate predictions
        if test_df is not None and len(test_df) > args.sequence_length:
            print(f"\n🔮 Generating predictions...")
            predictions = predictor.predict(test_df)
            print(f"   Generated {len(predictions)} predictions")
            
            # Show sample predictions
            if len(predictions) >= 5:
                print(f"\n   Sample predictions (first 5):")
                test_processed = predictor.prepare_features(test_df, fit_scaler=False)
                actual_values = test_processed[args.target_column].iloc[args.sequence_length:args.sequence_length+5]
                test_dates = test_df['date'].iloc[args.sequence_length:args.sequence_length+5]
                
                for date, actual, pred in zip(test_dates, actual_values, predictions[:5]):
                    error = abs(actual - pred)
                    error_pct = (error / actual * 100) if actual != 0 else 0
                    print(f"   {date}: Actual={actual:.2f}, Predicted={pred:.2f}, Error={error:.2f} ({error_pct:.1f}%)")
            
            # Save predictions if requested
            if args.predictions_output:
                print(f"\n💾 Saving predictions to: {args.predictions_output}")
                test_processed = predictor.prepare_features(test_df, fit_scaler=False)
                
                # Create predictions DataFrame
                pred_df = pd.DataFrame({
                    'date': test_df['date'].iloc[args.sequence_length:args.sequence_length+len(predictions)],
                    f'actual_{args.target_column}': test_processed[args.target_column].iloc[args.sequence_length:args.sequence_length+len(predictions)],
                    f'predicted_{args.target_column}': predictions,
                })
                pred_df['error'] = pred_df[f'actual_{args.target_column}'] - pred_df[f'predicted_{args.target_column}']
                pred_df['error_pct'] = (pred_df['error'] / pred_df[f'actual_{args.target_column}'] * 100).round(2)
                
                pred_df.to_csv(args.predictions_output, index=False)
                print(f"   Saved {len(pred_df)} predictions")
        
        # Save model if requested
        if args.save_model:
            print(f"\n💾 Saving model to: {args.save_model}")
            # Note: Model saving functionality would need to be implemented in the base predictor
            print(f"   ⚠️  Model saving not yet implemented in base predictor")
        
        print(f"\n✅ Stock forecasting completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()