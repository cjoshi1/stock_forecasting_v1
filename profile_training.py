#!/usr/bin/env python3
"""
Training Pipeline Performance Profiler

This script profiles the entire training pipeline to identify the real bottlenecks.
"""

import time
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intraday_forecasting.predictor import IntradayPredictor
from intraday_forecasting.preprocessing.market_data import (
    load_intraday_data, prepare_intraday_for_training
)
from tf_predictor.core.utils import split_time_series

def time_function(func, *args, **kwargs):
    """Time a function execution and return result and duration."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    return result, duration

def profile_training_pipeline(data_path: str, num_rows: int = None):
    """Profile the complete training pipeline."""
    print("="*60)
    print("🔍 Training Pipeline Performance Profile")
    print("="*60)

    # 1. Load data
    print(f"\n📊 Loading data...")
    df, load_time = time_function(load_intraday_data, data_path)

    if num_rows:
        df = df.head(num_rows)

    print(f"   Loaded {len(df)} rows in {load_time:.2f}s")

    # 2. Prepare data for training (includes feature engineering)
    print(f"\n🔄 Preparing data for training...")
    preparation_result, prep_time = time_function(
        prepare_intraday_for_training, df, 'close', '1h', 'timestamp', 'CRYPTO', True
    )
    df_processed = preparation_result['data']
    print(f"   Data preparation completed in {prep_time:.2f}s")

    # 3. Split data
    print(f"\n🔄 Splitting data...")
    split_result, split_time = time_function(
        split_time_series, df_processed, 200, 100
    )
    train_df, val_df, test_df = split_result
    print(f"   Data splitting completed in {split_time:.2f}s")
    print(f"   Train: {len(train_df)}, Val: {len(val_df) if val_df is not None else 0}, Test: {len(test_df) if test_df is not None else 0}")

    # 4. Initialize model
    print(f"\n🧠 Initializing predictor...")
    model, init_time = time_function(
        IntradayPredictor,
        target_column='close',
        timeframe='1h',
        country='CRYPTO',
        sequence_length=12,  # Small sequence length for testing
        d_token=32,
        n_layers=1,
        n_heads=2,
        dropout=0.1,
        verbose=True
    )
    print(f"   Model initialization completed in {init_time:.2f}s")

    # 5. Profile the feature preparation during training
    print(f"\n🔄 Profiling feature preparation in model.fit...")
    start_fit = time.time()

    # This is where the bottleneck likely occurs
    print("   Calling model.prepare_data on training set...")
    X_train, y_train = model.prepare_data(train_df, fit_scaler=True)
    prepare_train_time = time.time() - start_fit
    print(f"   Training data preparation: {prepare_train_time:.2f}s")

    if val_df is not None:
        print("   Calling model.prepare_data on validation set...")
        val_start = time.time()
        X_val, y_val = model.prepare_data(val_df, fit_scaler=False)
        prepare_val_time = time.time() - val_start
        print(f"   Validation data preparation: {prepare_val_time:.2f}s")
    else:
        prepare_val_time = 0

    # 6. Profile actual model training (just 2 epochs)
    print(f"\n🏋️ Training model (2 epochs only)...")
    training_start = time.time()
    model.fit(
        df=train_df,
        val_df=val_df,
        epochs=2,  # Just 2 epochs to test
        batch_size=32,
        learning_rate=1e-3,
        patience=10,
        verbose=True
    )
    training_time = time.time() - training_start
    print(f"   Model training (2 epochs): {training_time:.2f}s")

    # Summary
    total_time = (load_time + prep_time + split_time + init_time +
                  prepare_train_time + prepare_val_time + training_time)

    print(f"\n📊 Performance Summary:")
    print(f"   Data loading: {load_time:.2f}s")
    print(f"   Data preparation (features): {prep_time:.2f}s")
    print(f"   Data splitting: {split_time:.2f}s")
    print(f"   Model initialization: {init_time:.2f}s")
    print(f"   Training data prep (sequences): {prepare_train_time:.2f}s")
    print(f"   Validation data prep (sequences): {prepare_val_time:.2f}s")
    print(f"   Model training (2 epochs): {training_time:.2f}s")
    print(f"   Total time: {total_time:.2f}s")

    # Identify bottlenecks
    print(f"\n🐛 Bottleneck Analysis:")
    times = {
        'Training data prep (sequences)': prepare_train_time,
        'Model training (2 epochs)': training_time,
        'Data preparation (features)': prep_time,
        'Validation data prep (sequences)': prepare_val_time,
        'Data loading': load_time,
        'Data splitting': split_time,
        'Model initialization': init_time
    }

    sorted_times = sorted(times.items(), key=lambda x: x[1], reverse=True)
    for i, (step, duration) in enumerate(sorted_times):
        pct = (duration / total_time) * 100
        print(f"   #{i+1} {step}: {duration:.2f}s ({pct:.1f}%)")

    return total_time

if __name__ == "__main__":
    data_path = "/Users/chinmay/code/get_stock_data/downloaded_data/BTC-USD_1h_1y_20250915_124207.csv"

    # Test with progressively larger datasets
    for num_rows in [500, 1000, 2000]:
        print(f"\n{'='*20} Testing with {num_rows} rows {'='*20}")
        try:
            total_time = profile_training_pipeline(data_path, num_rows)
            print(f"✅ {num_rows} rows completed in {total_time:.2f}s")

            if num_rows == 2000:
                # Estimate full dataset time
                full_rows = 8761
                estimated_time = (total_time / num_rows) * full_rows
                print(f"\n📈 Estimated time for full dataset ({full_rows} rows): {estimated_time:.1f}s ({estimated_time/60:.1f} minutes)")

        except Exception as e:
            print(f"❌ Error during profiling with {num_rows} rows: {e}")
            import traceback
            traceback.print_exc()
            break