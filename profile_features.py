#!/usr/bin/env python3
"""
Feature Engineering Performance Profiler

This script profiles the intraday feature engineering pipeline to identify bottlenecks.
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path

# Import the feature engineering modules
from intraday_forecasting.preprocessing.market_data import load_intraday_data
from intraday_forecasting.preprocessing.intraday_features import create_intraday_features
from intraday_forecasting.preprocessing.timeframe_utils import prepare_intraday_data

def time_function(func, *args, **kwargs):
    """Time a function execution and return result and duration."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    return result, duration

def profile_feature_engineering(data_path: str, num_rows: int = None):
    """Profile the feature engineering pipeline."""
    print("="*60)
    print("🔍 Feature Engineering Performance Profile")
    print("="*60)

    # 1. Load data
    print(f"\n📊 Loading data from: {data_path}")
    df, load_time = time_function(load_intraday_data, data_path)

    if num_rows:
        df = df.head(num_rows)

    print(f"   Loaded {len(df)} rows in {load_time:.2f}s")

    # 2. Profile data preparation (resampling)
    print(f"\n🔄 Profiling data preparation...")
    df_prepared, prep_time = time_function(
        prepare_intraday_data, df, '1h', 'timestamp', 'CRYPTO'
    )
    print(f"   Data preparation: {prep_time:.2f}s ({len(df_prepared)} rows)")

    # 3. Profile individual feature creation steps
    print(f"\n⚙️ Profiling feature creation steps...")

    # Import individual feature functions
    from intraday_forecasting.preprocessing.intraday_features import (
        create_intraday_time_features,
        create_intraday_price_features,
        create_intraday_volume_features
    )
    from tf_predictor.preprocessing.time_features import (
        create_lag_features,
        create_rolling_features
    )

    df_working = df_prepared.copy()

    # Time features
    df_working, time_features_duration = time_function(
        create_intraday_time_features, df_working, 'timestamp', 'CRYPTO'
    )
    print(f"   Time features: {time_features_duration:.2f}s")

    # Price features
    df_working, price_features_duration = time_function(
        create_intraday_price_features, df_working
    )
    print(f"   Price features: {price_features_duration:.2f}s")

    # Volume features
    df_working, volume_features_duration = time_function(
        create_intraday_volume_features, df_working
    )
    print(f"   Volume features: {volume_features_duration:.2f}s")

    # Lag features (this might be slow)
    print("   Creating lag features...")
    lag_start = time.time()
    # Create lag features for close price only
    for lag in [1, 2, 3, 5]:
        df_working = create_lag_features(df_working, 'close', [lag])
    lag_duration = time.time() - lag_start
    print(f"   Lag features: {lag_duration:.2f}s")

    # Rolling features (this might be very slow)
    print("   Creating rolling features...")
    rolling_start = time.time()
    # Create rolling features for close price
    df_working = create_rolling_features(df_working, 'close', [5, 10, 20])
    rolling_duration = time.time() - rolling_start
    print(f"   Rolling features: {rolling_duration:.2f}s")

    # 4. Profile complete feature creation
    print(f"\n🎯 Profiling complete feature pipeline...")
    df_final, complete_duration = time_function(
        create_intraday_features, df_prepared, 'close', 'timestamp', 'CRYPTO', True
    )

    # Summary
    print(f"\n📊 Performance Summary:")
    print(f"   Data loading: {load_time:.2f}s")
    print(f"   Data preparation: {prep_time:.2f}s")
    print(f"   Time features: {time_features_duration:.2f}s")
    print(f"   Price features: {price_features_duration:.2f}s")
    print(f"   Volume features: {volume_features_duration:.2f}s")
    print(f"   Lag features: {lag_duration:.2f}s")
    print(f"   Rolling features: {rolling_duration:.2f}s")
    print(f"   Complete pipeline: {complete_duration:.2f}s")

    total_time = (load_time + prep_time + time_features_duration +
                  price_features_duration + volume_features_duration +
                  lag_duration + rolling_duration)

    print(f"   Individual sum: {total_time:.2f}s")
    print(f"   Rows processed: {len(df_final)}")
    print(f"   Features created: {len(df_final.columns)}")
    print(f"   Processing rate: {len(df_final)/complete_duration:.1f} rows/sec")

    # Identify bottlenecks
    print(f"\n🐛 Bottleneck Analysis:")
    times = {
        'Rolling features': rolling_duration,
        'Lag features': lag_duration,
        'Price features': price_features_duration,
        'Volume features': volume_features_duration,
        'Time features': time_features_duration,
        'Data preparation': prep_time,
        'Data loading': load_time
    }

    sorted_times = sorted(times.items(), key=lambda x: x[1], reverse=True)
    for i, (step, duration) in enumerate(sorted_times[:3]):
        print(f"   #{i+1} Slowest: {step} ({duration:.2f}s)")

    return df_final, complete_duration

if __name__ == "__main__":
    # Test with different dataset sizes
    data_path = "/Users/chinmay/code/get_stock_data/downloaded_data/BTC-USD_1h_1y_20250915_124207.csv"

    # Start with small dataset
    print("Testing with 500 rows...")
    try:
        df_500, time_500 = profile_feature_engineering(data_path, 500)
        print(f"\n✅ 500 rows completed in {time_500:.2f}s")

        print("\nTesting with 1000 rows...")
        df_1000, time_1000 = profile_feature_engineering(data_path, 1000)
        print(f"\n✅ 1000 rows completed in {time_1000:.2f}s")

        # Estimate full dataset time
        full_rows = 8761  # Known size
        estimated_time = (time_1000 / 1000) * full_rows
        print(f"\n📈 Estimated time for full dataset ({full_rows} rows): {estimated_time:.1f}s ({estimated_time/60:.1f} minutes)")

    except Exception as e:
        print(f"❌ Error during profiling: {e}")
        import traceback
        traceback.print_exc()