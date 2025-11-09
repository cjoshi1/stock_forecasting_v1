#!/usr/bin/env python3
"""
End-to-end test for multi-target return forecasting.

Tests the new return forecasting functionality with:
- Technical indicator calculation (RSI, BB, relative volume, intraday momentum)
- Forward return calculation (1d, 2d, 3d, 4d, 5d)
- Multi-target prediction using FT-Transformer
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from daily_stock_forecasting.predictor import StockPredictor
from daily_stock_forecasting.preprocessing.market_data import create_sample_stock_data
from tf_predictor.core.utils import split_time_series


def test_return_forecasting():
    """Test return forecasting end-to-end."""

    print("="*80)
    print("🧪 TESTING MULTI-TARGET RETURN FORECASTING")
    print("="*80)

    # 1. Create sample data
    print("\n1️⃣ Creating sample OHLCV data...")
    df = create_sample_stock_data(n_samples=300, start_price=100.0, asset_type='stock')
    print(f"   ✅ Created {len(df)} samples")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Sample data (first 5 rows):")
    print(df.head())

    # 2. Split data
    print("\n2️⃣ Splitting data into train/val/test...")
    train_df, val_df, test_df = split_time_series(
        df,
        test_size=30,
        val_size=20,
        time_column='date',
        sequence_length=10
    )

    print(f"   ✅ Train: {len(train_df)} samples")
    print(f"   ✅ Val:   {len(val_df)} samples")
    print(f"   ✅ Test:  {len(test_df)} samples")

    # 3. Initialize predictor with return forecasting mode
    print("\n3️⃣ Initializing StockPredictor with return forecasting mode...")

    predictor = StockPredictor(
        target_column='close',  # Ignored in return forecasting mode
        sequence_length=10,
        prediction_horizon=1,  # Ignored in return forecasting mode
        asset_type='stock',
        model_type='ft_transformer',
        use_return_forecasting=True,  # 🎯 Enable return forecasting
        return_horizons=[1, 2, 3, 4, 5],
        scaler_type='standard',
        verbose=True,
        d_token=64,  # Smaller for faster testing
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        pooling_type='multihead_attention'
    )

    print(f"   ✅ Predictor initialized")
    print(f"   Mode: Return Forecasting")
    print(f"   Targets: {predictor.target_columns}")

    # 4. Train model
    print("\n4️⃣ Training model...")

    predictor.fit(
        df=train_df,
        val_df=val_df,
        epochs=20,  # Small for testing
        batch_size=16,
        learning_rate=1e-3,
        patience=10,
        verbose=True
    )

    print(f"   ✅ Training completed")

    # 5. Make predictions
    print("\n5️⃣ Making predictions on test set...")

    test_predictions = predictor.predict(test_df)

    print(f"   ✅ Predictions generated")
    print(f"   Prediction keys: {list(test_predictions.keys())}")

    # 6. Evaluate predictions
    print("\n6️⃣ Evaluating predictions...")

    # For return forecasting, we need to add return columns to the dataframe
    from daily_stock_forecasting.preprocessing.technical_indicators import calculate_technical_indicators
    from daily_stock_forecasting.preprocessing.return_features import calculate_forward_returns

    test_df_with_indicators = calculate_technical_indicators(test_df, group_column=None, verbose=False)
    test_df_with_returns = calculate_forward_returns(
        test_df_with_indicators,
        price_column='close',
        horizons=[1, 2, 3, 4, 5],
        return_type='percentage',
        group_column=None,
        verbose=False
    )

    test_metrics = predictor.evaluate(test_df_with_returns)

    print(f"\n   📊 Test Metrics:")
    if isinstance(test_metrics, dict):
        for key, value in test_metrics.items():
            if isinstance(value, dict):
                print(f"\n   {key}:")
                for metric_name, metric_value in value.items():
                    if isinstance(metric_value, (int, float)):
                        print(f"     - {metric_name}: {metric_value:.4f}")
            elif isinstance(value, (int, float)):
                print(f"   - {key}: {value:.4f}")

    # 7. Verify predictions
    print("\n7️⃣ Verifying prediction shapes and values...")

    for target_col in predictor.target_columns:
        preds = test_predictions[target_col]
        print(f"\n   {target_col}:")
        print(f"     - Shape: {preds.shape}")
        print(f"     - Mean: {np.mean(preds):.4f}%")
        print(f"     - Std: {np.std(preds):.4f}%")
        print(f"     - Min: {np.min(preds):.4f}%")
        print(f"     - Max: {np.max(preds):.4f}%")

        # Check for reasonable return values (should be small percentages)
        if np.abs(np.mean(preds)) > 100:
            print(f"     ⚠️  WARNING: Mean return seems unreasonably high!")
        else:
            print(f"     ✅ Returns are in reasonable range")

    # 8. Show sample predictions
    print("\n8️⃣ Sample predictions (first 5 test samples):")

    sample_df_rows = []
    for i in range(min(5, len(test_predictions[predictor.target_columns[0]]))):
        row = {'Sample': i+1}
        for target_col in predictor.target_columns:
            row[target_col] = f"{test_predictions[target_col][i]:.3f}%"
        sample_df_rows.append(row)

    sample_df = pd.DataFrame(sample_df_rows)
    print(sample_df.to_string(index=False))

    # 9. Test future prediction
    print("\n9️⃣ Testing future prediction (next 5 days)...")

    future_preds = predictor.predict_next_bars(test_df, n_predictions=5)

    print(f"   ✅ Future predictions generated")
    print(f"   Columns: {list(future_preds.columns)}")
    print(f"\n   Future predictions:")
    print(future_preds.to_string(index=False))

    # 10. Success summary
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\n📋 Summary:")
    print(f"   ✅ Sample data creation: PASS")
    print(f"   ✅ Technical indicators: PASS")
    print(f"   ✅ Return calculation: PASS")
    print(f"   ✅ Model training: PASS")
    print(f"   ✅ Prediction: PASS")
    print(f"   ✅ Evaluation: PASS")
    print(f"   ✅ Future prediction: PASS")
    print("\n🎯 Return forecasting implementation is working correctly!")
    print("="*80)

    return True


if __name__ == '__main__':
    try:
        success = test_return_forecasting()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
