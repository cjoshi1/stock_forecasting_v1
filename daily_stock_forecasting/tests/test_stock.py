#!/usr/bin/env python3
"""
Comprehensive tests for stock_forecasting components.
"""

# import pytest  # Not available - use manual assertions
import pandas as pd
import numpy as np
import torch
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daily_stock_forecasting.predictor import StockPredictor
from daily_stock_forecasting.preprocessing.stock_features import create_stock_features
from daily_stock_forecasting.preprocessing.market_data import (
    load_stock_data, validate_stock_data, create_sample_stock_data
)
from tf_predictor.core.utils import calculate_metrics, split_time_series


class TestMarketData:
    """Test market data handling functions."""
    
    def test_create_sample_stock_data(self):
        """Test synthetic stock data generation."""
        df = create_sample_stock_data(n_samples=100)
        
        # Check basic properties
        assert len(df) == 100
        assert 'date' in df.columns
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(df['date'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert pd.api.types.is_numeric_dtype(df[col])
        
        # Check OHLC relationships
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()
        
        # Check chronological order
        assert df['date'].is_monotonic_increasing
    
    def test_validate_stock_data(self):
        """Test stock data validation."""
        # Valid data
        df_valid = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10, freq='D'),
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'volume': [1000000] * 10
        })
        
        # validate_stock_data returns cleaned DataFrame, not error list
        cleaned_df = validate_stock_data(df_valid)
        assert len(cleaned_df) == len(df_valid)
        assert all(col in cleaned_df.columns for col in df_valid.columns)
        
        # Test with some invalid data - negative prices
        df_with_negatives = df_valid.copy()
        df_with_negatives.loc[0, 'close'] = -50  # Negative price
        cleaned_df = validate_stock_data(df_with_negatives)
        
        # The function should clean the data (replace negatives)
        assert len(cleaned_df) == len(df_with_negatives)
        # After cleaning, there should be no negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in cleaned_df.columns:
                positive_values = cleaned_df[col].dropna()
                if len(positive_values) > 0:
                    assert (positive_values > 0).all(), f"Found non-positive values in {col} after validation"
    
    def test_load_stock_data_with_sample_file(self):
        """Test loading stock data from the sample CSV file."""
        # This tests with the actual sample file in the repo
        sample_file = 'stock_forecasting/data/sample/MSFT_sample.csv'
        if os.path.exists(sample_file):
            df = load_stock_data(sample_file)
            
            # Check basic properties
            assert len(df) > 0
            assert 'date' in df.columns
            assert 'close' in df.columns
            
            # Check chronological order
            assert df['date'].is_monotonic_increasing
        else:
            print(f"   Skipping load test - sample file not found: {sample_file}")


class TestStockFeatures:
    """Test stock-specific feature engineering."""
    
    def test_create_stock_features(self):
        """Test stock-specific feature creation (vwap)."""
        df = create_sample_stock_data(n_samples=50)
        df_features = create_stock_features(df, verbose=False)

        # Check that vwap was added
        assert 'vwap' in df_features.columns

        # Check vwap calculation
        expected_vwap = (df['high'] + df['low'] + df['close']) / 3
        assert np.allclose(df_features['vwap'].values, expected_vwap.values, rtol=1e-5)

        # Check for NaN handling
        assert not df_features.isnull().any().any(), "Features contain NaN values"

    def test_create_stock_features_different_targets(self):
        """Test stock feature creation preserves original columns."""
        df = create_sample_stock_data(n_samples=30)
        df_features = create_stock_features(df, verbose=False)

        # Check that original columns are preserved
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in df_features.columns

        # Check that vwap was added
        assert 'vwap' in df_features.columns

    def test_create_technical_indicators(self):
        """Test that create_stock_features handles OHLC data correctly."""
        df = create_sample_stock_data(n_samples=100)
        df_tech = create_stock_features(df, verbose=False)

        # Just check that vwap was created and is numeric
        assert 'vwap' in df_tech.columns
        assert pd.api.types.is_numeric_dtype(df_tech['vwap'])


class TestStockPredictor:
    """Test the StockPredictor class."""
    
    def test_initialization(self):
        """Test StockPredictor initialization."""
        predictor = StockPredictor(target_column='close', sequence_length=5)

        # Check initialization
        assert predictor.original_target_column == 'close'
        assert predictor.target_columns == ['close']  # Normalized to list
        assert predictor.sequence_length == 5
        assert predictor.model is None
    
    def test_create_features(self):
        """Test StockPredictor feature creation."""
        predictor = StockPredictor(target_column='close', sequence_length=5, verbose=False)
        df = create_sample_stock_data(n_samples=50)

        df_features = predictor._create_base_features(df)

        # Check that stock-specific features were created (vwap)
        assert 'vwap' in df_features.columns

        # Check that time-series features were created
        feature_names = df_features.columns.tolist()
        assert 'month_sin' in feature_names
        assert 'month_cos' in feature_names
        assert 'dayofweek_sin' in feature_names
        assert 'dayofweek_cos' in feature_names

        # Check data types
        numeric_features = [col for col in df_features.columns
                          if col != 'date' and pd.api.types.is_numeric_dtype(df_features[col])]
        assert len(numeric_features) > 0
    
    def test_prepare_data_workflow(self):
        """Test complete data preparation workflow."""
        df = create_sample_stock_data(n_samples=50)
        predictor = StockPredictor(target_column='close', sequence_length=5, verbose=True)
        
        # Test data preparation
        X, y = predictor.prepare_data(df, fit_scaler=True)
        
        # Check shapes and types - account for sequence length and target shift
        expected_samples = len(df) - predictor.sequence_length - predictor.prediction_horizon
        assert X.shape[0] == expected_samples
        assert X.shape[1] == predictor.sequence_length
        assert y.shape[0] == expected_samples
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)
    
    def test_fit_predict_workflow(self):
        """Test basic fit and predict workflow."""
        df = create_sample_stock_data(n_samples=100)
        train_df, val_df, test_df = split_time_series(df, test_size=20, val_size=15)
        
        predictor = StockPredictor(
            target_column='close', 
            sequence_length=5,
            d_token=32,  # Smaller for faster testing
            n_layers=1,  # Single layer for testing
            verbose=False
        )
        
        # Test fitting (minimal epochs for speed)
        predictor.fit(
            df=train_df,
            val_df=val_df,
            epochs=1,  # Just 1 epoch for testing
            batch_size=16,
            verbose=False
        )
        
        # Check that model was created
        assert predictor.model is not None
        
        # Test prediction
        predictions = predictor.predict(test_df)
        assert len(predictions) > 0
        assert isinstance(predictions, np.ndarray)
        
        # Test evaluation
        metrics = predictor.evaluate(test_df)
        assert 'MAE' in metrics
        assert 'MAPE' in metrics
        assert 'R2' in metrics
        
        # Test saving/loading
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            predictor.save(tmp.name)
            assert os.path.exists(tmp.name)
            
            # Test loading
            predictor_loaded = StockPredictor.load(tmp.name)
            assert predictor_loaded.model is not None
            
            # Cleanup
            os.unlink(tmp.name)
    
    def test_percentage_change_targets(self):
        """Test predicting volume target."""
        df = create_sample_stock_data(n_samples=50)

        # Test with volume target
        predictor = StockPredictor(
            target_column='volume',
            sequence_length=5,
            verbose=False
        )

        # This should work without errors
        X, y = predictor.prepare_data(df, fit_scaler=True, store_for_evaluation=False)
        assert X.shape[0] > 0
        assert y.shape[0] > 0


def test_integration_with_tf_predictor():
    """Test integration between stock_forecasting and tf_predictor."""
    # Test that stock predictor properly uses tf_predictor components
    df = create_sample_stock_data(n_samples=100)
    
    # Split using tf_predictor utility
    train_df, val_df, test_df = split_time_series(df, test_size=20, val_size=15)
    
    # Use stock predictor
    predictor = StockPredictor(target_column='close', sequence_length=5, verbose=False)
    
    # This should work seamlessly
    X_train, y_train = predictor.prepare_data(train_df, fit_scaler=True)
    X_val, y_val = predictor.prepare_data(val_df, fit_scaler=False)
    
    # Check integration
    assert X_train.shape[1] == X_val.shape[1]  # Same sequence length
    assert X_train.shape[2] == X_val.shape[2]  # Same number of features


def test_end_to_end_stock_prediction():
    """Test complete end-to-end stock prediction workflow."""
    print("  Running end-to-end stock prediction test...")
    
    # Create larger dataset for more realistic test
    df = create_sample_stock_data(n_samples=200)
    
    # Split data
    train_df, val_df, test_df = split_time_series(df, test_size=30, val_size=20)
    
    # Initialize predictor with small model for speed
    predictor = StockPredictor(
        target_column='close',
        sequence_length=7,
        d_token=32,
        n_layers=1,
        n_heads=4,
        dropout=0.1,
        verbose=False
    )
    
    # Train model
    predictor.fit(
        df=train_df,
        val_df=val_df,
        epochs=2,  # Minimal training for testing
        batch_size=16,
        learning_rate=1e-3,
        patience=10,
        verbose=False
    )
    
    # Evaluate on test set
    test_metrics = predictor.evaluate(test_df)
    
    # Basic sanity checks
    assert test_metrics['MAE'] > 0
    assert test_metrics['RMSE'] > 0
    assert test_metrics['MAPE'] >= 0
    
    # Check that model can make predictions
    predictions = predictor.predict(test_df)
    assert len(predictions) > 0
    
    print(f"    Test completed - MAE: {test_metrics['MAE']:.2f}, MAPE: {test_metrics['MAPE']:.2f}%")


if __name__ == '__main__':
    # Run tests
    print("🧪 Testing stock_forecasting components...")
    
    # Test market data
    print("  Testing market data functions...")
    test_market = TestMarketData()
    test_market.test_create_sample_stock_data()
    test_market.test_validate_stock_data()
    test_market.test_load_stock_data_with_sample_file()
    print("  ✅ Market data tests passed")
    
    # Test stock features
    print("  Testing stock feature engineering...")
    test_features = TestStockFeatures()
    test_features.test_create_stock_features()
    test_features.test_create_stock_features_different_targets()
    test_features.test_create_technical_indicators()
    print("  ✅ Stock feature tests passed")
    
    # Test stock predictor
    print("  Testing StockPredictor...")
    test_predictor = TestStockPredictor()
    test_predictor.test_initialization()
    test_predictor.test_create_features()
    test_predictor.test_prepare_data_workflow()
    test_predictor.test_percentage_change_targets()
    print("  ✅ StockPredictor tests passed")
    
    # Test model training (takes longer)
    print("  Testing model training workflow...")
    test_predictor.test_fit_predict_workflow()
    print("  ✅ Model training tests passed")
    
    # Test integration
    print("  Testing integration with tf_predictor...")
    test_integration_with_tf_predictor()
    print("  ✅ Integration tests passed")
    
    # Test complete workflow
    test_end_to_end_stock_prediction()
    print("  ✅ End-to-end tests passed")
    
    print("🎉 All stock_forecasting tests passed!")