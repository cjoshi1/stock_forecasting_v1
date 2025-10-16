"""
Comprehensive test suite for intraday forecasting system.

Tests all components: data loading, feature engineering, model training, 
and prediction functionality.
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from intraday_forecasting import (
    IntradayPredictor,
    create_sample_intraday_data,
    load_intraday_data,
    validate_intraday_data,
    prepare_intraday_for_training,
    create_intraday_features,
    create_intraday_time_features,
    create_intraday_price_features,
    create_intraday_volume_features,
    filter_market_hours,
    resample_ohlcv,
    get_timeframe_config,
    get_supported_timeframes,
    get_country_market_hours
)
from tf_predictor.core.utils import split_time_series


class TestIntradayDataGeneration(unittest.TestCase):
    """Test intraday data generation and validation."""
    
    def test_create_sample_intraday_data(self):
        """Test sample data generation."""
        df = create_sample_intraday_data(n_days=2)
        
        # Check basic structure
        self.assertGreater(len(df), 0)
        self.assertIn('timestamp', df.columns)
        self.assertIn('open', df.columns)
        self.assertIn('high', df.columns)
        self.assertIn('low', df.columns)
        self.assertIn('close', df.columns)
        self.assertIn('volume', df.columns)
        
        # Check OHLC relationships
        self.assertTrue((df['high'] >= df['low']).all())
        self.assertTrue((df['high'] >= df['open']).all())
        self.assertTrue((df['high'] >= df['close']).all())
        self.assertTrue((df['low'] <= df['open']).all())
        self.assertTrue((df['low'] <= df['close']).all())
        
        # Check positive values
        self.assertTrue((df['open'] > 0).all())
        self.assertTrue((df['high'] > 0).all())
        self.assertTrue((df['low'] > 0).all())
        self.assertTrue((df['close'] > 0).all())
        self.assertTrue((df['volume'] >= 0).all())
        
        # Check timestamps are during market hours
        market_hours = df[(df['timestamp'].dt.time >= pd.to_datetime('09:30').time()) & 
                         (df['timestamp'].dt.time < pd.to_datetime('16:00').time())]
        self.assertEqual(len(market_hours), len(df))
    
    def test_create_sample_with_oi(self):
        """Test sample data generation with open interest."""
        df = create_sample_intraday_data(n_days=1, include_oi=True)
        
        self.assertIn('open_interest', df.columns)
        self.assertTrue((df['open_interest'] >= 0).all())
    
    def test_validate_intraday_data(self):
        """Test data validation functionality."""
        # Create valid data
        df = create_sample_intraday_data(n_days=1)
        df_validated = validate_intraday_data(df)
        
        self.assertGreater(len(df_validated), 0)
        
        # Test with invalid data
        df_invalid = df.copy()
        df_invalid.loc[0, 'high'] = df_invalid.loc[0, 'low'] - 1  # Invalid OHLC
        df_validated_invalid = validate_intraday_data(df_invalid)
        
        # Should remove invalid row
        self.assertLess(len(df_validated_invalid), len(df_invalid))


class TestTimeframeUtils(unittest.TestCase):
    """Test timeframe utilities."""
    
    def test_get_supported_timeframes(self):
        """Test getting supported timeframes."""
        timeframes = get_supported_timeframes()
        
        self.assertIn('1min', timeframes)
        self.assertIn('5min', timeframes)
        self.assertIn('15min', timeframes)
        self.assertIn('1h', timeframes)
    
    
    def test_get_country_market_hours(self):
        """Test getting country market hours."""
        us_config = get_country_market_hours('US')
        india_config = get_country_market_hours('INDIA')
        
        # Check US configuration
        self.assertEqual(us_config['open'].hour, 9)
        self.assertEqual(us_config['open'].minute, 30)
        self.assertEqual(us_config['close'].hour, 16)
        self.assertEqual(us_config['close'].minute, 0)
        
        # Check India configuration  
        self.assertEqual(india_config['open'].hour, 9)
        self.assertEqual(india_config['open'].minute, 15)
        self.assertEqual(india_config['close'].hour, 15)
        self.assertEqual(india_config['close'].minute, 30)
    
    def test_get_timeframe_config(self):
        """Test timeframe configuration."""
        for timeframe in ['1min', '5min', '15min', '1h']:
            config = get_timeframe_config(timeframe)
            
            self.assertIn('resample_rule', config)
            self.assertIn('sequence_length', config)
            self.assertIn('description', config)
            self.assertGreater(config['sequence_length'], 0)
    
    def test_filter_market_hours(self):
        """Test market hours filtering."""
        # Create data with pre-market and after-hours
        timestamps = pd.date_range('2023-01-01 08:00', '2023-01-01 18:00', freq='1H')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'close': [100] * len(timestamps)
        })
        
        # Test US market hours
        df_us = filter_market_hours(df, country='US')
        self.assertTrue((df_us['timestamp'].dt.hour >= 9).all())
        self.assertTrue((df_us['timestamp'].dt.hour < 16).all())
        
        # Test India market hours
        df_india = filter_market_hours(df, country='INDIA')
        self.assertTrue((df_india['timestamp'].dt.hour >= 9).all())
        self.assertTrue((df_india['timestamp'].dt.hour < 16).all())  # Less than 4 PM
    
    def test_resample_ohlcv(self):
        """Test OHLCV resampling."""
        df = create_sample_intraday_data(n_days=1)
        
        # Test 5-minute resampling
        df_5min = resample_ohlcv(df, '5min')
        
        self.assertLess(len(df_5min), len(df))  # Should have fewer bars
        self.assertIn('open', df_5min.columns)
        self.assertIn('high', df_5min.columns)
        self.assertIn('low', df_5min.columns)
        self.assertIn('close', df_5min.columns)
        self.assertIn('volume', df_5min.columns)
        
        # Check OHLC relationships still valid
        self.assertTrue((df_5min['high'] >= df_5min['low']).all())
        self.assertTrue((df_5min['high'] >= df_5min['open']).all())
        self.assertTrue((df_5min['high'] >= df_5min['close']).all())


class TestIntradayFeatures(unittest.TestCase):
    """Test intraday feature engineering."""
    
    def setUp(self):
        """Set up test data."""
        self.df = create_sample_intraday_data(n_days=2)
    
    def test_create_intraday_time_features(self):
        """Test time-based feature creation."""
        # Test for both countries
        for country in ['US', 'INDIA']:
            with self.subTest(country=country):
                df_time = create_intraday_time_features(self.df, country=country)
                
                # Check new time features exist
                expected_features = [
                    'hour', 'minute', 'dayofweek',
                    'minutes_since_open', 'minutes_until_close',
                    'is_market_open', 'is_opening_hour', 'is_lunch_hour',
                    'is_closing_hour', 'is_power_hour', 'is_weekend',
                    'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
                    'dayofweek_sin', 'dayofweek_cos'
                ]
                
                for feature in expected_features:
                    self.assertIn(feature, df_time.columns, f"Missing feature: {feature}")
                
                # Check value ranges
                self.assertTrue((df_time['hour'] >= 0).all())
                self.assertTrue((df_time['hour'] <= 23).all())
                self.assertTrue((df_time['minute'] >= 0).all())
                self.assertTrue((df_time['minute'] <= 59).all())
                self.assertTrue((df_time['is_market_open'].isin([0, 1])).all())
    
    def test_create_intraday_price_features(self):
        """Test price-based feature creation."""
        df_price = create_intraday_price_features(self.df)
        
        expected_features = [
            'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
            'range_pct', 'typical_price', 'volume_ratio',
            'volatility_5', 'volatility_15', 'momentum_5', 'momentum_15'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, df_price.columns, f"Missing feature: {feature}")
        
        # Check ratio features are positive
        self.assertTrue((df_price['high_low_ratio'] >= 1.0).all())
        self.assertTrue((df_price['close_open_ratio'] > 0).all())
    
    def test_create_intraday_volume_features(self):
        """Test volume-based feature creation."""
        df_volume = create_intraday_volume_features(self.df)
        
        expected_features = [
            'volume_sma_5', 'volume_sma_20', 'volume_momentum',
            'volume_rate', 'price_volume_trend', 'volume_percentile'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, df_volume.columns, f"Missing feature: {feature}")
        
        # Check volume features are reasonable
        self.assertTrue((df_volume['volume_sma_5'] >= 0).all())
        self.assertTrue((df_volume['volume_sma_20'] >= 0).all())
    
    def test_create_comprehensive_features(self):
        """Test comprehensive feature creation."""
        for country in ['US', 'INDIA']:
            with self.subTest(country=country):
                df_features = create_intraday_features(self.df, country=country, verbose=False)
                
                # Should have significantly more features than original
                original_features = len(self.df.columns)
                new_features = len(df_features.columns)
                self.assertGreater(new_features, original_features * 5)
                
                # Should not have NaN values
                self.assertEqual(df_features.isnull().sum().sum(), 0)


class TestIntradayPredictor(unittest.TestCase):
    """Test IntradayPredictor functionality."""
    
    def setUp(self):
        """Set up test data and predictor."""
        self.df = create_sample_intraday_data(n_days=5)
        
        # Prepare data
        result = prepare_intraday_for_training(
            self.df, target_column='close', timeframe='5min', country='US', verbose=False
        )
        self.df_processed = result['data']
        
        # Split data
        self.train_df, self.val_df, self.test_df = split_time_series(
            self.df_processed, test_size=20, val_size=15
        )
        
        # Initialize predictor
        self.predictor = IntradayPredictor(
            target_column='close',
            timeframe='5min',
            country='US',  # Test with US market
            d_token=32,  # Small for fast testing
            n_layers=2,
            n_heads=4,
            verbose=False
        )
    
    def test_predictor_initialization(self):
        """Test predictor initialization."""
        self.assertEqual(self.predictor.target_column, 'close')
        self.assertEqual(self.predictor.timeframe, '5min')
        self.assertEqual(self.predictor.country, 'US')
        self.assertIsNotNone(self.predictor.timeframe_config)
        self.assertIsNotNone(self.predictor.market_config)
    
    def test_get_timeframe_info(self):
        """Test timeframe information retrieval."""
        info = self.predictor.get_timeframe_info()
        
        self.assertIn('timeframe', info)
        self.assertIn('description', info)
        self.assertIn('sequence_length', info)
        self.assertIn('country', info)
        self.assertIn('market_hours', info)
        self.assertIn('timezone', info)
        self.assertEqual(info['timeframe'], '5min')
        self.assertEqual(info['country'], 'US')
    
    def test_create_features(self):
        """Test feature creation through predictor."""
        df_features = self.predictor.create_features(self.train_df, fit_scaler=True)
        
        self.assertGreater(len(df_features.columns), len(self.train_df.columns))
        self.assertEqual(len(df_features), len(self.train_df))
    
    def test_model_training(self):
        """Test model training."""
        if len(self.train_df) < 50:  # Skip if insufficient data
            self.skipTest("Insufficient training data")
        
        # Train with minimal epochs for speed
        self.predictor.fit(
            df=self.train_df,
            val_df=self.val_df,
            epochs=3,
            batch_size=8,
            verbose=False
        )
        
        self.assertIsNotNone(self.predictor.model)
        self.assertTrue(hasattr(self.predictor, 'feature_columns'))
    
    def test_predictions(self):
        """Test prediction functionality."""
        if len(self.train_df) < 50:
            self.skipTest("Insufficient training data")
        
        # Train model
        self.predictor.fit(
            df=self.train_df,
            val_df=self.val_df,
            epochs=3,
            batch_size=8,
            verbose=False
        )
        
        # Make predictions
        predictions = self.predictor.predict(self.test_df)
        
        self.assertGreater(len(predictions), 0)
        self.assertTrue(all(np.isfinite(predictions)))
    
    def test_predict_next_bars(self):
        """Test future prediction functionality."""
        if len(self.train_df) < 50:
            self.skipTest("Insufficient training data")
        
        # Train model
        self.predictor.fit(
            df=self.train_df,
            val_df=self.val_df,
            epochs=3,
            batch_size=8,
            verbose=False
        )
        
        # Predict next bars
        future_df = self.predictor.predict_next_bars(self.test_df, n_predictions=3)

        self.assertEqual(len(future_df), 3)
        self.assertIn(self.predictor.timestamp_col, future_df.columns)
        self.assertIn(f'predicted_{self.predictor.original_target_column}', future_df.columns)

    def test_logger_initialization(self):
        """Test logger initialization."""
        self.assertIsNotNone(self.predictor.logger)
        self.assertEqual(self.predictor.logger.name, 'intraday_forecasting.predictor')
        """Test future prediction functionality."""
        if len(self.train_df) < 50:
            self.skipTest("Insufficient training data")
        
        # Train model
        self.predictor.fit(
            df=self.train_df,
            val_df=self.val_df,
            epochs=3,
            batch_size=8,
            verbose=False
        )
        
        # Predict next bars
        future_df = self.predictor.predict_next_bars(self.test_df, n_predictions=3)

        self.assertEqual(len(future_df), 3)
        self.assertIn(self.predictor.timestamp_col, future_df.columns)
        self.assertIn(f'predicted_{self.predictor.original_target_column}', future_df.columns)
    
    def test_evaluation_metrics(self):
        """Test model evaluation."""
        if len(self.train_df) < 50:
            self.skipTest("Insufficient training data")
        
        # Train model
        self.predictor.fit(
            df=self.train_df,
            val_df=self.val_df,
            epochs=3,
            batch_size=8,
            verbose=False
        )
        
        # Evaluate
        metrics = self.predictor.evaluate(self.test_df)
        
        expected_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'R2']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertFalse(np.isnan(metrics[metric]))


class TestIntegration(unittest.TestCase):
    """Test end-to-end integration."""
    
    def test_full_pipeline(self):
        """Test complete pipeline from data to predictions."""
        # 1. Generate data
        df = create_sample_intraday_data(n_days=3)
        
        # 2. Prepare for training
        result = prepare_intraday_for_training(
            df, target_column='close', timeframe='5min', verbose=False
        )
        df_processed = result['data']
        
        # 3. Split data
        train_df, val_df, test_df = split_time_series(
            df_processed, test_size=15, val_size=10
        )
        
        if len(train_df) < 30:
            self.skipTest("Insufficient data for integration test")
        
        # 4. Train model
        predictor = IntradayPredictor(
            target_column='close',
            timeframe='5min',
            d_token=16,  # Very small for speed
            n_layers=1,
            verbose=False
        )
        
        predictor.fit(
            df=train_df,
            val_df=val_df,
            epochs=2,
            batch_size=4,
            verbose=False
        )
        
        # 5. Make predictions
        predictions = predictor.predict(test_df)
        
        # 6. Evaluate
        metrics = predictor.evaluate(test_df)
        
        # Verify results
        self.assertGreater(len(predictions), 0)
        self.assertIn('MAPE', metrics)
        self.assertFalse(np.isnan(metrics['MAPE']))
    
    def test_multiple_timeframes(self):
        """Test training models on different timeframes."""
        df = create_sample_intraday_data(n_days=7)
        
        timeframes = ['5min', '15min']  # Test subset for speed
        
        for timeframe in timeframes:
            with self.subTest(timeframe=timeframe):
                # Prepare data
                result = prepare_intraday_for_training(
                    df, target_column='close', timeframe=timeframe, country='US', verbose=False
                )
                df_processed = result['data']
                
                if len(df_processed) < 50:
                    continue  # Skip if insufficient data
                
                # Split and train
                train_df, val_df, test_df = split_time_series(
                    df_processed, test_size=10, val_size=8
                )
                
                predictor = IntradayPredictor(
                    target_column='close',
                    timeframe=timeframe,
                    country='US',
                    d_token=16,
                    n_layers=1,
                    verbose=False
                )
                
                predictor.fit(
                    df=train_df, val_df=val_df,
                    epochs=2, batch_size=4,
                    verbose=False
                )
                
                predictions = predictor.predict(test_df)
                self.assertGreater(len(predictions), 0)
    
    def test_multiple_countries(self):
        """Test training models for different countries."""
        df = create_sample_intraday_data(n_days=5)
        
        countries = ['US', 'INDIA']
        
        for country in countries:
            with self.subTest(country=country):
                # Prepare data for specific country
                result = prepare_intraday_for_training(
                    df, target_column='close', timeframe='5min', country=country, verbose=False
                )
                df_processed = result['data']
                
                if len(df_processed) < 30:
                    continue  # Skip if insufficient data
                
                # Split and train
                train_df, val_df, test_df = split_time_series(
                    df_processed, test_size=8, val_size=6
                )
                
                predictor = IntradayPredictor(
                    target_column='close',
                    timeframe='5min',
                    country=country,
                    d_token=16,
                    n_layers=1,
                    verbose=False
                )
                
                # Verify country is properly set
                self.assertEqual(predictor.country, country)
                info = predictor.get_timeframe_info()
                self.assertEqual(info['country'], country)
                
                predictor.fit(
                    df=train_df, val_df=val_df,
                    epochs=2, batch_size=4,
                    verbose=False
                )
                
                predictions = predictor.predict(test_df)
                self.assertGreater(len(predictions), 0)


class TestDataIO(unittest.TestCase):
    """Test data input/output functionality."""
    
    def test_load_intraday_data(self):
        """Test loading intraday data from CSV."""
        # Create temporary CSV file
        df = create_sample_intraday_data(n_days=1)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Load the data
            loaded_df = load_intraday_data(temp_file)
            
            # Verify structure
            self.assertEqual(len(loaded_df), len(df))
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded_df['timestamp']))
            
            # Check required columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                self.assertIn(col, loaded_df.columns)
                
        finally:
            os.unlink(temp_file)
    
    def test_load_nonexistent_file(self):
        """Test error handling for non-existent files."""
        with self.assertRaises(FileNotFoundError):
            load_intraday_data('nonexistent_file.csv')


def run_specific_test(test_class, test_method=None):
    """Run a specific test class or method."""
    if test_method:
        suite = unittest.TestSuite()
        suite.addTest(test_class(test_method))
    else:
        suite = unittest.TestLoader().loadTestsFromTestClass(test_class)
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == '__main__':
    # Run all tests
    print("="*60)
    print("🧪 Running Intraday Forecasting Test Suite")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestIntradayDataGeneration,
        TestTimeframeUtils,
        TestIntradayFeatures,
        TestIntradayPredictor,
        TestIntegration,
        TestDataIO
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.splitlines()[-1]}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.splitlines()[-1]}")
    
    print(f"{'='*60}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
