#!/usr/bin/env python3
"""
Comprehensive tests for tf_predictor core components.
"""

# import pytest  # Not available - use manual assertions
import pandas as pd
import numpy as np
import torch
from tf_predictor.core.predictor import TimeSeriesPredictor
from tf_predictor.core.utils import calculate_metrics, split_time_series
from tf_predictor.preprocessing.time_features import (
    create_date_features, create_sequences, create_lag_features, 
    create_rolling_features
)


class SimpleTestPredictor(TimeSeriesPredictor):
    """Simple test implementation of TimeSeriesPredictor."""
    
    def create_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """Simple feature engineering - just use numeric columns."""
        df_processed = df.copy()

        # Add date features if date column exists
        if 'date' in df_processed.columns:
            df_processed = create_date_features(df_processed, 'date')

        # Add some basic features
        if 'value' in df_processed.columns:
            df_processed['value_lag_1'] = df_processed['value'].shift(1)
            df_processed['value_ma_3'] = df_processed['value'].rolling(3).mean()

        # Fill NaN values
        df_processed = df_processed.fillna(0)

        # Create target variable(s) based on prediction horizon
        # Use target_column attribute since that's what's available in base class
        original_target = getattr(self, 'original_target_column', self.target_column)
        if self.prediction_horizon == 1:
            # Single horizon
            target_col = f"{original_target}_target_h1"
            df_processed[target_col] = df_processed[original_target].shift(-1)
            df_processed = df_processed.dropna(subset=[target_col])
        else:
            # Multi-horizon
            target_columns = []
            for h in range(1, self.prediction_horizon + 1):
                col_name = f"{original_target}_target_h{h}"
                df_processed[col_name] = df_processed[original_target].shift(-h)
                target_columns.append(col_name)
            df_processed = df_processed.dropna(subset=target_columns)

        return df_processed


def create_sample_timeseries(n_samples=100, start_date='2020-01-01'):
    """Create sample time series data for testing."""
    dates = pd.date_range(start=start_date, periods=n_samples, freq='D')
    
    # Generate synthetic time series with trend and noise
    np.random.seed(42)  # For reproducibility
    trend = np.linspace(100, 200, n_samples)
    noise = np.random.normal(0, 10, n_samples)
    seasonal = 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 30)  # 30-day cycle
    
    values = trend + seasonal + noise
    
    return pd.DataFrame({
        'date': dates,
        'value': values,
        'feature_1': np.random.uniform(-1, 1, n_samples),
        'feature_2': np.random.uniform(0, 100, n_samples)
    })


class TestTimeFeatures:
    """Test time series feature engineering functions."""
    
    def test_create_date_features(self):
        """Test date feature creation."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10, freq='D'),
            'value': range(10)
        })
        
        df_with_features = create_date_features(df, 'date')
        
        # Check that date features were created
        expected_features = ['year', 'month', 'day', 'dayofweek', 'quarter', 
                           'is_weekend', 'month_sin', 'month_cos', 
                           'dayofweek_sin', 'dayofweek_cos']
        
        for feature in expected_features:
            assert feature in df_with_features.columns, f"Missing feature: {feature}"
        
        # Check data types
        assert df_with_features['year'].dtype in [np.int64, np.int32]
        assert df_with_features['month_sin'].dtype == np.float64
        assert df_with_features['is_weekend'].dtype in [np.int64, np.int32]
    
    def test_create_sequences(self):
        """Test sequence creation for time series."""
        df = create_sample_timeseries(20)
        df_processed = create_date_features(df, 'date')
        
        sequence_length = 5
        sequences, targets = create_sequences(df_processed, sequence_length, 'value')
        
        # Check shapes
        expected_samples = len(df) - sequence_length
        assert sequences.shape[0] == expected_samples
        assert sequences.shape[1] == sequence_length
        assert targets.shape[0] == expected_samples
        
        # Check that sequences are sequential
        # First sequence should contain the first 5 rows of features
        numeric_cols = [col for col in df_processed.columns 
                       if col != 'value' and pd.api.types.is_numeric_dtype(df_processed[col])]
        
        assert sequences.shape[2] == len(numeric_cols)
    
    def test_create_lag_features(self):
        """Test lag feature creation."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        df_with_lags = create_lag_features(df, 'value', [1, 2, 3])
        
        # Check that lag features were created
        assert 'value_lag_1' in df_with_lags.columns
        assert 'value_lag_2' in df_with_lags.columns
        assert 'value_lag_3' in df_with_lags.columns
        
        # Check values
        assert df_with_lags['value_lag_1'].iloc[1] == 1  # Second row should have first row's value
        assert df_with_lags['value_lag_2'].iloc[2] == 1  # Third row should have first row's value
    
    def test_create_rolling_features(self):
        """Test rolling window feature creation."""
        df = pd.DataFrame({
            'value': range(10)
        })
        
        df_with_rolling = create_rolling_features(df, 'value', [3, 5])
        
        # Check that rolling features were created
        expected_features = ['value_rolling_mean_3', 'value_rolling_std_3',
                           'value_rolling_min_3', 'value_rolling_max_3',
                           'value_rolling_mean_5', 'value_rolling_std_5',
                           'value_rolling_min_5', 'value_rolling_max_5']
        
        for feature in expected_features:
            assert feature in df_with_rolling.columns, f"Missing feature: {feature}"
    


class TestUtils:
    """Test utility functions."""
    
    def test_calculate_metrics(self):
        """Test metric calculations."""
        y_true = np.array([100, 110, 105, 115, 120])
        y_pred = np.array([102, 108, 107, 113, 118])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # Check that all expected metrics are present
        expected_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        # Check that values are reasonable
        assert metrics['MAE'] > 0
        assert metrics['MSE'] > 0
        assert metrics['RMSE'] == np.sqrt(metrics['MSE'])
        assert 0 <= metrics['R2'] <= 1
        assert 0 <= metrics['Directional_Accuracy'] <= 100
    
    def test_split_time_series(self):
        """Test time series data splitting."""
        df = create_sample_timeseries(100)
        
        train_df, val_df, test_df = split_time_series(df, test_size=20, val_size=10)
        
        # Check sizes
        assert len(train_df) == 70  # 100 - 20 - 10
        assert len(val_df) == 10
        assert len(test_df) == 20
        
        # Check chronological order
        assert train_df['date'].max() <= val_df['date'].min()
        assert val_df['date'].max() <= test_df['date'].min()
    
    def test_split_time_series_no_validation(self):
        """Test time series splitting without validation set."""
        df = create_sample_timeseries(50)
        
        train_df, val_df, test_df = split_time_series(df, test_size=15, val_size=None)
        
        # Check sizes
        assert len(train_df) == 35  # 50 - 15
        assert val_df is None
        assert len(test_df) == 15


class TestTimeSeriesPredictor:
    """Test the abstract TimeSeriesPredictor base class."""
    
    def test_initialization(self):
        """Test predictor initialization."""
        predictor = SimpleTestPredictor(target_column='value', sequence_length=5)
        
        assert predictor.target_column == 'value'
        assert predictor.sequence_length == 5
        assert predictor.model is None
        assert predictor.feature_columns is None
    
    def test_prepare_features(self):
        """Test feature preparation and scaling."""
        df = create_sample_timeseries(50)
        predictor = SimpleTestPredictor(target_column='value', sequence_length=5)
        predictor.verbose = True
        
        # First call should fit the scaler
        df_processed = predictor.prepare_features(df, fit_scaler=True)
        
        # Check that features were created
        assert len(df_processed.columns) > len(df.columns)
        assert predictor.feature_columns is not None
        
        # Check that numeric features were scaled (mean ~0, std ~1)
        # Skip constant features like 'year' which have std=0
        numeric_cols = [col for col in predictor.feature_columns[:5]]  # Check first few
        for col in numeric_cols:
            if col in df_processed.columns:
                mean = df_processed[col].mean()
                std = df_processed[col].std()
                # Skip constant features (std = 0)
                if std > 1e-6:  
                    assert abs(mean) < 0.1, f"Column {col} not properly scaled (mean={mean})"
                    assert abs(std - 1.0) < 0.1, f"Column {col} not properly scaled (std={std})"
    
    def test_prepare_data(self):
        """Test data preparation for model training."""
        df = create_sample_timeseries(50)
        predictor = SimpleTestPredictor(target_column='value', sequence_length=5)
        
        X, y = predictor.prepare_data(df, fit_scaler=True)

        # Check shapes - account for sequence length and target shift
        # With sequence_length=5 and target shift of 1, we lose sequence_length + target_shift rows
        expected_samples = len(df) - predictor.sequence_length - predictor.prediction_horizon
        assert X.shape[0] == expected_samples
        assert X.shape[1] == predictor.sequence_length
        assert y.shape[0] == expected_samples
        
        # Check data types
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert X.dtype == torch.float32
        assert y.dtype == torch.float32
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        df = create_sample_timeseries(5)  # Too small for sequence_length=5
        predictor = SimpleTestPredictor(target_column='value', sequence_length=5)
        
        try:
            predictor.prepare_data(df, fit_scaler=True)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Need at least" in str(e)


def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    # Create sample data
    df = create_sample_timeseries(100)
    
    # Split data
    train_df, val_df, test_df = split_time_series(df, test_size=20, val_size=15)
    
    # Initialize predictor
    predictor = SimpleTestPredictor(target_column='value', sequence_length=5)
    predictor.verbose = False
    
    # Prepare training data
    X_train, y_train = predictor.prepare_data(train_df, fit_scaler=True)
    
    # Prepare validation data (using fitted scaler)
    X_val, y_val = predictor.prepare_data(val_df, fit_scaler=False)
    
    # Check that all data has consistent shapes
    assert X_train.shape[1] == X_val.shape[1]  # Same sequence length
    assert X_train.shape[2] == X_val.shape[2]  # Same number of features
    
    # Test metrics calculation
    y_pred_sample = torch.randn_like(y_val)
    metrics = calculate_metrics(y_val.numpy(), y_pred_sample.numpy())
    
    assert 'MAE' in metrics
    assert 'MAPE' in metrics
    assert 'R2' in metrics


if __name__ == '__main__':
    # Run basic tests
    print("🧪 Testing tf_predictor core components...")
    
    # Test time features
    print("  Testing time features...")
    test_time_features = TestTimeFeatures()
    test_time_features.test_create_date_features()
    test_time_features.test_create_sequences()
    test_time_features.test_create_lag_features()
    test_time_features.test_create_rolling_features()
    print("  ✅ Time features tests passed")
    
    # Test utils
    print("  Testing utils...")
    test_utils = TestUtils()
    test_utils.test_calculate_metrics()
    test_utils.test_split_time_series()
    test_utils.test_split_time_series_no_validation()
    print("  ✅ Utils tests passed")
    
    # Test predictor
    print("  Testing TimeSeriesPredictor...")
    test_predictor = TestTimeSeriesPredictor()
    test_predictor.test_initialization()
    test_predictor.test_prepare_features()
    test_predictor.test_prepare_data()
    test_predictor.test_insufficient_data()
    print("  ✅ TimeSeriesPredictor tests passed")
    
    # Test end-to-end
    print("  Testing end-to-end workflow...")
    test_end_to_end_workflow()
    print("  ✅ End-to-end workflow test passed")
    
    print("🎉 All tf_predictor tests passed!")