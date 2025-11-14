"""
Integration tests for Rossmann forecasting module.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import pandas as pd
import numpy as np
from rossman_forecasting.predictor import RossmannPredictor
from rossman_forecasting.preprocessing import apply_rossmann_preprocessing
from rossman_forecasting.utils import rmspe, calculate_all_metrics


def create_sample_data(n_stores=3, n_days=100):
    """Create sample Rossmann-like data for testing."""
    dates = pd.date_range('2015-01-01', periods=n_days)
    stores = range(1, n_stores + 1)

    data = []
    for store in stores:
        for date in dates:
            data.append({
                'Store': store,
                'Date': date,
                'DayOfWeek': date.dayofweek + 1,
                'Sales': np.random.randint(1000, 8000),
                'Customers': np.random.randint(100, 1000),
                'Open': 1,
                'Promo': np.random.randint(0, 2),
                'StateHoliday': '0',
                'SchoolHoliday': 0
            })

    return pd.DataFrame(data)


def create_sample_store_data(n_stores=3):
    """Create sample store metadata."""
    data = []
    for store in range(1, n_stores + 1):
        data.append({
            'Store': store,
            'StoreType': np.random.choice(['a', 'b', 'c', 'd']),
            'Assortment': np.random.choice(['a', 'b', 'c']),
            'CompetitionDistance': np.random.randint(100, 10000),
            'CompetitionOpenSinceMonth': np.random.randint(1, 13),
            'CompetitionOpenSinceYear': np.random.randint(2010, 2015),
            'Promo2': np.random.randint(0, 2),
            'Promo2SinceWeek': np.random.randint(1, 53) if np.random.rand() > 0.5 else np.nan,
            'Promo2SinceYear': 2014 if np.random.rand() > 0.5 else np.nan,
            'PromoInterval': 'Jan,Apr,Jul,Oct' if np.random.rand() > 0.5 else np.nan
        })

    return pd.DataFrame(data)


def test_rmspe_metric():
    """Test RMSPE calculation."""
    y_true = np.array([100, 200, 300, 0, 500])
    y_pred = np.array([110, 190, 310, 50, 480])

    # RMSPE excludes zeros by default
    rmspe_val = rmspe(y_true, y_pred)
    assert rmspe_val > 0
    assert rmspe_val < 1  # Should be a reasonable percentage


def test_predictor_initialization():
    """Test RossmannPredictor initialization."""
    predictor = RossmannPredictor(
        target_column='Sales',
        sequence_length=7,
        prediction_horizon=1,
        d_token=32,
        n_layers=2,
        n_heads=4,
        verbose=False
    )

    assert predictor.target_column == 'Sales'
    assert predictor.sequence_length == 7
    assert predictor.prediction_horizon == 1


def test_feature_engineering():
    """Test Rossmann feature engineering."""
    # Create sample data
    sales_df = create_sample_data(n_stores=2, n_days=50)
    store_df = create_sample_store_data(n_stores=2)

    # Simple config
    config = {
        'version': 'test_v1',
        'competition': {'enabled': True, 'fill_distance_missing': 'median', 'calculate_months_since': True},
        'promotion': {'enabled': True, 'use_promo2': True},
        'holidays': {'enabled': True, 'encode_state_holiday': True},
        'store': {'enabled': True, 'encode_store_type': True, 'encode_assortment': True},
        'domain_features': {'sales_per_customer': True},
        'filtering': {'remove_closed_stores': True}
    }

    # Apply preprocessing
    processed = apply_rossmann_preprocessing(sales_df, store_df, config, verbose=False)

    # Check that data was processed
    assert len(processed) > 0
    assert 'MonthsSinceCompetition' in processed.columns
    assert 'SalesPerCustomer' in processed.columns


def test_end_to_end_simple():
    """Test end-to-end training with sample data."""
    # Create sample data
    train_df = create_sample_data(n_stores=2, n_days=80)
    val_df = create_sample_data(n_stores=2, n_days=20)
    store_df = create_sample_store_data(n_stores=2)

    # Apply preprocessing
    config = {
        'version': 'test_v1',
        'competition': {'enabled': True, 'fill_distance_missing': 'median', 'calculate_months_since': True},
        'promotion': {'enabled': False},
        'holidays': {'enabled': False},
        'store': {'enabled': True, 'encode_store_type': True, 'encode_assortment': True},
        'domain_features': {'sales_per_customer': True},
        'filtering': {'remove_closed_stores': True}
    }

    train_processed = apply_rossmann_preprocessing(train_df, store_df, config, verbose=False)
    val_processed = apply_rossmann_preprocessing(val_df, store_df, config, verbose=False)

    # Initialize predictor
    predictor = RossmannPredictor(
        target_column='Sales',
        sequence_length=5,
        prediction_horizon=1,
        group_columns='Store',
        d_token=32,
        n_layers=2,
        n_heads=4,
        verbose=False
    )

    # Train
    predictor.fit(train_processed, val_processed, epochs=2, verbose=False)

    # Predict
    predictions = predictor.predict(val_processed)

    # Check predictions
    assert len(predictions) > 0
    assert all(predictions >= 0)  # Sales should be non-negative

    # Evaluate
    metrics = predictor.evaluate(val_processed)

    # Check metrics
    assert 'RMSPE' in metrics or ('overall' in metrics and 'RMSPE' in metrics.get('overall', {}))


if __name__ == '__main__':
    # Run tests
    print("Running Rossmann forecasting tests...")
    print("-" * 60)

    test_rmspe_metric()
    print("✅ test_rmspe_metric passed")

    test_predictor_initialization()
    print("✅ test_predictor_initialization passed")

    test_feature_engineering()
    print("✅ test_feature_engineering passed")

    test_end_to_end_simple()
    print("✅ test_end_to_end_simple passed")

    print("-" * 60)
    print("✅ All tests passed!")
