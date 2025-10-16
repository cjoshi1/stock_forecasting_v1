import unittest
import pandas as pd
from daily_stock_forecasting.predictor import StockPredictor

class TestStockPredictor(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
            'open': [100, 102, 101, 105, 107],
            'high': [102, 104, 103, 107, 109],
            'low': [99, 101, 100, 104, 106],
            'close': [101, 103, 102, 106, 108],
            'volume': [1000, 1100, 1050, 1150, 1200]
        })
        self.predictor = StockPredictor()

    def test_create_features(self):
        # Test feature creation
        features = self.predictor.create_features(self.sample_data)
        self.assertIsInstance(features, pd.DataFrame)

    def test_predict_next_bars(self):
        # Test prediction
        predictions = self.predictor.predict_next_bars(self.sample_data, n_predictions=3)
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(len(predictions), 3)

if __name__ == '__main__':
    unittest.main()
