"""
Stock forecasting application using FT-Transformer.

This application provides specialized tools for stock price prediction,
including stock-specific feature engineering and visualization.
"""

from .predictor import StockPredictor
from .preprocessing.stock_features import create_stock_features, create_technical_indicators
from .preprocessing.market_data import load_stock_data, validate_stock_data, create_sample_stock_data
from .visualization.stock_charts import create_comprehensive_plots, print_performance_summary

__version__ = "1.0.0"

__all__ = [
    "StockPredictor",
    "create_stock_features",
    "create_technical_indicators", 
    "load_stock_data",
    "validate_stock_data",
    "create_sample_stock_data",
    "create_comprehensive_plots",
    "print_performance_summary"
]