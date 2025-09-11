"""
Stock prediction using FT-Transformer.

A specialized wrapper for stock price prediction with OHLCV data,
including automatic feature engineering and model management.
"""

import pandas as pd
import numpy as np
from tf_predictor import TimeSeriesPredictor
from .preprocessing.stock_features import create_stock_features


class StockPredictor(TimeSeriesPredictor):
    """FT-Transformer wrapper for stock price prediction."""
    
    def __init__(
        self,
        target_column: str = 'close',
        sequence_length: int = 5,  # Number of historical days to use
        **ft_kwargs
    ):
        """
        Args:
            target_column: Which column to predict ('close', 'open', etc.)
            sequence_length: Number of historical days to use for prediction
            **ft_kwargs: FT-Transformer hyperparameters
        """
        super().__init__(
            target_column=target_column,
            sequence_length=sequence_length,
            **ft_kwargs
        )
    
    def create_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Create stock-specific features from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data and optional date column
            fit_scaler: Whether to fit the scaler (True for training data)
            
        Returns:
            processed_df: DataFrame with engineered features
        """
        return create_stock_features(df, self.target_column, self.verbose)