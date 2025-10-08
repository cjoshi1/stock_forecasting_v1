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
        use_essential_only: bool = False,  # Use only essential features
        prediction_horizon: int = 1,  # Number of steps ahead to predict
        **ft_kwargs
    ):
        """
        Args:
            target_column: Which column to predict ('close', 'open', etc.)
            sequence_length: Number of historical days to use for prediction
            use_essential_only: If True, only use essential features (volume, typical_price, seasonal)
            prediction_horizon: Number of steps ahead to predict (1 = next step)
            **ft_kwargs: FT-Transformer hyperparameters
        """
        # Handle target column naming for single vs multi-horizon
        if prediction_horizon == 1:
            shifted_target_name = f"{target_column}_target_h1"
            super().__init__(
                target_column=shifted_target_name,  # Single target for training
                sequence_length=sequence_length,
                **ft_kwargs
            )
        else:
            # Multi-horizon: pass the base name, predictor will handle multiple targets
            super().__init__(
                target_column=target_column,  # Base target name, will be extended
                sequence_length=sequence_length,
                **ft_kwargs
            )

        # Store original target info
        self.original_target_column = target_column
        self.use_essential_only = use_essential_only
        self.prediction_horizon = prediction_horizon
    
    def create_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Create stock-specific features from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data and optional date column
            fit_scaler: Whether to fit the scaler (True for training data)
            
        Returns:
            processed_df: DataFrame with engineered features
        """
        return create_stock_features(df, self.original_target_column, self.verbose, self.use_essential_only, self.prediction_horizon)