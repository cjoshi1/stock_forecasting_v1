"""
Stock prediction using FT-Transformer.

A specialized wrapper for stock price prediction with OHLCV data,
including automatic feature engineering and model management.
"""

import pandas as pd
import numpy as np
from typing import Optional
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
        asset_type: str = 'stock',  # 'stock' or 'crypto'
        group_column: Optional[str] = None,  # Column for group-based scaling
        **ft_kwargs
    ):
        """
        Args:
            target_column: Which column to predict ('close', 'open', etc.)
            sequence_length: Number of historical days to use for prediction
            use_essential_only: If True, only use essential features (volume, typical_price, seasonal)
            prediction_horizon: Number of steps ahead to predict (1 = next step)
            asset_type: Type of asset - 'stock' (5-day week) or 'crypto' (7-day week)
            group_column: Optional column for group-based scaling (e.g., 'symbol' for multi-stock datasets)
            **ft_kwargs: FT-Transformer hyperparameters
        """
        # Handle target column naming for single vs multi-horizon
        if prediction_horizon == 1:
            shifted_target_name = f"{target_column}_target_h1"
            super().__init__(
                target_column=shifted_target_name,  # Single target for training
                sequence_length=sequence_length,
                group_column=group_column,
                **ft_kwargs
            )
        else:
            # Multi-horizon: pass the base name, predictor will handle multiple targets
            super().__init__(
                target_column=target_column,  # Base target name, will be extended
                sequence_length=sequence_length,
                group_column=group_column,
                **ft_kwargs
            )

        # Store original target info
        self.original_target_column = target_column
        self.use_essential_only = use_essential_only
        self.prediction_horizon = prediction_horizon
        self.asset_type = asset_type
    
    def create_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Create stock-specific features from OHLCV data.

        Args:
            df: DataFrame with OHLCV data and optional date column
            fit_scaler: Whether to fit the scaler (True for training data)

        Returns:
            processed_df: DataFrame with engineered features
        """
        return create_stock_features(df, self.original_target_column, self.verbose, self.use_essential_only, self.prediction_horizon, self.asset_type)

    def predict_next_bars(self, df: pd.DataFrame, n_predictions: int = 1) -> pd.DataFrame:
        """
        Predict next N days for stock/crypto trading.

        For single-horizon (prediction_horizon=1): Returns simple predictions
        For multi-horizon (prediction_horizon>1): Returns predictions for each horizon as separate columns

        Args:
            df: DataFrame with recent stock data
            n_predictions: Number of future days to predict

        Returns:
            DataFrame with predicted values and dates
            - Single-horizon: columns [date, predicted_{target}]
            - Multi-horizon: columns [date, predicted_{target}_h1, predicted_{target}_h2, ...]
        """
        # Get base predictions
        predictions = self.predict(df)

        if len(predictions) == 0:
            return pd.DataFrame()

        # Generate future dates based on asset type
        last_date = df['date'].iloc[-1]

        if self.asset_type == 'crypto':
            # Crypto: 24/7 trading, use calendar days
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=n_predictions,
                freq='D'
            )
        else:
            # Stock: skip weekends, use business days
            future_dates = pd.bdate_range(
                start=last_date + pd.Timedelta(days=1),
                periods=n_predictions
            )

        # Handle single vs multi-horizon predictions
        if self.prediction_horizon == 1:
            # Single-horizon: predictions is 1D array
            result = pd.DataFrame({
                'date': future_dates,
                f'predicted_{self.original_target_column}': predictions[:n_predictions]
            })
        else:
            # Multi-horizon: predictions is 2D array (n_samples, n_horizons)
            # Create columns for each horizon
            result_dict = {'date': future_dates}

            for h in range(self.prediction_horizon):
                horizon_num = h + 1
                # Extract predictions for this specific horizon
                horizon_predictions = predictions[:n_predictions, h]
                result_dict[f'predicted_{self.original_target_column}_h{horizon_num}'] = horizon_predictions

            result = pd.DataFrame(result_dict)

        return result