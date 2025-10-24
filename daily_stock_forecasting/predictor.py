"""
Stock prediction using FT-Transformer.

A specialized wrapper for stock price prediction with OHLCV data,
including automatic feature engineering and model management.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Union
from tf_predictor import TimeSeriesPredictor
from .preprocessing.stock_features import create_stock_features


class StockPredictor(TimeSeriesPredictor):
    """FT-Transformer wrapper for stock price prediction."""

    def __init__(
        self,
        target_column: Union[str, list] = 'close',
        sequence_length: int = 5,  # Number of historical days to use
        prediction_horizon: int = 1,  # Number of steps ahead to predict
        asset_type: str = 'stock',  # 'stock' or 'crypto'
        model_type: str = 'ft',  # 'ft' or 'csn'
        group_column: Optional[str] = None,  # Column for group-based scaling
        **ft_kwargs
    ):
        """
        Args:
            target_column: Which column(s) to predict
                          - str: Single target (e.g., 'close')
                          - List[str]: Multiple targets (e.g., ['close', 'volume'])
            sequence_length: Number of historical days to use for prediction
            prediction_horizon: Number of steps ahead to predict (1 = next step)
            asset_type: Type of asset - 'stock' (5-day week) or 'crypto' (7-day week)
            model_type: Model architecture ('ft' for FT-Transformer, 'csn' for CSNTransformer)
            group_column: Optional column for group-based scaling (e.g., 'symbol' for multi-stock datasets)
            **ft_kwargs: Model hyperparameters (architecture-specific)
        """
        # Validate model type
        if model_type not in ['ft', 'csn']:
            raise ValueError(f"Unsupported model_type: {model_type}. Use: 'ft' or 'csn'")

        # Store original target info before any transformation
        self.original_target_column = target_column
        self.prediction_horizon = prediction_horizon
        self.asset_type = asset_type
        self.model_type = model_type

        # Initialize logger with a specific format
        self.logger = self._initialize_logger()

        # Initialize base predictor based on model_type
        if model_type == 'csn':
            from tf_predictor.core.csn_predictor import CSNPredictor
            # Initialize CSNPredictor directly
            CSNPredictor.__init__(
                self,
                target_column=target_column,
                sequence_length=sequence_length,
                prediction_horizon=prediction_horizon,
                group_column=group_column,
                **ft_kwargs
            )
        else:
            # Default to FT-Transformer
            # Pass target_column as-is to base class
            # The base TimeSeriesPredictor will handle normalization
            super().__init__(
                target_column=target_column,
                sequence_length=sequence_length,
                prediction_horizon=prediction_horizon,
                group_column=group_column,
                **ft_kwargs
            )
    
    def _initialize_logger(self):
        """Initialize and return a logger with a specific format."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    def create_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Create stock-specific features from OHLCV data.

        Args:
            df: DataFrame with OHLCV data and optional date column
            fit_scaler: Whether to fit the scaler (True for training data)

        Returns:
            processed_df: DataFrame with engineered features
        """
        return create_stock_features(
            df=df,
            target_column=self.original_target_column,
            verbose=self.verbose,
            prediction_horizon=self.prediction_horizon,
            asset_type=self.asset_type,
            group_column=self.group_column
        )

    def predict_next_bars(self, df: pd.DataFrame, n_predictions: int = 1) -> pd.DataFrame:
        """
        Predict next N days for stock/crypto trading.

        For single-target, single-horizon: Returns simple predictions
        For single-target, multi-horizon: Returns predictions for each horizon as separate columns
        For multi-target: Returns dict of predictions for each target

        Args:
            df: DataFrame with recent stock data
            n_predictions: Number of future days to predict

        Returns:
            DataFrame with predicted values and dates
            - Single-target, single-horizon: columns [date, predicted_{target}]
            - Single-target, multi-horizon: columns [date, predicted_{target}_h1, predicted_{target}_h2, ...]
            - Multi-target: columns [date, predicted_{target1}, predicted_{target2}, ...] or with _h{n} suffixes
        """
        try:
            # Get base predictions
            self.logger.info(f"Starting prediction for next {n_predictions} bars")
            predictions = self.predict(df)

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

            # Handle multi-target vs single-target
            if isinstance(predictions, dict):
                # Multi-target mode: predictions is a dict
                result_dict = {'date': future_dates}

                for target_name, target_preds in predictions.items():
                    if len(target_preds) == 0:
                        continue

                    if self.prediction_horizon == 1:
                        # Single-horizon: 1D array
                        result_dict[f'predicted_{target_name}'] = target_preds[:n_predictions]
                    else:
                        # Multi-horizon: 2D array
                        for h in range(self.prediction_horizon):
                            horizon_num = h + 1
                            horizon_predictions = target_preds[:n_predictions, h]
                            result_dict[f'predicted_{target_name}_h{horizon_num}'] = horizon_predictions

                result = pd.DataFrame(result_dict)
            else:
                # Single-target mode: predictions is an array
                if len(predictions) == 0:
                    return pd.DataFrame()

                if self.prediction_horizon == 1:
                    # Single-horizon: predictions is 1D array
                    result = pd.DataFrame({
                        'date': future_dates,
                        f'predicted_{self.original_target_column}': predictions[:n_predictions]
                    })
                else:
                    # Multi-horizon: predictions is 2D array (n_samples, n_horizons)
                    result_dict = {'date': future_dates}

                    for h in range(self.prediction_horizon):
                        horizon_num = h + 1
                        horizon_predictions = predictions[:n_predictions, h]
                        result_dict[f'predicted_{self.original_target_column}_h{horizon_num}'] = horizon_predictions

                    result = pd.DataFrame(result_dict)

            self.logger.info("Prediction completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Exception occurred during prediction for {n_predictions} bars", exc_info=True)
            raise RuntimeError("Prediction failed due to an unexpected error.") from e
