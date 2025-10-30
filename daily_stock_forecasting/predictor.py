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
        model_type: str = 'ft_transformer_cls',  # 'ft_transformer_cls' or 'csn_transformer_cls'
        group_columns: Optional[Union[str, list]] = None,  # Column(s) for group-based scaling
        categorical_columns: Optional[Union[str, list]] = None,  # Column(s) for categorical features
        scaler_type: str = 'standard',  # Scaler type for normalization
        use_lagged_target_features: bool = False,  # Include target in input sequences
        d_model: int = 128,  # Token embedding dimension (renamed from d_token)
        num_heads: int = 8,  # Number of attention heads (renamed from n_heads)
        num_layers: int = 3,  # Number of transformer layers (renamed from n_layers)
        dropout: float = 0.1,  # Dropout rate
        **kwargs
    ):
        """
        Args:
            target_column: Which column(s) to predict
                          - str: Single target (e.g., 'close')
                          - List[str]: Multiple targets (e.g., ['close', 'volume'])
            sequence_length: Number of historical days to use for prediction
            prediction_horizon: Number of steps ahead to predict (1 = next step)
            asset_type: Type of asset - 'stock' (5-day week) or 'crypto' (7-day week)
            model_type: Model architecture ('ft_transformer_cls' for FT-Transformer, 'csn_transformer_cls' for CSNTransformer)
            group_columns: Optional column(s) for group-based scaling (e.g., 'symbol' for multi-stock datasets)
            categorical_columns: Optional column(s) to encode and pass as categorical features
            scaler_type: Type of scaler ('standard', 'minmax', 'robust', 'maxabs', 'onlymax')
            use_lagged_target_features: Whether to include target columns in input sequences
            d_model: Token embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            **kwargs: Additional model hyperparameters
        """
        # Validate model type
        if model_type not in ['ft_transformer_cls', 'csn_transformer_cls']:
            raise ValueError(f"Unsupported model_type: {model_type}. Use: 'ft_transformer_cls' or 'csn_transformer_cls'")

        # Store original target info before any transformation
        self.original_target_column = target_column
        self.prediction_horizon = prediction_horizon
        self.asset_type = asset_type
        self.model_type = model_type

        # Initialize logger with a specific format
        self.logger = self._initialize_logger()

        # Initialize base predictor with model_type (no mapping needed - names match factory)
        super().__init__(
            target_column=target_column,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            group_columns=group_columns,
            categorical_columns=categorical_columns,
            model_type=model_type,
            scaler_type=scaler_type,
            use_lagged_target_features=use_lagged_target_features,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs
        )
    
    def _initialize_logger(self):
        """Initialize and return a logger with a specific format."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Prepare features by creating stock-specific features and time-series features.

        Args:
            df: DataFrame with OHLCV data and optional date column
            fit_scaler: Whether to fit the scaler (True for training data)

        Returns:
            processed_df: DataFrame with engineered features
        """
        # First create stock-specific features (technical indicators, etc.)
        # Note: group_columns could be a list, but create_stock_features expects single column or None
        # Pass first group column if available, otherwise None
        group_col_for_features = None
        if self.group_columns:
            group_col_for_features = self.group_columns[0] if isinstance(self.group_columns, list) else self.group_columns

        df_with_stock_features = create_stock_features(
            df=df,
            target_column=self.original_target_column,
            verbose=self.verbose,
            prediction_horizon=self.prediction_horizon,
            asset_type=self.asset_type,
            group_column=group_col_for_features
        )

        # Then call parent's prepare_features to add time-series features
        return super().prepare_features(df_with_stock_features, fit_scaler)

    def predict(self, df: pd.DataFrame, return_group_info: bool = False):
        """
        Override predict to store properly preprocessed dataframe for evaluation.

        The issue: StockPredictor.prepare_features() calls create_stock_features()
        which creates shifted targets, but then prepare_data() calls
        create_shifted_targets() AGAIN. We need to store the FINAL processed
        dataframe (after both preprocessing steps) for evaluation alignment.

        Args:
            df: DataFrame with raw stock data
            return_group_info: If True, return (predictions, group_indices)

        Returns:
            predictions: Array or dict of predictions (same as parent)
            group_indices: List of group values (only if return_group_info=True)
        """
        # Preprocess exactly as prepare_data() does it
        # Step 1: Call prepare_features (applies stock features + base features)
        df_processed = self.prepare_features(df.copy(), fit_scaler=False)

        # Step 2: Apply the same target shifting that prepare_data() does
        # This is critical - prepare_data() calls create_shifted_targets() which
        # overwrites the targets created by create_stock_features()
        from tf_predictor.preprocessing.time_features import create_shifted_targets
        group_col_for_shift = self.categorical_columns if self.categorical_columns else self.group_columns
        df_processed = create_shifted_targets(
            df_processed,
            target_column=self.target_columns,
            prediction_horizon=self.prediction_horizon,
            group_column=group_col_for_shift,
            verbose=False
        )

        # Store for evaluation - this now matches what prepare_data() produces
        self._last_processed_df = df_processed

        # Call parent predict (which will call prepare_data internally)
        return super().predict(df, return_group_info)

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
