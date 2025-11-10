"""
Stock prediction using FT-Transformer.

A specialized wrapper for stock price prediction with OHLCV data,
including automatic feature engineering and model management.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Union, List
from tf_predictor import TimeSeriesPredictor
from .preprocessing.stock_features import create_stock_features
from .preprocessing.technical_indicators import calculate_technical_indicators
from .preprocessing.return_features import calculate_forward_returns, get_return_column_names


class StockPredictor(TimeSeriesPredictor):
    """FT-Transformer wrapper for stock price prediction."""

    def __init__(
        self,
        target_column: Union[str, list] = 'close',
        sequence_length: int = 5,  # Number of historical days to use
        prediction_horizon: int = 1,  # Number of steps ahead to predict
        asset_type: str = 'stock',  # 'stock' or 'crypto'
        model_type: str = 'ft_transformer',  # 'ft_transformer' or 'csn_transformer'
        group_columns: Optional[Union[str, list]] = None,  # Column(s) for group-based scaling
        categorical_columns: Optional[Union[str, list]] = None,  # Column(s) for categorical features
        scaler_type: str = 'standard',  # Scaler type for normalization
        use_lagged_target_features: bool = False,  # Include target in input sequences
        use_return_forecasting: bool = False,  # Enable multi-target return forecasting mode
        return_horizons: Optional[List[int]] = None,  # Return horizons (default: [1,2,3,4,5])
        verbose: bool = False,  # Whether to print detailed processing information
        d_token: int = 128,  # Token embedding dimension
        n_heads: int = 8,  # Number of attention heads
        n_layers: int = 3,  # Number of transformer layers
        dropout: float = 0.1,  # Dropout rate
        **kwargs
    ):
        """
        Args:
            target_column: Which column(s) to predict
                          - str: Single target (e.g., 'close')
                          - List[str]: Multiple targets (e.g., ['close', 'volume'])
                          - Ignored if use_return_forecasting=True
            sequence_length: Number of historical days to use for prediction
            prediction_horizon: Number of steps ahead to predict (1 = next step)
                               - Ignored if use_return_forecasting=True
            asset_type: Type of asset - 'stock' (5-day week) or 'crypto' (7-day week)
            model_type: Model architecture ('ft_transformer' for FT-Transformer, 'csn_transformer' for CSN-Transformer)
            group_columns: Optional column(s) for group-based scaling (e.g., 'symbol' for multi-stock datasets)
            categorical_columns: Optional column(s) to encode and pass as categorical features
            scaler_type: Type of scaler ('standard', 'minmax', 'robust', 'maxabs', 'onlymax')
            use_lagged_target_features: Whether to include target columns in input sequences
            use_return_forecasting: Enable multi-target return forecasting mode
                                   - Automatically calculates technical indicators as features
                                   - Predicts forward returns at multiple horizons
            return_horizons: List of return horizons in days (default: [1,2,3,4,5])
            verbose: Whether to print detailed processing information
            d_token: Token embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
            **kwargs: Additional model hyperparameters
        """
        # Validate model type
        if model_type not in ['ft_transformer', 'csn_transformer']:
            raise ValueError(f"Unsupported model_type: {model_type}. Use: 'ft_transformer' or 'csn_transformer'")

        # Handle return forecasting mode
        self.use_return_forecasting = use_return_forecasting
        self.return_horizons = return_horizons if return_horizons is not None else [1, 2, 3, 4, 5]

        if use_return_forecasting:
            # Override target and prediction_horizon for return forecasting
            target_column = get_return_column_names(self.return_horizons)
            prediction_horizon = 1  # Returns are pre-calculated, so horizon=1

            if verbose:
                print(f"\n🎯 Return Forecasting Mode Enabled")
                print(f"   Targets: {target_column}")
                print(f"   Input Features: close, relative_volume, intraday_momentum, rsi_14, bb_position")

        # Store original target info before any transformation
        self.original_target_column = target_column
        self.prediction_horizon = prediction_horizon
        self.asset_type = asset_type
        self.model_type = model_type

        # Initialize logger with a specific format
        self.logger = self._initialize_logger()

        # Initialize base predictor with model_type
        super().__init__(
            target_column=target_column,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            group_columns=group_columns,
            categorical_columns=categorical_columns,
            model_type=model_type,
            scaler_type=scaler_type,
            use_lagged_target_features=use_lagged_target_features,
            verbose=verbose,
            d_token=d_token,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            **kwargs
        )
    
    def _initialize_logger(self):
        """Initialize and return a logger with a specific format."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

    def _create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Override base class to add stock-specific preprocessing before time-series features.

        Args:
            df: DataFrame with OHLCV data and optional date column

        Returns:
            processed_df: DataFrame with stock-specific and time-series features
        """
        if self.use_return_forecasting:
            # Return forecasting mode: calculate technical indicators and returns
            if self.verbose:
                print(f"\n🔧 Processing features for return forecasting...")

            # Determine group column to use (if any)
            group_col = None
            if self.group_columns:
                group_col = self.group_columns[0] if isinstance(self.group_columns, list) else self.group_columns

            # Step 1: Calculate technical indicators
            df_with_indicators = calculate_technical_indicators(
                df=df,
                group_column=group_col,
                verbose=self.verbose
            )

            # Step 2: Calculate forward returns (targets)
            df_with_returns = calculate_forward_returns(
                df=df_with_indicators,
                price_column='close',
                horizons=self.return_horizons,
                return_type='percentage',
                group_column=group_col,
                verbose=self.verbose
            )

            # Step 3: Select only the required input features
            # Input features: close, relative_volume, intraday_momentum, rsi_14, bb_position
            feature_columns = ['close', 'relative_volume', 'intraday_momentum', 'rsi_14', 'bb_position']
            return_columns = get_return_column_names(self.return_horizons)

            # Keep date and group columns if they exist
            columns_to_keep = []
            if 'date' in df_with_returns.columns:
                columns_to_keep.append('date')
            if group_col and group_col in df_with_returns.columns:
                columns_to_keep.append(group_col)

            columns_to_keep.extend(feature_columns)
            columns_to_keep.extend(return_columns)

            # Filter to only necessary columns
            df_filtered = df_with_returns[columns_to_keep].copy()

            if self.verbose:
                print(f"\n   Selected Features: {feature_columns}")
                print(f"   Target Columns: {return_columns}")
                print(f"   Total rows: {len(df_filtered)}")

                # Drop rows with NaN in features or targets (warm-up period)
                rows_before = len(df_filtered)
                df_filtered = df_filtered.dropna()
                rows_after = len(df_filtered)

                if rows_before != rows_after:
                    print(f"   ⚠️  Dropped {rows_before - rows_after} rows with NaN (indicator warm-up period)")
                    print(f"   ✅ Valid data: {rows_after} rows")

            # Call parent's _create_base_features to add time-series encoding
            return super()._create_base_features(df_filtered)

        else:
            # Standard mode: use original stock features
            df_with_stock_features = create_stock_features(
                df=df,
                verbose=self.verbose
            )

            # Then call parent's _create_base_features to add time-series features
            return super()._create_base_features(df_with_stock_features)

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
