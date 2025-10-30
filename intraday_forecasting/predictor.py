"""
IntradayPredictor class for high-frequency trading forecasting.

Extends the generic TimeSeriesPredictor with intraday-specific feature engineering
and timeframe-aware functionality.
"""

import pandas as pd
from typing import Optional, Union

# Import base predictor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tf_predictor import TimeSeriesPredictor

# Import intraday-specific modules
from .preprocessing.intraday_features import create_intraday_features
from .preprocessing.timeframe_utils import get_timeframe_config, get_country_market_hours, validate_country


class IntradayPredictor(TimeSeriesPredictor):
    """
    Intraday-specific time series predictor using FT-Transformer.
    
    Handles minute-level to hourly forecasting with intraday market patterns,
    time-of-day effects, and market microstructure features.
    """
    
    def __init__(self, target_column: Union[str, list] = 'close', timeframe: str = '5min', model_type: str = 'ft_transformer_cls',
                 timestamp_col: str = 'timestamp', country: str = 'US',
                 prediction_horizon: int = 1, group_columns: Optional[Union[str, list]] = None,
                 categorical_columns: Optional[Union[str, list]] = None,
                 scaler_type: str = 'standard',
                 use_lagged_target_features: bool = False,
                 lag_periods: list = None,
                 d_model: int = 128, num_heads: int = 8, num_layers: int = 3,
                 dropout: float = 0.1, verbose: bool = False, **kwargs):
        """
        Initialize IntradayPredictor.

        Args:
            target_column: Column(s) to predict
                          - str: Single target (e.g., 'close')
                          - List[str]: Multiple targets (e.g., ['close', 'volume'])
            timeframe: Trading timeframe ('1min', '5min', '15min', '1h')
            model_type: Model architecture ('ft_transformer_cls' for FT-Transformer, 'csn_transformer_cls' for CSNTransformer)
            timestamp_col: Name of timestamp column
            country: Country code ('US' or 'INDIA')
            prediction_horizon: Number of steps ahead to predict (1=single, >1=multi-horizon)
            group_columns: Optional column(s) for group-based scaling (e.g., 'symbol' for multi-stock datasets)
            categorical_columns: Optional column(s) to encode and pass as categorical features
            scaler_type: Type of scaler ('standard', 'minmax', 'robust', 'maxabs', 'onlymax')
            use_lagged_target_features: Whether to include target columns in input sequences
            lag_periods: List of lag periods for target features (e.g., [1, 2, 3, 5, 10])
            d_model: Token embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            verbose: Whether to print detailed logs
            **kwargs: Additional arguments passed to base predictor
        """
        # Validate country
        if not validate_country(country):
            raise ValueError(f"Unsupported country: {country}. Use: US or INDIA")

        # Validate model type
        if model_type not in ['ft_transformer_cls', 'csn_transformer_cls']:
            raise ValueError(f"Unsupported model_type: {model_type}. Use: 'ft_transformer_cls' or 'csn_transformer_cls'")

        # Get timeframe-specific configuration
        self.timeframe = timeframe
        self.timestamp_col = timestamp_col
        self.country = country
        config = get_timeframe_config(timeframe)
        market_config = get_country_market_hours(country)

        # Use timeframe-specific sequence length if not provided
        sequence_length = kwargs.pop('sequence_length', config['sequence_length'])

        # Store model type for later use
        self.model_type = model_type

        # Store original target info before any transformation
        self.original_target_column = target_column

        # Normalize target_column to list for processing
        if isinstance(target_column, str):
            target_columns_list = [target_column]
            is_multi_target = False
        else:
            target_columns_list = list(target_column)
            is_multi_target = len(target_columns_list) > 1

        # Handle target column naming for multi-target support
        # For the base class, we pass the target_column as-is (str or list)
        # The base TimeSeriesPredictor will handle the normalization
        shifted_target_name = target_column

        # Initialize base predictor with model_type (no mapping needed - names match factory)
        super().__init__(
            target_column=shifted_target_name,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            group_columns=group_columns,
            categorical_columns=categorical_columns,
            model_type=model_type,
            scaler_type=scaler_type,
            use_lagged_target_features=use_lagged_target_features,
            lag_periods=lag_periods,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            verbose=verbose,
            **kwargs
        )

        # Keep original target info
        self.prediction_horizon = prediction_horizon

        self.timeframe_config = config
        self.market_config = market_config

        if verbose:
            model_name = "CSNTransformer" if model_type == 'csn_transformer_cls' else "FT-Transformer"
            targets_text = ', '.join(target_columns_list) if is_multi_target else target_columns_list[0]
            print(f"Initialized IntradayPredictor for {timeframe} forecasting")
            print(f"  - Target(s): {targets_text}")
            print(f"  - Model Type: {model_name}")
            print(f"  - Country: {country} ({market_config['description']})")
            print(f"  - Market Hours: {market_config['open']} - {market_config['close']}")
            print(f"  - Sequence length: {sequence_length} {timeframe} bars")
            print(f"  - Model: {d_model}d x {num_layers}L x {num_heads}H")

    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Prepare features by creating intraday-specific features and time-series features.

        Args:
            df: DataFrame with intraday OHLCV data
            fit_scaler: Whether to fit the feature scaler

        Returns:
            DataFrame with engineered features
        """
        # First create intraday-specific features (microstructure, time-of-day effects, etc.)
        # Note: group_columns could be a list, but create_intraday_features expects single column or None
        # Pass first group column if available, otherwise None
        group_col_for_features = None
        if self.group_columns:
            group_col_for_features = self.group_columns[0] if isinstance(self.group_columns, list) else self.group_columns

        df_with_intraday_features = create_intraday_features(
            df=df,
            target_column=self.original_target_column,  # Use original, not shifted
            timestamp_col=self.timestamp_col,
            country=self.country,
            timeframe=self.timeframe,
            prediction_horizon=self.prediction_horizon,
            verbose=self.verbose,
            group_column=group_col_for_features  # Pass group_column to preserve it
        )

        # Then call parent's prepare_features to add time-series features
        return super().prepare_features(df_with_intraday_features, fit_scaler)
    
    def get_timeframe_info(self) -> dict:
        """
        Get information about the current timeframe configuration.
        
        Returns:
            Dictionary with timeframe information
        """
        return {
            'timeframe': self.timeframe,
            'description': self.timeframe_config['description'],
            'sequence_length': self.sequence_length,
            'recommended_sequence_length': self.timeframe_config['sequence_length'],
            'resample_rule': self.timeframe_config['resample_rule'],
            'country': self.country,
            'market_hours': f"{self.market_config['open']} - {self.market_config['close']}",
            'timezone': self.market_config['timezone']
        }
    
    def predict_next_bars(self, df: pd.DataFrame, n_predictions: int = 1) -> pd.DataFrame:
        """
        Predict next N bars for intraday trading.

        For single-target, single-horizon: Returns simple predictions
        For single-target, multi-horizon: Returns predictions for each horizon as separate columns
        For multi-target: Returns dict of predictions for each target

        Args:
            df: DataFrame with recent intraday data
            n_predictions: Number of future bars to predict

        Returns:
            DataFrame with predicted values and timestamps
            - Single-target, single-horizon: columns [timestamp, predicted_{target}]
            - Single-target, multi-horizon: columns [timestamp, predicted_{target}_h1, predicted_{target}_h2, ...]
            - Multi-target: columns [timestamp, predicted_{target1}, predicted_{target2}, ...] or with _h{n} suffixes
        """
        try:
            # Get base predictions
            predictions = self.predict(df)
        except Exception as e:
            self.logger.error("Error during prediction", exc_info=True)
            raise RuntimeError("Prediction failed due to an unexpected error.") from e

        # Generate future timestamps
        last_timestamp = df[self.timestamp_col].iloc[-1]

        freq_map = {
            '1min': '1T',
            '5min': '5T',
            '15min': '15T',
            '1h': '1H'
        }

        freq = freq_map.get(self.timeframe, '5T')
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(freq),
            periods=n_predictions,
            freq=freq
        )

        # Handle multi-target vs single-target
        if isinstance(predictions, dict):
            # Multi-target mode: predictions is a dict
            result_dict = {self.timestamp_col: future_timestamps}

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
                    self.timestamp_col: future_timestamps,
                    f'predicted_{self.original_target_column}': predictions[:n_predictions]
                })
            else:
                # Multi-horizon: predictions is 2D array (n_samples, n_horizons)
                result_dict = {self.timestamp_col: future_timestamps}

                for h in range(self.prediction_horizon):
                    horizon_num = h + 1
                    horizon_predictions = predictions[:n_predictions, h]
                    result_dict[f'predicted_{self.original_target_column}_h{horizon_num}'] = horizon_predictions

                result = pd.DataFrame(result_dict)

        return result
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance analysis for intraday features.
        
        Returns:
            DataFrame with feature importance scores (if available)
        """
        # This would require additional analysis - placeholder for future implementation
        if self.verbose:
            print("Feature importance analysis not yet implemented")
        return None
    
    def summary(self) -> str:
        """
        Get summary of the intraday predictor configuration.
        
        Returns:
            String summary of predictor settings
        """
        base_summary = super().summary() if hasattr(super(), 'summary') else ""
        
        intraday_info = f"""
Intraday Configuration:
  - Timeframe: {self.timeframe} ({self.timeframe_config['description']})
  - Country: {self.country} ({self.market_config['description']})
  - Market Hours: {self.market_config['open']} - {self.market_config['close']} {self.market_config['timezone']}
  - Sequence Length: {self.sequence_length} bars
  - Timestamp Column: {self.timestamp_col}
        """
        
        return base_summary + intraday_info
