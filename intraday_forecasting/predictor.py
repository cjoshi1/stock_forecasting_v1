"""
IntradayPredictor class for high-frequency trading forecasting.

Extends the generic TimeSeriesPredictor with intraday-specific feature engineering
and timeframe-aware functionality.
"""

import pandas as pd
from typing import Optional

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
    
    def __init__(self, target_column: str = 'close', timeframe: str = '5min', model_type: str = 'ft',
                 timestamp_col: str = 'timestamp', country: str = 'US',
                 prediction_horizon: int = 1,
                 d_token: int = 128, n_layers: int = 3, n_heads: int = 8,
                 dropout: float = 0.1, verbose: bool = False, **kwargs):
        """
        Initialize IntradayPredictor.

        Args:
            target_column: Column to predict
            timeframe: Trading timeframe ('1min', '5min', '15min', '1h')
            model_type: Model architecture ('ft' for FT-Transformer, 'csn' for CSNTransformer)
            timestamp_col: Name of timestamp column
            country: Country code ('US' or 'INDIA')
            prediction_horizon: Number of steps ahead to predict (1=single, >1=multi-horizon)
            d_token: Token embedding dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            dropout: Dropout rate
            verbose: Whether to print detailed logs
            **kwargs: Additional arguments passed to base predictor
        """
        # Validate country
        if not validate_country(country):
            raise ValueError(f"Unsupported country: {country}. Use: US or INDIA")

        # Validate model type
        if model_type not in ['ft', 'csn']:
            raise ValueError(f"Unsupported model_type: {model_type}. Use: 'ft' or 'csn'")

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

        # Handle target column naming for single vs multi-horizon (same as StockPredictor)
        if prediction_horizon == 1:
            shifted_target_name = f"{target_column}_target_h1"
        else:
            # Multi-horizon: use base target name
            shifted_target_name = target_column

        # Initialize base predictor based on model_type
        if model_type == 'csn':
            from tf_predictor.core.csn_predictor import CSNPredictor
            # Initialize CSNPredictor directly
            CSNPredictor.__init__(
                self,
                target_column=shifted_target_name,
                sequence_length=sequence_length,
                prediction_horizon=prediction_horizon,
                d_model=d_token,  # CSNPredictor uses d_model instead of d_token
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout,
                verbose=verbose,
                **kwargs
            )
        else:
            # Default to FT-Transformer
            super().__init__(
                target_column=shifted_target_name,
                sequence_length=sequence_length,
                prediction_horizon=prediction_horizon,
                d_token=d_token,
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout,
                verbose=verbose,
                **kwargs
            )

        # Store original target info (same as StockPredictor)
        self.original_target_column = target_column
        self.prediction_horizon = prediction_horizon
        
        self.timeframe_config = config
        self.market_config = market_config
        
        if verbose:
            model_name = "CSNTransformer" if model_type == 'csn' else "FT-Transformer"
            print(f"Initialized IntradayPredictor for {timeframe} forecasting")
            print(f"  - Target: {target_column}")
            print(f"  - Model Type: {model_name}")
            print(f"  - Country: {country} ({market_config['description']})")
            print(f"  - Market Hours: {market_config['open']} - {market_config['close']}")
            print(f"  - Sequence length: {sequence_length} {timeframe} bars")
            print(f"  - Model: {d_token}d x {n_layers}L x {n_heads}H")
    
    def create_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Create intraday-specific features for the dataset.

        Args:
            df: DataFrame with intraday OHLCV data
            fit_scaler: Whether to fit the feature scaler

        Returns:
            DataFrame with engineered features
        """
        return create_intraday_features(
            df=df,
            target_column=self.original_target_column,  # Use original, not shifted
            timestamp_col=self.timestamp_col,
            country=self.country,
            timeframe=self.timeframe,
            prediction_horizon=self.prediction_horizon,
            verbose=self.verbose
        )
    
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
        
        Args:
            df: DataFrame with recent intraday data
            n_predictions: Number of future bars to predict
            
        Returns:
            DataFrame with predicted values and timestamps
        """
        # Get base predictions
        predictions = self.predict(df)
        
        if len(predictions) == 0:
            return pd.DataFrame()
        
        # Create result DataFrame with future timestamps
        last_timestamp = df[self.timestamp_col].iloc[-1]
        
        # Generate future timestamps based on timeframe
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
        
        # Create result DataFrame
        result = pd.DataFrame({
            self.timestamp_col: future_timestamps,
            f'predicted_{self.target_column}': predictions[:n_predictions]
        })
        
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