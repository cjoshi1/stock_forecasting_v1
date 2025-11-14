"""
Rossmann Store Sales Predictor.

Extends TimeSeriesPredictor with Rossmann-specific features and RMSPE evaluation.
"""
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict
from tf_predictor.core.predictor import TimeSeriesPredictor
from .utils.metrics import rmspe, calculate_all_metrics


class RossmannPredictor(TimeSeriesPredictor):
    """
    Time series predictor specialized for Rossmann Store Sales forecasting.

    This predictor extends the generic TimeSeriesPredictor with:
    - RMSPE evaluation metric (Kaggle competition metric)
    - Store-specific scaling (group_columns='Store')
    - Rossmann domain knowledge

    Note: Rossmann-specific features (competition, promo, holidays, store type)
    must be added BEFORE passing data to this predictor. This class only handles
    the time-series modeling part.
    """

    def __init__(
        self,
        target_column: str = 'Sales',
        sequence_length: int = 14,
        prediction_horizon: int = 1,
        group_columns: Optional[Union[str, List[str]]] = 'Store',
        model_type: str = 'ft_transformer',
        pooling_type: str = 'multihead_attention',
        d_token: int = 128,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        scaler_type: str = 'standard',
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize Rossmann predictor.

        Args:
            target_column: Target variable ('Sales')
            sequence_length: Number of historical days for prediction
            prediction_horizon: Steps ahead to predict
            group_columns: Column for group-based scaling (default: 'Store')
            model_type: Model architecture ('ft_transformer' or 'csn_transformer')
            pooling_type: Pooling strategy for sequence aggregation
            d_token: Token embedding dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            dropout: Dropout rate
            scaler_type: Scaler type ('standard', 'minmax', 'robust', etc.)
            verbose: Print progress messages
            **kwargs: Additional arguments passed to TimeSeriesPredictor
        """
        super().__init__(
            target_column=target_column,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            group_columns=group_columns,
            model_type=model_type,
            scaler_type=scaler_type,
            verbose=verbose,
            d_token=d_token,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            pooling_type=pooling_type,
            **kwargs
        )

        if verbose:
            print("\n🏪 Rossmann Store Sales Predictor")
            print(f"   Target: {target_column}")
            print(f"   Sequence length: {sequence_length} days")
            print(f"   Prediction horizon: {prediction_horizon} day(s)")
            print(f"   Group-based scaling: {group_columns}")

    def _create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Override to add sorting and date features.

        Rossmann-specific features should already be in the dataframe.
        This method only ensures proper sorting and adds time-series features.

        Args:
            df: DataFrame with Rossmann features already added

        Returns:
            DataFrame with base time-series features
        """
        df = df.copy()

        # Ensure Date column exists
        if 'Date' not in df.columns:
            raise ValueError("Date column is required for Rossmann forecasting")

        # Ensure Store column exists for group-based operations
        if 'Store' not in df.columns:
            raise ValueError("Store column is required for Rossmann forecasting")

        # Sort by Store and Date to ensure temporal order
        df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

        # Call parent's base feature creation (adds date features automatically)
        df = super()._create_base_features(df)

        return df

    def evaluate(
        self,
        df: pd.DataFrame,
        per_group: bool = False,
        predictions=None,
        group_indices=None,
        export_csv: Optional[str] = None
    ) -> Dict:
        """
        Evaluate model with standard metrics + RMSPE.

        Args:
            df: DataFrame with raw data
            per_group: If True, return per-store metrics
            predictions: Optional pre-computed predictions
            group_indices: Optional pre-computed group indices
            export_csv: Optional path to export predictions

        Returns:
            Dictionary with metrics including RMSPE
        """
        # Get standard metrics from parent
        metrics = super().evaluate(
            df,
            per_group=per_group,
            predictions=predictions,
            group_indices=group_indices,
            export_csv=export_csv
        )

        # Add RMSPE metric (Kaggle competition metric)
        if predictions is None:
            predictions = self.predict(df)

        # Get actuals
        if hasattr(self, '_last_processed_df') and self._last_processed_df is not None:
            target_col = self.target_columns[0] if isinstance(self.target_columns, list) else self.target_columns
            actuals = self._last_processed_df[target_col].values[:len(predictions)]
        else:
            actuals = df[self.target_column].values[:len(predictions)]

        # Calculate RMSPE
        rmspe_value = rmspe(actuals, predictions.flatten() if isinstance(predictions, np.ndarray) else predictions)

        # Add RMSPE to metrics
        if isinstance(metrics, dict):
            if 'overall' in metrics:
                metrics['overall']['RMSPE'] = rmspe_value
            else:
                metrics['RMSPE'] = rmspe_value

        return metrics

    def predict_with_dates(
        self,
        df: pd.DataFrame,
        inference_mode: bool = False
    ) -> pd.DataFrame:
        """
        Make predictions and return with Store IDs and dates.

        Useful for creating Kaggle submissions or analyzing predictions.

        Args:
            df: Input dataframe
            inference_mode: Whether to use inference mode (no target needed)

        Returns:
            DataFrame with columns: Store, Date, Predicted_Sales
        """
        predictions = self.predict(df, inference_mode=inference_mode)

        # Get Store and Date from input
        result_df = pd.DataFrame({
            'Store': df['Store'].values[:len(predictions)],
            'Date': df['Date'].values[:len(predictions)],
            'Predicted_Sales': predictions.flatten() if isinstance(predictions, np.ndarray) else predictions
        })

        return result_df
