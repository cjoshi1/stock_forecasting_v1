"""
Automatic feature detection for CSNTransformer.

Separates features into categorical and numerical categories,
with special handling for seasonal sin/cos pairs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
import re


class FeatureDetector:
    """Automatically detects and categorizes features for CSNTransformer."""

    def __init__(self,
                 categorical_threshold: int = 20,
                 seasonal_pattern: str = r'(.+)_(sin|cos)$'):
        """
        Args:
            categorical_threshold: Max unique values for a column to be considered categorical
            seasonal_pattern: Regex pattern to detect seasonal features (sin/cos pairs)
        """
        self.categorical_threshold = categorical_threshold
        self.seasonal_pattern = re.compile(seasonal_pattern)

    def detect_features(self, df: pd.DataFrame,
                       exclude_cols: List[str] = None) -> Dict[str, any]:
        """
        Detect and categorize features in the dataframe.

        Args:
            df: Input dataframe
            exclude_cols: Columns to exclude from feature detection

        Returns:
            Dictionary with categorized features and metadata
        """
        if exclude_cols is None:
            exclude_cols = []

        # Get feature columns (exclude timestamp, target, etc.)
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Detect seasonal features (sin/cos columns)
        seasonal_features = self._detect_seasonal_features(feature_cols)

        # Categorize features
        categorical_features = {}
        numerical_features = []

        for col in feature_cols:
            if self._is_seasonal_feature(col):
                # Treat seasonal features as categorical
                # Estimate unique values for sin/cos features
                unique_vals = self._estimate_seasonal_unique_values(df[col])
                categorical_features[col] = unique_vals
                seasonal_features.add(col)
            elif self._is_categorical(df[col]):
                # Get unique values for categorical features
                unique_vals = df[col].nunique()
                categorical_features[col] = unique_vals
            else:
                numerical_features.append(col)

        return {
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'seasonal_features': list(seasonal_features),
            'feature_summary': {
                'total_features': len(feature_cols),
                'categorical_count': len(categorical_features),
                'numerical_count': len(numerical_features),
                'seasonal_count': len(seasonal_features)
            }
        }

    def _detect_seasonal_pairs(self, feature_cols: List[str]) -> Dict[str, Tuple[str, str]]:
        """Detect sin/cos pairs in feature columns."""
        seasonal_pairs = {}
        sin_cols = set()
        cos_cols = set()

        # Find all sin and cos columns
        for col in feature_cols:
            match = self.seasonal_pattern.match(col)
            if match:
                base_name, suffix = match.groups()
                if suffix == 'sin':
                    sin_cols.add((base_name, col))
                elif suffix == 'cos':
                    cos_cols.add((base_name, col))

        # Match sin/cos pairs
        sin_dict = dict(sin_cols)
        cos_dict = dict(cos_cols)

        for base_name in sin_dict.keys():
            if base_name in cos_dict:
                seasonal_pairs[base_name] = (sin_dict[base_name], cos_dict[base_name])

        return seasonal_pairs

    def _detect_seasonal_features(self, feature_cols: List[str]) -> Set[str]:
        """Detect individual seasonal features (sin/cos columns)."""
        seasonal_features = set()

        for col in feature_cols:
            if self._is_seasonal_feature(col):
                seasonal_features.add(col)

        return seasonal_features

    def _is_seasonal_feature(self, col: str) -> bool:
        """Check if a column is a seasonal feature (sin or cos)."""
        match = self.seasonal_pattern.match(col)
        return match is not None

    def _estimate_seasonal_unique_values(self, series: pd.Series) -> int:
        """Estimate unique values for individual sin/cos features."""
        # For sin/cos features, estimate based on the resolution
        # Since values are between -1 and 1, we can estimate discretization levels

        # Remove NaN values
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return 24  # Default assumption

        # Count approximately unique values by rounding to reasonable precision
        rounded_values = np.round(clean_series.values, decimals=4)
        unique_count = len(np.unique(rounded_values))

        # For sin/cos, typical values range based on time granularity
        # Hour: ~24 values, Minute: ~60 values, etc.
        return min(unique_count, 100)  # Cap at reasonable maximum

    def _is_categorical(self, series: pd.Series) -> bool:
        """Determine if a series should be treated as categorical."""
        # Handle missing values
        series_clean = series.dropna()

        if len(series_clean) == 0:
            return False

        # Check data type
        if series.dtype == 'object' or series.dtype.name == 'category':
            return True

        if series.dtype == 'bool':
            return True

        # For numeric types, check number of unique values
        unique_count = series_clean.nunique()

        # If very few unique values, likely categorical
        if unique_count <= self.categorical_threshold:
            # Additional check: if all values are integers and range is reasonable
            if series.dtype in ['int64', 'int32', 'int16', 'int8']:
                value_range = series_clean.max() - series_clean.min()
                # If range is much larger than unique count, probably not categorical
                if value_range > unique_count * 10:
                    return False
            return True

        return False

    def _estimate_seasonal_resolution(self, sin_series: pd.Series, cos_series: pd.Series) -> int:
        """Estimate the resolution of seasonal features based on sin/cos values."""
        # Calculate the number of unique angles (approximately)
        # This is a rough estimation based on the granularity of the data

        # Combine sin and cos to estimate unique angles
        combined = np.column_stack([sin_series.values, cos_series.values])

        # Remove NaN rows
        valid_mask = ~(np.isnan(combined).any(axis=1))
        if not valid_mask.any():
            return 24  # Default assumption

        valid_combined = combined[valid_mask]

        # Estimate unique angles by rounding and counting unique pairs
        # Round to reasonable precision to account for floating point errors
        rounded = np.round(valid_combined, decimals=4)
        unique_pairs = np.unique(rounded, axis=0)

        # Return the number of unique seasonal positions
        return len(unique_pairs)

    def prepare_categorical_inputs(self,
                                 df: pd.DataFrame,
                                 categorical_features: Dict[str, int],
                                 seasonal_features: List[str]) -> Dict[str, np.ndarray]:
        """
        Prepare categorical inputs for the model.

        Args:
            df: Input dataframe
            categorical_features: Dictionary of categorical feature names and their vocab sizes
            seasonal_features: List of seasonal feature names

        Returns:
            Dictionary of categorical inputs ready for embedding
        """
        categorical_inputs = {}

        for feature_name in categorical_features.keys():
            if feature_name in seasonal_features:
                # Handle seasonal features (individual sin/cos)
                categorical_inputs[feature_name] = self._convert_sincos_to_categorical(
                    df[feature_name].values, categorical_features[feature_name]
                )
            else:
                # Handle regular categorical features
                categorical_inputs[feature_name] = self._encode_categorical(df[feature_name].values)

        return categorical_inputs

    def _convert_seasonal_to_categorical(self, sin_vals: np.ndarray, cos_vals: np.ndarray, num_categories: int) -> np.ndarray:
        """Convert sin/cos values back to categorical indices."""
        # Calculate angles from sin/cos
        angles = np.arctan2(sin_vals, cos_vals)

        # Normalize to [0, 2π]
        angles = (angles + 2 * np.pi) % (2 * np.pi)

        # Convert to categorical indices [0, num_categories-1]
        categorical_indices = (angles / (2 * np.pi) * num_categories).astype(int)

        # Handle edge case where angle is exactly 2π
        categorical_indices = np.clip(categorical_indices, 0, num_categories - 1)

        return categorical_indices

    def _convert_sincos_to_categorical(self, sincos_vals: np.ndarray, num_categories: int) -> np.ndarray:
        """Convert individual sin or cos values to categorical indices."""
        # Normalize sin/cos values from [-1, 1] to [0, 1]
        normalized = (sincos_vals + 1) / 2

        # Convert to categorical indices [0, num_categories-1]
        categorical_indices = (normalized * num_categories).astype(int)

        # Handle edge case where value is exactly 1.0
        categorical_indices = np.clip(categorical_indices, 0, num_categories - 1)

        return categorical_indices

    def _encode_categorical(self, values: np.ndarray) -> np.ndarray:
        """Encode categorical values to integer indices."""
        # Simple label encoding
        unique_vals = np.unique(values[~pd.isna(values)])
        val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}

        # Encode values, use 0 for NaN values
        encoded = np.array([val_to_idx.get(val, len(unique_vals)) for val in values])
        return encoded