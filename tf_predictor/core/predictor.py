"""
Generic Time Series Predictor using FT-Transformer.

A base class for time series prediction that can be extended for different domains.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import gc
from typing import Optional, Dict, Any, Tuple, Union
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod

from .ft_model import FTTransformerPredictor, SequenceFTTransformerPredictor


class TimeSeriesPredictor(ABC):
    """Generic FT-Transformer wrapper for time series prediction."""
    
    def __init__(
        self,
        target_column: Union[str, list],
        sequence_length: int = 5,
        prediction_horizon: int = 1,
        group_column: Optional[str] = None,
        **ft_kwargs
    ):
        """
        Args:
            target_column: Name(s) of target column(s) to predict
                          - str: Single-target prediction (e.g., 'close')
                          - List[str]: Multi-target prediction (e.g., ['close', 'volume'])
            sequence_length: Number of historical time steps to use for prediction
            prediction_horizon: Number of steps ahead to predict (1=single, >1=multi-horizon)
            group_column: Optional column name for group-based scaling (e.g., 'symbol')
                         If provided, each unique value gets its own scaler
            **ft_kwargs: FT-Transformer hyperparameters
        """
        # Normalize target_column to list for uniform handling
        if isinstance(target_column, str):
            self.target_columns = [target_column]
            self.is_multi_target = False
        else:
            self.target_columns = list(target_column)
            self.is_multi_target = len(self.target_columns) > 1

        # Keep backward compatibility
        self.target_column = self.target_columns[0] if not self.is_multi_target else self.target_columns

        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.group_column = group_column
        self.ft_kwargs = ft_kwargs
        self.num_targets = len(self.target_columns)

        # Will be set during training
        self.model = None

        # Single-group scalers (used when group_column=None)
        self.scaler = StandardScaler()  # For features

        # Target scalers structure depends on single vs multi-target
        # Single-target mode (backward compatible):
        #   - single-horizon: self.target_scaler (StandardScaler)
        #   - multi-horizon: self.target_scalers (List[StandardScaler])
        # Multi-target mode:
        #   - single-horizon: self.target_scalers_dict (Dict[target_name, StandardScaler])
        #   - multi-horizon: self.target_scalers_dict (Dict[target_name, List[StandardScaler]])
        if not self.is_multi_target:
            self.target_scaler = StandardScaler()  # For single target, single horizon
            self.target_scalers = []  # For single target, multi-horizon
        else:
            self.target_scalers_dict = {}  # For multi-target

        # Multi-group scalers (used when group_column is provided)
        # Structure: {group_value: StandardScaler} for features
        # For targets:
        #   - Single-target: {group_value: StandardScaler}
        #   - Multi-target: {group_value: Dict[target_name, StandardScaler]}
        self.group_feature_scalers = {}  # Dict[group_value, StandardScaler]
        self.group_target_scalers = {}   # Dict[group_value, StandardScaler or Dict]

        self.feature_columns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training history
        self.history = {'train_loss': [], 'val_loss': []}
        self.verbose = False  # Will be set during training

        # Feature caching to avoid recomputation
        self._feature_cache = {}
        self._cache_enabled = True
    
    @abstractmethod
    def create_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Create domain-specific features from raw data.
        
        Args:
            df: DataFrame with raw data
            fit_scaler: Whether to fit the scaler (True for training data)
            
        Returns:
            processed_df: DataFrame with engineered features
        """
        pass
    
    def _get_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Generate a hash key for DataFrame caching."""
        import hashlib
        # Use shape, column names, and sample of data for hash
        key_data = f"{df.shape}_{list(df.columns)}_{df.iloc[0].to_dict() if len(df) > 0 else {}}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Prepare features by calling domain-specific feature creation and handling scaling.
        Uses caching to avoid recomputing features for the same data.

        Args:
            df: DataFrame with raw data
            fit_scaler: Whether to fit the scaler (True for training data)

        Returns:
            processed_df: DataFrame with scaled features
        """
        # Generate cache key
        cache_key = f"{self._get_dataframe_hash(df)}_{fit_scaler}"

        # Check cache first
        if self._cache_enabled and cache_key in self._feature_cache:
            if self.verbose:
                print(f"   Using cached features for dataset ({len(df)} rows)")
            return self._feature_cache[cache_key].copy()

        # Sort by group and time if group_column is specified (BEFORE feature creation and scaling)
        if self.group_column is not None and self.group_column in df.columns:
            # Detect time-related column for sorting
            time_column = None
            possible_time_cols = ['timestamp', 'date', 'datetime', 'time', 'Date', 'Timestamp', 'DateTime']
            for col in possible_time_cols:
                if col in df.columns:
                    time_column = col
                    break

            # Sort dataframe by group and time to ensure temporal order
            if time_column:
                if self.verbose:
                    print(f"   Sorting data by '{self.group_column}' and '{time_column}' to ensure temporal order")
                df = df.sort_values([self.group_column, time_column]).reset_index(drop=True)
            else:
                if self.verbose:
                    print(f"   No time column detected. Assuming data is already sorted in temporal order within groups.")

        # Create domain-specific features
        df_processed = self.create_features(df, fit_scaler)

        # Get all feature columns (excluding target and non-numeric columns)
        feature_cols = []

        # Build exclusion set: all target columns (original and shifted)
        exclude_cols = set(self.target_columns)

        # Also exclude shifted target columns for all targets
        for target_col in self.target_columns:
            # Add all shifted horizon columns
            for h in range(1, self.prediction_horizon + 1):
                exclude_cols.add(f"{target_col}_target_h{h}")

        for col in df_processed.columns:
            # Exclude all target-related columns
            if col not in exclude_cols:
                # Only include numeric columns for scaling
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    feature_cols.append(col)
                elif self.verbose:
                    print(f"   Excluding non-numeric column from scaling: {col} (dtype: {df_processed[col].dtype})")
            elif self.verbose and col in exclude_cols:
                if col in self.target_columns or '_target_h' in col:
                    pass  # Don't print for every shifted column, too verbose
        
        if self.feature_columns is None:
            # First time: establish the numeric feature columns from training data
            self.feature_columns = feature_cols
        else:
            # For validation/test data: only use features that were created during training
            # This handles cases where smaller datasets can't generate all features
            available_features = [col for col in self.feature_columns if col in df_processed.columns]
            missing_features = [col for col in self.feature_columns if col not in df_processed.columns]
            
            if missing_features:
                if self.verbose:
                    print(f"   Warning: {len(missing_features)} features missing in current dataset (likely due to insufficient data size)")
                    print(f"   Missing: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
                
                # Add missing features as zero-filled columns
                for feature in missing_features:
                    df_processed[feature] = 0.0
                    if self.verbose:
                        print(f"   Added zero-filled feature: {feature}")
        
        # Scale numeric features - route to appropriate scaling method
        if len(self.feature_columns) > 0:
            if self.group_column is None:
                # Single-group scaling (original behavior)
                df_processed = self._scale_features_single(df_processed, fit_scaler)
            else:
                # Multi-group scaling (new behavior)
                df_processed = self._scale_features_grouped(df_processed, fit_scaler)

        # Cache the result
        if self._cache_enabled:
            self._feature_cache[cache_key] = df_processed.copy()
            if self.verbose:
                print(f"   Cached features for dataset ({len(df)} rows)")

        return df_processed

    def clear_feature_cache(self):
        """Clear the feature cache to free memory."""
        self._feature_cache.clear()
        if self.verbose:
            print("   Feature cache cleared")

    def disable_feature_cache(self):
        """Disable feature caching."""
        self._cache_enabled = False
        self.clear_feature_cache()

    def enable_feature_cache(self):
        """Enable feature caching."""
        self._cache_enabled = True

    def _scale_features_single(self, df_processed: pd.DataFrame, fit_scaler: bool) -> pd.DataFrame:
        """
        Scale features using single scaler (original behavior).

        Args:
            df_processed: DataFrame with engineered features
            fit_scaler: Whether to fit the scaler

        Returns:
            DataFrame with scaled features
        """
        # Scale only the numeric features
        if len(self.feature_columns) > 0:
            if fit_scaler:
                df_processed[self.feature_columns] = self.scaler.fit_transform(
                    df_processed[self.feature_columns]
                )
            else:
                df_processed[self.feature_columns] = self.scaler.transform(
                    df_processed[self.feature_columns]
                )

        return df_processed

    def _scale_features_grouped(self, df_processed: pd.DataFrame, fit_scaler: bool) -> pd.DataFrame:
        """
        Scale features separately per group.

        Each unique value in group_column gets its own scaler. This ensures
        that different entities (e.g., different stock symbols) are scaled
        independently, preserving their individual statistical properties.

        Args:
            df_processed: DataFrame with engineered features
            fit_scaler: Whether to fit the scalers

        Returns:
            DataFrame with group-scaled features
        """
        if self.group_column not in df_processed.columns:
            raise ValueError(
                f"Group column '{self.group_column}' not found in dataframe. "
                f"Available columns: {list(df_processed.columns)}"
            )

        # Create a copy to avoid modifying original
        df_scaled = df_processed.copy()

        # Get unique groups
        unique_groups = df_processed[self.group_column].unique()

        if self.verbose:
            print(f"   Scaling features for {len(unique_groups)} groups: {unique_groups}")

        # Scale each group separately
        for group_value in unique_groups:
            # Get mask for this group
            group_mask = df_processed[self.group_column] == group_value
            group_size = group_mask.sum()

            if group_size == 0:
                continue

            # Get data for this group
            group_data = df_processed.loc[group_mask, self.feature_columns]

            if fit_scaler:
                # Create and fit new scaler for this group
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(group_data)
                self.group_feature_scalers[group_value] = scaler

                if self.verbose:
                    print(f"   Group '{group_value}': fitted scaler on {group_size} samples")
            else:
                # Use existing scaler for this group
                if group_value not in self.group_feature_scalers:
                    raise ValueError(
                        f"No scaler found for group '{group_value}'. "
                        f"Make sure to fit on training data first."
                    )
                scaled_data = self.group_feature_scalers[group_value].transform(group_data)

            # Update the scaled data for this group
            df_scaled.loc[group_mask, self.feature_columns] = scaled_data

        return df_scaled

    def _prepare_data_grouped(self, df_processed: pd.DataFrame, fit_scaler: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequential data with group-based target scaling.

        Args:
            df_processed: DataFrame with features already scaled by group
            fit_scaler: Whether to fit new target scalers or use existing ones

        Returns:
            X: Sequences tensor of shape (n_samples, sequence_length, n_features)
            y: Target tensor of shape (n_samples,) for single-horizon or (n_samples, prediction_horizon) for multi-horizon
        """
        from ..preprocessing.time_features import create_sequences

        # Note: Data is already sorted by group and time in prepare_features()
        # so temporal order is guaranteed at this point

        # Validate target columns exist
        if self.is_multi_target:
            # Multi-target: check all targets
            if self.prediction_horizon == 1:
                expected_targets = {target_col: [f"{target_col}_target_h1"]
                                   for target_col in self.target_columns}
            else:
                expected_targets = {target_col: [f"{target_col}_target_h{h}"
                                                 for h in range(1, self.prediction_horizon + 1)]
                                   for target_col in self.target_columns}

            # Use first target for sequence creation
            first_target = self.target_columns[0]
            target_cols = expected_targets[first_target]
        else:
            # Single-target: existing behavior
            if self.prediction_horizon == 1:
                expected_target = f"{self.target_column}_target_h1" if not self.target_column.endswith('_target_h1') else self.target_column
                if expected_target not in df_processed.columns:
                    raise ValueError(f"Single horizon target column '{expected_target}' not found")
                target_cols = [expected_target]
            else:
                target_cols = [f"{self.target_column}_target_h{h}" for h in range(1, self.prediction_horizon + 1)]
                missing = [col for col in target_cols if col not in df_processed.columns]
                if missing:
                    raise ValueError(f"Multi-horizon target columns {missing} not found")

        # Process each group separately
        unique_groups = df_processed[self.group_column].unique()
        all_sequences = []
        all_targets = []
        group_indices = []  # Track which group each sequence belongs to

        for group_value in unique_groups:
            group_mask = df_processed[self.group_column] == group_value
            group_df = df_processed[group_mask].copy()

            # Check if group has enough data for sequences
            if len(group_df) <= self.sequence_length:
                if self.verbose:
                    print(f"  Warning: Skipping group '{group_value}' - insufficient data ({len(group_df)} <= {self.sequence_length})")
                continue

            # Create sequences for this group using first target column
            sequences, targets_h1 = create_sequences(group_df, self.sequence_length, target_cols[0], self.feature_columns)

            if self.is_multi_target:
                # Multi-target mode
                if fit_scaler:
                    # Initialize dict for this group
                    self.group_target_scalers[group_value] = {}

                all_scaled_dict = {}  # {target_name: scaled_targets}

                for target_col in self.target_columns:
                    if self.prediction_horizon == 1:
                        # Single-horizon for this target
                        target_col_name = f"{target_col}_target_h1"
                        target_values = group_df[target_col_name].values[self.sequence_length:]
                        target_values = target_values.reshape(-1, 1)

                        if fit_scaler:
                            scaler = StandardScaler()
                            targets_scaled = scaler.fit_transform(target_values).flatten()
                            self.group_target_scalers[group_value][target_col] = scaler
                        else:
                            if group_value not in self.group_target_scalers or target_col not in self.group_target_scalers[group_value]:
                                raise ValueError(f"No scaler found for group '{group_value}', target '{target_col}'")
                            targets_scaled = self.group_target_scalers[group_value][target_col].transform(target_values).flatten()

                        all_scaled_dict[target_col] = targets_scaled

                    else:
                        # Multi-horizon for this target
                        target_cols_list = [f"{target_col}_target_h{h}"
                                           for h in range(1, self.prediction_horizon + 1)]

                        # Extract all horizon values
                        all_horizons = []
                        for tcol in target_cols_list:
                            target_values = group_df[tcol].values[self.sequence_length:]
                            all_horizons.append(target_values)

                        # Stack into matrix: (samples, horizons)
                        targets_matrix = np.column_stack(all_horizons)

                        if fit_scaler:
                            # Scale each horizon separately for this target
                            self.group_target_scalers[group_value][target_col] = []
                            scaled_horizons = []
                            for h in range(self.prediction_horizon):
                                scaler = StandardScaler()
                                scaled = scaler.fit_transform(targets_matrix[:, h].reshape(-1, 1)).flatten()
                                self.group_target_scalers[group_value][target_col].append(scaler)
                                scaled_horizons.append(scaled)
                            targets_scaled = np.column_stack(scaled_horizons)
                        else:
                            # Use existing scalers
                            if group_value not in self.group_target_scalers or target_col not in self.group_target_scalers[group_value]:
                                raise ValueError(f"No scalers found for group '{group_value}', target '{target_col}'")
                            scaled_horizons = []
                            for h, scaler in enumerate(self.group_target_scalers[group_value][target_col]):
                                scaled = scaler.transform(targets_matrix[:, h].reshape(-1, 1)).flatten()
                                scaled_horizons.append(scaled)
                            targets_scaled = np.column_stack(scaled_horizons)

                        all_scaled_dict[target_col] = targets_scaled

                # Concatenate all targets: single-horizon -> [samples, num_targets]
                #                          multi-horizon -> [samples, num_targets * horizons]
                if self.prediction_horizon == 1:
                    # Stack as columns: each target is one column
                    y_combined = np.column_stack([all_scaled_dict[tc] for tc in self.target_columns])
                else:
                    # Flatten: [close_h1, close_h2, ..., volume_h1, volume_h2, ...]
                    y_list = []
                    for target_col in self.target_columns:
                        # all_scaled_dict[target_col] is (samples, horizons)
                        for h in range(self.prediction_horizon):
                            y_list.append(all_scaled_dict[target_col][:, h])
                    y_combined = np.column_stack(y_list)

                all_sequences.append(sequences)
                all_targets.append(y_combined)
                group_indices.extend([group_value] * len(sequences))

            else:
                # Single-target mode (original behavior)
                if self.prediction_horizon == 1:
                    # Single horizon - scale targets
                    targets = targets_h1.reshape(-1, 1)

                    if fit_scaler:
                        scaler = StandardScaler()
                        targets_scaled = scaler.fit_transform(targets).flatten()
                        self.group_target_scalers[group_value] = scaler
                    else:
                        if group_value not in self.group_target_scalers:
                            raise ValueError(f"No fitted target scaler found for group '{group_value}'")
                        targets_scaled = self.group_target_scalers[group_value].transform(targets).flatten()

                    all_sequences.append(sequences)
                    all_targets.append(targets_scaled)
                    # Track which group these sequences belong to
                    group_indices.extend([group_value] * len(sequences))

                else:
                    # Multi-horizon - extract all target values and scale
                    targets_list = []
                    for target_col in target_cols:
                        target_values = group_df[target_col].values[self.sequence_length:]
                        targets_list.append(target_values)

                    # Stack into matrix: (samples, horizons)
                    targets_matrix = np.column_stack(targets_list)

                    if fit_scaler:
                        # Create ONE scaler for this group (all horizons together)
                        scaler = StandardScaler()
                        targets_scaled = scaler.fit_transform(targets_matrix)
                        self.group_target_scalers[group_value] = scaler
                    else:
                        if group_value not in self.group_target_scalers:
                            raise ValueError(f"No fitted target scaler found for group '{group_value}'")
                        targets_scaled = self.group_target_scalers[group_value].transform(targets_matrix)

                    all_sequences.append(sequences)
                    all_targets.append(targets_scaled)
                    # Track which group these sequences belong to
                    group_indices.extend([group_value] * len(sequences))

        # Concatenate all groups
        if len(all_sequences) == 0:
            raise ValueError(f"No groups had sufficient data (need > {self.sequence_length} samples per group)")

        X_combined = np.vstack(all_sequences)

        # Handle target concatenation based on single vs multi-target mode
        if self.is_multi_target:
            # Multi-target: all_targets contains 2D arrays
            # For single-horizon: each array is (samples, num_targets)
            # For multi-horizon: each array is (samples, num_targets * horizons)
            y_combined = np.vstack(all_targets)
        else:
            # Single-target: original behavior
            # For single-horizon: all_targets contains 1D arrays -> use concatenate
            # For multi-horizon: all_targets contains 2D arrays -> use vstack
            y_combined = np.vstack(all_targets) if self.prediction_horizon > 1 else np.concatenate(all_targets)

        # Store group indices for inverse transform during prediction
        self._last_group_indices = group_indices

        # Convert to tensors
        X = torch.tensor(X_combined, dtype=torch.float32)
        y = torch.tensor(y_combined, dtype=torch.float32)

        if self.verbose:
            print(f"  Created {len(X)} sequences from {len(unique_groups)} groups")
            print(f"  X shape: {X.shape}, y shape: {y.shape}")

        return X, y

    def prepare_data(self, df: pd.DataFrame, fit_scaler: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare sequential data for model training/inference.

        Returns:
            X: Sequences tensor of shape (n_samples, sequence_length, n_features)
            y: Target tensor of shape (n_samples,) (None for inference)
        """
        # First prepare features (includes group-based or single scaling)
        df_processed = self.prepare_features(df, fit_scaler)

        # Route to appropriate data preparation method
        if self.group_column is not None:
            # Group-based sequence creation and target scaling
            return self._prepare_data_grouped(df_processed, fit_scaler)

        # Single-group data preparation (original behavior)
        # Check if we have enough data for sequences
        if len(df_processed) <= self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length + 1} samples for sequence_length={self.sequence_length}, got {len(df_processed)}")
        
        # Handle target validation for single vs multi-target and single vs multi-horizon
        if self.is_multi_target:
            # Multi-target: check all targets have their shifted columns
            if self.prediction_horizon == 1:
                expected_targets = {target_col: [f"{target_col}_target_h1"]
                                   for target_col in self.target_columns}
            else:
                expected_targets = {target_col: [f"{target_col}_target_h{h}"
                                                 for h in range(1, self.prediction_horizon + 1)]
                                   for target_col in self.target_columns}

            # Check all expected columns exist
            for target_col, shifted_cols in expected_targets.items():
                missing = [col for col in shifted_cols if col not in df_processed.columns]
                if missing:
                    raise ValueError(f"Target columns {missing} not found for target '{target_col}'")

            # Use first target's first horizon for sequence creation
            actual_target = expected_targets[self.target_columns[0]][0]
        else:
            # Single-target: existing behavior
            if self.prediction_horizon == 1:
                expected_target = f"{self.target_columns[0]}_target_h1"
                if expected_target not in df_processed.columns:
                    raise ValueError(f"Single horizon target column '{expected_target}' not found")
                actual_target = expected_target
            else:
                target_columns = [f"{self.target_columns[0]}_target_h{h}"
                                 for h in range(1, self.prediction_horizon + 1)]
                missing_targets = [col for col in target_columns if col not in df_processed.columns]
                if missing_targets:
                    raise ValueError(f"Multi-horizon target columns {missing_targets} not found")
                actual_target = target_columns[0]

        # Create sequences - this will reduce our sample count
        if actual_target in df_processed.columns:
            from ..preprocessing.time_features import create_sequences
            
            # For training: create sequences with targets
            sequences, targets = create_sequences(df_processed, self.sequence_length, actual_target, self.feature_columns)

            if self.is_multi_target:
                # Multi-target handling
                all_scaled_dict = {}   # {target_name: scaled_targets}

                for target_col in self.target_columns:
                    if self.prediction_horizon == 1:
                        # Single horizon for this target
                        target_col_name = f"{target_col}_target_h1"
                        target_values = df_processed[target_col_name].values[self.sequence_length:]
                        target_values = target_values.reshape(-1, 1)

                        if fit_scaler:
                            scaler = StandardScaler()
                            targets_scaled = scaler.fit_transform(target_values).flatten()
                            self.target_scalers_dict[target_col] = scaler
                        else:
                            if target_col not in self.target_scalers_dict:
                                raise ValueError(f"No scaler found for target '{target_col}'")
                            targets_scaled = self.target_scalers_dict[target_col].transform(target_values).flatten()

                        all_scaled_dict[target_col] = targets_scaled

                    else:
                        # Multi-horizon for this target
                        target_cols_list = [f"{target_col}_target_h{h}"
                                           for h in range(1, self.prediction_horizon + 1)]

                        # Extract all horizon values
                        all_horizons = []
                        for tcol in target_cols_list:
                            target_values = df_processed[tcol].values[self.sequence_length:]
                            all_horizons.append(target_values)

                        # Stack into matrix: (samples, horizons)
                        targets_matrix = np.column_stack(all_horizons)

                        if fit_scaler:
                            # Scale each horizon separately for this target
                            self.target_scalers_dict[target_col] = []
                            scaled_horizons = []
                            for h in range(self.prediction_horizon):
                                scaler = StandardScaler()
                                scaled = scaler.fit_transform(targets_matrix[:, h].reshape(-1, 1)).flatten()
                                self.target_scalers_dict[target_col].append(scaler)
                                scaled_horizons.append(scaled)
                            targets_scaled = np.column_stack(scaled_horizons)
                        else:
                            # Use existing scalers
                            if target_col not in self.target_scalers_dict:
                                raise ValueError(f"No scalers found for target '{target_col}'")
                            scaled_horizons = []
                            for h, scaler in enumerate(self.target_scalers_dict[target_col]):
                                scaled = scaler.transform(targets_matrix[:, h].reshape(-1, 1)).flatten()
                                scaled_horizons.append(scaled)
                            targets_scaled = np.column_stack(scaled_horizons)

                        all_scaled_dict[target_col] = targets_scaled

                # Concatenate all targets: single-horizon -> [samples, num_targets]
                #                          multi-horizon -> [samples, num_targets * horizons]
                if self.prediction_horizon == 1:
                    # Stack as columns: each target is one column
                    y_combined = np.column_stack([all_scaled_dict[tc] for tc in self.target_columns])
                else:
                    # Flatten: [close_h1, close_h2, ..., volume_h1, volume_h2, ...]
                    y_list = []
                    for target_col in self.target_columns:
                        # all_scaled_dict[target_col] is (samples, horizons)
                        for h in range(self.prediction_horizon):
                            y_list.append(all_scaled_dict[target_col][:, h])
                    y_combined = np.column_stack(y_list)

                # Convert to tensors
                X = torch.tensor(sequences, dtype=torch.float32)
                y = torch.tensor(y_combined, dtype=torch.float32)

            elif self.prediction_horizon == 1:
                # Single-target, single horizon target scaling
                targets = targets.reshape(-1, 1)
                if fit_scaler:
                    targets_scaled = self.target_scaler.fit_transform(targets)
                else:
                    targets_scaled = self.target_scaler.transform(targets)

                # Convert to tensors
                X = torch.tensor(sequences, dtype=torch.float32)  # (n_samples, seq_len, n_features)
                y = torch.tensor(targets_scaled.flatten(), dtype=torch.float32)

            else:
                # Single-target, multi-horizon target handling - MEMORY OPTIMIZED
                target_columns = [f"{self.target_column}_target_h{h}" for h in range(1, self.prediction_horizon + 1)]

                # Extract target values directly without re-creating sequences
                # This is much more memory efficient than calling create_sequences multiple times
                all_targets = []
                for target_col in target_columns:
                    # Extract target values starting from sequence_length (matching sequence indexing)
                    target_values = df_processed[target_col].values[self.sequence_length:]
                    all_targets.append(target_values)

                # Stack into matrix: (samples, horizons)
                targets_matrix = np.column_stack(all_targets)

                # Scale each horizon separately
                if fit_scaler:
                    self.target_scalers = []
                    scaled_targets = []
                    for h in range(self.prediction_horizon):
                        scaler = StandardScaler()
                        scaled = scaler.fit_transform(targets_matrix[:, h].reshape(-1, 1)).flatten()
                        self.target_scalers.append(scaler)
                        scaled_targets.append(scaled)
                else:
                    scaled_targets = []
                    for h, scaler in enumerate(self.target_scalers):
                        scaled = scaler.transform(targets_matrix[:, h].reshape(-1, 1)).flatten()
                        scaled_targets.append(scaled)

                # Convert to tensors
                X = torch.tensor(sequences, dtype=torch.float32)  # (n_samples, seq_len, n_features)
                y = torch.tensor(np.column_stack(scaled_targets), dtype=torch.float32)  # (n_samples, horizons)
            
            if self.verbose:
                print(f"   Created {len(sequences)} sequences of length {self.sequence_length}")
                print(f"   Each sequence has {sequences.shape[2]} features")
                print(f"   Total tensor shape: {X.shape}")
            
            return X, y
        else:
            # For inference: create sequences without targets (use last available value as dummy target)
            # We'll create sequences up to the last available data point
            dummy_targets = df_processed[self.feature_columns[0]]  # Use first feature as dummy target
            temp_df = df_processed.copy()
            temp_df['__dummy_target__'] = dummy_targets
            
            from ..preprocessing.time_features import create_sequences
            sequences, _ = create_sequences(temp_df, self.sequence_length, '__dummy_target__', self.feature_columns)
            X = torch.tensor(sequences, dtype=torch.float32)
            
            return X, None
    
    def fit(
        self, 
        df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        patience: int = 10,
        verbose: bool = True
    ):
        """
        Train the time series predictor.
        
        Args:
            df: Training DataFrame
            val_df: Optional validation DataFrame  
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            patience: Early stopping patience
            verbose: Whether to print training progress
        """
        # Prepare data (target column validation happens inside prepare_data)
        # The target column might be an engineered feature created during create_features
        X_train, y_train = self.prepare_data(df, fit_scaler=True)

        if val_df is not None:
            X_val, y_val = self.prepare_data(val_df, fit_scaler=False)
            # Save validation group indices for later use in progress reporting
            if self.group_column is not None and hasattr(self, '_last_group_indices'):
                self._last_val_group_indices = self._last_group_indices.copy()
        else:
            X_val, y_val = None, None
        
        # Initialize model - use sequence model
        if len(X_train.shape) == 3:  # Sequence data: (batch, seq_len, features)
            _, seq_len, num_features = X_train.shape
            # Filter out invalid kwargs for model initialization
            model_kwargs = {k: v for k, v in self.ft_kwargs.items() if k not in ['verbose']}

            # Calculate total output size
            if self.is_multi_target:
                # Multi-target: output num_targets * prediction_horizon values
                total_output_size = self.num_targets * self.prediction_horizon
            else:
                # Single-target: output prediction_horizon values
                total_output_size = self.prediction_horizon

            self.model = SequenceFTTransformerPredictor(
                num_numerical=num_features,
                cat_cardinalities=[],  # All features are numerical for now
                sequence_length=seq_len,
                n_classes=1,  # Regression
                prediction_horizons=total_output_size,
                **model_kwargs
            ).to(self.device)
        else:  # Single timestep data: (batch, features) - fallback to original model
            num_features = X_train.shape[1]
            # Filter out invalid kwargs for model initialization
            model_kwargs = {k: v for k, v in self.ft_kwargs.items() if k not in ['verbose']}

            # Calculate total output size
            if self.is_multi_target:
                total_output_size = self.num_targets * self.prediction_horizon
            else:
                total_output_size = self.prediction_horizon

            self.model = FTTransformerPredictor(
                num_numerical=num_features,
                cat_cardinalities=[],  # All features are numerical for now
                n_classes=1,  # Regression
                prediction_horizons=total_output_size,
                **model_kwargs
            ).to(self.device)
        
        # Training setup
        dataset = TensorDataset(X_train, y_train)
        # Optimize batch size based on data size for better throughput
        optimal_batch_size = min(batch_size, max(32, len(X_train) // 20))
        dataloader = DataLoader(dataset, batch_size=optimal_batch_size, shuffle=True, num_workers=0, pin_memory=False)

        # Use more aggressive optimizer settings for faster convergence
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate,
                                     weight_decay=1e-5, eps=1e-6, betas=(0.9, 0.999))
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=max(2, patience//3), min_lr=1e-6
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.verbose = verbose
        if verbose:
            print(f"Training FT-Transformer for {self.target_column} prediction")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Training samples: {len(X_train)} (batch_size: {optimal_batch_size})")
            if X_val is not None:
                print(f"Validation samples: {len(X_val)}")

        # Mixed precision training for speed
        use_amp = torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # Training loop with optimizations
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device, non_blocking=True), batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                # Use mixed precision training if available
                if use_amp:
                    with torch.cuda.amp.autocast():
                        if len(batch_x.shape) == 3:  # Sequence data
                            outputs = self.model(batch_x).squeeze()
                        else:  # Non-sequence data
                            outputs = self.model(batch_x, None).squeeze()
                        loss = criterion(outputs, batch_y)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Handle both sequence and non-sequence models
                    if len(batch_x.shape) == 3:  # Sequence data
                        outputs = self.model(batch_x).squeeze()
                    else:  # Non-sequence data
                        outputs = self.model(batch_x, None).squeeze()

                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    optimizer.step()

                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(dataloader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation with optimizations
            val_loss = None
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_device = X_val.to(self.device, non_blocking=True)
                    y_val_device = y_val.to(self.device, non_blocking=True)

                    # Use mixed precision for validation too
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            if len(X_val.shape) == 3:  # Sequence data
                                val_outputs = self.model(X_val_device).squeeze()
                            else:  # Non-sequence data
                                val_outputs = self.model(X_val_device, None).squeeze()
                            val_loss = criterion(val_outputs, y_val_device).item()
                    else:
                        if len(X_val.shape) == 3:  # Sequence data
                            val_outputs = self.model(X_val_device).squeeze()
                        else:  # Non-sequence data
                            val_outputs = self.model(X_val_device, None).squeeze()
                        val_loss = criterion(val_outputs, y_val_device).item()

                    self.history['val_loss'].append(val_loss)

                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                scheduler.step(avg_train_loss)
            
            # Optimized progress reporting - only detailed metrics occasionally
            if verbose:
                if X_val is not None and val_loss is not None:
                    # Only calculate detailed metrics every 5 epochs or on last epoch
                    if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                        # Reuse validation outputs to avoid recomputation
                        val_pred_scaled_np = val_outputs.cpu().numpy()
                        val_actual_scaled_np = y_val_device.cpu().numpy()

                        # Inverse transform based on group-based vs single-group mode
                        if self.group_column is not None:
                            # Group-based inverse transform
                            # Multi-target group-based mode: skip detailed metrics during training
                            if self.is_multi_target:
                                print(f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")
                                continue

                            val_pred = np.zeros_like(val_pred_scaled_np)
                            val_actual = np.zeros_like(val_actual_scaled_np)

                            # Use stored group indices from validation data preparation
                            if hasattr(self, '_last_val_group_indices'):
                                for group_value in self.group_target_scalers.keys():
                                    group_mask = np.array([g == group_value for g in self._last_val_group_indices])
                                    if not group_mask.any():
                                        continue

                                    if self.prediction_horizon == 1:
                                        val_pred[group_mask] = self.group_target_scalers[group_value].inverse_transform(
                                            val_pred_scaled_np[group_mask].reshape(-1, 1)
                                        ).flatten()
                                        val_actual[group_mask] = self.group_target_scalers[group_value].inverse_transform(
                                            val_actual_scaled_np[group_mask].reshape(-1, 1)
                                        ).flatten()
                                    else:
                                        val_pred[group_mask] = self.group_target_scalers[group_value].inverse_transform(
                                            val_pred_scaled_np[group_mask]
                                        )
                                        val_actual[group_mask] = self.group_target_scalers[group_value].inverse_transform(
                                            val_actual_scaled_np[group_mask]
                                        )
                            else:
                                # Fallback: just show loss without detailed metrics
                                print(f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")
                                continue

                            val_pred = val_pred.flatten() if self.prediction_horizon == 1 else val_pred
                            val_actual = val_actual.flatten() if self.prediction_horizon == 1 else val_actual
                        else:
                            # Single-group inverse transform
                            if self.is_multi_target:
                                # Multi-target mode: don't inverse transform for progress reporting
                                # Just use scaled values for loss comparison
                                # TODO: Add multi-target metrics in future
                                print(f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")
                                continue
                            elif self.prediction_horizon == 1:
                                # Single-target, single-horizon
                                val_pred = self.target_scaler.inverse_transform(val_pred_scaled_np.reshape(-1, 1)).flatten()
                                val_actual = self.target_scaler.inverse_transform(val_actual_scaled_np.reshape(-1, 1)).flatten()
                            else:
                                # Single-target, multi-horizon: inverse transform each horizon
                                val_pred_list = []
                                val_actual_list = []
                                for h in range(self.prediction_horizon):
                                    val_pred_list.append(
                                        self.target_scalers[h].inverse_transform(val_pred_scaled_np[:, h].reshape(-1, 1)).flatten()
                                    )
                                    val_actual_list.append(
                                        self.target_scalers[h].inverse_transform(val_actual_scaled_np[:, h].reshape(-1, 1)).flatten()
                                    )
                                val_pred = np.column_stack(val_pred_list)
                                val_actual = np.column_stack(val_actual_list)

                        # Calculate MAPE and MAE (for single horizon or average across horizons)
                        if self.prediction_horizon == 1:
                            val_mae = np.mean(np.abs(val_actual - val_pred))
                            val_mape = np.mean(np.abs((val_actual - val_pred) / (val_actual + 1e-8))) * 100
                        else:
                            # Average metrics across all horizons
                            val_mae = np.mean(np.abs(val_actual - val_pred))
                            val_mape = np.mean(np.abs((val_actual - val_pred) / (val_actual + 1e-8))) * 100

                        print(f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}, Val MAE = ${val_mae:.2f}, Val MAPE = {val_mape:.2f}%")
                    else:
                        # Minimal progress indication
                        print(f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.6f}")
        
        if verbose:
            print("Training completed!")
    
    def predict(self, df: pd.DataFrame, return_group_info: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
        """
        Make predictions on new data.

        Args:
            df: DataFrame with same structure as training data
            return_group_info: If True and group_column is set, returns (predictions, group_indices)

        Returns:
            predictions: Numpy array of predictions (in original scale)
            group_indices: List of group values for each prediction (only if return_group_info=True)
        """
        if self.model is None:
            raise RuntimeError("Model must be trained first. Call fit().")

        # Clear feature cache before prediction to free memory
        self._feature_cache.clear()
        gc.collect()  # Force garbage collection to free memory

        X, _ = self.prepare_data(df, fit_scaler=False)

        self.model.eval()
        with torch.no_grad():
            # Use batched inference to avoid OOM errors
            batch_size = 256  # Process predictions in smaller batches
            num_samples = X.shape[0]
            all_predictions = []

            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                X_batch = X[i:batch_end]

                # Handle both sequence and non-sequence models
                if len(X_batch.shape) == 3:  # Sequence data
                    batch_preds = self.model(X_batch.to(self.device))
                else:  # Non-sequence data
                    batch_preds = self.model(X_batch.to(self.device), None)

                # Move to CPU immediately to free GPU memory
                all_predictions.append(batch_preds.cpu())

                # Clear GPU cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Concatenate all batch predictions
            predictions_scaled = torch.cat(all_predictions, dim=0).numpy()

            # Handle group-based vs single-group inverse transform
            if self.group_column is not None:
                # Group-based inverse transform
                if not hasattr(self, '_last_group_indices') or len(self._last_group_indices) != len(predictions_scaled):
                    raise RuntimeError("Group indices not available or mismatched. This shouldn't happen.")

                if self.is_multi_target:
                    # Multi-target group-based inverse transform
                    # predictions_scaled shape: (n_samples, num_targets) or (n_samples, num_targets * horizons)

                    predictions_dict = {}  # {target_name: predictions_array}

                    for idx, target_col in enumerate(self.target_columns):
                        # Initialize array for this target's predictions
                        if self.prediction_horizon == 1:
                            target_preds = np.zeros(len(predictions_scaled))
                        else:
                            target_preds = np.zeros((len(predictions_scaled), self.prediction_horizon))

                        # Inverse transform each group separately
                        for group_value in self.group_target_scalers.keys():
                            group_mask = np.array([g == group_value for g in self._last_group_indices])
                            if not group_mask.any():
                                continue

                            if self.prediction_horizon == 1:
                                # Single-horizon: extract column for this target
                                col_idx = idx
                                group_preds_scaled = predictions_scaled[group_mask, col_idx].reshape(-1, 1)
                                group_preds_original = self.group_target_scalers[group_value][target_col].inverse_transform(group_preds_scaled).flatten()
                                target_preds[group_mask] = group_preds_original
                            else:
                                # Multi-horizon: extract horizons for this target
                                # Layout: [close_h1, close_h2, ..., volume_h1, volume_h2, ...]
                                start_idx = idx * self.prediction_horizon
                                end_idx = start_idx + self.prediction_horizon

                                # Extract all horizons for this target in this group
                                group_horizons_scaled = predictions_scaled[group_mask, start_idx:end_idx]

                                # Inverse transform each horizon separately
                                group_horizons_original = np.zeros_like(group_horizons_scaled)
                                for h in range(self.prediction_horizon):
                                    col_idx = start_idx + h - start_idx  # Relative index within group_horizons_scaled
                                    horizon_scaled = group_horizons_scaled[:, col_idx].reshape(-1, 1)
                                    horizon_original = self.group_target_scalers[group_value][target_col][h].inverse_transform(horizon_scaled).flatten()
                                    group_horizons_original[:, col_idx] = horizon_original

                                target_preds[group_mask] = group_horizons_original

                        predictions_dict[target_col] = target_preds

                    if return_group_info:
                        return predictions_dict, self._last_group_indices
                    else:
                        return predictions_dict

                else:
                    # Single-target group-based inverse transform
                    predictions = np.zeros_like(predictions_scaled)

                    # Inverse transform each prediction using its group's scaler
                    for group_value in self.group_target_scalers.keys():
                        # Find indices for this group
                        group_mask = np.array([g == group_value for g in self._last_group_indices])
                        if not group_mask.any():
                            continue

                        group_preds_scaled = predictions_scaled[group_mask]

                        if self.prediction_horizon == 1:
                            # Single horizon
                            group_preds_scaled = group_preds_scaled.reshape(-1, 1)
                            group_preds_original = self.group_target_scalers[group_value].inverse_transform(group_preds_scaled)
                            predictions[group_mask] = group_preds_original.flatten()
                        else:
                            # Multi-horizon: use same scaler for all horizons in this group
                            group_preds_original = self.group_target_scalers[group_value].inverse_transform(group_preds_scaled)
                            predictions[group_mask] = group_preds_original

                    final_predictions = predictions.flatten() if self.prediction_horizon == 1 else predictions

                    if return_group_info:
                        return final_predictions, self._last_group_indices
                    else:
                        return final_predictions

            else:
                # Single-group inverse transform (original behavior)
                if self.is_multi_target:
                    # Multi-target inverse transform
                    # predictions_scaled shape: (n_samples, num_targets * horizons)
                    #   For single-horizon: (n_samples, num_targets)
                    #   For multi-horizon: (n_samples, num_targets * prediction_horizon)

                    predictions_dict = {}  # {target_name: predictions_array}

                    for idx, target_col in enumerate(self.target_columns):
                        if self.prediction_horizon == 1:
                            # Single-horizon: extract column for this target
                            target_preds_scaled = predictions_scaled[:, idx].reshape(-1, 1)
                            target_preds = self.target_scalers_dict[target_col].inverse_transform(target_preds_scaled).flatten()
                            predictions_dict[target_col] = target_preds
                        else:
                            # Multi-horizon: extract horizons for this target
                            # Layout: [close_h1, close_h2, ..., volume_h1, volume_h2, ...]
                            start_idx = idx * self.prediction_horizon
                            end_idx = start_idx + self.prediction_horizon

                            # Inverse transform each horizon for this target
                            horizons_list = []
                            for h in range(self.prediction_horizon):
                                col_idx = start_idx + h
                                horizon_preds_scaled = predictions_scaled[:, col_idx].reshape(-1, 1)
                                horizon_preds = self.target_scalers_dict[target_col][h].inverse_transform(horizon_preds_scaled).flatten()
                                horizons_list.append(horizon_preds)

                            # Stack into (n_samples, horizons) for this target
                            predictions_dict[target_col] = np.column_stack(horizons_list)

                    return predictions_dict

                elif self.prediction_horizon == 1:
                    # Single-target, single horizon: reshape to (n_samples, 1) for inverse transform
                    predictions_scaled = predictions_scaled.reshape(-1, 1)
                    predictions = self.target_scaler.inverse_transform(predictions_scaled)
                    return predictions.flatten()
                else:
                    # Single-target, multi-horizon: predictions_scaled shape is (n_samples, horizons)
                    # Inverse transform each horizon separately using its own scaler
                    predictions_list = []
                    for h in range(self.prediction_horizon):
                        horizon_preds = predictions_scaled[:, h].reshape(-1, 1)
                        horizon_preds_original = self.target_scalers[h].inverse_transform(horizon_preds)
                        predictions_list.append(horizon_preds_original.flatten())

                    # Stack predictions: (n_samples, horizons)
                    predictions = np.column_stack(predictions_list)
                    return predictions

    def predict_from_features(self, df_processed: pd.DataFrame) -> np.ndarray:
        """
        Make predictions from already-processed features.

        Note: This method does not support group-based scaling. Use predict() instead.

        Args:
            df_processed: DataFrame with preprocessed features

        Returns:
            predictions: Numpy array of predictions (in original scale)
        """
        if self.model is None:
            raise RuntimeError("Model must be trained first. Call fit().")

        # Group-based mode is not supported in this optimized path
        if self.group_column is not None:
            raise NotImplementedError(
                "predict_from_features() does not support group-based scaling. "
                "Please use predict() instead."
            )

        # Skip feature preprocessing since it's already done
        from ..preprocessing.time_features import create_sequences

        # Create sequences directly from processed features
        if self.target_column in df_processed.columns:
            sequences, _ = create_sequences(df_processed, self.sequence_length, self.target_column, self.feature_columns)
        else:
            # Use first feature as dummy target for sequence creation
            dummy_target = df_processed[self.feature_columns[0]]
            temp_df = df_processed.copy()
            temp_df['__dummy_target__'] = dummy_target
            sequences, _ = create_sequences(temp_df, self.sequence_length, '__dummy_target__', self.feature_columns)

        X = torch.tensor(sequences, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            if len(X.shape) == 3:  # Sequence data
                predictions_scaled = self.model(X.to(self.device))
            else:  # Non-sequence data
                predictions_scaled = self.model(X.to(self.device), None)

            # Convert to numpy
            predictions_scaled = predictions_scaled.cpu().numpy()

            # Handle single vs multi-horizon predictions
            if self.prediction_horizon == 1:
                # Single horizon: reshape to (n_samples, 1) for inverse transform
                predictions_scaled = predictions_scaled.reshape(-1, 1)
                predictions = self.target_scaler.inverse_transform(predictions_scaled)
                return predictions.flatten()
            else:
                # Multi-horizon: predictions_scaled shape is (n_samples, horizons)
                # Inverse transform each horizon separately using its own scaler
                predictions_list = []
                for h in range(self.prediction_horizon):
                    horizon_preds = predictions_scaled[:, h].reshape(-1, 1)
                    horizon_preds_original = self.target_scalers[h].inverse_transform(horizon_preds)
                    predictions_list.append(horizon_preds_original.flatten())

                # Stack predictions: (n_samples, horizons)
                predictions = np.column_stack(predictions_list)
                return predictions

    def evaluate(self, df: pd.DataFrame, per_group: bool = False) -> Dict:
        """
        Evaluate model performance.

        Returns nested dict structure based on configuration:
        - Single-horizon, no groups: {'MAE': ..., 'RMSE': ..., ...}
        - Multi-horizon, no groups: {'overall': {...}, 'horizon_1': {...}, ...}
        - Single-horizon with groups (per_group=True): {'overall': {...}, 'AAPL': {...}, 'GOOGL': {...}, ...}
        - Multi-horizon with groups (per_group=True): {
            'overall': {'overall': {...}, 'horizon_1': {...}, ...},
            'AAPL': {'overall': {...}, 'horizon_1': {...}, ...},
            'GOOGL': {...}, ...
          }

        Args:
            df: DataFrame with raw data (will be processed to extract target)
            per_group: If True and group_column is set, return per-group metrics breakdown

        Returns:
            metrics: Dictionary of evaluation metrics (structure depends on configuration)
        """
        # Clear feature cache before evaluation to free memory
        self._feature_cache.clear()
        gc.collect()  # Force garbage collection to free memory

        # Process features first to get the target column
        df_processed = self.prepare_features(df, fit_scaler=False)

        # Now check if target column(s) exist after processing
        if self.is_multi_target:
            # Check all target columns exist
            missing_targets = [col for col in self.target_columns if col not in df_processed.columns]
            if missing_targets:
                raise ValueError(f"Target columns {missing_targets} not found after feature engineering")
        else:
            # Single-target check
            if self.target_column not in df_processed.columns:
                raise ValueError(f"Target column '{self.target_column}' not found after feature engineering")

        # Check if we should do per-group evaluation
        if per_group and self.group_column is not None:
            return self._evaluate_per_group(df, df_processed)
        else:
            # Standard evaluation (backward compatible)
            return self._evaluate_standard(df_processed)

    def _evaluate_standard(self, df_processed: pd.DataFrame) -> Dict:
        """Standard evaluation without per-group breakdown."""
        predictions = self.predict(df_processed.copy())  # Returns dict for multi-target

        # Check if multi-target
        if self.is_multi_target:
            # Multi-target evaluation: return metrics per target
            from ..core.utils import calculate_metrics, calculate_metrics_multi_horizon

            metrics_dict = {}

            for target_col in self.target_columns:
                # Get actual values for this target
                if self.sequence_length > 1:
                    actual = df_processed[target_col].values[self.sequence_length:]
                else:
                    actual = df_processed[target_col].values

                target_predictions = predictions[target_col]

                # Handle single vs multi-horizon
                if self.prediction_horizon == 1:
                    # Single-horizon
                    min_len = min(len(actual), len(target_predictions))
                    actual = actual[:min_len]
                    target_predictions = target_predictions[:min_len]

                    metrics_dict[target_col] = calculate_metrics(actual, target_predictions)
                else:
                    # Multi-horizon
                    # For multi-horizon, we need predictions.shape[0] actual future values
                    # where each prediction uses up to prediction_horizon future values
                    num_preds = target_predictions.shape[0]
                    # Calculate how many actual values we need
                    needed_actual_len = num_preds + self.prediction_horizon - 1
                    # Only use predictions for which we have enough actual values
                    if len(actual) < needed_actual_len:
                        # Not enough actual values - trim predictions
                        valid_pred_count = len(actual) - self.prediction_horizon + 1
                        if valid_pred_count <= 0:
                            # Skip this target if we don't have enough data
                            continue
                        num_preds = valid_pred_count

                    actual_aligned = actual[:num_preds + self.prediction_horizon - 1]
                    predictions_aligned = target_predictions[:num_preds]

                    metrics_dict[target_col] = calculate_metrics_multi_horizon(
                        actual_aligned,
                        predictions_aligned,
                        self.prediction_horizon
                    )

            return metrics_dict

        else:
            # Single-target evaluation (existing behavior)
            # For sequences, we need to align the actual values with predictions
            if self.sequence_length > 1:
                actual = df_processed[self.target_column].values[self.sequence_length:]
            else:
                actual = df_processed[self.target_column].values

            # Handle single vs multi-horizon evaluation
            if self.prediction_horizon == 1:
                # Single-horizon: return simple metrics dict (backward compatible)
                from ..core.utils import calculate_metrics

                # Ensure same length
                min_len = min(len(actual), len(predictions))
                actual = actual[:min_len]
                predictions = predictions[:min_len]

                return calculate_metrics(actual, predictions)
            else:
                # Multi-horizon: return nested dict with per-horizon metrics
                from ..core.utils import calculate_metrics_multi_horizon

                # For multi-horizon, predictions is 2D: (n_samples, horizons)
                # Align actual values - we need enough actual values for all horizons
                min_len = min(len(actual), predictions.shape[0])
                actual_aligned = actual[:min_len + self.prediction_horizon - 1]  # Extra values for future horizons
                predictions_aligned = predictions[:min_len]

                return calculate_metrics_multi_horizon(
                    actual_aligned,
                    predictions_aligned,
                    self.prediction_horizon
                )

    def _evaluate_per_group(self, df: pd.DataFrame, df_processed: pd.DataFrame) -> Dict:
        """
        Evaluate performance per group (e.g., per stock symbol).

        Returns nested dict with overall metrics plus per-group breakdown.
        """
        from ..core.utils import calculate_metrics, calculate_metrics_multi_horizon

        # Get predictions with group information
        predictions, group_indices = self.predict(df_processed.copy(), return_group_info=True)

        # Get unique groups
        unique_groups = sorted(set(group_indices))

        # Check if multi-target
        if self.is_multi_target:
            # Multi-target per-group evaluation
            # Structure: {
            #   'overall': {target: metrics_dict, ...},
            #   'group1': {target: metrics_dict, ...},
            #   ...
            # }
            all_metrics = {}

            for target_col in self.target_columns:
                # Get actual values for this target
                if self.sequence_length > 1:
                    actual_base = df_processed[target_col].values[self.sequence_length:]
                else:
                    actual_base = df_processed[target_col].values

                target_predictions = predictions[target_col]

                # Calculate per-group metrics for this target
                for group_value in unique_groups:
                    # Find indices for this group
                    group_mask = np.array([g == group_value for g in group_indices])
                    group_pred_indices = np.where(group_mask)[0]

                    if self.prediction_horizon == 1:
                        # Single-horizon
                        group_preds = target_predictions[group_mask]
                        group_actual = actual_base[group_pred_indices]
                        min_len = min(len(group_actual), len(group_preds))

                        group_metrics = calculate_metrics(group_actual[:min_len], group_preds[:min_len])
                    else:
                        # Multi-horizon
                        group_preds = target_predictions[group_mask, :]
                        group_actual_aligned = actual_base[group_pred_indices[0]:group_pred_indices[-1] + self.prediction_horizon]
                        min_len = min(len(group_actual_aligned) - self.prediction_horizon + 1, group_preds.shape[0])

                        group_metrics = calculate_metrics_multi_horizon(
                            group_actual_aligned[:min_len + self.prediction_horizon - 1],
                            group_preds[:min_len],
                            self.prediction_horizon
                        )

                    # Store metrics in nested structure
                    group_key = str(group_value)
                    if group_key not in all_metrics:
                        all_metrics[group_key] = {}
                    all_metrics[group_key][target_col] = group_metrics

                # Calculate overall metrics for this target
                if self.prediction_horizon == 1:
                    min_len = min(len(actual_base), len(target_predictions))
                    overall_target_metrics = calculate_metrics(actual_base[:min_len], target_predictions[:min_len])
                else:
                    min_len = min(len(actual_base), target_predictions.shape[0])
                    actual_aligned = actual_base[:min_len + self.prediction_horizon - 1]
                    overall_target_metrics = calculate_metrics_multi_horizon(
                        actual_aligned,
                        target_predictions[:min_len],
                        self.prediction_horizon
                    )

                # Store overall metrics for this target
                if 'overall' not in all_metrics:
                    all_metrics['overall'] = {}
                all_metrics['overall'][target_col] = overall_target_metrics

            return all_metrics

        else:
            # Single-target per-group evaluation (existing behavior)
            # Prepare actual values
            if self.sequence_length > 1:
                actual_base = df_processed[self.target_column].values[self.sequence_length:]
            else:
                actual_base = df_processed[self.target_column].values

            # Storage for per-group metrics
            all_metrics = {}

            # Calculate metrics per group
            for group_value in unique_groups:
                # Find indices for this group
                group_mask = np.array([g == group_value for g in group_indices])
                group_preds = predictions[group_mask] if self.prediction_horizon == 1 else predictions[group_mask, :]

                # Get actual values for this group
                group_pred_indices = np.where(group_mask)[0]

                if self.prediction_horizon == 1:
                    # Single-horizon
                    group_actual = actual_base[group_pred_indices]
                    min_len = min(len(group_actual), len(group_preds))

                    group_metrics = calculate_metrics(group_actual[:min_len], group_preds[:min_len])
                    all_metrics[str(group_value)] = group_metrics

                else:
                    # Multi-horizon
                    group_actual_aligned = actual_base[group_pred_indices[0]:group_pred_indices[-1] + self.prediction_horizon]
                    min_len = min(len(group_actual_aligned) - self.prediction_horizon + 1, group_preds.shape[0])

                    group_metrics = calculate_metrics_multi_horizon(
                        group_actual_aligned[:min_len + self.prediction_horizon - 1],
                        group_preds[:min_len],
                        self.prediction_horizon
                    )
                    all_metrics[str(group_value)] = group_metrics

            # Calculate overall metrics across all groups
            if self.prediction_horizon == 1:
                # Single-horizon overall
                min_len = min(len(actual_base), len(predictions))
                overall_metrics = calculate_metrics(actual_base[:min_len], predictions[:min_len])
            else:
                # Multi-horizon overall
                min_len = min(len(actual_base), predictions.shape[0])
                actual_aligned = actual_base[:min_len + self.prediction_horizon - 1]
                overall_metrics = calculate_metrics_multi_horizon(
                    actual_aligned,
                    predictions[:min_len],
                    self.prediction_horizon
                )

            # Combine into final structure
            result = {'overall': overall_metrics}
            result.update(all_metrics)

            return result

    def evaluate_from_features(self, df_processed: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance from already-processed features.

        Args:
            df_processed: DataFrame with preprocessed features

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        if self.target_column not in df_processed.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in processed features")

        # Use predict() instead of predict_from_features() if group-based scaling is enabled
        if self.group_column is not None:
            predictions = self.predict(df_processed)
        else:
            predictions = self.predict_from_features(df_processed)

        # Get the original target column name (which should NOT be scaled in df_processed)
        original_target = getattr(self, 'original_target_column', self.target_column)

        # For sequences, align actual values with predictions
        # Use the original unscaled target column from df_processed
        if hasattr(df_processed, 'iloc'):
            actual = df_processed[original_target].iloc[self.sequence_length:self.sequence_length + len(predictions)].values
        else:
            actual = df_processed[original_target].values

        # Ensure same length (take minimum to be safe)
        min_len = min(len(actual), len(predictions))
        actual = actual[:min_len]
        predictions = predictions[:min_len]

        from ..core.utils import calculate_metrics
        return calculate_metrics(actual, predictions)

    def save(self, path: str):
        """Save the trained model and preprocessors."""
        if self.model is None:
            raise RuntimeError("No model to save. Train first.")

        state = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'num_features': len(self.feature_columns),  # Save the actual number of features used
            'target_column': self.target_column,
            'target_columns': self.target_columns,  # Save list
            'is_multi_target': self.is_multi_target,  # Save flag
            'num_targets': self.num_targets,  # Save count
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,  # Explicitly save
            'ft_kwargs': self.ft_kwargs,
            'history': self.history,
            # Group-based scaling
            'group_column': self.group_column,
            'group_feature_scalers': self.group_feature_scalers,
            'group_target_scalers': self.group_target_scalers
        }

        # Add target scalers based on mode
        if not self.is_multi_target:
            # Single-target scalers
            state['target_scaler'] = self.target_scaler
            state['target_scalers'] = self.target_scalers
        else:
            # Multi-target scalers dict
            state['target_scalers_dict'] = self.target_scalers_dict

        torch.save(state, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, **kwargs):
        """Load a saved model."""
        state = torch.load(path, map_location='cpu', weights_only=False)

        # Determine target_column format (backward compatible)
        if 'target_columns' in state:
            target_column = state['target_columns']
        else:
            # Backward compatibility: single target
            target_column = state['target_column']

        # Create predictor with correct parameters
        predictor = cls(
            target_column=target_column,
            sequence_length=state.get('sequence_length', 5),
            prediction_horizon=state.get('prediction_horizon', 1),
            group_column=state.get('group_column', None),
            **state['ft_kwargs']
        )

        # Restore feature scaler and feature columns
        predictor.scaler = state['scaler']
        predictor.feature_columns = state['feature_columns']
        predictor.history = state.get('history', {'train_loss': [], 'val_loss': []})

        # Restore target scalers based on mode
        if predictor.is_multi_target:
            # Multi-target: restore target_scalers_dict
            predictor.target_scalers_dict = state.get('target_scalers_dict', {})
        else:
            # Single-target: restore target_scaler and target_scalers
            predictor.target_scaler = state.get('target_scaler', StandardScaler())
            predictor.target_scalers = state.get('target_scalers', [])

        # Restore group scalers (if present)
        predictor.group_feature_scalers = state.get('group_feature_scalers', {})
        predictor.group_target_scalers = state.get('group_target_scalers', {})

        # Recreate model with correct output size
        # Use saved num_features to ensure model architecture matches the saved weights
        num_features = state.get('num_features', len(predictor.feature_columns))

        # Calculate total output size
        if predictor.is_multi_target:
            total_output_size = predictor.num_targets * predictor.prediction_horizon
        else:
            total_output_size = predictor.prediction_horizon

        # Filter out non-model parameters from ft_kwargs
        model_kwargs = {k: v for k, v in predictor.ft_kwargs.items()
                       if k not in ['verbose']}

        if predictor.sequence_length > 1:
            predictor.model = SequenceFTTransformerPredictor(
                num_numerical=num_features,
                cat_cardinalities=[],
                sequence_length=predictor.sequence_length,
                n_classes=1,
                prediction_horizons=total_output_size,
                **model_kwargs
            )
        else:
            predictor.model = FTTransformerPredictor(
                num_numerical=num_features,
                cat_cardinalities=[],
                n_classes=1,
                prediction_horizons=total_output_size,
                **model_kwargs
            )
        predictor.model.load_state_dict(state['model_state_dict'])
        predictor.model.to(predictor.device)

        print(f"Model loaded from {path}")
        return predictor