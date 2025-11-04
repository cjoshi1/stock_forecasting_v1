"""
Generic Time Series Predictor using FT-Transformer.

A base class for time series prediction that can be extended for different domains.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import gc
import math
from typing import Optional, Dict, Any, Tuple, Union, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from .base.model_factory import ModelFactory
from ..preprocessing.time_features import create_date_features, create_shifted_targets
from ..preprocessing.scaler_factory import ScalerFactory


class TimeSeriesPredictor:
    """Generic FT-Transformer wrapper for time series prediction."""
    
    def __init__(
        self,
        target_column: Union[str, list],
        sequence_length: int = 5,
        prediction_horizon: int = 1,
        group_columns: Optional[Union[str, list]] = None,
        categorical_columns: Optional[Union[str, list]] = None,
        model_type: str = 'ft_transformer',
        scaler_type: str = 'standard',
        use_lagged_target_features: bool = False,
        verbose: bool = False,
        **model_kwargs
    ):
        """
        Args:
            target_column: Name(s) of target column(s) to predict
                          - str: Single-target prediction (e.g., 'close')
                          - List[str]: Multi-target prediction (e.g., ['close', 'volume'])
            sequence_length: Number of historical time steps to use for prediction
            prediction_horizon: Number of steps ahead to predict (1=single, >1=multi-horizon)
            group_columns: Column(s) for group-based scaling and sequence boundaries
                          - str: Single column (e.g., 'symbol')
                          - List[str]: Multiple columns (e.g., ['symbol', 'sector'])
                          - None: No grouping
            categorical_columns: Column(s) to encode and pass to model as categorical features
                                - str: Single column
                                - List[str]: Multiple columns
                                - None: No categorical features
            model_type: Type of model to use ('ft_transformer', 'csn_transformer', etc.)
            scaler_type: Type of scaler to use for normalization
                        - 'standard': StandardScaler (mean=0, std=1) - default
                        - 'minmax': MinMaxScaler (range [0, 1])
                        - 'robust': RobustScaler (median and IQR, robust to outliers)
                        - 'maxabs': MaxAbsScaler (range [-1, 1], preserves sparsity)
                        - 'onlymax': OnlyMaxScaler (divide by max only, no shifting)
            use_lagged_target_features: Whether to include target columns in input sequences
                                       If True, enables autoregressive modeling by including target
                                       values in the sequence window
            verbose: Whether to print detailed processing information
            **model_kwargs: Model-specific hyperparameters (d_model, num_heads, num_layers, etc.)
        """
        # Normalize target_column to list for uniform handling
        if isinstance(target_column, str):
            self.target_columns = [target_column]
            self.is_multi_target = False
        else:
            self.target_columns = list(target_column)
            self.is_multi_target = len(self.target_columns) > 1

        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        # Normalize group_columns to list
        if group_columns is None:
            self.group_columns = []
        elif isinstance(group_columns, str):
            self.group_columns = [group_columns]
        else:
            self.group_columns = list(group_columns)

        # Normalize categorical_columns to list
        if categorical_columns is None:
            self.categorical_columns = []
        elif isinstance(categorical_columns, str):
            self.categorical_columns = [categorical_columns]
        else:
            self.categorical_columns = list(categorical_columns)

        # Ensure all group_columns are in categorical_columns
        for col in self.group_columns:
            if col not in self.categorical_columns:
                self.categorical_columns.append(col)

        self.model_type = model_type
        self.scaler_type = scaler_type
        self.use_lagged_target_features = use_lagged_target_features
        self.model_kwargs = model_kwargs
        self.num_targets = len(self.target_columns)

        # Categorical feature encoding
        self.cat_encoders = {}  # {col_name: LabelEncoder}
        self.cat_cardinalities = []  # [vocab_size1, vocab_size2, ...]
        self.numerical_columns = None  # Will be set during training

        # Will be set during training
        self.model = None

        # Single-group scalers (used when group_columns is empty)
        self.scaler = ScalerFactory.create_scaler(scaler_type)  # For features

        # Target scalers structure - now uses per-horizon scalers
        # Both single-target and multi-target use target_scalers_dict
        # Dict keys are shifted target column names (e.g., 'close_target_h1', 'close_target_h2')
        # This allows each horizon to have its own scaler for better accuracy
        self.target_scalers_dict = {}  # Will be populated during fit with per-horizon scalers

        # Multi-group scalers (used when group_columns is provided)
        # Structure: {group_key: Scaler} for features
        # group_key can be:
        #   - str: Single column (e.g., 'AAPL')
        #   - tuple: Multiple columns (e.g., ('AAPL', 'Tech'))
        # For targets:
        #   - Single-target: {group_key: StandardScaler}
        #   - Multi-target: {group_key: Dict[target_name, StandardScaler]}
        self.group_feature_scalers = {}  # Dict[group_key, StandardScaler]
        self.group_target_scalers = {}   # Dict[group_key, StandardScaler or Dict]

        self.feature_columns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training history
        self.history = {'train_loss': [], 'val_loss': []}
        self.verbose = verbose  # Set from constructor parameter

        # Feature caching to avoid recomputation
        self._feature_cache = {}
        self._cache_enabled = True

    def _get_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Generate a hash key for DataFrame caching."""
        import hashlib
        # Use shape, column names, and sample of data for hash
        key_data = f"{df.shape}_{list(df.columns)}_{df.iloc[0].to_dict() if len(df) > 0 else {}}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _create_group_key(self, row_or_df):
        """
        Create a composite group key from multiple columns.

        Args:
            row_or_df: Can be a Series (single row) or DataFrame

        Returns:
            - If no group columns: None
            - If single group column: scalar value (e.g., 'AAPL')
            - If multiple group columns: tuple (e.g., ('AAPL', 'Tech'))

        Examples:
            >>> self.group_columns = ['symbol']
            >>> _create_group_key(df.iloc[0])
            'AAPL'

            >>> self.group_columns = ['symbol', 'sector']
            >>> _create_group_key(df.iloc[0])
            ('AAPL', 'Tech')
        """
        if not self.group_columns:
            return None

        if isinstance(row_or_df, pd.Series):
            # Single row
            values = tuple(row_or_df[col] for col in self.group_columns)
            # If single column, return scalar instead of tuple
            return values[0] if len(self.group_columns) == 1 else values
        else:
            # DataFrame - return Series of keys
            if len(self.group_columns) == 1:
                return row_or_df[self.group_columns[0]]
            else:
                return row_or_df[self.group_columns].apply(tuple, axis=1)

    def _detect_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect time-related column in DataFrame."""
        possible_time_cols = ['timestamp', 'date', 'datetime', 'time', 'Date', 'Timestamp', 'DateTime']
        for col in possible_time_cols:
            if col in df.columns:
                return col
        return None

    def _encode_categorical_features(self, df: pd.DataFrame, fit_encoders: bool = False) -> pd.DataFrame:
        """
        Label encode categorical features.

        Args:
            df: DataFrame with categorical columns
            fit_encoders: Whether to fit new encoders (True for training data)

        Returns:
            DataFrame with encoded categorical columns
        """
        if not self.categorical_columns:
            return df

        df = df.copy()

        for col in self.categorical_columns:
            if col not in df.columns:
                raise ValueError(f"Categorical column '{col}' not found in DataFrame")

            if fit_encoders:
                # Fit new encoder
                encoder = LabelEncoder()
                # Convert to string to handle any data type
                df[col] = df[col].astype(str)
                encoder.fit(df[col])
                self.cat_encoders[col] = encoder

                # Store cardinality (number of unique categories)
                cardinality = len(encoder.classes_)
                if self.verbose:
                    print(f"   Encoded '{col}': {cardinality} unique categories")
            else:
                # Use existing encoder
                if col not in self.cat_encoders:
                    raise ValueError(f"No encoder found for categorical column '{col}'. "
                                   f"Must fit on training data first.")
                encoder = self.cat_encoders[col]
                df[col] = df[col].astype(str)

            # Transform
            try:
                df[col] = encoder.transform(df[col])
            except ValueError as e:
                # Handle unseen categories
                unseen_categories = set(df[col].unique()) - set(encoder.classes_)
                raise ValueError(
                    f"Unseen categories in column '{col}': {unseen_categories}. "
                    f"Known categories: {set(encoder.classes_)}. "
                    f"Cannot handle unseen categories during prediction."
                ) from e

        # Compute and store cardinalities (after all encoders are fitted)
        if fit_encoders:
            self.cat_cardinalities = [
                len(self.cat_encoders[col].classes_)
                for col in self.categorical_columns
            ]

            if self.verbose:
                print(f"   Categorical cardinalities: {dict(zip(self.categorical_columns, self.cat_cardinalities))}")

        return df

    def _create_sequences_with_categoricals(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        numerical_columns: List[str],
        categorical_columns: List[str]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for numerical features and extract categorical features.

        For numerical features: Creates 3D sequences using sliding window (standard time series approach)
        For categorical features: Extracts values from the LAST timestep of each sequence (static per sequence)

        Args:
            df: DataFrame with encoded features (numerical and categorical already prepared)
            sequence_length: Length of lookback window
            numerical_columns: List of numerical feature columns
            categorical_columns: List of categorical feature columns (already label encoded)

        Returns:
            (X_num, X_cat) tuple:
                X_num: (num_sequences, seq_len, num_numerical) - 3D numerical sequences
                X_cat: (num_sequences, num_categorical) - 2D categorical features or None if no categorical columns

        Example:
            If df has 100 rows, sequence_length=10:
            - X_num will have shape (90, 10, num_numerical) - 90 sequences of length 10
            - X_cat will have shape (90, num_categorical) - categorical from last timestep of each sequence
        """
        from ..preprocessing.time_features import create_input_variable_sequence

        # Step 1: Create numerical sequences (3D)
        X_num = create_input_variable_sequence(
            df,
            sequence_length,
            feature_columns=numerical_columns
        )

        # Step 2: Extract categorical from last timestep of each sequence (2D)
        if categorical_columns:
            # For each sequence starting at index i, we use categorical from index i+sequence_length-1 (last timestep)
            # The sequences start at index sequence_length (first sequence uses rows 0 to sequence_length-1)
            # So categorical values should be extracted from indices: sequence_length-1, sequence_length, ..., len(df)-2
            # which simplifies to: df[categorical_columns].values[sequence_length-1:len(df)-1]

            # However, create_input_variable_sequence creates sequences for indices sequence_length to len(df)
            # Each sequence i corresponds to df rows [i-sequence_length:i]
            # The last timestep of sequence i is at df index i-1
            # So for num_sequences sequences, we need categorical from indices [sequence_length-1:sequence_length-1+num_sequences]

            num_sequences = len(X_num)
            # Extract categorical values: for sequence i (where i goes from 0 to num_sequences-1),
            # we need categorical from df index (sequence_length - 1 + i)
            cat_indices = np.arange(sequence_length - 1, sequence_length - 1 + num_sequences)
            X_cat = df[categorical_columns].values[cat_indices]

            # Verify shape consistency
            assert X_cat.shape[0] == X_num.shape[0], \
                f"Mismatch: X_num has {X_num.shape[0]} sequences but X_cat has {X_cat.shape[0]}"
        else:
            X_cat = None

        return X_num, X_cat

    def _create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create base features without encoding or scaling.

        This method can be overridden by subclasses to add domain-specific features
        (e.g., StockPredictor adds vwap).

        Steps:
        1. Sort by group/time
        2. Create date-based features (year, month, cyclical encodings)

        Args:
            df: Raw input dataframe

        Returns:
            DataFrame with base features (unencoded, unscaled)
        """
        df = df.copy()

        # Sort by group columns and time
        if self.group_columns:
            # Verify all group columns exist
            missing_cols = [col for col in self.group_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Group columns {missing_cols} not found in DataFrame. Available: {list(df.columns)}")

            # Detect time-related column for sorting
            time_column = self._detect_time_column(df)

            # Sort dataframe by group columns and time to ensure temporal order
            if time_column:
                sort_columns = self.group_columns + [time_column]
                df = df.sort_values(sort_columns).reset_index(drop=True)

                if self.verbose:
                    group_str = ' + '.join(self.group_columns)
                    print(f"   Sorted data by {group_str} and '{time_column}' to ensure temporal order within groups")
            else:
                # No time column - just sort by group columns
                df = df.sort_values(self.group_columns).reset_index(drop=True)
                if self.verbose:
                    group_str = ' + '.join(self.group_columns)
                    print(f"   Sorted data by {group_str}. Warning: No time column detected.")

        # Create time-series features (date-based features)
        time_column = self._detect_time_column(df)

        if time_column:
            # Pass first group column to create_date_features
            group_col_for_date_features = self.group_columns[0] if self.group_columns else None
            df = create_date_features(df, time_column, group_column=group_col_for_date_features)

        return df

    def _determine_numerical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Determine which columns are numerical features for sequences.

        Excludes:
        - Shifted target columns (close_target_h1, etc.) - these are Y, not X
        - Categorical columns - handled separately
        - Original target columns (if use_lagged_target_features=False)

        Args:
            df: DataFrame with all features

        Returns:
            Same dataframe (sets self.numerical_columns, self.feature_columns as side effect)
        """
        # Build exclusion set
        exclude_cols = set()

        # Always exclude shifted target columns (these are Y, not X)
        for target_col in self.target_columns:
            for h in range(1, self.prediction_horizon + 1):
                exclude_cols.add(f"{target_col}_target_h{h}")

        # Exclude categorical columns from numerical features (handled separately)
        exclude_cols.update(self.categorical_columns)

        # Decide whether to exclude original target columns
        if not self.use_lagged_target_features:
            # Exclude targets: model won't see close[t], close[t-1], etc. in sequence
            exclude_cols.update(self.target_columns)
            if self.verbose:
                print(f"\n   Target columns EXCLUDED from sequence: {self.target_columns}")
                print(f"   Model will NOT have autoregressive information")
        else:
            # Include targets: model will see close[t], close[t-1], etc. in sequence
            if self.verbose:
                print(f"\n   Target columns INCLUDED in sequence: {self.target_columns}")
                print(f"   Model will have autoregressive information (use_lagged_target_features=True)")

        # Select numerical feature columns
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols:
                # Only include numeric columns
                if pd.api.types.is_numeric_dtype(df[col]):
                    feature_cols.append(col)
                elif self.verbose:
                    print(f"   Excluding non-numeric column: {col} (dtype: {df[col].dtype})")
            elif self.verbose and '_target_h' in col:
                pass  # Don't print for every shifted column, too verbose

        if self.numerical_columns is None:
            # First time: establish the numeric feature columns from training data
            self.numerical_columns = feature_cols
            self.feature_columns = feature_cols
        else:
            # For validation/test data: only use features created during training
            # This handles cases where smaller datasets can't generate all features
            missing_features = [col for col in self.numerical_columns if col not in df.columns]

            if missing_features:
                if self.verbose:
                    print(f"   Warning: {len(missing_features)} features missing in current dataset")
                    print(f"   Missing: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")

                # Add missing features as zero-filled columns
                for feature in missing_features:
                    df[feature] = 0.0
                    if self.verbose:
                        print(f"   Added zero-filled feature: {feature}")

        return df

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

    def _scale_features_single(self, df_processed: pd.DataFrame, fit_scaler: bool, shifted_target_columns: list = None) -> pd.DataFrame:
        """
        Scale features and shifted targets using single scaler.

        Args:
            df_processed: DataFrame with engineered features and shifted targets
            fit_scaler: Whether to fit the scaler
            shifted_target_columns: List of shifted target column names to scale (e.g., ['close_target_h1', 'close_target_h2'])

        Returns:
            DataFrame with scaled features and targets
        """
        # Scale numerical input features
        if len(self.feature_columns) > 0:
            if fit_scaler:
                df_processed[self.feature_columns] = self.scaler.fit_transform(
                    df_processed[self.feature_columns]
                )
            else:
                df_processed[self.feature_columns] = self.scaler.transform(
                    df_processed[self.feature_columns]
                )

        # Scale shifted target columns (each horizon separately)
        if shifted_target_columns:
            for shifted_col in shifted_target_columns:
                if shifted_col in df_processed.columns:
                    values = df_processed[shifted_col].values.reshape(-1, 1)

                    if fit_scaler:
                        scaler = ScalerFactory.create_scaler(self.scaler_type)
                        df_processed[shifted_col] = scaler.fit_transform(values).flatten()
                        # Store scaler per-horizon
                        self.target_scalers_dict[shifted_col] = scaler
                    else:
                        if shifted_col not in self.target_scalers_dict:
                            raise ValueError(f"No scaler found for {shifted_col}")
                        df_processed[shifted_col] = self.target_scalers_dict[shifted_col].transform(values).flatten()

        return df_processed

    def _scale_features_grouped(self, df_processed: pd.DataFrame, fit_scaler: bool, shifted_target_columns: list = None) -> pd.DataFrame:
        """
        Scale features and shifted targets separately per group (supports multi-column grouping).

        Each unique group (single column or composite) gets its own scaler. This ensures
        that different entities (e.g., different stock symbols, or symbol+sector combinations)
        are scaled independently, preserving their individual statistical properties.

        Args:
            df_processed: DataFrame with engineered features and shifted targets
            fit_scaler: Whether to fit the scalers
            shifted_target_columns: List of shifted target column names to scale

        Returns:
            DataFrame with group-scaled features and targets
        """
        # Verify all group columns exist
        missing_cols = [col for col in self.group_columns if col not in df_processed.columns]
        if missing_cols:
            raise ValueError(
                f"Group columns {missing_cols} not found in dataframe. "
                f"Available columns: {list(df_processed.columns)}"
            )

        # Create a copy to avoid modifying original
        df_scaled = df_processed.copy()

        # Ensure numerical columns are float type to avoid dtype incompatibility warnings
        for col in self.numerical_columns:
            if col in df_scaled.columns:
                df_scaled[col] = df_scaled[col].astype('float64')

        # Create composite group key
        df_scaled['_group_key'] = self._create_group_key(df_scaled)

        # Get unique groups
        unique_groups = df_scaled['_group_key'].unique()

        if self.verbose:
            print(f"   Scaling features for {len(unique_groups)} groups")
            if len(unique_groups) <= 10:
                print(f"   Groups: {unique_groups}")

        # Scale each group separately
        for group_key in unique_groups:
            # Get mask for this group
            group_mask = df_scaled['_group_key'] == group_key
            group_size = group_mask.sum()

            if group_size == 0:
                continue

            # Get data for this group (use numerical_columns)
            group_data = df_scaled.loc[group_mask, self.numerical_columns]

            if fit_scaler:
                # Create and fit new scaler for this group
                scaler = ScalerFactory.create_scaler(self.scaler_type)
                scaled_data = scaler.fit_transform(group_data)
                self.group_feature_scalers[group_key] = scaler

                if self.verbose:
                    print(f"   Group {group_key}: fitted scaler on {group_size} samples")
            else:
                # Use existing scaler for this group
                if group_key not in self.group_feature_scalers:
                    raise ValueError(
                        f"No scaler found for group {group_key}. "
                        f"Make sure to fit on training data first."
                    )
                scaled_data = self.group_feature_scalers[group_key].transform(group_data)

            # Update the scaled data for this group (convert to DataFrame to preserve dtypes)
            df_scaled.loc[group_mask, self.numerical_columns] = pd.DataFrame(
                scaled_data,
                columns=self.numerical_columns,
                index=df_scaled[group_mask].index
            )

        # Scale shifted target columns per group (each horizon separately)
        if shifted_target_columns:
            # Re-add group key for target scaling
            df_scaled['_group_key'] = self._create_group_key(df_scaled)

            for group_key in unique_groups:
                group_mask = df_scaled['_group_key'] == group_key

                for shifted_col in shifted_target_columns:
                    if shifted_col in df_scaled.columns:
                        values = df_scaled.loc[group_mask, shifted_col].values.reshape(-1, 1)

                        if fit_scaler:
                            scaler = ScalerFactory.create_scaler(self.scaler_type)
                            df_scaled.loc[group_mask, shifted_col] = scaler.fit_transform(values).flatten()
                            # Store per group, per horizon
                            if group_key not in self.group_target_scalers:
                                self.group_target_scalers[group_key] = {}
                            self.group_target_scalers[group_key][shifted_col] = scaler
                        else:
                            if group_key not in self.group_target_scalers or shifted_col not in self.group_target_scalers[group_key]:
                                raise ValueError(f"No scaler found for group {group_key}, target {shifted_col}")
                            scaler = self.group_target_scalers[group_key][shifted_col]
                            df_scaled.loc[group_mask, shifted_col] = scaler.transform(values).flatten()

        # Remove temporary column
        df_scaled = df_scaled.drop(columns=['_group_key'])

        return df_scaled

    def _prepare_data_grouped(self, df_processed: pd.DataFrame, fit_scaler: bool) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Create sequences and extract targets for grouped data.

        Note: Targets are already scaled (done in prepare_data step 6), so this method
        just extracts them without scaling.

        Args:
            df_processed: DataFrame with features and targets already scaled by group
            fit_scaler: Unused (kept for compatibility, targets already scaled)

        Returns:
            X: If categorical_columns exist: Tuple of (X_num, X_cat) where
                  X_num: (n_samples, sequence_length, n_numerical_features)
                  X_cat: (n_samples, n_categorical_features)
               Otherwise: X_num tensor only
            y: Target tensor of shape (n_samples, n_targets * n_horizons) - already scaled
        """

        # Note: Data is already sorted by group and time in _create_base_features()
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
            # Single-target: use first target column
            target_col = self.target_columns[0]
            if self.prediction_horizon == 1:
                expected_target = f"{target_col}_target_h1" if not target_col.endswith('_target_h1') else target_col
                if expected_target not in df_processed.columns:
                    raise ValueError(f"Single horizon target column '{expected_target}' not found")
                target_cols = [expected_target]
            else:
                target_cols = [f"{target_col}_target_h{h}" for h in range(1, self.prediction_horizon + 1)]
                missing = [col for col in target_cols if col not in df_processed.columns]
                if missing:
                    raise ValueError(f"Multi-horizon target columns {missing} not found")

        # Process each group separately
        # Create temporary composite key column
        df_processed = df_processed.copy()
        df_processed['_group_key'] = self._create_group_key(df_processed)
        unique_groups = df_processed['_group_key'].unique()

        all_sequences_num = []  # Numerical sequences
        all_sequences_cat = []  # Categorical features (if any)
        all_targets = []
        group_indices = []  # Track which group each sequence belongs to

        # Determine numerical feature columns (exclude categoricals)
        numerical_feature_cols = [col for col in self.feature_columns if col not in self.categorical_columns]

        for group_value in unique_groups:
            group_mask = df_processed['_group_key'] == group_value
            group_df = df_processed[group_mask].copy()
            # Remove temporary column from group_df
            group_df = group_df.drop(columns=['_group_key'])

            # Check if group has enough data for sequences
            if len(group_df) <= self.sequence_length:
                if self.verbose:
                    print(f"  Warning: Skipping group '{group_value}' - insufficient data ({len(group_df)} <= {self.sequence_length})")
                continue

            # Create sequences with separate numerical and categorical handling
            sequences_num, sequences_cat = self._create_sequences_with_categoricals(
                group_df,
                self.sequence_length,
                numerical_feature_cols,
                self.categorical_columns
            )

            # Extract already-scaled target values (no scaling needed - done in step 6)
            y_list = []
            for target_col in self.target_columns:
                for h in range(1, self.prediction_horizon + 1):
                    shifted_col = f"{target_col}_target_h{h}"
                    # Extract values after sequence offset (already scaled)
                    # Updated offset: sequence_length - 1 due to new sequence creation logic
                    target_values = group_df[shifted_col].values[self.sequence_length - 1:]
                    y_list.append(target_values)

            # Combine targets
            if len(y_list) == 1:
                y_combined = y_list[0]
            else:
                y_combined = np.column_stack(y_list)

            all_sequences_num.append(sequences_num)
            if sequences_cat is not None:
                all_sequences_cat.append(sequences_cat)
            all_targets.append(y_combined)
            group_indices.extend([group_value] * len(sequences_num))

        # Concatenate all groups
        if len(all_sequences_num) == 0:
            raise ValueError(f"No groups had sufficient data (need > {self.sequence_length} samples per group)")

        X_num_combined = np.vstack(all_sequences_num)

        # Concatenate categorical if present
        if len(all_sequences_cat) > 0:
            X_cat_combined = np.vstack(all_sequences_cat)
        else:
            X_cat_combined = None

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
        X_num = torch.tensor(X_num_combined, dtype=torch.float32)
        y = torch.tensor(y_combined, dtype=torch.float32)

        if X_cat_combined is not None:
            X_cat = torch.tensor(X_cat_combined, dtype=torch.long)  # Categorical indices are integers
            X = (X_num, X_cat)
            if self.verbose:
                print(f"  Created {len(X_num)} sequences from {len(unique_groups)} groups")
                print(f"  X_num shape: {X_num.shape}, X_cat shape: {X_cat.shape}, y shape: {y.shape}")
        else:
            X = X_num
            if self.verbose:
                print(f"  Created {len(X_num)} sequences from {len(unique_groups)} groups")
                print(f"  X shape: {X_num.shape}, y shape: {y.shape}")

        return X, y

    def prepare_data(self, df: pd.DataFrame, fit_scaler: bool = False, store_for_evaluation: bool = False) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Prepare sequential data for model training/inference.

        Pipeline:
        1. Create base features (sorting, date features) - can be overridden by subclasses
        2. Create shifted targets
        3. STORE unscaled/unencoded dataframe (if store_for_evaluation=True)
        4. Encode categorical features
        5. Determine numerical columns
        6. Scale numerical features AND shifted targets (per-horizon)
        7. Create sequences and extract already-scaled targets

        Args:
            df: Input dataframe
            fit_scaler: Whether to fit scalers (True for training data)
            store_for_evaluation: If True, stores unscaled/unencoded dataframe in self._last_processed_df
                                 for evaluation alignment. Should be True during predict(), False during fit().

        Returns:
            X: If categorical_columns exist: Tuple of (X_num, X_cat) where
                  X_num: (n_samples, sequence_length, n_numerical_features)
                  X_cat: (n_samples, n_categorical_features)
               Otherwise: X_num tensor only
            y: Target tensor of shape (n_samples,) (None for inference)
        """
        # Step 1: Create base features (sorting, date features)
        # Can be overridden by subclasses (e.g., StockPredictor adds vwap)
        df_features = self._create_base_features(df)

        # Step 2: Create shifted target columns
        # Use categorical_columns for shifting to prevent data leakage across all categorical boundaries
        # Falls back to group_columns if categorical_columns not specified
        group_col_for_shift = self.categorical_columns if self.categorical_columns else self.group_columns
        df_with_targets = create_shifted_targets(
            df_features,
            target_column=self.target_columns,
            prediction_horizon=self.prediction_horizon,
            group_column=group_col_for_shift,
            verbose=self.verbose
        )

        # Step 3: STORE unscaled/unencoded dataframe for evaluation
        if store_for_evaluation:
            self._last_processed_df = df_with_targets.copy()
            if self.verbose:
                print(f"   Stored unscaled dataframe for evaluation ({len(df_with_targets)} rows)")

        # Step 4: Encode categorical features
        df_encoded = self._encode_categorical_features(df_with_targets, fit_encoders=fit_scaler)

        # Step 5: Determine numerical columns (excludes shifted targets, categoricals)
        df_encoded = self._determine_numerical_columns(df_encoded)

        # Step 6: Collect shifted target column names for scaling
        shifted_target_columns = []
        for target_col in self.target_columns:
            for h in range(1, self.prediction_horizon + 1):
                shifted_col = f"{target_col}_target_h{h}"
                if shifted_col in df_encoded.columns:
                    shifted_target_columns.append(shifted_col)

        # Step 6: Scale numerical features AND shifted targets (per-horizon)
        if self.group_columns:
            df_scaled = self._scale_features_grouped(df_encoded, fit_scaler, shifted_target_columns)
        else:
            df_scaled = self._scale_features_single(df_encoded, fit_scaler, shifted_target_columns)

        # Step 7: Create sequences and extract already-scaled targets
        if self.group_columns:
            # Group-based sequence creation (no target scaling - already done in step 6)
            return self._prepare_data_grouped(df_scaled, fit_scaler=False)

        # Non-grouped path: single-group data preparation
        # Check if we have enough data for sequences
        if len(df_scaled) <= self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length + 1} samples for sequence_length={self.sequence_length}, got {len(df_scaled)}")

        # Create sequences using _create_sequences_with_categoricals (same as grouped path)
        numerical_cols = [col for col in self.feature_columns if col not in (self.categorical_columns or [])]
        X_num, X_cat = self._create_sequences_with_categoricals(
            df_scaled,
            self.sequence_length,
            numerical_cols,
            self.categorical_columns
        )

        # Extract already-scaled target values (no scaling needed - done in step 6)
        y_list = []
        for target_col in self.target_columns:
            for h in range(1, self.prediction_horizon + 1):
                shifted_col = f"{target_col}_target_h{h}"
                if shifted_col not in df_scaled.columns:
                    raise ValueError(f"Target column '{shifted_col}' not found")
                # Extract values after sequence offset
                # Updated offset: sequence_length - 1 due to new sequence creation logic
                target_values = df_scaled[shifted_col].values[self.sequence_length - 1:]
                y_list.append(target_values)

        # Combine targets
        if len(y_list) == 1:
            y = y_list[0]
        else:
            y = np.column_stack(y_list)

        # Convert to tensors
        X_num_tensor = torch.tensor(X_num, dtype=torch.float32)
        if X_cat is not None:
            X_cat_tensor = torch.tensor(X_cat, dtype=torch.long)
            X = (X_num_tensor, X_cat_tensor)
        else:
            X = X_num_tensor

        y_tensor = torch.tensor(y, dtype=torch.float32)

        if self.verbose:
            print(f"   Created {len(X_num)} sequences of length {self.sequence_length}")
            if isinstance(X, tuple):
                print(f"   X_num shape: {X[0].shape}, X_cat shape: {X[1].shape}")
            else:
                print(f"   X shape: {X.shape}")
            print(f"   y shape: {y_tensor.shape}")

        return X, y_tensor
    
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
        # Don't store dataframe during training (store_for_evaluation=False)
        X_train, y_train = self.prepare_data(df, fit_scaler=True, store_for_evaluation=False)

        if val_df is not None:
            X_val, y_val = self.prepare_data(val_df, fit_scaler=False, store_for_evaluation=False)
            # Save validation group indices for later use in progress reporting
            if self.group_columns and hasattr(self, '_last_group_indices'):
                self._last_val_group_indices = self._last_group_indices.copy()
        else:
            X_val, y_val = None, None
        
        # Initialize model using factory pattern
        # Handle tuple input (X_num, X_cat) for categorical models
        if isinstance(X_train, tuple):
            X_num_train, X_cat_train = X_train
            _, seq_len, num_numerical = X_num_train.shape
            num_categorical = X_cat_train.shape[1] if X_cat_train is not None else 0
        else:
            # Original behavior: single tensor
            if len(X_train.shape) == 3:  # Sequence data: (batch, seq_len, features)
                _, seq_len, num_features = X_train.shape
            else:  # Single timestep data: (batch, features)
                seq_len = 1  # Treat as sequence of length 1
                num_features = X_train.shape[1]
            num_numerical = num_features
            num_categorical = 0

        # Filter out invalid kwargs for model initialization
        # 'verbose' is a general parameter, not for model init
        model_kwargs = {k: v for k, v in self.model_kwargs.items() if k not in ['verbose']}

        # Calculate total output size
        if self.is_multi_target:
            # Multi-target: output num_targets * prediction_horizon values
            total_output_size = self.num_targets * self.prediction_horizon
        else:
            # Single-target: output prediction_horizon values
            total_output_size = self.prediction_horizon

        # Create model using factory
        # For _cls models, pass additional categorical parameters
        if self.model_type.endswith('_cls'):
            # For CLS models, calculate total features for display
            num_features = num_numerical + num_categorical
            self.model = ModelFactory.create_model(
                model_type=self.model_type,
                sequence_length=seq_len,
                num_numerical=num_numerical,
                num_categorical=num_categorical,
                cat_cardinalities=self.cat_cardinalities,
                output_dim=total_output_size,
                **model_kwargs
            ).to(self.device)
        else:
            # Standard models use num_features parameter
            num_features = num_numerical + num_categorical
            self.model = ModelFactory.create_model(
                model_type=self.model_type,
                sequence_length=seq_len,
                num_features=num_features,
                output_dim=total_output_size,
                **model_kwargs
            ).to(self.device)

        if self.verbose:
            print(f"\n   Created {self.model_type} model:")
            print(f"   - Input: sequences of length {seq_len} with {num_features} features")
            print(f"   - Output: {total_output_size} values ({'multi-target' if self.is_multi_target else 'single-target'}, {'multi-horizon' if self.prediction_horizon > 1 else 'single-horizon'})")
            print(f"   - Parameters: {self.model.get_num_parameters():,}")
            print(f"   - Embedding dim: {self.model.get_embedding_dim()}")
        
        # Training setup
        # Handle tuple input (X_num, X_cat) for categorical models
        if isinstance(X_train, tuple):
            X_num_train, X_cat_train = X_train
            dataset = TensorDataset(X_num_train, X_cat_train, y_train)
            optimal_batch_size = min(batch_size, max(32, len(X_num_train) // 20))
        else:
            dataset = TensorDataset(X_train, y_train)
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
            target_desc = ', '.join(self.target_columns) if self.is_multi_target else self.target_columns[0]
            print(f"Training FT-Transformer for {target_desc} prediction")
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
            for batch_data in dataloader:
                # Unpack batch based on dataset format
                if isinstance(X_train, tuple):
                    # Categorical model: (X_num, X_cat, y)
                    batch_x_num, batch_x_cat, batch_y = batch_data
                    batch_x_num = batch_x_num.to(self.device, non_blocking=True)
                    batch_x_cat = batch_x_cat.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                else:
                    # Standard model: (X, y)
                    batch_x, batch_y = batch_data
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                # Use mixed precision training if available
                if use_amp:
                    with torch.cuda.amp.autocast():
                        if isinstance(X_train, tuple):
                            # CLS models with categorical features
                            outputs = self.model(batch_x_num, batch_x_cat).squeeze()
                        else:
                            # Standard models
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
                    # Handle both categorical and standard models
                    if isinstance(X_train, tuple):
                        # CLS models with categorical features
                        outputs = self.model(batch_x_num, batch_x_cat).squeeze()
                    else:
                        # Standard models
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
                    # Handle tuple input for categorical models
                    if isinstance(X_val, tuple):
                        X_num_val, X_cat_val = X_val
                        X_num_val_device = X_num_val.to(self.device, non_blocking=True)
                        X_cat_val_device = X_cat_val.to(self.device, non_blocking=True)
                    else:
                        X_val_device = X_val.to(self.device, non_blocking=True)

                    y_val_device = y_val.to(self.device, non_blocking=True)

                    # Use mixed precision for validation too
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            if isinstance(X_val, tuple):
                                # CLS models with categorical features
                                val_outputs = self.model(X_num_val_device, X_cat_val_device).squeeze()
                            else:
                                # Standard models
                                if len(X_val.shape) == 3:  # Sequence data
                                    val_outputs = self.model(X_val_device).squeeze()
                                else:  # Non-sequence data
                                    val_outputs = self.model(X_val_device, None).squeeze()
                            val_loss = criterion(val_outputs, y_val_device).item()
                    else:
                        if isinstance(X_val, tuple):
                            # CLS models with categorical features
                            val_outputs = self.model(X_num_val_device, X_cat_val_device).squeeze()
                        else:
                            # Standard models
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
                        if self.group_columns:
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

                                    target_col = self.target_columns[0]  # Single-target mode in this path
                                    if self.prediction_horizon == 1:
                                        shifted_col = f"{target_col}_target_h1"
                                        val_pred[group_mask] = self.group_target_scalers[group_value][shifted_col].inverse_transform(
                                            val_pred_scaled_np[group_mask].reshape(-1, 1)
                                        ).flatten()
                                        val_actual[group_mask] = self.group_target_scalers[group_value][shifted_col].inverse_transform(
                                            val_actual_scaled_np[group_mask].reshape(-1, 1)
                                        ).flatten()
                                    else:
                                        # Multi-horizon: inverse transform each horizon separately
                                        group_pred_scaled = val_pred_scaled_np[group_mask]
                                        group_actual_scaled = val_actual_scaled_np[group_mask]
                                        for h in range(self.prediction_horizon):
                                            shifted_col = f"{target_col}_target_h{h+1}"
                                            val_pred[group_mask, h] = self.group_target_scalers[group_value][shifted_col].inverse_transform(
                                                group_pred_scaled[:, h].reshape(-1, 1)
                                            ).flatten()
                                            val_actual[group_mask, h] = self.group_target_scalers[group_value][shifted_col].inverse_transform(
                                                group_actual_scaled[:, h].reshape(-1, 1)
                                            ).flatten()
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
                                target_col = self.target_columns[0]
                                shifted_col = f"{target_col}_target_h1"
                                val_pred = self.target_scalers_dict[shifted_col].inverse_transform(val_pred_scaled_np.reshape(-1, 1)).flatten()
                                val_actual = self.target_scalers_dict[shifted_col].inverse_transform(val_actual_scaled_np.reshape(-1, 1)).flatten()
                            else:
                                # Single-target, multi-horizon: inverse transform each horizon separately
                                target_col = self.target_columns[0]
                                val_pred = np.zeros_like(val_pred_scaled_np)
                                val_actual = np.zeros_like(val_actual_scaled_np)
                                for h in range(self.prediction_horizon):
                                    shifted_col = f"{target_col}_target_h{h+1}"
                                    val_pred[:, h] = self.target_scalers_dict[shifted_col].inverse_transform(
                                        val_pred_scaled_np[:, h].reshape(-1, 1)
                                    ).flatten()
                                    val_actual[:, h] = self.target_scalers_dict[shifted_col].inverse_transform(
                                        val_actual_scaled_np[:, h].reshape(-1, 1)
                                    ).flatten()

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
            self.logger.error("Model must be trained first. Call fit().")
            raise RuntimeError("Model must be trained first. Call fit().")

        # Clear feature cache before prediction to free memory
        self._feature_cache.clear()
        gc.collect()  # Force garbage collection to free memory

        # Store processed dataframe for evaluation alignment (store_for_evaluation=True)
        # This stores the dataframe after feature engineering and target shifting, but before encoding/scaling
        X, _ = self.prepare_data(df, fit_scaler=False, store_for_evaluation=True)

        self.model.eval()
        with torch.no_grad():
            # Use batched inference to avoid OOM errors
            batch_size = 256  # Process predictions in smaller batches

            # Handle tuple input for categorical models
            if isinstance(X, tuple):
                X_num, X_cat = X
                num_samples = X_num.shape[0]
            else:
                num_samples = X.shape[0]

            all_predictions = []

            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)

                # Handle both categorical and standard models
                if isinstance(X, tuple):
                    # CLS models with categorical features
                    X_num_batch = X_num[i:batch_end].to(self.device)
                    X_cat_batch = X_cat[i:batch_end].to(self.device)
                    batch_preds = self.model(X_num_batch, X_cat_batch)
                else:
                    # Standard models
                    X_batch = X[i:batch_end]
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
            if self.group_columns:
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

                                # Use per-horizon scaler
                                shifted_col = f"{target_col}_target_h1"
                                group_preds_original = self.group_target_scalers[group_value][shifted_col].inverse_transform(group_preds_scaled).flatten()
                                target_preds[group_mask] = group_preds_original
                            else:
                                # Multi-horizon: extract horizons for this target
                                # Layout: [close_h1, close_h2, ..., volume_h1, volume_h2, ...]
                                start_idx = idx * self.prediction_horizon
                                end_idx = start_idx + self.prediction_horizon

                                # Extract all horizons for this target in this group
                                group_horizons_scaled = predictions_scaled[group_mask, start_idx:end_idx]

                                # Inverse transform each horizon separately using per-horizon scalers
                                group_horizons_original = np.zeros_like(group_horizons_scaled)
                                for h in range(self.prediction_horizon):
                                    shifted_col = f"{target_col}_target_h{h+1}"
                                    horizon_scaled = group_horizons_scaled[:, h].reshape(-1, 1)
                                    group_horizons_original[:, h] = self.group_target_scalers[group_value][shifted_col].inverse_transform(horizon_scaled).flatten()

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
                        target_col = self.target_columns[0]  # Single-target mode

                        if self.prediction_horizon == 1:
                            # Single horizon
                            group_preds_scaled = group_preds_scaled.reshape(-1, 1)
                            shifted_col = f"{target_col}_target_h1"
                            group_preds_original = self.group_target_scalers[group_value][shifted_col].inverse_transform(group_preds_scaled)
                            predictions[group_mask] = group_preds_original.flatten()
                        else:
                            # Multi-horizon: inverse transform each horizon separately
                            group_preds_original = np.zeros_like(group_preds_scaled)
                            for h in range(self.prediction_horizon):
                                shifted_col = f"{target_col}_target_h{h+1}"
                                horizon_scaled = group_preds_scaled[:, h].reshape(-1, 1)
                                group_preds_original[:, h] = self.group_target_scalers[group_value][shifted_col].inverse_transform(horizon_scaled).flatten()
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
                            shifted_col = f"{target_col}_target_h1"
                            target_preds = self.target_scalers_dict[shifted_col].inverse_transform(target_preds_scaled).flatten()
                            predictions_dict[target_col] = target_preds
                        else:
                            # Multi-horizon: extract horizons for this target
                            # Layout: [close_h1, close_h2, ..., volume_h1, volume_h2, ...]
                            start_idx = idx * self.prediction_horizon
                            end_idx = start_idx + self.prediction_horizon

                            # Inverse transform each horizon separately using per-horizon scalers
                            horizons_scaled = predictions_scaled[:, start_idx:end_idx]
                            horizons_original = np.zeros_like(horizons_scaled)
                            for h in range(self.prediction_horizon):
                                shifted_col = f"{target_col}_target_h{h+1}"
                                horizon_scaled = horizons_scaled[:, h].reshape(-1, 1)
                                horizons_original[:, h] = self.target_scalers_dict[shifted_col].inverse_transform(horizon_scaled).flatten()

                            # Stack into (n_samples, horizons) for this target
                            predictions_dict[target_col] = horizons_original

                    return predictions_dict

                elif self.prediction_horizon == 1:
                    # Single-target, single horizon: reshape to (n_samples, 1) for inverse transform
                    target_col = self.target_columns[0]
                    shifted_col = f"{target_col}_target_h1"
                    predictions_scaled = predictions_scaled.reshape(-1, 1)
                    predictions = self.target_scalers_dict[shifted_col].inverse_transform(predictions_scaled)
                    return predictions.flatten()
                else:
                    # Single-target, multi-horizon: predictions_scaled shape is (n_samples, horizons)
                    # Inverse transform each horizon separately using per-horizon scalers
                    target_col = self.target_columns[0]
                    predictions = np.zeros_like(predictions_scaled)
                    for h in range(self.prediction_horizon):
                        shifted_col = f"{target_col}_target_h{h+1}"
                        horizon_scaled = predictions_scaled[:, h].reshape(-1, 1)
                        predictions[:, h] = self.target_scalers_dict[shifted_col].inverse_transform(horizon_scaled).flatten()
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
        if self.group_columns:
            raise NotImplementedError(
                "predict_from_features() does not support group-based scaling. "
                "Please use predict() instead."
            )

        # Skip feature preprocessing since it's already done
        from ..preprocessing.time_features import create_input_variable_sequence

        # Create sequences directly from processed features
        sequences = create_input_variable_sequence(df_processed, self.sequence_length, self.feature_columns)

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
            target_col = self.target_columns[0]  # Single-target mode (no multi-target support in this method)
            if self.prediction_horizon == 1:
                # Single horizon: reshape to (n_samples, 1) for inverse transform
                shifted_col = f"{target_col}_target_h1"
                predictions_scaled = predictions_scaled.reshape(-1, 1)
                predictions = self.target_scalers_dict[shifted_col].inverse_transform(predictions_scaled)
                return predictions.flatten()
            else:
                # Multi-horizon: predictions_scaled shape is (n_samples, horizons)
                # Inverse transform each horizon separately using per-horizon scalers
                predictions = np.zeros_like(predictions_scaled)
                for h in range(self.prediction_horizon):
                    shifted_col = f"{target_col}_target_h{h+1}"
                    horizon_scaled = predictions_scaled[:, h].reshape(-1, 1)
                    predictions[:, h] = self.target_scalers_dict[shifted_col].inverse_transform(horizon_scaled).flatten()
                return predictions

    def evaluate(self, df: pd.DataFrame, per_group: bool = False, predictions=None, group_indices=None,
                 export_csv: Optional[str] = None) -> Dict:
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
            predictions: Optional pre-computed predictions to avoid reprocessing (default: None)
            group_indices: Optional pre-computed group indices to avoid reprocessing (default: None)
            export_csv: Optional path to export predictions and actuals to CSV (default: None)
                       Format: group,date,dataset,{target}_actual,{target}_pred_h{N},...
                       Supports multi-group, multi-target, multi-horizon

        Returns:
            metrics: Dictionary of evaluation metrics (structure depends on configuration)
        """
        # Clear feature cache before evaluation to free memory
        self._feature_cache.clear()
        gc.collect()  # Force garbage collection to free memory

        # Validate that target column(s) exist in raw dataframe
        if self.is_multi_target:
            # Check all target columns exist
            missing_targets = [col for col in self.target_columns if col not in df.columns]
            if missing_targets:
                raise ValueError(f"Target columns {missing_targets} not found in dataframe")
        else:
            # Single-target check
            if self.target_columns[0] not in df.columns:
                raise ValueError(f"Target column '{self.target_columns[0]}' not found in dataframe")

        # Get predictions if not provided (avoids reprocessing if predictions are passed in)
        if predictions is None:
            if per_group and self.group_columns:
                predictions, group_indices = self.predict(df, return_group_info=True)
            else:
                predictions = self.predict(df)

        # Check if we should do per-group evaluation
        if per_group and self.group_columns:
            result = self._evaluate_per_group(df, predictions=predictions, group_indices=group_indices)
        else:
            # Standard evaluation - predict() handles all preprocessing
            result = self._evaluate_standard(df, predictions=predictions)

        # Export predictions to CSV if requested (pass metrics for summary section)
        if export_csv:
            self._export_predictions_csv(df, predictions, group_indices, export_csv, metrics=result)

        # Clear cached processed dataframe to free memory
        if hasattr(self, '_last_processed_df'):
            del self._last_processed_df

        return result

    def _evaluate_standard(self, df: pd.DataFrame, predictions=None) -> Dict:
        """
        Standard evaluation without per-group breakdown.

        Args:
            df: DataFrame with raw data
            predictions: Optional pre-computed predictions (if None, will call predict())
        """
        # Use provided predictions or compute them
        if predictions is None:
            predictions = self.predict(df)  # Returns dict for multi-target, predict() handles all preprocessing

        # Check if multi-target
        if self.is_multi_target:
            # Multi-target evaluation: return metrics per target
            from ..core.utils import calculate_metrics

            # Use stored processed dataframe (has shifted target columns)
            if not hasattr(self, '_last_processed_df') or self._last_processed_df is None:
                raise RuntimeError(
                    "_last_processed_df not available for evaluation. "
                    "This is a bug - predict() should have set _last_processed_df."
                )

            # New offset: sequence_length - 1 (due to new sequence creation logic)
            offset = self.sequence_length - 1

            metrics_dict = {}

            for target_col in self.target_columns:
                target_predictions = predictions[target_col]  # Dict of predictions

                # Handle single vs multi-horizon
                if self.prediction_horizon == 1:
                    # Single-horizon: extract from shifted target column
                    shifted_col = f"{target_col}_target_h1"
                    actual = self._last_processed_df[shifted_col].values[offset:]

                    # Validate alignment
                    if len(actual) != len(target_predictions):
                        raise ValueError(
                            f"Alignment error for {target_col}: "
                            f"{len(actual)} actuals vs {len(target_predictions)} predictions"
                        )

                    metrics_dict[target_col] = calculate_metrics(actual, target_predictions)
                else:
                    # Multi-horizon: extract each horizon separately
                    horizon_metrics = {}

                    for h in range(1, self.prediction_horizon + 1):
                        shifted_col = f"{target_col}_target_h{h}"
                        horizon_actual = self._last_processed_df[shifted_col].values[offset:]
                        horizon_pred = target_predictions[:, h-1]

                        # Validate alignment
                        if len(horizon_actual) != len(horizon_pred):
                            raise ValueError(
                                f"Alignment error for {target_col}, horizon {h}: "
                                f"{len(horizon_actual)} actuals vs {len(horizon_pred)} predictions"
                            )

                        horizon_metrics[f'horizon_{h}'] = calculate_metrics(horizon_actual, horizon_pred)

                    # Overall for this target across all horizons
                    all_actual = np.concatenate([
                        self._last_processed_df[f"{target_col}_target_h{h}"].values[offset:]
                        for h in range(1, self.prediction_horizon + 1)
                    ])
                    all_pred = target_predictions.flatten()
                    horizon_metrics['overall'] = calculate_metrics(all_actual, all_pred)

                    metrics_dict[target_col] = horizon_metrics

            return metrics_dict

        else:
            # Single-target evaluation
            # Use stored processed dataframe (has shifted target columns)
            if not hasattr(self, '_last_processed_df') or self._last_processed_df is None:
                raise RuntimeError(
                    "_last_processed_df not available for evaluation. "
                    "This is a bug - predict() should have set _last_processed_df."
                )

            target_col = self.target_columns[0]

            # New offset: sequence_length - 1 (due to new sequence creation logic)
            offset = self.sequence_length - 1

            # Handle single vs multi-horizon evaluation
            if self.prediction_horizon == 1:
                # Single-horizon: extract from shifted target column
                from ..core.utils import calculate_metrics

                shifted_col = f"{target_col}_target_h1"
                actual = self._last_processed_df[shifted_col].values[offset:]

                # Validate alignment
                if len(actual) != len(predictions):
                    raise ValueError(
                        f"Alignment error: {len(actual)} actuals vs {len(predictions)} predictions. "
                        f"This indicates a bug in the evaluation pipeline."
                    )

                return calculate_metrics(actual, predictions)
            else:
                # Multi-horizon: extract each horizon from shifted columns
                from ..core.utils import calculate_metrics

                horizons_actuals = []
                for h in range(1, self.prediction_horizon + 1):
                    shifted_col = f"{target_col}_target_h{h}"
                    horizon_actual = self._last_processed_df[shifted_col].values[offset:]
                    horizons_actuals.append(horizon_actual)

                # Stack into 2D: (n_samples, horizons)
                actual_2d = np.column_stack(horizons_actuals)

                # Validate alignment
                if actual_2d.shape != predictions.shape:
                    raise ValueError(
                        f"Shape mismatch: actuals {actual_2d.shape} vs predictions {predictions.shape}. "
                        f"This indicates a bug in the evaluation pipeline."
                    )

                # Calculate metrics - simple per-horizon approach
                metrics = {}

                # Overall metrics (flatten all horizons)
                all_actual = actual_2d.flatten()
                all_pred = predictions.flatten()
                metrics['overall'] = calculate_metrics(all_actual, all_pred)

                # Per-horizon metrics
                for h in range(self.prediction_horizon):
                    horizon_key = f'horizon_{h+1}'
                    metrics[horizon_key] = calculate_metrics(actual_2d[:, h], predictions[:, h])

                return metrics

    def _evaluate_per_group(self, df: pd.DataFrame, predictions=None, group_indices=None) -> Dict:
        """
        Evaluate performance per group (e.g., per stock symbol).

        Args:
            df: DataFrame with raw data
            predictions: Optional pre-computed predictions (if None, will call predict())
            group_indices: Optional pre-computed group indices (if None, will call predict())

        Returns nested dict with overall metrics plus per-group breakdown.
        """
        from ..core.utils import calculate_metrics

        # Use provided predictions or compute them
        if predictions is None or group_indices is None:
            # Get predictions with group information - predict() handles all preprocessing
            predictions, group_indices = self.predict(df, return_group_info=True)

        # Get unique groups
        unique_groups = sorted(set(group_indices))

        # We need to map group_value (label encoded int) back to original group column values
        if self.group_columns:
            # Get the group column name
            group_col_name = self.group_columns[0]  # Assuming single group column for now

            # Get encoder for this group column
            if group_col_name in self.cat_encoders:
                encoder = self.cat_encoders[group_col_name]
                # Map encoded values back to original strings
                group_value_to_name = {i: name for i, name in enumerate(encoder.classes_)}
            else:
                # Fallback: use group values as-is (shouldn't happen in normal flow)
                group_value_to_name = {gv: gv for gv in unique_groups}
        else:
            group_value_to_name = {gv: gv for gv in unique_groups}

        # Check if _last_processed_df is available
        if not hasattr(self, '_last_processed_df') or self._last_processed_df is None:
            raise RuntimeError(
                "Processed dataframe not available for evaluation. "
                "This is a bug - predict() should have set _last_processed_df."
            )

        # New offset: sequence_length - 1 (due to new sequence creation logic)
        offset = self.sequence_length - 1

        # Check if multi-target
        if self.is_multi_target:
            # Multi-target per-group evaluation
            all_metrics = {}

            for target_col in self.target_columns:
                target_predictions = predictions[target_col]

                # Calculate per-group metrics for this target
                for group_value in unique_groups:
                    group_name = group_value_to_name[group_value]

                    # Get processed data for this group
                    group_df_processed = self._last_processed_df[
                        self._last_processed_df[group_col_name] == group_name
                    ].copy()

                    # Get predictions for this group
                    group_mask = np.array([g == group_value for g in group_indices])

                    if self.prediction_horizon == 1:
                        # Single-horizon
                        shifted_col = f"{target_col}_target_h1"
                        group_actual = group_df_processed[shifted_col].values[offset:]
                        group_preds = target_predictions[group_mask]

                        # Validate alignment
                        if len(group_actual) != len(group_preds):
                            raise ValueError(
                                f"Alignment error for group {group_name}, {shifted_col}: "
                                f"Expected {len(group_preds)} actuals but found {len(group_actual)}"
                            )

                        group_metrics = calculate_metrics(group_actual, group_preds)
                    else:
                        # Multi-horizon
                        group_preds = target_predictions[group_mask, :]
                        horizon_metrics = {}

                        for h in range(1, self.prediction_horizon + 1):
                            shifted_col = f"{target_col}_target_h{h}"
                            horizon_actual = group_df_processed[shifted_col].values[offset:]
                            horizon_pred = group_preds[:, h-1]

                            # Validate alignment
                            if len(horizon_actual) != len(horizon_pred):
                                raise ValueError(
                                    f"Alignment error for group {group_name}, {shifted_col}: "
                                    f"Expected {len(horizon_pred)} actuals but found {len(horizon_actual)}"
                                )

                            horizon_metrics[f'horizon_{h}'] = calculate_metrics(horizon_actual, horizon_pred)

                        # Overall metrics for this group across all horizons
                        all_horizons_actual = []
                        all_horizons_pred = []
                        for h in range(1, self.prediction_horizon + 1):
                            shifted_col = f"{target_col}_target_h{h}"
                            all_horizons_actual.append(group_df_processed[shifted_col].values[offset:])
                            all_horizons_pred.append(group_preds[:, h-1])

                        all_actual_flat = np.concatenate(all_horizons_actual)
                        all_pred_flat = np.concatenate(all_horizons_pred)
                        horizon_metrics['overall'] = calculate_metrics(all_actual_flat, all_pred_flat)

                        group_metrics = horizon_metrics

                    # Store metrics in nested structure
                    group_key = str(group_value)
                    if group_key not in all_metrics:
                        all_metrics[group_key] = {}
                    all_metrics[group_key][target_col] = group_metrics

                # Calculate overall metrics for this target (across all groups)
                # For grouped data, extract actuals per-group and concatenate
                if self.prediction_horizon == 1:
                    shifted_col = f"{target_col}_target_h1"

                    # Extract actuals per group in the correct order
                    all_actuals_list = []
                    for group_value in unique_groups:
                        group_name = group_value_to_name[group_value]
                        group_df = self._last_processed_df[
                            self._last_processed_df[group_col_name] == group_name
                        ]
                        group_actual = group_df[shifted_col].values[offset:]
                        all_actuals_list.append(group_actual)

                    all_actual = np.concatenate(all_actuals_list)

                    if len(all_actual) != len(target_predictions):
                        raise ValueError(
                            f"Overall alignment error for {target_col}: "
                            f"{len(all_actual)} actuals vs {len(target_predictions)} predictions"
                        )

                    overall_target_metrics = calculate_metrics(all_actual, target_predictions)
                else:
                    horizon_metrics = {}

                    for h in range(1, self.prediction_horizon + 1):
                        shifted_col = f"{target_col}_target_h{h}"

                        # Extract actuals per group in the correct order
                        all_actuals_list = []
                        for group_value in unique_groups:
                            group_name = group_value_to_name[group_value]
                            group_df = self._last_processed_df[
                                self._last_processed_df[group_col_name] == group_name
                            ]
                            group_actual = group_df[shifted_col].values[offset:]
                            all_actuals_list.append(group_actual)

                        horizon_actual = np.concatenate(all_actuals_list)
                        horizon_pred = target_predictions[:, h-1]

                        if len(horizon_actual) != len(horizon_pred):
                            raise ValueError(
                                f"Overall alignment error for {target_col}, horizon {h}: "
                                f"{len(horizon_actual)} actuals vs {len(horizon_pred)} predictions"
                            )

                        horizon_metrics[f'horizon_{h}'] = calculate_metrics(horizon_actual, horizon_pred)

                    # Overall across all horizons
                    all_horizons_actual = []
                    all_horizons_pred = []
                    for h in range(1, self.prediction_horizon + 1):
                        shifted_col = f"{target_col}_target_h{h}"

                        # Extract actuals per group
                        actuals_for_horizon = []
                        for group_value in unique_groups:
                            group_name = group_value_to_name[group_value]
                            group_df = self._last_processed_df[
                                self._last_processed_df[group_col_name] == group_name
                            ]
                            group_actual = group_df[shifted_col].values[offset:]
                            actuals_for_horizon.append(group_actual)

                        all_horizons_actual.append(np.concatenate(actuals_for_horizon))
                        all_horizons_pred.append(target_predictions[:, h-1])

                    all_actual_flat = np.concatenate(all_horizons_actual)
                    all_pred_flat = np.concatenate(all_horizons_pred)
                    horizon_metrics['overall'] = calculate_metrics(all_actual_flat, all_pred_flat)

                    overall_target_metrics = horizon_metrics

                # Store overall metrics for this target
                if 'overall' not in all_metrics:
                    all_metrics['overall'] = {}
                all_metrics['overall'][target_col] = overall_target_metrics

            return all_metrics

        else:
            # Single-target per-group evaluation
            target_col = self.target_columns[0]
            all_metrics = {}

            # Calculate metrics per group
            for group_value in unique_groups:
                group_name = group_value_to_name[group_value]

                # Get processed data for this group
                group_df_processed = self._last_processed_df[
                    self._last_processed_df[group_col_name] == group_name
                ].copy()

                # Get predictions for this group
                group_mask = np.array([g == group_value for g in group_indices])

                if self.prediction_horizon == 1:
                    # Single-horizon
                    shifted_col = f"{target_col}_target_h1"
                    group_actual = group_df_processed[shifted_col].values[offset:]
                    group_preds = predictions[group_mask]

                    # Validate alignment
                    if len(group_actual) != len(group_preds):
                        raise ValueError(
                            f"Alignment error for group {group_name}, {shifted_col}: "
                            f"Expected {len(group_preds)} actuals but found {len(group_actual)}"
                        )

                    group_metrics = calculate_metrics(group_actual, group_preds)
                else:
                    # Multi-horizon
                    group_preds = predictions[group_mask, :]
                    horizon_metrics = {}

                    # Extract and evaluate each horizon independently
                    for h in range(1, self.prediction_horizon + 1):
                        shifted_col = f"{target_col}_target_h{h}"
                        horizon_actual = group_df_processed[shifted_col].values[offset:]
                        horizon_pred = group_preds[:, h-1]

                        # Validate alignment
                        if len(horizon_actual) != len(horizon_pred):
                            raise ValueError(
                                f"Alignment error for group {group_name}, {shifted_col}: "
                                f"Expected {len(horizon_pred)} actuals but found {len(horizon_actual)}"
                            )

                        horizon_metrics[f'horizon_{h}'] = calculate_metrics(horizon_actual, horizon_pred)

                    # Calculate overall metrics across all horizons for this group
                    all_horizons_actual = []
                    all_horizons_pred = []
                    for h in range(1, self.prediction_horizon + 1):
                        shifted_col = f"{target_col}_target_h{h}"
                        all_horizons_actual.append(group_df_processed[shifted_col].values[offset:])
                        all_horizons_pred.append(group_preds[:, h-1])

                    all_actual_flat = np.concatenate(all_horizons_actual)
                    all_pred_flat = np.concatenate(all_horizons_pred)
                    horizon_metrics['overall'] = calculate_metrics(all_actual_flat, all_pred_flat)

                    group_metrics = horizon_metrics

                all_metrics[str(group_value)] = group_metrics

            # Calculate overall metrics across all groups
            # For grouped data, we must extract actuals per-group and concatenate in the same order as predictions
            # Predictions are ordered: [group0_samples, group1_samples, ...]
            if self.prediction_horizon == 1:
                shifted_col = f"{target_col}_target_h1"

                # Extract actuals per group in the correct order
                all_actuals_list = []
                for group_value in unique_groups:
                    group_name = group_value_to_name[group_value]
                    group_df = self._last_processed_df[
                        self._last_processed_df[group_col_name] == group_name
                    ]
                    group_actual = group_df[shifted_col].values[offset:]
                    all_actuals_list.append(group_actual)

                all_actual = np.concatenate(all_actuals_list)

                if len(all_actual) != len(predictions):
                    raise ValueError(
                        f"Overall alignment error: {len(all_actual)} actuals vs {len(predictions)} predictions"
                    )

                overall_metrics = calculate_metrics(all_actual, predictions)
            else:
                # Multi-horizon overall
                horizon_metrics = {}

                for h in range(1, self.prediction_horizon + 1):
                    shifted_col = f"{target_col}_target_h{h}"

                    # Extract actuals per group in the correct order
                    all_actuals_list = []
                    for group_value in unique_groups:
                        group_name = group_value_to_name[group_value]
                        group_df = self._last_processed_df[
                            self._last_processed_df[group_col_name] == group_name
                        ]
                        group_actual = group_df[shifted_col].values[offset:]
                        all_actuals_list.append(group_actual)

                    horizon_actual = np.concatenate(all_actuals_list)
                    horizon_pred = predictions[:, h-1]

                    if len(horizon_actual) != len(horizon_pred):
                        raise ValueError(
                            f"Overall alignment error for {shifted_col}: "
                            f"{len(horizon_actual)} actuals vs {len(horizon_pred)} predictions"
                        )

                    horizon_metrics[f'horizon_{h}'] = calculate_metrics(horizon_actual, horizon_pred)

                # Overall across all horizons
                all_horizons_actual = []
                all_horizons_pred = []
                for h in range(1, self.prediction_horizon + 1):
                    shifted_col = f"{target_col}_target_h{h}"

                    # Extract actuals per group
                    actuals_for_horizon = []
                    for group_value in unique_groups:
                        group_name = group_value_to_name[group_value]
                        group_df = self._last_processed_df[
                            self._last_processed_df[group_col_name] == group_name
                        ]
                        group_actual = group_df[shifted_col].values[offset:]
                        actuals_for_horizon.append(group_actual)

                    all_horizons_actual.append(np.concatenate(actuals_for_horizon))
                    all_horizons_pred.append(predictions[:, h-1])

                all_actual_flat = np.concatenate(all_horizons_actual)
                all_pred_flat = np.concatenate(all_horizons_pred)
                horizon_metrics['overall'] = calculate_metrics(all_actual_flat, all_pred_flat)

                overall_metrics = horizon_metrics

            # Combine results
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
        target_col = self.target_columns[0]
        if target_col not in df_processed.columns:
            raise ValueError(f"Target column '{target_col}' not found in processed features")

        # Use predict() instead of predict_from_features() if group-based scaling is enabled
        if self.group_columns:
            predictions = self.predict(df_processed)
        else:
            predictions = self.predict_from_features(df_processed)

        # For sequences, align actual values with predictions
        if hasattr(df_processed, 'iloc'):
            actual = df_processed[target_col].iloc[self.sequence_length:self.sequence_length + len(predictions)].values
        else:
            actual = df_processed[target_col].values

        # Ensure same length (take minimum to be safe)
        min_len = min(len(actual), len(predictions))
        actual = actual[:min_len]
        predictions = predictions[:min_len]

        from ..core.utils import calculate_metrics
        return calculate_metrics(actual, predictions)

    def _export_predictions_csv(self, df: pd.DataFrame, predictions, group_indices, csv_path: str, metrics: Dict = None):
        """
        Export predictions, actuals, and metrics summary to CSV.

        CSV has two sections:
        1. METRICS SUMMARY (top): dataset, group, target, horizon, metric, value
        2. PREDICTIONS (bottom): [group], date/timestamp, {target}_actual_h1, {target}_pred_h1, ...

        Sections are separated by a blank row.

        For multi-horizon predictions, each horizon gets its own actual and predicted columns:
            - {target}_actual_h1: actual value at t+1
            - {target}_pred_h1: predicted value for t+1
            - {target}_actual_h2: actual value at t+2
            - {target}_pred_h2: predicted value for t+2
            - etc.

        Args:
            df: Raw input DataFrame
            predictions: Predictions (array, 2D array, or dict for multi-target)
            group_indices: Group indices (if using groups)
            csv_path: Path to save CSV file
            metrics: Evaluation metrics dict (for summary section)
        """
        from pathlib import Path
        from datetime import datetime

        # Determine time column (prefer 'date', fallback to 'timestamp')
        time_column = None
        for col in ['date', 'timestamp', 'time']:
            if col in df.columns:
                time_column = col
                break

        if time_column is None:
            print(f"   Warning: No date/timestamp column found. CSV will not include time information.")

        # Get configuration
        has_groups = self.group_columns is not None
        is_multi_target = self.is_multi_target
        prediction_horizon = self.prediction_horizon
        sequence_length = self.sequence_length

        # Get target columns
        if is_multi_target:
            target_columns = self.target_columns
        else:
            target_columns = [self.target_columns[0] if isinstance(self.target_columns, list) else self.target_columns]

        # Get group column name
        group_col_name = None
        if has_groups:
            if isinstance(self.group_columns, list) and len(self.group_columns) > 0:
                group_col_name = self.group_columns[0]
            elif isinstance(self.group_columns, str):
                group_col_name = self.group_columns

        # Extract time values (offset by sequence_length)
        if time_column:
            time_values = df[time_column].values[sequence_length:]
        else:
            time_values = None

        # Extract group values (offset by sequence_length)
        if has_groups and group_col_name and group_col_name in df.columns:
            group_values = df[group_col_name].values[sequence_length:]
        else:
            group_values = None

        # Extract actuals for each target (offset by sequence_length)
        actuals_dict = {}
        for target_col in target_columns:
            if target_col in df.columns:
                actuals_dict[target_col] = df[target_col].values[sequence_length:]
            else:
                actuals_dict[target_col] = None

        # Determine number of samples
        if is_multi_target:
            first_target = target_columns[0]
            n_samples = len(predictions[first_target])
        else:
            n_samples = len(predictions) if prediction_horizon == 1 else len(predictions)

        # Build results
        results = []
        for i in range(n_samples):
            row = {}

            # Add group if applicable
            if group_values is not None and i < len(group_values):
                row[group_col_name] = group_values[i]
            elif group_indices is not None and i < len(group_indices):
                row[group_col_name] = group_indices[i]

            # Add time
            if time_values is not None and i < len(time_values):
                row[time_column] = time_values[i]

            # Add actuals and predictions for each target
            for target_col in target_columns:
                # Get predictions for this target
                if is_multi_target:
                    target_preds = predictions[target_col]
                else:
                    target_preds = predictions

                # Add actual values and predictions for each horizon
                if prediction_horizon == 1:
                    # Single-horizon: one actual, one prediction
                    if actuals_dict[target_col] is not None and i < len(actuals_dict[target_col]):
                        row[f'{target_col}_actual_h1'] = actuals_dict[target_col][i]
                    else:
                        row[f'{target_col}_actual_h1'] = None

                    if i < len(target_preds):
                        row[f'{target_col}_pred_h1'] = target_preds[i]
                    else:
                        row[f'{target_col}_pred_h1'] = None
                else:
                    # Multi-horizon: separate actual and prediction for each horizon
                    for h in range(prediction_horizon):
                        horizon_num = h + 1

                        # Actual for horizon h is at position i + h
                        # (for sample at time t, h1 actual is at t+1, h2 is at t+2, etc.)
                        if actuals_dict[target_col] is not None and (i + h) < len(actuals_dict[target_col]):
                            row[f'{target_col}_actual_h{horizon_num}'] = actuals_dict[target_col][i + h]
                        else:
                            row[f'{target_col}_actual_h{horizon_num}'] = None

                        # Prediction for horizon h
                        if i < len(target_preds):
                            row[f'{target_col}_pred_h{horizon_num}'] = target_preds[i, h]
                        else:
                            row[f'{target_col}_pred_h{horizon_num}'] = None

            results.append(row)

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Sort by group (if exists) and time
        sort_cols = []
        if group_col_name and group_col_name in results_df.columns:
            sort_cols.append(group_col_name)
        if time_column and time_column in results_df.columns:
            sort_cols.append(time_column)

        if sort_cols:
            results_df = results_df.sort_values(sort_cols).reset_index(drop=True)

        # Create output directory if needed
        csv_path_obj = Path(csv_path)
        csv_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Build metrics summary section
        metrics_rows = []
        if metrics:
            metrics_rows = self._flatten_metrics_for_csv(metrics, dataset='test')

        # Write combined CSV: metrics summary + blank row + predictions
        with open(csv_path, 'w') as f:
            if metrics_rows:
                # Write metrics summary section
                metrics_df = pd.DataFrame(metrics_rows)
                metrics_df.to_csv(f, index=False)
                # Write blank row separator
                f.write('\n')

            # Write predictions section
            results_df.to_csv(f, index=False)

        print(f"\n✅ Predictions exported to CSV:")
        print(f"   File: {csv_path}")
        if metrics_rows:
            print(f"   Metrics rows: {len(metrics_rows)}")
        print(f"   Prediction rows: {len(results_df)}")
        print(f"   Prediction columns: {len(results_df.columns)}")
        if group_col_name and group_col_name in results_df.columns:
            print(f"   Groups: {results_df[group_col_name].nunique()}")
        print(f"   Targets: {', '.join(target_columns)}")
        print(f"   Horizons: {prediction_horizon}")

    def _flatten_metrics_for_csv(self, metrics: Dict, dataset: str = 'test') -> list:
        """
        Flatten nested metrics dict into rows for CSV export.

        Handles various metric structures:
        - Single-target, single-horizon: {'MAE': 0.5, 'RMSE': 0.7, ...}
        - Multi-horizon: {'overall': {...}, 'horizon_1': {...}, ...}
        - Multi-target: {'close': {...}, 'volume': {...}}
        - Multi-group: {'overall': {...}, 'AAPL': {...}, ...}

        Returns:
            List of dicts with columns: dataset, group, target, horizon, metric, value
        """
        rows = []

        # Determine if multi-target
        is_multi_target = self.is_multi_target
        is_multi_horizon = self.prediction_horizon > 1

        # Check if per-group metrics (keys are group names)
        has_per_group = False
        if isinstance(metrics, dict) and 'overall' in metrics and len(metrics) > 1:
            # Could be multi-horizon or multi-group
            # If 'overall' value is a dict with 'MAE' etc, it's not multi-group
            overall_val = metrics['overall']
            if isinstance(overall_val, dict) and 'MAE' not in overall_val:
                # Could be multi-horizon overall metrics {'overall': {...}, 'horizon_1': {...}}
                # Check if other keys are group names or horizon names
                other_keys = [k for k in metrics.keys() if k != 'overall']
                if other_keys and not any(k.startswith('horizon_') for k in other_keys[:1]):
                    has_per_group = True

        if has_per_group:
            # Multi-group metrics
            for group_name, group_metrics in metrics.items():
                if is_multi_target:
                    # Multi-group, multi-target
                    for target_name, target_metrics in group_metrics.items():
                        rows.extend(self._extract_metric_rows(
                            target_metrics, dataset, group_name, target_name
                        ))
                else:
                    # Multi-group, single-target
                    target_name = target_columns[0] if len(target_columns) > 0 else self.target_columns[0]
                    rows.extend(self._extract_metric_rows(
                        group_metrics, dataset, group_name, target_name
                    ))
        elif is_multi_target:
            # Multi-target (no groups)
            for target_name, target_metrics in metrics.items():
                rows.extend(self._extract_metric_rows(
                    target_metrics, dataset, 'overall', target_name
                ))
        else:
            # Single-target (no groups)
            target_name = self.target_columns[0] if isinstance(self.target_columns, list) else self.target_columns
            rows.extend(self._extract_metric_rows(
                metrics, dataset, 'overall', target_name
            ))

        return rows

    def _extract_metric_rows(self, metrics_dict: Dict, dataset: str, group: str, target: str) -> list:
        """
        Extract metric rows from a single target's metrics dict.

        Handles:
        - Single-horizon: {'MAE': 0.5, 'RMSE': 0.7, ...}
        - Multi-horizon: {'overall': {...}, 'horizon_1': {...}, 'horizon_2': {...}}
        """
        rows = []

        # Check if multi-horizon structure
        if 'overall' in metrics_dict and isinstance(metrics_dict['overall'], dict):
            # Multi-horizon metrics
            for horizon_key, horizon_metrics in metrics_dict.items():
                if isinstance(horizon_metrics, dict):
                    for metric_name, metric_value in horizon_metrics.items():
                        rows.append({
                            'dataset': dataset,
                            'group': group,
                            'target': target,
                            'horizon': horizon_key,  # 'overall', 'horizon_1', 'horizon_2', etc.
                            'metric': metric_name,
                            'value': metric_value
                        })
        else:
            # Single-horizon metrics (flat dict)
            for metric_name, metric_value in metrics_dict.items():
                if isinstance(metric_value, (int, float)):
                    rows.append({
                        'dataset': dataset,
                        'group': group,
                        'target': target,
                        'horizon': '1',
                        'metric': metric_name,
                        'value': metric_value
                    })

        return rows

    def save(self, path: str):
        """Save the trained model and preprocessors."""
        if self.model is None:
            raise RuntimeError("No model to save. Train first.")

        state = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'scaler_type': self.scaler_type,  # Save scaler type
            'feature_columns': self.feature_columns,
            'num_features': len(self.feature_columns),  # Save the actual number of features used
            'target_columns': self.target_columns,  # Save list
            'is_multi_target': self.is_multi_target,  # Save flag
            'num_targets': self.num_targets,  # Save count
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,  # Explicitly save
            'model_kwargs': self.model_kwargs,  # Save model kwargs
            'model_type': self.model_type,  # Save model type
            'use_lagged_target_features': self.use_lagged_target_features,  # Save lagged target feature flag
            'history': self.history,
            # Group-based scaling (save new format)
            'group_columns': self.group_columns,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'group_feature_scalers': self.group_feature_scalers,
            'group_target_scalers': self.group_target_scalers,
            # Categorical encoders (needed for CLS models)
            'cat_encoders': self.cat_encoders,
            'cat_cardinalities': self.cat_cardinalities
        }

        # Add per-horizon target scalers (used for both single and multi-target)
        state['target_scalers_dict'] = self.target_scalers_dict

        torch.save(state, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, **kwargs):
        """Load a saved model."""
        state = torch.load(path, map_location='cpu', weights_only=False)

        # Create predictor with correct parameters
        predictor = cls(
            target_column=state['target_columns'],
            sequence_length=state['sequence_length'],
            prediction_horizon=state['prediction_horizon'],
            group_columns=state['group_columns'],
            categorical_columns=state['categorical_columns'],
            model_type=state['model_type'],
            scaler_type=state['scaler_type'],
            use_lagged_target_features=state['use_lagged_target_features'],
            **state['model_kwargs']
        )

        # Restore feature scaler and feature columns
        predictor.scaler = state['scaler']
        predictor.feature_columns = state['feature_columns']
        predictor.numerical_columns = state['numerical_columns']
        predictor.history = state['history']

        # Restore per-horizon target scalers (now used for both single and multi-target)
        predictor.target_scalers_dict = state['target_scalers_dict']

        # Restore group scalers
        predictor.group_feature_scalers = state['group_feature_scalers']
        predictor.group_target_scalers = state['group_target_scalers']

        # Restore categorical encoders and cardinalities (needed for CLS models)
        if 'cat_encoders' in state:
            predictor.cat_encoders = state['cat_encoders']
        if 'cat_cardinalities' in state:
            predictor.cat_cardinalities = state['cat_cardinalities']

        # Recreate model with correct output size
        num_features = state['num_features']

        # Calculate total output size
        if predictor.is_multi_target:
            total_output_size = predictor.num_targets * predictor.prediction_horizon
        else:
            total_output_size = predictor.prediction_horizon

        # Filter out non-model parameters from model_kwargs
        # 'verbose' is a general parameter, not for model init
        model_kwargs = {k: v for k, v in predictor.model_kwargs.items()
                       if k not in ['verbose']}

        # Recreate model using factory
        if predictor.model_type.endswith('_cls'):
            # CLS models need separate numerical/categorical counts
            num_numerical = len(predictor.numerical_columns) if predictor.numerical_columns else num_features
            num_categorical = len(predictor.categorical_columns) if predictor.categorical_columns else 0

            predictor.model = ModelFactory.create_model(
                model_type=predictor.model_type,
                sequence_length=predictor.sequence_length if predictor.sequence_length > 1 else 1,
                num_numerical=num_numerical,
                num_categorical=num_categorical,
                cat_cardinalities=predictor.cat_cardinalities,
                output_dim=total_output_size,
                **model_kwargs
            )
        else:
            # Standard models use num_features
            predictor.model = ModelFactory.create_model(
                model_type=predictor.model_type,
                sequence_length=predictor.sequence_length if predictor.sequence_length > 1 else 1,
                num_features=num_features,
                output_dim=total_output_size,
                **model_kwargs
            )

        predictor.model.load_state_dict(state['model_state_dict'])
        predictor.model.to(predictor.device)

        print(f"Model loaded from {path}")
        return predictor
