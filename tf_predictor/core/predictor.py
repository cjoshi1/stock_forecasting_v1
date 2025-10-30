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

        # Target scalers structure depends on single vs multi-target
        # Single-target mode:
        #   - single-horizon: self.target_scaler (Scaler)
        #   - multi-horizon: self.target_scaler (Scaler) - SAME scaler for all horizons
        # Multi-target mode:
        #   - single-horizon: self.target_scalers_dict (Dict[target_name, Scaler])
        #   - multi-horizon: self.target_scalers_dict (Dict[target_name, Scaler]) - ONE scaler per variable
        if not self.is_multi_target:
            self.target_scaler = ScalerFactory.create_scaler(scaler_type)  # For single target
        else:
            self.target_scalers_dict = {}  # For multi-target (will be populated with scalers)

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
        self.verbose = False  # Will be set during training

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

    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Prepare features by creating time-series features and handling scaling.
        Uses caching to avoid recomputing features for the same data.

        This method can be overridden by subclasses to add domain-specific features.

        Note: This method does not modify the input dataframe (creates a copy internally).

        Args:
            df: DataFrame with raw data
            fit_scaler: Whether to fit the scaler (True for training data)

        Returns:
            processed_df: DataFrame with scaled features
        """
        # Generate cache key BEFORE copying (for efficiency)
        cache_key = f"{self._get_dataframe_hash(df)}_{fit_scaler}"

        # Check cache first
        if self._cache_enabled and cache_key in self._feature_cache:
            if self.verbose:
                print(f"   Using cached features for dataset ({len(df)} rows)")
            return self._feature_cache[cache_key].copy()

        # Create a copy to avoid modifying the input dataframe
        df = df.copy()

        # Sort by group columns and time (BEFORE feature creation and scaling)
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
            df_processed = create_date_features(df, time_column, group_column=group_col_for_date_features)
        else:
            df_processed = df.copy()

        # Encode categorical features (BEFORE scaling numerical features)
        df_processed = self._encode_categorical_features(df_processed, fit_encoders=fit_scaler)

        # Get all feature columns
        # The key logic: use_lagged_target_features controls whether targets are in the sequence
        feature_cols = []

        # Build exclusion set: shifted target columns + categorical columns (always excluded from numerical features)
        exclude_cols = set()
        for target_col in self.target_columns:
            # Always exclude shifted horizon columns (these are Y, not X)
            for h in range(1, self.prediction_horizon + 1):
                exclude_cols.add(f"{target_col}_target_h{h}")

        # Exclude categorical columns from numerical features (they'll be handled separately)
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
        for col in df_processed.columns:
            if col not in exclude_cols:
                # Only include numeric columns
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    feature_cols.append(col)
                elif self.verbose:
                    print(f"   Excluding non-numeric column: {col} (dtype: {df_processed[col].dtype})")
            elif self.verbose and '_target_h' in col:
                pass  # Don't print for every shifted column, too verbose
        
        if self.numerical_columns is None:
            # First time: establish the numeric feature columns from training data
            self.numerical_columns = feature_cols
            self.feature_columns = feature_cols
        else:
            # For validation/test data: only use features that were created during training
            # This handles cases where smaller datasets can't generate all features
            available_features = [col for col in self.numerical_columns if col in df_processed.columns]
            missing_features = [col for col in self.numerical_columns if col not in df_processed.columns]

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
        if len(self.numerical_columns) > 0:
            if not self.group_columns:
                # Single-group scaling (no grouping)
                df_processed = self._scale_features_single(df_processed, fit_scaler)
            else:
                # Multi-group scaling
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
        Scale features separately per group (supports multi-column grouping).

        Each unique group (single column or composite) gets its own scaler. This ensures
        that different entities (e.g., different stock symbols, or symbol+sector combinations)
        are scaled independently, preserving their individual statistical properties.

        Args:
            df_processed: DataFrame with engineered features
            fit_scaler: Whether to fit the scalers

        Returns:
            DataFrame with group-scaled features
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

        # Remove temporary column
        df_scaled = df_scaled.drop(columns=['_group_key'])

        return df_scaled

    def _prepare_data_grouped(self, df_processed: pd.DataFrame, fit_scaler: bool) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Prepare sequential data with group-based target scaling.

        Args:
            df_processed: DataFrame with features already scaled by group
            fit_scaler: Whether to fit new target scalers or use existing ones

        Returns:
            X: If categorical_columns exist: Tuple of (X_num, X_cat) where
                  X_num: (n_samples, sequence_length, n_numerical_features)
                  X_cat: (n_samples, n_categorical_features)
               Otherwise: X_num tensor only
            y: Target tensor of shape (n_samples,) for single-horizon or (n_samples, prediction_horizon) for multi-horizon
        """

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
                            scaler = ScalerFactory.create_scaler(self.scaler_type)
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
                            # Use ONE scaler for all horizons of this target (shared statistics)
                            scaler = ScalerFactory.create_scaler(self.scaler_type)
                            targets_scaled = scaler.fit_transform(targets_matrix)
                            self.group_target_scalers[group_value][target_col] = scaler
                        else:
                            # Use existing scaler
                            if group_value not in self.group_target_scalers or target_col not in self.group_target_scalers[group_value]:
                                raise ValueError(f"No scaler found for group '{group_value}', target '{target_col}'")
                            targets_scaled = self.group_target_scalers[group_value][target_col].transform(targets_matrix)

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

                all_sequences_num.append(sequences_num)
                if sequences_cat is not None:
                    all_sequences_cat.append(sequences_cat)
                all_targets.append(y_combined)
                group_indices.extend([group_value] * len(sequences_num))

            else:
                # Single-target mode (original behavior)
                if self.prediction_horizon == 1:
                    # Single horizon - scale targets
                    target_col_name = target_cols[0]
                    targets_h1 = group_df[target_col_name].values[self.sequence_length:]
                    targets = targets_h1.reshape(-1, 1)

                    if fit_scaler:
                        scaler = ScalerFactory.create_scaler(self.scaler_type)
                        targets_scaled = scaler.fit_transform(targets).flatten()
                        self.group_target_scalers[group_value] = scaler
                    else:
                        if group_value not in self.group_target_scalers:
                            raise ValueError(f"No fitted target scaler found for group '{group_value}'")
                        targets_scaled = self.group_target_scalers[group_value].transform(targets).flatten()

                    all_sequences_num.append(sequences_num)
                    if sequences_cat is not None:
                        all_sequences_cat.append(sequences_cat)
                    all_targets.append(targets_scaled)
                    # Track which group these sequences belong to
                    group_indices.extend([group_value] * len(sequences_num))

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
                        scaler = ScalerFactory.create_scaler(self.scaler_type)
                        targets_scaled = scaler.fit_transform(targets_matrix)
                        self.group_target_scalers[group_value] = scaler
                    else:
                        if group_value not in self.group_target_scalers:
                            raise ValueError(f"No fitted target scaler found for group '{group_value}'")
                        targets_scaled = self.group_target_scalers[group_value].transform(targets_matrix)

                    all_sequences_num.append(sequences_num)
                    if sequences_cat is not None:
                        all_sequences_cat.append(sequences_cat)
                    all_targets.append(targets_scaled)
                    # Track which group these sequences belong to
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

    def prepare_data(self, df: pd.DataFrame, fit_scaler: bool = False) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Prepare sequential data for model training/inference.

        Note: This method does not modify the input dataframe (creates a copy internally).

        Returns:
            X: If categorical_columns exist: Tuple of (X_num, X_cat) where
                  X_num: (n_samples, sequence_length, n_numerical_features)
                  X_cat: (n_samples, n_categorical_features)
               Otherwise: X_num tensor only
            y: Target tensor of shape (n_samples,) (None for inference)
        """
        # Create a copy to avoid modifying the input dataframe
        df = df.copy()

        # First prepare features (includes group-based or single scaling)
        df_processed = self.prepare_features(df, fit_scaler)

        # Create shifted target columns
        # Use categorical_columns for shifting to prevent data leakage across all categorical boundaries
        # Falls back to group_columns if categorical_columns not specified
        group_col_for_shift = self.categorical_columns if self.categorical_columns else self.group_columns
        df_processed = create_shifted_targets(
            df_processed,
            target_column=self.target_columns,
            prediction_horizon=self.prediction_horizon,
            group_column=group_col_for_shift,
            verbose=self.verbose
        )

        # Route to appropriate data preparation method
        if self.group_columns:
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
            # Single-target mode
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
            from ..preprocessing.time_features import create_input_variable_sequence

            # For training: create sequences (input variables only)
            sequences = create_input_variable_sequence(df_processed, self.sequence_length, self.feature_columns)

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
                            scaler = ScalerFactory.create_scaler(self.scaler_type)
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
                            # Use ONE scaler for all horizons of this target (shared statistics)
                            scaler = ScalerFactory.create_scaler(self.scaler_type)
                            targets_scaled = scaler.fit_transform(targets_matrix)
                            self.target_scalers_dict[target_col] = scaler
                        else:
                            # Use existing scaler
                            if target_col not in self.target_scalers_dict:
                                raise ValueError(f"No scaler found for target '{target_col}'")
                            targets_scaled = self.target_scalers_dict[target_col].transform(targets_matrix)

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
                # Extract target values manually (after sequence_length offset)
                target_col_name = f"{self.target_columns[0]}_target_h1"
                targets = df_processed[target_col_name].values[self.sequence_length:].reshape(-1, 1)
                if fit_scaler:
                    targets_scaled = self.target_scaler.fit_transform(targets)
                else:
                    targets_scaled = self.target_scaler.transform(targets)

                # Convert to tensors
                X = torch.tensor(sequences, dtype=torch.float32)  # (n_samples, seq_len, n_features)
                y = torch.tensor(targets_scaled.flatten(), dtype=torch.float32)

            else:
                # Single-target, multi-horizon target handling - MEMORY OPTIMIZED
                target_columns = [f"{self.target_columns[0]}_target_h{h}" for h in range(1, self.prediction_horizon + 1)]

                # Extract target values directly without re-creating sequences
                # This is much more memory efficient than calling create_input_variable_sequence multiple times
                all_targets = []
                for target_col in target_columns:
                    # Extract target values starting from sequence_length (matching sequence indexing)
                    target_values = df_processed[target_col].values[self.sequence_length:]
                    all_targets.append(target_values)

                # Stack into matrix: (samples, horizons)
                targets_matrix = np.column_stack(all_targets)

                # Use ONE scaler for all horizons (shared statistics)
                if fit_scaler:
                    self.target_scaler = ScalerFactory.create_scaler(self.scaler_type)
                    targets_scaled = self.target_scaler.fit_transform(targets_matrix)
                else:
                    targets_scaled = self.target_scaler.transform(targets_matrix)

                # Convert to tensors
                X = torch.tensor(sequences, dtype=torch.float32)  # (n_samples, seq_len, n_features)
                y = torch.tensor(targets_scaled, dtype=torch.float32)  # (n_samples, horizons)
            
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

            from ..preprocessing.time_features import create_input_variable_sequence
            sequences = create_input_variable_sequence(temp_df, self.sequence_length, self.feature_columns)
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
                                # Single-target, multi-horizon: use single scaler for all horizons
                                val_pred = self.target_scaler.inverse_transform(val_pred_scaled_np)
                                val_actual = self.target_scaler.inverse_transform(val_actual_scaled_np)

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

        X, _ = self.prepare_data(df, fit_scaler=False)

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
                                group_preds_original = self.group_target_scalers[group_value][target_col].inverse_transform(group_preds_scaled).flatten()
                                target_preds[group_mask] = group_preds_original
                            else:
                                # Multi-horizon: extract horizons for this target
                                # Layout: [close_h1, close_h2, ..., volume_h1, volume_h2, ...]
                                start_idx = idx * self.prediction_horizon
                                end_idx = start_idx + self.prediction_horizon

                                # Extract all horizons for this target in this group
                                group_horizons_scaled = predictions_scaled[group_mask, start_idx:end_idx]

                                # Inverse transform all horizons together using single scaler
                                group_horizons_original = self.group_target_scalers[group_value][target_col].inverse_transform(group_horizons_scaled)

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

                            # Inverse transform all horizons together using single scaler
                            horizons_scaled = predictions_scaled[:, start_idx:end_idx]
                            horizons_original = self.target_scalers_dict[target_col].inverse_transform(horizons_scaled)

                            # Stack into (n_samples, horizons) for this target
                            predictions_dict[target_col] = horizons_original

                    return predictions_dict

                elif self.prediction_horizon == 1:
                    # Single-target, single horizon: reshape to (n_samples, 1) for inverse transform
                    predictions_scaled = predictions_scaled.reshape(-1, 1)
                    predictions = self.target_scaler.inverse_transform(predictions_scaled)
                    return predictions.flatten()
                else:
                    # Single-target, multi-horizon: predictions_scaled shape is (n_samples, horizons)
                    # Inverse transform all horizons together using single scaler
                    predictions = self.target_scaler.inverse_transform(predictions_scaled)
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
            if self.prediction_horizon == 1:
                # Single horizon: reshape to (n_samples, 1) for inverse transform
                predictions_scaled = predictions_scaled.reshape(-1, 1)
                predictions = self.target_scaler.inverse_transform(predictions_scaled)
                return predictions.flatten()
            else:
                # Multi-horizon: predictions_scaled shape is (n_samples, horizons)
                # Inverse transform all horizons together using single scaler
                predictions = self.target_scaler.inverse_transform(predictions_scaled)
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

        # Check if we should do per-group evaluation
        if per_group and self.group_columns:
            return self._evaluate_per_group(df)
        else:
            # Standard evaluation - predict() handles all preprocessing
            return self._evaluate_standard(df)

    def _evaluate_standard(self, df: pd.DataFrame) -> Dict:
        """Standard evaluation without per-group breakdown."""
        predictions = self.predict(df)  # Returns dict for multi-target, predict() handles all preprocessing

        # Check if multi-target
        if self.is_multi_target:
            # Multi-target evaluation: return metrics per target
            from ..core.utils import calculate_metrics, calculate_metrics_multi_horizon

            metrics_dict = {}

            for target_col in self.target_columns:
                # Get actual values for this target from raw dataframe
                if self.sequence_length > 1:
                    actual = df[target_col].values[self.sequence_length:]
                else:
                    actual = df[target_col].values

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
            # Single-target evaluation
            # For sequences, we need to align the actual values with predictions
            target_col = self.target_columns[0]
            if self.sequence_length > 1:
                actual = df[target_col].values[self.sequence_length:]
            else:
                actual = df[target_col].values

            # Handle single vs multi-horizon evaluation
            if self.prediction_horizon == 1:
                # Single-horizon: return simple metrics dict
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

    def _evaluate_per_group(self, df: pd.DataFrame) -> Dict:
        """
        Evaluate performance per group (e.g., per stock symbol).

        Returns nested dict with overall metrics plus per-group breakdown.
        """
        from ..core.utils import calculate_metrics, calculate_metrics_multi_horizon

        # Get predictions with group information - predict() handles all preprocessing
        predictions, group_indices = self.predict(df, return_group_info=True)

        # Get unique groups
        unique_groups = sorted(set(group_indices))

        # We need to map group_value (label encoded int) back to original group column values
        # First, decode the group values using the categorical encoder
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
                target_predictions = predictions[target_col]

                # Calculate per-group metrics for this target
                for group_value in unique_groups:
                    # Get the original group name
                    group_name = group_value_to_name[group_value]

                    # Filter DataFrame by this group to get actual values
                    group_df = df[df[group_col_name] == group_name].copy()

                    # Extract actual values for this group with sequence_length offset
                    if self.sequence_length > 1:
                        group_actual_full = group_df[target_col].values[self.sequence_length:]
                    else:
                        group_actual_full = group_df[target_col].values

                    # Get predictions for this group
                    group_mask = np.array([g == group_value for g in group_indices])

                    if self.prediction_horizon == 1:
                        # Single-horizon
                        group_preds = target_predictions[group_mask]

                        # Align actual values with predictions
                        min_len = min(len(group_actual_full), len(group_preds))
                        group_actual = group_actual_full[:min_len]
                        group_preds = group_preds[:min_len]

                        group_metrics = calculate_metrics(group_actual, group_preds)
                    else:
                        # Multi-horizon
                        group_preds = target_predictions[group_mask, :]

                        # For multi-horizon, we need enough actual values to cover all horizons
                        # Each prediction at index i needs actuals[i:i+prediction_horizon]
                        num_preds = group_preds.shape[0]
                        needed_actuals = num_preds + self.prediction_horizon - 1

                        if len(group_actual_full) >= needed_actuals:
                            group_actual_aligned = group_actual_full[:needed_actuals]
                        else:
                            # Not enough actuals - trim predictions
                            num_preds = max(0, len(group_actual_full) - self.prediction_horizon + 1)
                            group_actual_aligned = group_actual_full
                            group_preds = group_preds[:num_preds] if num_preds > 0 else group_preds[:0]

                        if len(group_preds) > 0:
                            group_metrics = calculate_metrics_multi_horizon(
                                group_actual_aligned,
                                group_preds,
                                self.prediction_horizon
                            )
                        else:
                            # Skip if no valid predictions
                            continue

                    # Store metrics in nested structure
                    group_key = str(group_value)
                    if group_key not in all_metrics:
                        all_metrics[group_key] = {}
                    all_metrics[group_key][target_col] = group_metrics

                # Calculate overall metrics for this target (using all groups combined)
                # Get all actual values across all groups
                if self.sequence_length > 1:
                    all_actual = df[target_col].values[self.sequence_length:]
                else:
                    all_actual = df[target_col].values

                if self.prediction_horizon == 1:
                    min_len = min(len(all_actual), len(target_predictions))
                    overall_target_metrics = calculate_metrics(all_actual[:min_len], target_predictions[:min_len])
                else:
                    num_preds = target_predictions.shape[0]
                    needed_actuals = num_preds + self.prediction_horizon - 1

                    if len(all_actual) >= needed_actuals:
                        actual_aligned = all_actual[:needed_actuals]
                        preds_aligned = target_predictions
                    else:
                        num_preds = max(0, len(all_actual) - self.prediction_horizon + 1)
                        actual_aligned = all_actual
                        preds_aligned = target_predictions[:num_preds] if num_preds > 0 else target_predictions[:0]

                    if len(preds_aligned) > 0:
                        overall_target_metrics = calculate_metrics_multi_horizon(
                            actual_aligned,
                            preds_aligned,
                            self.prediction_horizon
                        )
                    else:
                        continue

                # Store overall metrics for this target
                if 'overall' not in all_metrics:
                    all_metrics['overall'] = {}
                all_metrics['overall'][target_col] = overall_target_metrics

            return all_metrics

        else:
            # Single-target per-group evaluation
            target_col = self.target_columns[0]

            # Storage for per-group metrics
            all_metrics = {}

            # Calculate metrics per group
            for group_value in unique_groups:
                # Get the original group name
                group_name = group_value_to_name[group_value]

                # Filter DataFrame by this group to get actual values
                group_df = df[df[group_col_name] == group_name].copy()

                # Extract actual values for this group with sequence_length offset
                if self.sequence_length > 1:
                    group_actual_full = group_df[target_col].values[self.sequence_length:]
                else:
                    group_actual_full = group_df[target_col].values

                # Get predictions for this group
                group_mask = np.array([g == group_value for g in group_indices])
                group_preds = predictions[group_mask] if self.prediction_horizon == 1 else predictions[group_mask, :]

                if self.prediction_horizon == 1:
                    # Single-horizon
                    min_len = min(len(group_actual_full), len(group_preds))
                    group_actual = group_actual_full[:min_len]
                    group_preds = group_preds[:min_len]

                    group_metrics = calculate_metrics(group_actual, group_preds)
                    all_metrics[str(group_value)] = group_metrics

                else:
                    # Multi-horizon
                    num_preds = group_preds.shape[0]
                    needed_actuals = num_preds + self.prediction_horizon - 1

                    if len(group_actual_full) >= needed_actuals:
                        group_actual_aligned = group_actual_full[:needed_actuals]
                    else:
                        # Not enough actuals - trim predictions
                        num_preds = max(0, len(group_actual_full) - self.prediction_horizon + 1)
                        group_actual_aligned = group_actual_full
                        group_preds = group_preds[:num_preds] if num_preds > 0 else group_preds[:0]

                    if len(group_preds) > 0:
                        group_metrics = calculate_metrics_multi_horizon(
                            group_actual_aligned,
                            group_preds,
                            self.prediction_horizon
                        )
                        all_metrics[str(group_value)] = group_metrics

            # Calculate overall metrics across all groups
            if self.sequence_length > 1:
                all_actual = df[target_col].values[self.sequence_length:]
            else:
                all_actual = df[target_col].values

            if self.prediction_horizon == 1:
                # Single-horizon overall
                min_len = min(len(all_actual), len(predictions))
                overall_metrics = calculate_metrics(all_actual[:min_len], predictions[:min_len])
            else:
                # Multi-horizon overall
                num_preds = predictions.shape[0]
                needed_actuals = num_preds + self.prediction_horizon - 1

                if len(all_actual) >= needed_actuals:
                    actual_aligned = all_actual[:needed_actuals]
                    preds_aligned = predictions
                else:
                    num_preds = max(0, len(all_actual) - self.prediction_horizon + 1)
                    actual_aligned = all_actual
                    preds_aligned = predictions[:num_preds] if num_preds > 0 else predictions[:0]

                if len(preds_aligned) > 0:
                    overall_metrics = calculate_metrics_multi_horizon(
                        actual_aligned,
                        preds_aligned,
                        self.prediction_horizon
                    )
                else:
                    overall_metrics = {}

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
            'group_target_scalers': self.group_target_scalers
        }

        # Add target scalers based on mode
        if not self.is_multi_target:
            # Single-target scalers
            state['target_scaler'] = self.target_scaler
        else:
            # Multi-target scalers dict
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

        # Restore target scalers based on mode
        if predictor.is_multi_target:
            predictor.target_scalers_dict = state['target_scalers_dict']
        else:
            predictor.target_scaler = state['target_scaler']
            predictor.target_scalers = state['target_scalers']

        # Restore group scalers
        predictor.group_feature_scalers = state['group_feature_scalers']
        predictor.group_target_scalers = state['group_target_scalers']

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
