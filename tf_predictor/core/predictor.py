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
        target_column: str,
        sequence_length: int = 5,
        prediction_horizon: int = 1,
        group_column: Optional[str] = None,
        **ft_kwargs
    ):
        """
        Args:
            target_column: Name of the target column to predict
            sequence_length: Number of historical time steps to use for prediction
            prediction_horizon: Number of steps ahead to predict (1=single, >1=multi-horizon)
            group_column: Optional column name for group-based scaling (e.g., 'symbol')
                         If provided, each unique value gets its own scaler
            **ft_kwargs: FT-Transformer hyperparameters
        """
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.group_column = group_column
        self.ft_kwargs = ft_kwargs

        # Will be set during training
        self.model = None

        # Single-group scalers (used when group_column=None)
        self.scaler = StandardScaler()  # For features
        self.target_scaler = StandardScaler()  # For single target
        self.target_scalers = []  # For multi-horizon targets

        # Multi-group scalers (used when group_column is provided)
        # Structure: {group_value: StandardScaler} for features
        # Structure: {group_value: StandardScaler} for targets (same scaler for all horizons per group)
        self.group_feature_scalers = {}  # Dict[group_value, StandardScaler]
        self.group_target_scalers = {}   # Dict[group_value, StandardScaler]

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
        # Get original target column if it exists (for multi-horizon prediction)
        original_target = getattr(self, 'original_target_column', self.target_column)

        for col in df_processed.columns:
            # Exclude both the target column and the original column it came from
            if col != self.target_column and col != original_target:
                # Only include numeric columns for scaling
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    feature_cols.append(col)
                elif self.verbose:
                    print(f"   Excluding non-numeric column from scaling: {col} (dtype: {df_processed[col].dtype})")
            elif self.verbose and col == original_target and col != self.target_column:
                print(f"   Excluding original target column from scaling: {col}")
        
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
            sequences, targets_h1 = create_sequences(group_df, self.sequence_length, target_cols[0])

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
        y_combined = np.vstack(all_targets) if self.prediction_horizon > 1 else np.concatenate(all_targets)

        # Store group indices for inverse transform during prediction
        self._last_group_indices = group_indices

        # Convert to tensors
        X = torch.tensor(X_combined, dtype=torch.float32)

        if self.prediction_horizon == 1:
            y = torch.tensor(y_combined, dtype=torch.float32)
        else:
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
        
        # Handle single vs multi-horizon target validation
        if self.prediction_horizon == 1:
            expected_target = f"{self.target_column}_target_h1" if not self.target_column.endswith('_target_h1') else self.target_column
            if expected_target not in df_processed.columns:
                available_cols = list(df_processed.columns)
                raise ValueError(f"Single horizon target column '{expected_target}' not found after feature engineering.\n"
                               f"Available columns: {available_cols[:10]}{'...' if len(available_cols) > 10 else ''}")
            actual_target = expected_target
        else:
            # Multi-horizon: check all horizon targets exist
            target_columns = [f"{self.target_column}_target_h{h}" for h in range(1, self.prediction_horizon + 1)]
            missing_targets = [col for col in target_columns if col not in df_processed.columns]
            if missing_targets:
                available_cols = list(df_processed.columns)
                raise ValueError(f"Multi-horizon target columns {missing_targets} not found after feature engineering.\n"
                               f"Available columns: {available_cols[:10]}{'...' if len(available_cols) > 10 else ''}")
            actual_target = target_columns[0]  # Use first for sequence creation

        # Create sequences - this will reduce our sample count
        if actual_target in df_processed.columns:
            from ..preprocessing.time_features import create_sequences
            
            # For training: create sequences with targets
            sequences, targets = create_sequences(df_processed, self.sequence_length, actual_target)
            
            if self.prediction_horizon == 1:
                # Single horizon target scaling
                targets = targets.reshape(-1, 1)
                if fit_scaler:
                    targets_scaled = self.target_scaler.fit_transform(targets)
                else:
                    targets_scaled = self.target_scaler.transform(targets)

                # Convert to tensors
                X = torch.tensor(sequences, dtype=torch.float32)  # (n_samples, seq_len, n_features)
                y = torch.tensor(targets_scaled.flatten(), dtype=torch.float32)

            else:
                # Multi-horizon target handling - MEMORY OPTIMIZED
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
            sequences, _ = create_sequences(temp_df, self.sequence_length, '__dummy_target__')
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
        else:
            X_val, y_val = None, None
        
        # Initialize model - use sequence model
        if len(X_train.shape) == 3:  # Sequence data: (batch, seq_len, features)
            _, seq_len, num_features = X_train.shape
            # Filter out invalid kwargs for model initialization
            model_kwargs = {k: v for k, v in self.ft_kwargs.items() if k not in ['verbose']}
            self.model = SequenceFTTransformerPredictor(
                num_numerical=num_features,
                cat_cardinalities=[],  # All features are numerical for now
                sequence_length=seq_len,
                n_classes=1,  # Regression
                prediction_horizons=self.prediction_horizon,
                **model_kwargs
            ).to(self.device)
        else:  # Single timestep data: (batch, features) - fallback to original model
            num_features = X_train.shape[1]
            # Filter out invalid kwargs for model initialization
            model_kwargs = {k: v for k, v in self.ft_kwargs.items() if k not in ['verbose']}
            self.model = FTTransformerPredictor(
                num_numerical=num_features,
                cat_cardinalities=[],  # All features are numerical for now
                n_classes=1,  # Regression
                prediction_horizons=self.prediction_horizon,
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
                        val_pred_scaled_np = val_outputs.cpu().numpy().reshape(-1, 1)
                        val_pred = self.target_scaler.inverse_transform(val_pred_scaled_np).flatten()

                        val_actual_scaled_np = y_val_device.cpu().numpy().reshape(-1, 1)
                        val_actual = self.target_scaler.inverse_transform(val_actual_scaled_np).flatten()

                        # Calculate MAPE and MAE
                        val_mae = np.mean(np.abs(val_actual - val_pred))
                        val_mape = np.mean(np.abs((val_actual - val_pred) / val_actual)) * 100

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
            # Handle both sequence and non-sequence models
            if len(X.shape) == 3:  # Sequence data
                predictions_scaled = self.model(X.to(self.device))
            else:  # Non-sequence data
                predictions_scaled = self.model(X.to(self.device), None)

            # Convert to numpy
            predictions_scaled = predictions_scaled.cpu().numpy()

            # Handle group-based vs single-group inverse transform
            if self.group_column is not None:
                # Group-based inverse transform
                if not hasattr(self, '_last_group_indices') or len(self._last_group_indices) != len(predictions_scaled):
                    raise RuntimeError("Group indices not available or mismatched. This shouldn't happen.")

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

    def predict_from_features(self, df_processed: pd.DataFrame) -> np.ndarray:
        """
        Make predictions from already-processed features.

        Args:
            df_processed: DataFrame with preprocessed features

        Returns:
            predictions: Numpy array of predictions (in original scale)
        """
        if self.model is None:
            raise RuntimeError("Model must be trained first. Call fit().")

        # Skip feature preprocessing since it's already done
        from ..preprocessing.time_features import create_sequences

        # Create sequences directly from processed features
        if self.target_column in df_processed.columns:
            sequences, _ = create_sequences(df_processed, self.sequence_length, self.target_column)
        else:
            # Use first feature as dummy target for sequence creation
            dummy_target = df_processed[self.feature_columns[0]]
            temp_df = df_processed.copy()
            temp_df['__dummy_target__'] = dummy_target
            sequences, _ = create_sequences(temp_df, self.sequence_length, '__dummy_target__')

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

        # Now check if target column exists after processing
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
        predictions = self.predict(df_processed.copy())  # Use processed features

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

        # Prepare actual values
        if self.sequence_length > 1:
            actual_base = df_processed[self.target_column].values[self.sequence_length:]
        else:
            actual_base = df_processed[self.target_column].values

        # Get unique groups
        unique_groups = sorted(set(group_indices))

        # Storage for per-group metrics
        all_metrics = {}

        # Calculate metrics per group
        for group_value in unique_groups:
            # Find indices for this group
            group_mask = np.array([g == group_value for g in group_indices])
            group_preds = predictions[group_mask] if self.prediction_horizon == 1 else predictions[group_mask, :]

            # Get actual values for this group
            # We need to figure out the actual indices in the original data
            # The group_mask tells us which predictions belong to this group
            group_pred_indices = np.where(group_mask)[0]

            if self.prediction_horizon == 1:
                # Single-horizon
                group_actual = actual_base[group_pred_indices]
                min_len = min(len(group_actual), len(group_preds))

                group_metrics = calculate_metrics(group_actual[:min_len], group_preds[:min_len])
                all_metrics[str(group_value)] = group_metrics

            else:
                # Multi-horizon
                # For each prediction, we need actual values h steps ahead
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
            'target_scaler': self.target_scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'sequence_length': self.sequence_length,
            'ft_kwargs': self.ft_kwargs,
            'history': self.history,
            # Group-based scaling
            'group_column': self.group_column,
            'group_feature_scalers': self.group_feature_scalers,
            'group_target_scalers': self.group_target_scalers
        }
        torch.save(state, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, **kwargs):
        """Load a saved model."""
        state = torch.load(path, map_location='cpu')

        # Create predictor
        predictor = cls(
            target_column=state['target_column'],
            sequence_length=state.get('sequence_length', 5),
            group_column=state.get('group_column', None),  # Load group_column
            **state['ft_kwargs']
        )

        # Restore state
        predictor.scaler = state['scaler']
        predictor.target_scaler = state['target_scaler']
        predictor.feature_columns = state['feature_columns']
        predictor.history = state.get('history', {'train_loss': [], 'val_loss': []})

        # Restore group scalers (if present)
        predictor.group_feature_scalers = state.get('group_feature_scalers', {})
        predictor.group_target_scalers = state.get('group_target_scalers', {})
        
        # Recreate model - choose based on sequence_length
        num_features = len(predictor.feature_columns)
        
        # Filter out non-model parameters from ft_kwargs
        model_kwargs = {k: v for k, v in predictor.ft_kwargs.items() 
                       if k not in ['verbose']}
        
        if predictor.sequence_length > 1:
            predictor.model = SequenceFTTransformerPredictor(
                num_numerical=num_features,
                cat_cardinalities=[],
                sequence_length=predictor.sequence_length,
                n_classes=1,
                **model_kwargs
            )
        else:
            predictor.model = FTTransformerPredictor(
                num_numerical=num_features,
                cat_cardinalities=[],
                n_classes=1,
                **model_kwargs
            )
        predictor.model.load_state_dict(state['model_state_dict'])
        predictor.model.to(predictor.device)
        
        print(f"Model loaded from {path}")
        return predictor