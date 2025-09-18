"""
CSNPredictor: Categorical-Seasonal-Numerical Predictor for Time Series.

Extends the base TimeSeriesPredictor to use CSNTransformer architecture.
Maintains the same interface as FTTransformerPredictor for compatibility.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .predictor import TimeSeriesPredictor
from .csn_model import CSNTransformerPredictor
from .feature_detector import FeatureDetector


class CSNPredictor(TimeSeriesPredictor):
    """
    CSNTransformer-based predictor for time series forecasting.

    Uses separate processing for categorical and numerical features,
    with automatic feature detection and seasonal feature handling.
    """

    def __init__(self,
                 target_column: str,
                 sequence_length: int = 5,
                 d_model: int = 128,
                 n_layers: int = 3,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 categorical_threshold: int = 20,
                 **kwargs):
        """
        Args:
            target_column: Name of the target column to predict
            sequence_length: Number of historical time steps to use for prediction
            d_model: Model dimension for transformer
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            dropout: Dropout rate
            categorical_threshold: Max unique values for categorical classification
            **kwargs: Additional arguments passed to parent class
        """
        # Initialize parent class
        TimeSeriesPredictor.__init__(self, target_column, sequence_length, **kwargs)

        # CSN-specific parameters
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.categorical_threshold = categorical_threshold

        # Feature detection and processing
        self.feature_detector = FeatureDetector(categorical_threshold=categorical_threshold)
        self.feature_info = None
        self.categorical_encoders = {}

        # Scalers for numerical features only
        self.numerical_scaler = StandardScaler()

    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Prepare features with automatic categorical/numerical detection.

        Args:
            df: Input dataframe
            fit_scaler: Whether to fit the scaler (True for training data)

        Returns:
            Processed dataframe ready for model input
        """
        # Use parent class method for basic preparation
        df_processed = super().prepare_features(df, fit_scaler=False)

        # Detect features if not already done
        if self.feature_info is None:
            exclude_cols = [self.timestamp_col, self.target_column]
            self.feature_info = self.feature_detector.detect_features(
                df_processed, exclude_cols=exclude_cols
            )

            if self.verbose:
                self._print_feature_summary()

        # Fit numerical scaler if needed
        if fit_scaler and self.feature_info['numerical_features']:
            numerical_data = df_processed[self.feature_info['numerical_features']].values
            self.numerical_scaler.fit(numerical_data)

            if self.verbose:
                print(f"   Fitted numerical scaler for {len(self.feature_info['numerical_features'])} features")

        return df_processed

    def _print_feature_summary(self):
        """Print summary of detected features."""
        summary = self.feature_info['feature_summary']
        print(f"   Feature Detection Summary:")
        print(f"   - Total features: {summary['total_features']}")
        print(f"   - Categorical: {summary['categorical_count']}")
        print(f"   - Numerical: {summary['numerical_count']}")
        print(f"   - Seasonal features: {summary['seasonal_count']}")

        if self.feature_info['seasonal_features']:
            print(f"   - Seasonal features: {self.feature_info['seasonal_features']}")

    def _initialize_model(self, df_processed: pd.DataFrame):
        """Initialize the CSNTransformer model based on detected features."""
        if self.model is not None:
            return

        if self.feature_info is None:
            raise ValueError("Features must be detected before initializing model")

        # Get feature dimensions
        categorical_features = self.feature_info['categorical_features']
        num_numerical_features = len(self.feature_info['numerical_features'])

        if self.verbose:
            print(f"   Initializing CSNTransformer:")
            print(f"   - Categorical features: {len(categorical_features)}")
            print(f"   - Numerical features: {num_numerical_features}")
            print(f"   - Sequence length: {self.sequence_length}")
            print(f"   - Model dimension: {self.d_model}")

        # Create model
        self.model = CSNTransformerPredictor(
            categorical_features=categorical_features,
            num_numerical_features=num_numerical_features,
            sequence_length=self.sequence_length,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout
        ).to(self.device)

        if self.verbose:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"   Model parameters: {total_params:,}")

    def prepare_data(self, df: pd.DataFrame, fit_scaler: bool = True) -> Tuple[Dict, np.ndarray]:
        """
        Prepare data for CSNTransformer training/prediction.

        Args:
            df: Input dataframe
            fit_scaler: Whether to fit scalers

        Returns:
            Tuple of (inputs_dict, targets)
        """
        # Prepare features
        df_processed = self.prepare_features(df, fit_scaler=fit_scaler)

        # Initialize model if needed
        self._initialize_model(df_processed)

        # Prepare categorical inputs
        categorical_inputs = {}
        if self.feature_info['categorical_features']:
            categorical_inputs = self.feature_detector.prepare_categorical_inputs(
                df_processed,
                self.feature_info['categorical_features'],
                self.feature_info['seasonal_features']
            )

        # Prepare numerical inputs (sequences)
        numerical_inputs = None
        if self.feature_info['numerical_features']:
            numerical_data = df_processed[self.feature_info['numerical_features']].values

            # Scale numerical features
            if hasattr(self.numerical_scaler, 'scale_'):
                numerical_data = self.numerical_scaler.transform(numerical_data)

            # Create sequences
            numerical_inputs = self._create_sequences(numerical_data)

        # Prepare targets
        if self.target_column in df_processed.columns:
            targets = df_processed[self.target_column].values[self.sequence_length:]
            # Scale targets
            if fit_scaler:
                targets_reshaped = targets.reshape(-1, 1)
                self.target_scaler.fit(targets_reshaped)
                targets_scaled = self.target_scaler.transform(targets_reshaped).flatten()
            else:
                targets_reshaped = targets.reshape(-1, 1)
                targets_scaled = self.target_scaler.transform(targets_reshaped).flatten()
        else:
            targets_scaled = None

        # Prepare inputs dictionary
        inputs_dict = {
            'categorical': categorical_inputs,
            'numerical': numerical_inputs
        }

        return inputs_dict, targets_scaled

    def _create_sequences(self, numerical_data: np.ndarray) -> np.ndarray:
        """Create sequences from numerical data."""
        sequences = []
        for i in range(len(numerical_data) - self.sequence_length + 1):
            sequences.append(numerical_data[i:i + self.sequence_length])
        return np.array(sequences)

    def _create_data_loader(self, inputs_dict: Dict, targets: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Create DataLoader for CSNTransformer."""
        # Prepare categorical tensors
        categorical_tensors = {}
        if inputs_dict['categorical']:
            for feature_name, data in inputs_dict['categorical'].items():
                # Create sequences for categorical features (repeat for each timestep)
                cat_sequences = np.array([data[i:i + self.sequence_length]
                                        for i in range(len(data) - self.sequence_length + 1)])
                # For categorical features, we typically use the last timestep value
                categorical_tensors[feature_name] = torch.LongTensor(cat_sequences[:, -1])

        # Prepare numerical tensor
        if inputs_dict['numerical'] is not None:
            numerical_tensor = torch.FloatTensor(inputs_dict['numerical'])
        else:
            # Create dummy numerical tensor if no numerical features
            batch_size = len(targets) if targets is not None else 1
            numerical_tensor = torch.zeros(batch_size, self.sequence_length, 1)

        # Prepare target tensor
        target_tensor = torch.FloatTensor(targets) if targets is not None else None

        # Custom dataset for CSN inputs
        class CSNDataset(torch.utils.data.Dataset):
            def __init__(self, categorical_tensors, numerical_tensor, target_tensor):
                self.categorical = categorical_tensors
                self.numerical = numerical_tensor
                self.targets = target_tensor

            def __len__(self):
                return len(self.numerical)

            def __getitem__(self, idx):
                item = {
                    'categorical': {k: v[idx] for k, v in self.categorical.items()},
                    'numerical': self.numerical[idx]
                }
                if self.targets is not None:
                    item['target'] = self.targets[idx]
                return item

        dataset = CSNDataset(categorical_tensors, numerical_tensor, target_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def fit(self, df: pd.DataFrame, val_df: pd.DataFrame = None,
            epochs: int = 100, batch_size: int = 32, learning_rate: float = 1e-3,
            patience: int = 10, verbose: bool = True):
        """Train the CSNTransformer model."""
        self.verbose = verbose

        if verbose:
            print("Training CSNTransformer for time series prediction")

        # Prepare training data
        train_inputs, train_targets = self.prepare_data(df, fit_scaler=True)
        train_loader = self._create_data_loader(train_inputs, train_targets, batch_size, shuffle=True)

        # Prepare validation data
        val_loader = None
        if val_df is not None:
            val_inputs, val_targets = self.prepare_data(val_df, fit_scaler=False)
            val_loader = self._create_data_loader(val_inputs, val_targets, batch_size, shuffle=False)

        # Setup training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        if verbose:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model parameters: {total_params:,}")
            print(f"Training samples: {len(train_targets)} (batch_size: {batch_size})")
            if val_loader:
                print(f"Validation samples: {len(val_targets)}")

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()

                # Forward pass
                categorical_inputs = batch['categorical'] if batch['categorical'] else None
                numerical_inputs = batch['numerical']
                targets = batch['target']

                # Move to device
                if categorical_inputs:
                    categorical_inputs = {k: v.to(self.device) for k, v in categorical_inputs.items()}
                numerical_inputs = numerical_inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(categorical_inputs, numerical_inputs)
                loss = criterion(outputs.squeeze(), targets)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            val_loss = 0.0
            val_mae = 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        categorical_inputs = batch['categorical'] if batch['categorical'] else None
                        numerical_inputs = batch['numerical']
                        targets = batch['target']

                        # Move to device
                        if categorical_inputs:
                            categorical_inputs = {k: v.to(self.device) for k, v in categorical_inputs.items()}
                        numerical_inputs = numerical_inputs.to(self.device)
                        targets = targets.to(self.device)

                        outputs = self.model(categorical_inputs, numerical_inputs)
                        loss = criterion(outputs.squeeze(), targets)

                        val_loss += loss.item()

                        # Calculate MAE in original scale
                        outputs_original = self.target_scaler.inverse_transform(outputs.cpu().numpy().reshape(-1, 1)).flatten()
                        targets_original = self.target_scaler.inverse_transform(targets.cpu().numpy().reshape(-1, 1)).flatten()
                        val_mae += np.mean(np.abs(outputs_original - targets_original))

                val_loss /= len(val_loader)
                val_mae /= len(val_loader)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose:
                    print(f"Epoch {epoch + 1:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, Val MAE = ${val_mae:.2f}")

                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch + 1:3d}: Train Loss = {train_loss:.6f}")

        # Store training history
        self.history = {
            'train_loss': [train_loss],  # Simplified for now
            'val_loss': [val_loss] if val_loader else []
        }

        if verbose:
            print("Training completed!")

    def predict_from_features(self, df_processed: pd.DataFrame) -> np.ndarray:
        """Make predictions from preprocessed features."""
        self.model.eval()

        # Prepare inputs
        inputs_dict, _ = self.prepare_data(df_processed, fit_scaler=False)

        # Create data loader
        dummy_targets = np.zeros(len(inputs_dict['numerical']) if inputs_dict['numerical'] is not None else 1)
        data_loader = self._create_data_loader(inputs_dict, dummy_targets, batch_size=32, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                categorical_inputs = batch['categorical'] if batch['categorical'] else None
                numerical_inputs = batch['numerical']

                # Move to device
                if categorical_inputs:
                    categorical_inputs = {k: v.to(self.device) for k, v in categorical_inputs.items()}
                numerical_inputs = numerical_inputs.to(self.device)

                outputs = self.model(categorical_inputs, numerical_inputs)
                predictions.extend(outputs.cpu().numpy().flatten())

        # Convert back to original scale
        predictions = np.array(predictions)
        predictions_reshaped = predictions.reshape(-1, 1)
        predictions_original = self.target_scaler.inverse_transform(predictions_reshaped).flatten()

        return predictions_original