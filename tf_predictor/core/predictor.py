"""
Generic Time Series Predictor using FT-Transformer.

A base class for time series prediction that can be extended for different domains.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod

from .model import FTTransformerPredictor, SequenceFTTransformerPredictor


class TimeSeriesPredictor(ABC):
    """Generic FT-Transformer wrapper for time series prediction."""
    
    def __init__(
        self,
        target_column: str,
        sequence_length: int = 5,
        **ft_kwargs
    ):
        """
        Args:
            target_column: Name of the target column to predict
            sequence_length: Number of historical time steps to use for prediction
            **ft_kwargs: FT-Transformer hyperparameters
        """
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.ft_kwargs = ft_kwargs
        
        # Will be set during training
        self.model = None
        self.scaler = StandardScaler()  # For features
        self.target_scaler = StandardScaler()  # For target
        self.feature_columns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': []}
        self.verbose = False  # Will be set during training
    
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
    
    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Prepare features by calling domain-specific feature creation and handling scaling.
        
        Args:
            df: DataFrame with raw data
            fit_scaler: Whether to fit the scaler (True for training data)
            
        Returns:
            processed_df: DataFrame with scaled features
        """
        # Create domain-specific features
        df_processed = self.create_features(df, fit_scaler)
        
        # Get all feature columns (excluding target and non-numeric columns)
        feature_cols = []
        for col in df_processed.columns:
            if col != self.target_column:
                # Only include numeric columns for scaling
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    feature_cols.append(col)
                elif self.verbose:
                    print(f"   Excluding non-numeric column from scaling: {col} (dtype: {df_processed[col].dtype})")
        
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
        
        # Scale only the numeric features
        if len(self.feature_columns) > 0:
            if fit_scaler:
                df_processed[self.feature_columns] = self.scaler.fit_transform(df_processed[self.feature_columns])
            else:
                df_processed[self.feature_columns] = self.scaler.transform(df_processed[self.feature_columns])
        
        return df_processed
    
    def prepare_data(self, df: pd.DataFrame, fit_scaler: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare sequential data for model training/inference.
        
        Returns:
            X: Sequences tensor of shape (n_samples, sequence_length, n_features)
            y: Target tensor of shape (n_samples,) (None for inference)
        """
        # First prepare features
        df_processed = self.prepare_features(df, fit_scaler)
        
        # Check if we have enough data for sequences
        if len(df_processed) <= self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length + 1} samples for sequence_length={self.sequence_length}, got {len(df_processed)}")
        
        # Validate target column exists after feature engineering
        if self.target_column not in df_processed.columns:
            available_cols = list(df_processed.columns)
            raise ValueError(f"Target column '{self.target_column}' not found after feature engineering.\n"
                           f"Available columns: {available_cols[:10]}{'...' if len(available_cols) > 10 else ''}")
        
        # Create sequences - this will reduce our sample count
        if self.target_column in df_processed.columns:
            from ..preprocessing.time_features import create_sequences
            
            # For training: create sequences with targets
            sequences, targets = create_sequences(df_processed, self.sequence_length, self.target_column)
            
            # Scale target
            targets = targets.reshape(-1, 1)
            if fit_scaler:
                targets_scaled = self.target_scaler.fit_transform(targets)
            else:
                targets_scaled = self.target_scaler.transform(targets)
            
            # Convert to tensors
            X = torch.tensor(sequences, dtype=torch.float32)  # (n_samples, seq_len, n_features)
            y = torch.tensor(targets_scaled.flatten(), dtype=torch.float32)
            
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
                **model_kwargs
            ).to(self.device)
        
        # Training setup
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2, verbose=verbose
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.verbose = verbose
        if verbose:
            print(f"Training FT-Transformer for {self.target_column} prediction")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Training samples: {len(X_train)}")
            if X_val is not None:
                print(f"Validation samples: {len(X_val)}")
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Handle both sequence and non-sequence models
                if len(batch_x.shape) == 3:  # Sequence data
                    outputs = self.model(batch_x).squeeze()
                else:  # Non-sequence data  
                    outputs = self.model(batch_x, None).squeeze()  # No categorical features
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(dataloader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation
            val_loss = None
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    # Handle both sequence and non-sequence models
                    if len(X_val.shape) == 3:  # Sequence data
                        val_outputs = self.model(X_val.to(self.device)).squeeze()
                    else:  # Non-sequence data
                        val_outputs = self.model(X_val.to(self.device), None).squeeze()
                    
                    val_loss = criterion(val_outputs, y_val.to(self.device)).item()
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
            
            # Calculate MAPE and MAE for each epoch if verbose
            if verbose:
                # Get predictions for validation set to calculate MAPE/MAE
                if X_val is not None and y_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        if len(X_val.shape) == 3:  # Sequence data
                            val_pred_scaled = self.model(X_val.to(self.device)).squeeze()
                        else:  # Non-sequence data
                            val_pred_scaled = self.model(X_val.to(self.device), None).squeeze()
                        
                        # Transform back to original scale for MAPE/MAE calculation
                        val_pred_scaled_np = val_pred_scaled.cpu().numpy().reshape(-1, 1)
                        val_pred = self.target_scaler.inverse_transform(val_pred_scaled_np).flatten()
                        
                        val_actual_scaled_np = y_val.cpu().numpy().reshape(-1, 1)
                        val_actual = self.target_scaler.inverse_transform(val_actual_scaled_np).flatten()
                        
                        # Calculate MAPE and MAE
                        val_mae = np.mean(np.abs(val_actual - val_pred))
                        val_mape = np.mean(np.abs((val_actual - val_pred) / val_actual)) * 100
                        
                        print(f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}, Val MAE = ${val_mae:.2f}, Val MAPE = {val_mape:.2f}%")
                else:
                    print(f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.6f}")
        
        if verbose:
            print("Training completed!")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            df: DataFrame with same structure as training data
            
        Returns:
            predictions: Numpy array of predictions (in original scale)
        """
        if self.model is None:
            raise RuntimeError("Model must be trained first. Call fit().")
        
        X, _ = self.prepare_data(df, fit_scaler=False)
        
        self.model.eval()
        with torch.no_grad():
            # Handle both sequence and non-sequence models
            if len(X.shape) == 3:  # Sequence data
                predictions_scaled = self.model(X.to(self.device)).squeeze()
            else:  # Non-sequence data
                predictions_scaled = self.model(X.to(self.device), None).squeeze()
            
            # Transform back to original scale
            predictions_scaled = predictions_scaled.cpu().numpy().reshape(-1, 1)
            predictions = self.target_scaler.inverse_transform(predictions_scaled)
            return predictions.flatten()
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            df: DataFrame with raw data (will be processed to extract target)
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Process features first to get the target column
        df_processed = self.prepare_features(df, fit_scaler=False)
        
        # Now check if target column exists after processing
        if self.target_column not in df_processed.columns:
            raise ValueError(f"Target column '{self.target_column}' not found after feature engineering")
        
        predictions = self.predict(df)
        
        # For sequences, we need to align the actual values with predictions
        # Predictions correspond to targets starting from index sequence_length
        if self.sequence_length > 1:
            actual = df_processed[self.target_column].values[self.sequence_length:]
        else:
            actual = df_processed[self.target_column].values
        
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
            'history': self.history
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
            **state['ft_kwargs']
        )
        
        # Restore state
        predictor.scaler = state['scaler']
        predictor.target_scaler = state['target_scaler']
        predictor.feature_columns = state['feature_columns']
        predictor.history = state.get('history', {'train_loss': [], 'val_loss': []})
        
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