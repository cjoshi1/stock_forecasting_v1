"""
Multi-horizon prediction heads shared by both FT and CSN models.

This module provides unified prediction heads that work with both transformer architectures.
The same prediction logic is used regardless of the backbone model.
"""

import torch
import torch.nn as nn
from typing import Optional


class MultiHorizonHead(nn.Module):
    """
    Unified prediction head for single and multi-horizon forecasting.

    This head takes the CLS token output from the transformer backbone and
    produces predictions for one or more future time steps.

    For single-horizon (h=1): Predicts one value
    For multi-horizon (h>1): Predicts multiple future values simultaneously

    Args:
        d_input: Input dimension (typically d_token or d_model from backbone)
        prediction_horizons: Number of steps ahead to predict (1 = single, >1 = multi)
        hidden_dim: Optional hidden layer dimension for deeper prediction head
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        d_input: int,
        prediction_horizons: int = 1,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        self.prediction_horizons = prediction_horizons
        self.d_input = d_input

        if hidden_dim is not None:
            # Two-layer prediction head with hidden layer
            self.head = nn.Sequential(
                nn.Linear(d_input, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, prediction_horizons)
            )
        else:
            # Simple linear prediction head
            self.head = nn.Linear(d_input, prediction_horizons)

    def forward(self, cls_output: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions from CLS token output.

        Args:
            cls_output: CLS token representation [batch_size, d_input]

        Returns:
            Predictions tensor:
            - Single-horizon (h=1): [batch_size, 1] or [batch_size] after squeeze
            - Multi-horizon (h>1): [batch_size, prediction_horizons]
        """
        predictions = self.head(cls_output)  # [batch_size, prediction_horizons]

        # For single horizon, squeeze to [batch_size]
        if self.prediction_horizons == 1:
            predictions = predictions.squeeze(-1)  # [batch_size]

        return predictions


class RegressionHead(nn.Module):
    """
    Simple regression head for single-horizon prediction.

    A straightforward linear layer for regression tasks.
    This is equivalent to MultiHorizonHead with prediction_horizons=1
    but more explicit for single-output regression.

    Args:
        d_input: Input dimension
        dropout: Dropout rate
    """

    def __init__(self, d_input: int, dropout: float = 0.0):
        super().__init__()

        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_input, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, d_input]

        Returns:
            Predictions [batch_size, 1] or [batch_size] after squeeze
        """
        return self.head(x).squeeze(-1)


class ClassificationHead(nn.Module):
    """
    Classification head for categorical predictions.

    Provides a simple linear classifier with optional hidden layers.
    Can be used for future extensions to classification tasks.

    Args:
        d_input: Input dimension
        n_classes: Number of output classes
        hidden_dim: Optional hidden layer dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_input: int,
        n_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_classes = n_classes

        if hidden_dim is not None:
            # Two-layer classification head
            self.head = nn.Sequential(
                nn.Linear(d_input, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes)
            )
        else:
            # Simple linear classifier
            self.head = nn.Linear(d_input, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, d_input]

        Returns:
            Class logits [batch_size, n_classes]
        """
        return self.head(x)


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head for simultaneous regression and classification.

    Useful for models that need to predict multiple outputs simultaneously,
    such as predicting both price (regression) and trend direction (classification).

    Args:
        d_input: Input dimension
        regression_outputs: Number of regression outputs
        classification_outputs: Number of classification classes (0 = no classification)
        hidden_dim: Optional shared hidden layer dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_input: int,
        regression_outputs: int = 1,
        classification_outputs: int = 0,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        self.has_regression = regression_outputs > 0
        self.has_classification = classification_outputs > 0

        # Shared feature extractor (optional)
        if hidden_dim is not None:
            self.shared = nn.Sequential(
                nn.Linear(d_input, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            feature_dim = hidden_dim
        else:
            self.shared = nn.Identity()
            feature_dim = d_input

        # Task-specific heads
        if self.has_regression:
            self.regression_head = nn.Linear(feature_dim, regression_outputs)

        if self.has_classification:
            self.classification_head = nn.Linear(feature_dim, classification_outputs)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, d_input]

        Returns:
            Dictionary with keys:
            - 'regression': [batch_size, regression_outputs] (if enabled)
            - 'classification': [batch_size, classification_outputs] (if enabled)
        """
        # Shared feature extraction
        features = self.shared(x)

        outputs = {}

        if self.has_regression:
            outputs['regression'] = self.regression_head(features)

        if self.has_classification:
            outputs['classification'] = self.classification_head(features)

        return outputs


class DistributionHead(nn.Module):
    """
    Probabilistic prediction head for uncertainty estimation.

    Predicts both mean and variance for probabilistic forecasting.
    Useful for understanding prediction uncertainty.

    Args:
        d_input: Input dimension
        prediction_horizons: Number of horizons to predict
        hidden_dim: Optional hidden layer dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_input: int,
        prediction_horizons: int = 1,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        self.prediction_horizons = prediction_horizons

        # Shared feature extractor
        if hidden_dim is not None:
            self.shared = nn.Sequential(
                nn.Linear(d_input, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            feature_dim = hidden_dim
        else:
            self.shared = nn.Identity()
            feature_dim = d_input

        # Separate heads for mean and log-variance
        self.mean_head = nn.Linear(feature_dim, prediction_horizons)
        self.logvar_head = nn.Linear(feature_dim, prediction_horizons)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, d_input]

        Returns:
            Tuple of (mean, variance):
            - mean: [batch_size, prediction_horizons]
            - variance: [batch_size, prediction_horizons]
        """
        features = self.shared(x)

        mean = self.mean_head(features)
        # Predict log-variance for numerical stability, then exp to get variance
        log_var = self.logvar_head(features)
        variance = torch.exp(log_var)

        return mean, variance
