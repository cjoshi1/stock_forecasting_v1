"""
Custom metrics for Rossmann Store Sales forecasting.
"""
import numpy as np
from typing import Optional


def rmspe(y_true: np.ndarray, y_pred: np.ndarray, exclude_zeros: bool = True) -> float:
    """
    Calculate Root Mean Square Percentage Error (RMSPE).

    This is the official Kaggle evaluation metric for Rossmann Store Sales.

    Formula: RMSPE = sqrt(mean(((y_true - y_pred) / y_true)^2))

    Args:
        y_true: Actual values
        y_pred: Predicted values
        exclude_zeros: If True, exclude samples where y_true == 0 (Kaggle requirement)

    Returns:
        RMSPE value as a float

    Examples:
        >>> y_true = np.array([100, 200, 300, 0, 500])
        >>> y_pred = np.array([110, 190, 310, 50, 480])
        >>> rmspe(y_true, y_pred)  # Excludes the zero
        0.0565...
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    if exclude_zeros:
        # Kaggle requirement: exclude days where sales = 0
        mask = y_true != 0
        if mask.sum() == 0:
            return 0.0  # All zeros
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    # Calculate percentage errors
    percentage_errors = (y_true - y_pred) / y_true

    # Calculate RMSPE
    rmspe_value = np.sqrt(np.mean(percentage_errors ** 2))

    return rmspe_value


def mape(y_true: np.ndarray, y_pred: np.ndarray, exclude_zeros: bool = True) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        y_true: Actual values
        y_pred: Predicted values
        exclude_zeros: If True, exclude samples where y_true == 0

    Returns:
        MAPE value as a percentage
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if exclude_zeros:
        mask = y_true != 0
        if mask.sum() == 0:
            return 0.0
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error (RMSE).

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    return np.mean(np.abs(y_true - y_pred))


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate all relevant metrics for Rossmann forecasting.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary with all metrics
    """
    return {
        'RMSPE': rmspe(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
    }


if __name__ == '__main__':
    # Test the metrics
    y_true = np.array([100, 200, 300, 0, 500])
    y_pred = np.array([110, 190, 310, 50, 480])

    print("Test Metrics:")
    print("-" * 40)
    metrics = calculate_all_metrics(y_true, y_pred)
    for metric_name, value in metrics.items():
        print(f"{metric_name:<10} {value:>10.4f}")
