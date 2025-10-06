"""
Evaluation metrics
------------------

This module provides functions to compute common regression metrics such as
coefficient of determination (R²), root mean squared error (RMSE) and mean
absolute error (MAE). These metrics are used both for evaluating the quality
of the decomposition (reconstruction error) and the predictive performance of
machine learning models.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the coefficient of determination R² between true and predicted values."""
    return r2_score(y_true, y_pred)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))