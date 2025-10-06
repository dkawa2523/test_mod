"""
Linear regression model
----------------------

This module implements a simple linear regression wrapper that supports
multi‑output regression problems. It leverages scikit‑learn's
`LinearRegression` model and optionally wraps it in a `MultiOutputRegressor`
when the target is multi‑dimensional. The class includes methods to fit
the model, make predictions and compute R² and RMSE metrics.
"""

from typing import Optional, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from ..evaluation import compute_r2, compute_rmse, compute_mae


class LinearRegressionModel:
    """Multi‑output linear regression wrapper."""

    def __init__(self, fit_intercept: bool = True, n_jobs: Optional[int] = None):
        self.fit_intercept = fit_intercept
        self.n_jobs = n_jobs
        self.model: Optional[MultiOutputRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the linear regression model.

        Parameters
        ----------
        X : numpy.ndarray
            Input features of shape (n_samples, n_features).
        y : numpy.ndarray
            Target array. For multi‑output regression this should have shape
            (n_samples, n_outputs).
        """
        base_model = LinearRegression(fit_intercept=self.fit_intercept, n_jobs=self.n_jobs)
        if y.ndim == 1:
            # Single output
            self.model = base_model
            self.model.fit(X, y)
        else:
            # Multi‑output
            self.model = MultiOutputRegressor(base_model, n_jobs=self.n_jobs)
            self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for new inputs.

        Parameters
        ----------
        X : numpy.ndarray
            Input features of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Predicted targets of shape (n_samples, n_outputs).
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet")
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute evaluation metrics on a dataset.

        Returns
        -------
        r2 : float
            Coefficient of determination.
        rmse : float
            Root mean squared error.
        mae : float
            Mean absolute error.
        """
        y_pred = self.predict(X)
        # Flatten multi‑output for metrics that expect 1D arrays
        return (
            compute_r2(y_true, y_pred),
            compute_rmse(y_true, y_pred),
            compute_mae(y_true, y_pred),
        )