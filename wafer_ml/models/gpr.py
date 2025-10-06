"""
Gaussian process regression (GPR) model
--------------------------------------

This module implements a multi‑output Gaussian process regression model by
training independent GPR models for each output dimension. It uses
scikit‑learn's `GaussianProcessRegressor` and allows customisation of the
kernel via the configuration. GPR provides not only point predictions but
also predictive uncertainties (standard deviations), which is valuable for
assessing confidence intervals【601824961475775†L724-L755】.

GPR can be computationally expensive with large datasets because training
requires inversion of an N×N covariance matrix, giving O(N³) complexity【455717029507448†L125-L147】.
Therefore, it is best suited for moderate‑sized training sets or after
dimensionality reduction.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, RationalQuadratic, WhiteKernel

from ..evaluation import compute_r2, compute_rmse, compute_mae


def build_kernel(kernel_name: str, length_scale: float = 1.0, nu: float = 1.5) -> Any:
    """
    Build a kernel object from a name and parameters.

    Supported kernel names: 'RBF', 'Matern', 'RQ' (RationalQuadratic).
    """
    noise = WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-7, 1.0))

    if kernel_name.upper() == 'RBF':
        base = ConstantKernel(1.0, (1e-9, 1e9)) * RBF(
            length_scale=length_scale,
            length_scale_bounds=(1e-12, 1e5),
        )
    elif kernel_name.upper() == 'MATERN':
        base = ConstantKernel(1.0, (1e-9, 1e9)) * Matern(
            length_scale=length_scale,
            length_scale_bounds=(1e-12, 1e5),
            nu=nu,
        )
    elif kernel_name.upper() in ('RQ', 'RATIONALQUADRATIC'):
        base = ConstantKernel(1.0, (1e-9, 1e9)) * RationalQuadratic(
            length_scale=length_scale,
            alpha=1.0,
        )
    else:
        raise ValueError(f"Unsupported kernel name '{kernel_name}'. Choose from 'RBF', 'Matern', 'RQ'.")

    return base + noise


class GPRModel:
    """Multi‑output Gaussian process regression wrapper."""

    def __init__(
        self,
        kernel_name: str = 'RBF',
        length_scale: float = 1.0,
        nu: float = 1.5,
        alpha: float = 1e-10,
        scale_y: bool = True,
        normalize_y: bool = True,
        optimize_hyperparams: bool = False,
    ):
        self.kernel_name = kernel_name
        self.length_scale = length_scale
        self.nu = nu
        alpha_value = float(alpha) if isinstance(alpha, str) else alpha
        if isinstance(alpha_value, float) and alpha_value < 1e-8:
            alpha_value = 1e-8
        self.alpha = alpha_value
        self.scale_y = scale_y
        self.normalize_y = normalize_y
        self.optimize_hyperparams = optimize_hyperparams
        self.models: List[GaussianProcessRegressor] = []
        self.n_outputs: Optional[int] = None
        self._y_mean: Optional[np.ndarray] = None
        self._y_std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit independent GPR models for each output dimension.
        """
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.n_outputs = y.shape[1]

        if self.scale_y:
            mean = y.mean(axis=0)
            std = y.std(axis=0)
            std[std < 1e-12] = 1.0
            y = (y - mean) / std
            self._y_mean = mean
            self._y_std = std
        else:
            self._y_mean = None
            self._y_std = None
        kernel = build_kernel(self.kernel_name, self.length_scale, self.nu)
        self.models = []
        optimizer = 'fmin_l_bfgs_b' if self.optimize_hyperparams else None
        n_restarts = 2 if self.optimize_hyperparams else 0

        for j in range(self.n_outputs):
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.alpha,
                normalize_y=self.normalize_y,
                optimizer=optimizer,
                n_restarts_optimizer=n_restarts,
            )
            gp.fit(X, y[:, j])
            self.models.append(gp)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict mean outputs for new inputs.
        """
        if not self.models:
            raise RuntimeError("Model has not been fitted yet")
        preds = [gp.predict(X) for gp in self.models]
        preds = np.column_stack(preds)
        if self.scale_y and self._y_std is not None and self._y_mean is not None:
            preds = preds * self._y_std + self._y_mean
        return preds

    def predict_with_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation for each output.
        """
        if not self.models:
            raise RuntimeError("Model has not been fitted yet")
        means = []
        stds = []
        for gp in self.models:
            m, s = gp.predict(X, return_std=True)
            means.append(m)
            stds.append(s)
        means = np.column_stack(means)
        stds = np.column_stack(stds)
        if self.scale_y and self._y_std is not None and self._y_mean is not None:
            means = means * self._y_std + self._y_mean
            stds = stds * self._y_std
        return means, stds

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute R², RMSE and MAE metrics across outputs.
        """
        y_pred = self.predict(X)
        return (
            compute_r2(y_true, y_pred),
            compute_rmse(y_true, y_pred),
            compute_mae(y_true, y_pred),
        )
