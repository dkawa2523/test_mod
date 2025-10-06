"""
Zernike + Gaussian process regression model
-----------------------------------------

This module implements a hybrid approach that first decomposes spatial
distributions into Zernike polynomial coefficients and then models the
relationship between process conditions and those coefficients using
Gaussian process regression. At inference time the predicted coefficients
are used to reconstruct the distribution and, thanks to GPR, confidence
intervals can be obtained for each coefficient and propagated back to
spatial coordinates.

The method leverages the orthogonality and interpretability of Zernike
polynomials【990290441617999†L100-L115】 together with the probabilistic
predictions of GPR【601824961475775†L724-L755】.
"""

from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np

from ..features.zernike import fit_zernike, reconstruct_zernike, generate_nm_pairs, zernike_design_matrix
from ..models.gpr import GPRModel


class ZernikeGPRModel:
    """
    Hybrid model combining Zernike decomposition and GPR.
    """

    def __init__(
        self,
        max_order: int = 4,
        gpr_kernel_name: str = 'RBF',
        gpr_length_scale: float = 1.0,
        gpr_nu: float = 1.5,
        gpr_alpha: float = 1e-10,
        gpr_scale_y: bool = True,
        gpr_normalize_y: bool = True,
        gpr_optimize_hyperparams: bool = False,
        zernike_stabilize: bool = True,
        zernike_stabilize_center: str = 'mean',
        zernike_stabilize_scale: str = 'max_abs',
        zernike_stabilize_min_scale: float = 1e-8,
    ):
        self.max_order = max_order
        self.nm_pairs: Optional[List[Tuple[int, int]]] = None
        self.reference_center: Optional[Tuple[float, float]] = None
        self.reference_radius: Optional[float] = None
        self.gpr = GPRModel(
            kernel_name=gpr_kernel_name,
            length_scale=gpr_length_scale,
            nu=gpr_nu,
            alpha=gpr_alpha,
            scale_y=gpr_scale_y,
            normalize_y=gpr_normalize_y,
            optimize_hyperparams=gpr_optimize_hyperparams,
        )
        self.zernike_fit_kwargs = {
            "stabilize": zernike_stabilize,
            "stabilize_center": zernike_stabilize_center,
            "stabilize_scale": zernike_stabilize_scale,
            "stabilize_min_scale": zernike_stabilize_min_scale,
        }

    def _compute_coeff_matrix(
        self, distributions: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Compute Zernike coefficient matrix from a dictionary of distributions.
        Each value in the dict is a DataFrame or array with columns x, y, f.
        """
        coeff_list = []
        nm_pairs = None
        for dist in distributions.values():
            x = dist["x"].values
            y = dist["y"].values
            f = dist["f"].values
            polar_kwargs = _extract_polar_kwargs(dist)
            if self.reference_center is None and "center" in polar_kwargs:
                self.reference_center = polar_kwargs["center"]
            if self.reference_radius is None and "radius_scale" in polar_kwargs:
                self.reference_radius = polar_kwargs["radius_scale"]
            coeffs, nm_pairs_curr = fit_zernike(
                x,
                y,
                f,
                self.max_order,
                **polar_kwargs,
                **self.zernike_fit_kwargs,
            )
            if nm_pairs is None:
                nm_pairs = nm_pairs_curr
            coeff_list.append(coeffs)
        if nm_pairs is None:
            raise ValueError("No distributions provided")
        self.nm_pairs = nm_pairs
        return np.vstack(coeff_list), nm_pairs

    def fit(
        self,
        X_conditions: np.ndarray,
        distributions: Dict[str, np.ndarray],
    ) -> None:
        """
        Fit the Zernike+GPR model.

        Parameters
        ----------
        X_conditions : numpy.ndarray
            Condition features for each sample of shape (n_samples, n_features).
        distributions : dict
            Mapping from identifier to a DataFrame with columns x, y, f.
        """
        coeff_matrix, nm_pairs = self._compute_coeff_matrix(distributions)
        # Fit GPR models to each coefficient
        self.gpr.fit(X_conditions, coeff_matrix)

    def predict_coefficients(self, X_conditions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict Zernike coefficients and standard deviations using GPR.

        Returns
        -------
        mean_coeffs : numpy.ndarray
            Predicted mean coefficients of shape (n_samples, n_coeffs).
        std_coeffs : numpy.ndarray
            Predicted standard deviation of coefficients of shape (n_samples, n_coeffs).
        """
        mean_coeffs, std_coeffs = self.gpr.predict_with_std(X_conditions)
        return mean_coeffs, std_coeffs

    def reconstruct_distribution(
        self,
        x: np.ndarray,
        y: np.ndarray,
        coeffs: np.ndarray,
        **polar_kwargs,
    ) -> np.ndarray:
        """
        Reconstruct distribution values at coordinates using Zernike basis and given coefficients.
        """
        if self.nm_pairs is None:
            raise RuntimeError("Model has not been fitted yet; Zernike basis unknown")
        return reconstruct_zernike(x, y, coeffs, self.nm_pairs, **polar_kwargs)

    def predict_distribution(
        self,
        X_conditions: np.ndarray,
        template_distribution: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Predict full distributions for new conditions.

        Parameters
        ----------
        X_conditions : numpy.ndarray
            Condition features for the samples.
        template_distribution : dict
            A dictionary containing at least one distribution DataFrame to
            provide x and y coordinates for reconstruction. Only the x and y
            coordinates of the first element are used.

        Returns
        -------
        dict
            Mapping from sample index (as string) to predicted distribution
            DataFrame with columns x, y, f_pred, and optionally lower and
            upper confidence bounds.
        """
        mean_coeffs, std_coeffs = self.predict_coefficients(X_conditions)
        # Use coordinates from first template distribution
        example = next(iter(template_distribution.values()))
        polar_kwargs = _extract_polar_kwargs(example)
        if "center" not in polar_kwargs and self.reference_center is not None:
            polar_kwargs["center"] = self.reference_center
        if "radius_scale" not in polar_kwargs and self.reference_radius is not None:
            polar_kwargs["radius_scale"] = self.reference_radius
        x_coords = example["x"].values
        y_coords = example["y"].values
        preds = {}
        for i in range(mean_coeffs.shape[0]):
            coeffs = mean_coeffs[i]
            f_pred = self.reconstruct_distribution(
                x_coords,
                y_coords,
                coeffs,
                **polar_kwargs,
            )
            # Compute confidence interval per point: variance = sum sigma_j^2 * basis_j(x)^2
            if std_coeffs is not None and self.nm_pairs is not None:
                # Build design matrix once per call
                design = zernike_design_matrix(
                    x_coords,
                    y_coords,
                    self.nm_pairs,
                    **polar_kwargs,
                )
                var = np.sum((std_coeffs[i] ** 2) * (design ** 2), axis=1)
                std_f = np.sqrt(var)
                lower = f_pred - 1.96 * std_f
                upper = f_pred + 1.96 * std_f
                preds[str(i)] = {
                    "x": x_coords,
                    "y": y_coords,
                    "f_pred": f_pred,
                    "lower": lower,
                    "upper": upper,
                }
            else:
                preds[str(i)] = {
                    "x": x_coords,
                    "y": y_coords,
                    "f_pred": f_pred,
                }
        return preds
def _extract_polar_kwargs(df) -> Dict[str, np.ndarray]:
    """Collect polar metadata columns if available."""

    kwargs: Dict[str, np.ndarray] = {}
    if "r_norm" in df.columns:
        kwargs["r_norm"] = df["r_norm"].values
    if "theta" in df.columns:
        kwargs["theta"] = df["theta"].values
    if "cx" in df.columns and "cy" in df.columns:
        kwargs["center"] = (
            float(df["cx"].iloc[0]),
            float(df["cy"].iloc[0]),
        )
    if "radius_max" in df.columns:
        kwargs["radius_scale"] = float(df["radius_max"].iloc[0])
    return kwargs
