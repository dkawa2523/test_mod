"""
Pre‑processing utilities
-----------------------

This module defines a `Preprocessor` class that can apply common data
transformations such as missing value imputation, standardisation and
normalisation. The transformations are implemented using scikit‑learn
preprocessors and can be configured via dictionaries passed at initialisation.

The class is designed to be fitted on a training dataset and applied
consistently to validation/test or inference datasets.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

from scipy.interpolate import SmoothBivariateSpline


class Preprocessor:
    """Flexible pre‑processor for tabular data."""

    def __init__(
        self,
        standardize: bool = False,
        normalize: bool = False,
        impute: bool = True,
        impute_strategy: str = "mean",
    ):
        """
        Parameters
        ----------
        standardize : bool
            Whether to apply standard scaling (zero mean, unit variance).
        normalize : bool
            Whether to apply min‑max scaling to the range [0, 1]. If both
            standardize and normalize are True, standardisation is applied
            followed by normalisation.
        impute : bool
            Whether to impute missing values using `impute_strategy`.
        impute_strategy : str
            Strategy used by `SimpleImputer` to fill missing values (e.g.,
            'mean', 'median', 'most_frequent', 'constant').
        """
        self.standardize = standardize
        self.normalize = normalize
        self.impute = impute
        self.impute_strategy = impute_strategy
        self._imputer: Optional[SimpleImputer] = None
        self._scaler_std: Optional[StandardScaler] = None
        self._scaler_norm: Optional[MinMaxScaler] = None

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        """
        Fit the pre‑processing steps on the provided DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            Training data.

        Returns
        -------
        Preprocessor
            Returns self to allow chaining.
        """
        values = X.values.astype(float)
        # Impute missing values
        if self.impute:
            self._imputer = SimpleImputer(strategy=self.impute_strategy)
            values = self._imputer.fit_transform(values)
        # Standardisation
        if self.standardize:
            self._scaler_std = StandardScaler()
            values = self._scaler_std.fit_transform(values)
        # Normalisation
        if self.normalize:
            self._scaler_norm = MinMaxScaler()
            values = self._scaler_norm.fit_transform(values)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply the fitted transformations to new data.

        Parameters
        ----------
        X : pandas.DataFrame
            Data to transform.

        Returns
        -------
        numpy.ndarray
            Transformed array of shape (n_samples, n_features).
        """
        values = X.values.astype(float)
        if self.impute and self._imputer is not None:
            values = self._imputer.transform(values)
        if self.standardize and self._scaler_std is not None:
            values = self._scaler_std.transform(values)
        if self.normalize and self._scaler_norm is not None:
            values = self._scaler_norm.transform(values)
        return values

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit the pre‑processor on the data and return the transformed array.

        Parameters
        ----------
        X : pandas.DataFrame
            Training data.

        Returns
        -------
        numpy.ndarray
            Transformed array.
        """
        self.fit(X)
        return self.transform(X)

    def get_params(self) -> Dict[str, Any]:
        """Return the configuration of this preprocessor."""
        return {
            "standardize": self.standardize,
            "normalize": self.normalize,
            "impute": self.impute,
            "impute_strategy": self.impute_strategy,
        }


def _spline_resample_cartesian(
    df: pd.DataFrame,
    grid_resolution: int,
    smoothing: Optional[float],
    kx: int,
    ky: int,
    max_points: Optional[int],
) -> pd.DataFrame:
    """Resample a scattered distribution onto a regular Cartesian grid."""

    if grid_resolution < max(kx + 1, ky + 1):
        raise ValueError(
            "grid_resolution must be at least max(kx+1, ky+1) to construct a spline grid"
        )

    x = df["x"].values
    y = df["y"].values
    f = df["f"].values

    spline_kwargs = {"kx": kx, "ky": ky}
    if smoothing is not None:
        spline_kwargs["s"] = smoothing

    # Fit spline on scattered data
    spline = SmoothBivariateSpline(x, y, f, **spline_kwargs)

    # Build regular evaluation grid
    x_lin = np.linspace(np.min(x), np.max(x), grid_resolution)
    y_lin = np.linspace(np.min(y), np.max(y), grid_resolution)
    Xg, Yg = np.meshgrid(x_lin, y_lin)
    Z = spline.ev(Xg.ravel(), Yg.ravel()).reshape(Xg.shape)

    if max_points is not None and max_points > 0:
        total = Xg.size
        if total > max_points:
            stride = int(np.ceil(np.sqrt(total / max_points)))
            stride = max(stride, 1)
            Xg = Xg[::stride, ::stride]
            Yg = Yg[::stride, ::stride]
            Z = Z[::stride, ::stride]

    return pd.DataFrame({
        "x": Xg.ravel(),
        "y": Yg.ravel(),
        "f": Z.ravel(),
    })


def _compute_center_radius(
    x: np.ndarray,
    y: np.ndarray,
    center_override: Optional[Any],
) -> tuple[float, float, float]:
    """Return wafer centre and maximum radius."""

    if center_override is not None:
        try:
            cx, cy = map(float, center_override)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("distribution_spline.center must contain two numeric values") from exc
    else:
        cx = float(np.mean(x))
        cy = float(np.mean(y))

    radii = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = float(np.max(radii))
    if r_max == 0.0:
        r_max = 1.0
    return cx, cy, r_max


def _spline_resample_polar(
    df: pd.DataFrame,
    n_radii: int,
    n_angles: int,
    smoothing: Optional[float],
    kx: int,
    ky: int,
    radius_padding: float,
    center_override: Optional[Any],
    max_points: Optional[int],
    radius_limit: Optional[float],
) -> pd.DataFrame:
    """Resample distribution onto a polar grid constrained to the wafer disk."""

    if n_radii < 2 or n_angles < 4:
        raise ValueError("n_radii must be >=2 and n_angles >=4 for polar spline resampling")

    x = df["x"].values
    y = df["y"].values
    f = df["f"].values

    spline_kwargs = {"kx": kx, "ky": ky}
    if smoothing is not None:
        spline_kwargs["s"] = smoothing

    spline = SmoothBivariateSpline(x, y, f, **spline_kwargs)

    cx, cy, r_max = _compute_center_radius(x, y, center_override)
    data_limit = r_max * (1.0 + max(radius_padding, 0.0))
    if radius_limit is not None:
        r_limit = float(radius_limit)
    else:
        r_limit = data_limit
    if r_limit <= 0:
        raise ValueError("radius_limit must be positive when provided")
    effective_limit = min(r_limit, data_limit)

    radii = np.linspace(0.0, effective_limit, n_radii)
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)

    R, Theta = np.meshgrid(radii, angles)
    if max_points is not None and max_points > 0:
        total = R.size
        if total > max_points:
            stride_theta = int(np.ceil(np.sqrt(total / max_points)))
            stride_theta = max(stride_theta, 1)
            stride_r = int(np.ceil(total / (max_points * stride_theta)))
            stride_r = max(stride_r, 1)
            Theta = Theta[::stride_theta, ::stride_r]
            R = R[::stride_theta, ::stride_r]

    X = cx + R * np.cos(Theta)
    Y = cy + R * np.sin(Theta)
    Z = spline.ev(X.ravel(), Y.ravel()).reshape(X.shape)

    radius_grid = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask_limit = r_limit if radius_limit is not None else effective_limit
    mask = radius_grid <= (mask_limit + 1e-8)
    X = X[mask]
    Y = Y[mask]
    Z = Z[mask]
    radius_flat = radius_grid[mask]
    theta_flat = Theta[mask]

    resampled = pd.DataFrame({
        "x": X.ravel(),
        "y": Y.ravel(),
        "f": Z.ravel(),
        "r": radius_flat.ravel(),
        "theta": theta_flat.ravel(),
    })

    if radius_limit is not None and radius_limit > 0:
        resampled["r_norm"] = resampled["r"] / radius_limit
    else:
        max_r = resampled["r"].max()
        resampled["r_norm"] = resampled["r"] / (max_r if max_r > 0 else 1.0)

    resampled["cx"] = cx
    resampled["cy"] = cy
    radius_value = effective_limit if effective_limit > 0 else float(resampled["r"].max())
    resampled["radius_max"] = radius_value

    resampled = resampled.drop_duplicates(subset=("x", "y"), ignore_index=True)
    resampled = resampled.sort_values(by=["theta", "r"], kind="mergesort", ignore_index=True)
    return resampled


def preprocess_distributions_spline(
    distributions: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Apply spline interpolation to each distribution when enabled.

    When `mode` is ``"polar"`` the interpolation respects an optional
    `max_radius` specified in the configuration by clipping evaluations to
    that radius. Returned DataFrames include additional polar-aligned
    columns (`r`, `theta`, `r_norm`) so that downstream feature engineering
    can access the circular structure directly.
    """

    smoothing = config.get("smoothing", None)
    if smoothing is not None:
        smoothing = float(smoothing)
    kx = int(config.get("kx", 3))
    ky = int(config.get("ky", 3))

    mode = config.get("mode")
    if mode is None:
        mode = "polar" if "grid_resolution" not in config else "cartesian"
    mode = str(mode).lower()

    center_override = config.get("center")
    max_points = config.get("max_points")
    if max_points is not None:
        max_points = int(max_points)
    radius_limit = config.get("max_radius")
    if radius_limit is not None:
        radius_limit = float(radius_limit)

    processed: Dict[str, pd.DataFrame] = {}
    for key, df in distributions.items():
        try:
            if mode == "cartesian":
                grid_resolution = int(config.get("grid_resolution", 50))
                processed[key] = _spline_resample_cartesian(
                    df,
                    grid_resolution=grid_resolution,
                    smoothing=smoothing,
                    kx=kx,
                    ky=ky,
                    max_points=max_points,
                )
            else:
                n_radii = int(config.get("n_radii", 40))
                n_angles = int(config.get("n_angles", 180))
                radius_padding = float(config.get("radius_padding", 0.0))
                processed[key] = _spline_resample_polar(
                    df,
                    n_radii=n_radii,
                    n_angles=n_angles,
                    smoothing=smoothing,
                    kx=kx,
                    ky=ky,
                    radius_padding=radius_padding,
                    center_override=center_override,
                    max_points=max_points,
                    radius_limit=radius_limit,
                )
        except Exception as exc:  # noqa: BLE001 - provide context and continue
            print(
                f"[WARN] Spline preprocessing failed for '{key}': {exc}. Using original distribution."
            )
            processed[key] = df
    return processed
