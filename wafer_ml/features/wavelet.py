"""
Wavelet transform feature extraction
----------------------------------

This module provides utilities for performing 2D wavelet decomposition of a
spatial distribution. A discrete wavelet transform (DWT) is applied to a 2D
grid representation of the data to extract multi‑scale features. The
coefficients can be flattened into a feature vector for subsequent modelling.

The module also supports reconstruction of the distribution from wavelet
coefficients. If the input data are not on a regular grid, the points are
interpolated onto a grid via nearest neighbours for the purpose of the DWT.

Wavelet transforms are particularly effective at capturing localised
features and edges【334367983434783†L108-L116】 but may be less accurate for
extremely sharp micro‑cracks or very small scale variations. Wavelet
coefficients can serve as a compact representation for regression models.
"""

from typing import Any, List, Tuple
import numpy as np
import pandas as pd
import pywt
from scipy.interpolate import griddata


def _grid_from_scattered(
    x: np.ndarray,
    y: np.ndarray,
    f: np.ndarray,
    method: str = "linear",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a regular grid from scattered data using interpolation.

    Parameters
    ----------
    x, y, f : numpy.ndarray
        Scattered coordinates and values of length N.
    method : str
        Interpolation method for `scipy.interpolate.griddata` ('linear',
        'nearest', 'cubic').

    Returns
    -------
    Xg, Yg, Fg : numpy.ndarray
        Meshgrid arrays and interpolated values on a regular grid.
    """
    # Determine unique sorted coordinates
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    Xg, Yg = np.meshgrid(x_unique, y_unique)
    Fg = griddata((x, y), f, (Xg, Yg), method=method)
    # Replace NaN with nearest neighbour
    if np.any(np.isnan(Fg)):
        Fg_nn = griddata((x, y), f, (Xg, Yg), method="nearest")
        Fg = np.where(np.isnan(Fg), Fg_nn, Fg)
    return Xg, Yg, Fg


def wavelet_decompose(
    x: np.ndarray,
    y: np.ndarray,
    f: np.ndarray,
    wavelet: str,
    level: int,
) -> Tuple[np.ndarray, List[Any], Tuple[int, int]]:
    """
    Perform a 2D discrete wavelet transform on the input data.

    Parameters
    ----------
    x, y, f : numpy.ndarray
        Scattered coordinates and values.
    wavelet : str
        Name of the wavelet (e.g., 'db2', 'haar').
    level : int
        Decomposition level.

    Returns
    -------
    coeffs_flat : numpy.ndarray
        Flattened wavelet coefficient vector.
    coeffs_struct : list
        Nested list of coefficients as returned by pywt.wavedec2 (needed for
        reconstruction).
    grid_shape : tuple of int
        Shape of the underlying grid (rows, cols).
    """
    Xg, Yg, Fg = _grid_from_scattered(x, y, f)
    # Compute DWT
    coeffs = pywt.wavedec2(Fg, wavelet=wavelet, level=level)
    # Flatten coefficients
    coeffs_flat = []
    for i, c in enumerate(coeffs):
        if i == 0:
            coeffs_flat.append(c.flatten())
        else:
            for arr in c:
                coeffs_flat.append(arr.flatten())
    coeffs_flat = np.concatenate(coeffs_flat)
    return coeffs_flat, coeffs, Fg.shape


def wavelet_reconstruct(
    coeffs_struct: List[Any],
    wavelet: str,
) -> np.ndarray:
    """
    Reconstruct a 2D array from wavelet coefficients.

    Parameters
    ----------
    coeffs_struct : list
        Nested list of coefficients as returned by pywt.wavedec2.
    wavelet : str
        Wavelet name.

    Returns
    -------
    numpy.ndarray
        Reconstructed 2D array.
    """
    return pywt.waverec2(coeffs_struct, wavelet=wavelet)