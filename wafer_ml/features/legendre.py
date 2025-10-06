"""
Legendre polynomial feature extraction
------------------------------------

This module implements two‑dimensional Legendre polynomial decomposition for
spatial distributions. The 2D basis functions are products of 1D Legendre
polynomials in x and y directions, each defined on the interval [-1, 1].
Normalisation of the input coordinates to this interval is handled
internally. The resulting coefficients can be used as features for linear
regression or other machine learning models.
"""

from typing import List, Tuple
import numpy as np
from numpy.polynomial.legendre import Legendre


def normalise_to_unit_interval(values: np.ndarray) -> np.ndarray:
    """Linearly scale values to the range [-1, 1]."""
    vmin = np.min(values)
    vmax = np.max(values)
    if vmax == vmin:
        return np.zeros_like(values)
    return 2.0 * (values - vmin) / (vmax - vmin) - 1.0


def legendre_design_matrix(
    x: np.ndarray,
    y: np.ndarray,
    max_degree_x: int,
    max_degree_y: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Construct a design matrix of 2D Legendre basis functions.

    Parameters
    ----------
    x, y : numpy.ndarray
        Cartesian coordinates of shape (N,).
    max_degree_x, max_degree_y : int
        Maximum polynomial degree in x and y directions.

    Returns
    -------
    design : numpy.ndarray
        Matrix of shape (N, (max_degree_x+1) * (max_degree_y+1)).
    indices : list of tuple
        List of (n_x, n_y) pairs corresponding to each column.
    """
    # Normalize to [-1, 1]
    x_norm = normalise_to_unit_interval(x)
    y_norm = normalise_to_unit_interval(y)
    terms = []
    for nx in range(max_degree_x + 1):
        Pnx = Legendre.basis(nx)
        for ny in range(max_degree_y + 1):
            Pny = Legendre.basis(ny)
            terms.append((nx, ny, Pnx(x_norm), Pny(y_norm)))
    design = np.zeros((x.shape[0], len(terms)))
    indices: List[Tuple[int, int]] = []
    for idx, (nx, ny, px, py) in enumerate(terms):
        design[:, idx] = px * py
        indices.append((nx, ny))
    return design, indices


def fit_legendre(
    x: np.ndarray,
    y: np.ndarray,
    f: np.ndarray,
    max_degree_x: int,
    max_degree_y: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Fit a 2D Legendre polynomial expansion to the data via least squares.

    Parameters
    ----------
    x, y, f : numpy.ndarray
        Coordinates and values of shape (N,).
    max_degree_x, max_degree_y : int
        Maximum polynomial degrees.

    Returns
    -------
    coeffs : numpy.ndarray
        Coefficient vector.
    indices : list of tuple
        (n_x, n_y) pairs for each coefficient.
    """
    design, indices = legendre_design_matrix(x, y, max_degree_x, max_degree_y)
    coeffs, *_ = np.linalg.lstsq(design, f, rcond=None)
    return coeffs, indices


def reconstruct_legendre(
    x: np.ndarray,
    y: np.ndarray,
    coeffs: np.ndarray,
    indices: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Reconstruct function values from Legendre coefficients.

    Parameters
    ----------
    x, y : numpy.ndarray
        Points at which to reconstruct.
    coeffs : numpy.ndarray
        Coefficient vector.
    indices : list of tuple
        (n_x, n_y) pairs corresponding to coefficients.

    Returns
    -------
    numpy.ndarray
        Reconstructed values.
    """
    x_norm = normalise_to_unit_interval(x)
    y_norm = normalise_to_unit_interval(y)
    recon = np.zeros_like(x, dtype=float)
    # Precompute polynomial values for all unique degrees
    unique_degrees_x = sorted(set(nx for nx, _ in indices))
    unique_degrees_y = sorted(set(ny for _, ny in indices))
    poly_x = {nx: Legendre.basis(nx)(x_norm) for nx in unique_degrees_x}
    poly_y = {ny: Legendre.basis(ny)(y_norm) for ny in unique_degrees_y}
    for c, (nx, ny) in zip(coeffs, indices):
        recon += c * poly_x[nx] * poly_y[ny]
    return recon