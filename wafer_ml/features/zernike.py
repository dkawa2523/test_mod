"""
Zernike polynomial feature extraction
-----------------------------------

This module provides functions to compute Zernike polynomial bases and perform
least squares decomposition of an arbitrary spatial distribution on a unit
disk. The 2D Zernike polynomials form an orthogonal set over the unit
circle, which makes them suitable for capturing low‑order spatial
variations (e.g. tilt, defocus, astigmatism) in wafer distributions. This
implementation is adapted for arbitrary coordinate ranges by normalising the
input points to lie within the unit disk.

Functions in this module can:

* Generate Zernike radial polynomials and full polynomials.
* Assemble a design matrix for a set of (n, m) indices.
* Fit a Zernike expansion to data via least squares.
* Reconstruct the spatial distribution from fitted coefficients.

References
----------
The advantages of Zernike polynomials for modelling spatial errors in
semiconductor manufacturing were described in Semiengineering's discussion of
overlay modelling, which noted that orthogonality reduces collinearity and
improves stability【990290441617999†L100-L115】.
"""

import math
from typing import Iterable, List, Optional, Tuple

import numpy as np


def _radial_polynomial(n: int, m: int, r: np.ndarray) -> np.ndarray:
    """Compute the radial component R_nm(r) for Zernike polynomials."""
    m = abs(m)
    radial = np.zeros_like(r, dtype=float)
    for s in range((n - m) // 2 + 1):
        c = (
            (-1) ** s
            * math.factorial(n - s)
            / (
                math.factorial(s)
                * math.factorial((n + m) // 2 - s)
                * math.factorial((n - m) // 2 - s)
            )
        )
        radial += c * np.power(r, n - 2 * s)
    return radial


def zernike_polynomial(n: int, m: int, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Compute the (n, m) Zernike polynomial on polar coordinates.

    Parameters
    ----------
    n : int
        Radial degree (n >= 0).
    m : int
        Azimuthal frequency (|m| <= n and (n - |m|) is even).
    r : numpy.ndarray
        Radii normalised to [0, 1].
    theta : numpy.ndarray
        Angles in radians.

    Other Parameters
    ----------------
    r_norm : numpy.ndarray, optional
        Pre-computed normalised radii to reuse (must match the length of x).
    theta : numpy.ndarray, optional
        Pre-computed angles in radians corresponding to the coordinates.
    center : tuple of float, optional
        Explicit wafer centre to subtract before normalisation.
    radius_scale : float, optional
        Reference radius used when normalising coordinates.

    Other Parameters
    ----------------
    r_norm : numpy.ndarray, optional
        Pre-computed normalised radii for the coordinates.
    theta : numpy.ndarray, optional
        Pre-computed angles for the coordinates.
    center : tuple of float, optional
        Explicit wafer centre to subtract before normalisation.
    radius_scale : float, optional
        Reference radius used when normalising coordinates.

    Returns
    -------
    numpy.ndarray
        Values of Zernike polynomial of shape equal to r and theta.
    """
    R_nm = _radial_polynomial(n, m, r)
    if m > 0:
        return R_nm * np.cos(m * theta)
    elif m < 0:
        return R_nm * np.sin(-m * theta)
    else:
        return R_nm


def generate_nm_pairs(max_order: int) -> List[Tuple[int, int]]:
    """
    Generate a list of (n, m) indices up to a maximum radial order.

    The pairs satisfy 0 <= n <= max_order, |m| <= n and (n - |m|) is even.

    Returns
    -------
    list of tuple
        List of (n, m) index pairs in increasing order of n and m.
    """
    pairs = []
    for n in range(max_order + 1):
        for m in range(-n, n + 1, 2):
            if (n - abs(m)) % 2 == 0:
                pairs.append((n, m))
    return pairs


def normalise_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    center: Optional[Tuple[float, float]] = None,
    radius_scale: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalise x and y coordinates to the unit disk.

    The coordinates are linearly scaled such that the provided radius scale
    (or, if absent, the maximum observed radius) maps to 1.0.

    Parameters
    ----------
    x, y : numpy.ndarray
        Cartesian coordinates.
    center : tuple of float, optional
        Centre to subtract before computing radii. Defaults to (0, 0).
    radius_scale : float, optional
        Reference radius to use for normalisation. If None, the maximum
        radius from the supplied coordinates is used.

    Returns
    -------
    r_norm : numpy.ndarray
        Normalised radii in [0, 1].
    theta : numpy.ndarray
        Angles in [0, 2π).
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if center is not None:
        cx, cy = map(float, center)
        x_arr = x_arr - cx
        y_arr = y_arr - cy
    r = np.sqrt(x_arr ** 2 + y_arr ** 2)
    if radius_scale is not None:
        max_r = float(radius_scale)
    else:
        max_r = float(np.max(r))
    if max_r <= 0:
        max_r = 1.0
    r_norm = r / max_r
    theta = np.mod(np.arctan2(y_arr, x_arr), 2.0 * np.pi)
    return r_norm, theta


def zernike_design_matrix(
    x: np.ndarray,
    y: np.ndarray,
    nm_pairs: Iterable[Tuple[int, int]],
    r_norm: Optional[np.ndarray] = None,
    theta: Optional[np.ndarray] = None,
    center: Optional[Tuple[float, float]] = None,
    radius_scale: Optional[float] = None,
) -> np.ndarray:
    """
    Construct the Zernike design matrix for given coordinates and (n,m) pairs.

    Parameters
    ----------
    x, y : numpy.ndarray
        Cartesian coordinates of shape (N,).
    nm_pairs : iterable of tuple
        Indices of Zernike polynomials to include.
    r_norm : numpy.ndarray, optional
        Pre-computed normalised radii.
    theta : numpy.ndarray, optional
        Pre-computed angles in radians.
    center : tuple of float, optional
        Explicit wafer centre to subtract before normalisation.
    radius_scale : float, optional
        Reference radius for normalising coordinates.

    Returns
    -------
    numpy.ndarray
        Design matrix of shape (N, len(nm_pairs)).
    """
    if r_norm is None or theta is None:
        computed_r, computed_theta = normalise_coordinates(
            x,
            y,
            center=center,
            radius_scale=radius_scale,
        )
        if r_norm is None:
            r_norm = computed_r
        if theta is None:
            theta = computed_theta
    r_norm = np.asarray(r_norm, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if r_norm.shape[0] != x.shape[0] or theta.shape[0] != x.shape[0]:
        raise ValueError("r_norm and theta must match the length of x and y")
    design = np.empty((x.shape[0], len(nm_pairs)), dtype=float)
    for idx, (n, m) in enumerate(nm_pairs):
        design[:, idx] = zernike_polynomial(n, m, r_norm, theta)
    return design


def fit_zernike(
    x: np.ndarray,
    y: np.ndarray,
    f: np.ndarray,
    max_order: int,
    *,
    r_norm: Optional[np.ndarray] = None,
    theta: Optional[np.ndarray] = None,
    center: Optional[Tuple[float, float]] = None,
    radius_scale: Optional[float] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Fit a Zernike expansion to the given data via least squares.

    Parameters
    ----------
    x, y, f : numpy.ndarray
        Coordinates and values of shape (N,). Coordinates are not required to
        lie inside the unit disk; they will be normalised internally.
    max_order : int
        Maximum radial order n of the Zernike polynomials.

    Returns
    -------
    coeffs : numpy.ndarray
        Fitted coefficients of shape (K,), where K is the number of basis
        functions.
    nm_pairs : list of tuple
        The (n, m) pairs corresponding to the coefficients.
    """
    nm_pairs = generate_nm_pairs(max_order)
    design = zernike_design_matrix(
        x,
        y,
        nm_pairs,
        r_norm=r_norm,
        theta=theta,
        center=center,
        radius_scale=radius_scale,
    )
    # Solve least squares to find coefficients
    coeffs, *_ = np.linalg.lstsq(design, f, rcond=None)
    return coeffs, nm_pairs


def reconstruct_zernike(
    x: np.ndarray,
    y: np.ndarray,
    coeffs: np.ndarray,
    nm_pairs: List[Tuple[int, int]],
    *,
    r_norm: Optional[np.ndarray] = None,
    theta: Optional[np.ndarray] = None,
    center: Optional[Tuple[float, float]] = None,
    radius_scale: Optional[float] = None,
) -> np.ndarray:
    """
    Reconstruct the function values from Zernike coefficients at given points.

    Parameters
    ----------
    x, y : numpy.ndarray
        Cartesian coordinates at which to evaluate the reconstruction.
    coeffs : numpy.ndarray
        Coefficient vector of shape (K,).
    nm_pairs : list of tuple
        The (n,m) pairs corresponding to each coefficient.

    Returns
    -------
    numpy.ndarray
        Reconstructed values at the input coordinates.
    """
    design = zernike_design_matrix(
        x,
        y,
        nm_pairs,
        r_norm=r_norm,
        theta=theta,
        center=center,
        radius_scale=radius_scale,
    )
    coeffs_arr = np.asarray(coeffs, dtype=float).reshape(-1)
    return design @ coeffs_arr
