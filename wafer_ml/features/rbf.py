"""Radial basis function (RBF) feature extraction utilities."""

from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

def select_centers(
    x: np.ndarray,
    y: np.ndarray,
    n_centers: int,
    random_state: int = 0,
) -> np.ndarray:
    """Select RBF centres using k-means (mini-batch for large datasets)."""

    coords = np.column_stack((x, y))
    if coords.size == 0:
        raise ValueError("Cannot select RBF centres from empty coordinate set")
    if n_centers >= coords.shape[0]:
        return coords

    if coords.shape[0] > 20000:
        km = MiniBatchKMeans(
            n_clusters=n_centers,
            random_state=random_state,
            batch_size=4096,
            reassignment_ratio=0.01,
        )
    else:
        km = KMeans(n_clusters=n_centers, random_state=random_state)
    km.fit(coords)
    return km.cluster_centers_


def compute_normalisation_stats(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean and standard deviation for coordinate normalisation."""

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    mean = coords.mean(axis=0)
    std = coords.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def normalise_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalise coordinates using supplied mean and standard deviation."""

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if mean is None or std is None:
        return x_arr, y_arr
    norm_std = np.where(std < 1e-12, 1.0, std)
    return (x_arr - mean[0]) / norm_std[0], (y_arr - mean[1]) / norm_std[1]


def rbf_matrix(
    x: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Compute the RBF design matrix for given points and centres."""

    coords = np.column_stack((x, y))
    if coords.size == 0:
        raise ValueError("Cannot build RBF matrix with zero coordinates")
    diff = coords[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    return np.exp(-gamma * dist_sq)


def solve_rbf_weights(
    x: np.ndarray,
    y: np.ndarray,
    f: np.ndarray,
    centers: np.ndarray,
    gamma: float,
    *,
    coord_mean: Optional[np.ndarray] = None,
    coord_std: Optional[np.ndarray] = None,
    ridge: float = 0.0,
) -> np.ndarray:
    """Solve for RBF weights using optional normalisation and ridge regularisation."""

    x_norm, y_norm = normalise_coordinates(x, y, coord_mean, coord_std)
    design = rbf_matrix(x_norm, y_norm, centers, gamma)
    target = np.asarray(f, dtype=float)
    if target.shape[0] != design.shape[0]:
        raise ValueError("Target vector length must match number of coordinates")
    if ridge and ridge > 0.0:
        gram = design.T @ design
        np.fill_diagonal(gram, gram.diagonal() + ridge)
        weights = np.linalg.solve(gram, design.T @ target)
    else:
        weights, *_ = np.linalg.lstsq(design, target, rcond=None)
    return weights


def fit_rbf(
    x: np.ndarray,
    y: np.ndarray,
    f: np.ndarray,
    n_centers: int,
    gamma: float,
    random_state: int = 0,
    *,
    coord_mean: Optional[np.ndarray] = None,
    coord_std: Optional[np.ndarray] = None,
    ridge: float = 0.0,
    centres: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit an RBF representation using optional shared centres and normalisation."""

    if centres is None:
        x_norm, y_norm = normalise_coordinates(x, y, coord_mean, coord_std)
        centres = select_centers(x_norm, y_norm, n_centers, random_state=random_state)
    weights = solve_rbf_weights(
        x,
        y,
        f,
        centres,
        gamma,
        coord_mean=coord_mean,
        coord_std=coord_std,
        ridge=ridge,
    )
    return weights, centres


def reconstruct_rbf(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    centers: np.ndarray,
    gamma: float,
    *,
    coord_mean: Optional[np.ndarray] = None,
    coord_std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reconstruct function values from RBF weights.

    Parameters
    ----------
    x, y : numpy.ndarray
        Points at which to reconstruct.
    weights : numpy.ndarray
        Weight vector of shape (n_centers,).
    centers : numpy.ndarray
        RBF centre coordinates of shape (n_centers, 2).
    gamma : float
        Width parameter.

    Returns
    -------
    numpy.ndarray
        Reconstructed values.
    """
    x_norm, y_norm = normalise_coordinates(x, y, coord_mean, coord_std)
    design = rbf_matrix(x_norm, y_norm, centers, gamma)
    return design.dot(weights)
