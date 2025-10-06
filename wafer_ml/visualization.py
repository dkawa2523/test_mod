"""
Visualisation utilities
----------------------

This module contains helper functions for generating plots that summarise
decomposition and model performance. The plotting functions use Matplotlib
exclusively and avoid specifying colours so that default palettes apply.
Each function saves the resulting figure to disk at a specified path.
"""

import os
from typing import Dict, Iterable, Tuple, Optional

import matplotlib

# Use a non-interactive backend that works in headless environments.
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata
from matplotlib.patches import Circle

sns.set_style("whitegrid")


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    outfile: str,
    r2: Optional[float] = None,
) -> None:
    """
    Plot a scatter plot comparing true vs predicted values.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Arrays of the same shape.
    title : str
        Title of the plot.
    outfile : str
        Path to save the figure.
    r2 : float, optional
        R² score to annotate on the plot.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title(title)
    if r2 is not None:
        plt.text(0.05, 0.95, f"R² = {r2:.4f}", transform=plt.gca().transAxes, verticalalignment='top')
    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def plot_histogram(
    metrics: Dict[str, float],
    metric_name: str,
    outfile: str,
    title: Optional[str] = None,
) -> None:
    """
    Plot a bar chart of metric values for different methods.

    Parameters
    ----------
    metrics : dict
        Mapping from method name to metric value.
    metric_name : str
        Name of the metric being plotted.
    outfile : str
        Path to save the figure.
    title : str, optional
        Plot title.
    """
    methods = list(metrics.keys())
    values = [metrics[m] for m in methods]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=methods, y=values)
    plt.ylabel(metric_name)
    if title:
        plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def plot_distribution_comparison(
    x: np.ndarray,
    y: np.ndarray,
    f_true: np.ndarray,
    f_pred: np.ndarray,
    outfile: str,
    title: Optional[str] = None,
) -> None:
    """
    Plot original and reconstructed distributions side by side.

    Parameters
    ----------
    x, y : numpy.ndarray
        Coordinates of points.
    f_true, f_pred : numpy.ndarray
        True and predicted function values.
    outfile : str
        Path to save the figure.
    title : str, optional
        Overall title for the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    scatter = axes[0].scatter(x, y, c=f_true, s=15)
    axes[0].set_title('Original')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[1].scatter(x, y, c=f_pred, s=15)
    axes[1].set_title('Reconstruction')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.6)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def plot_confidence_intervals(
    prediction: Dict[str, np.ndarray],
    outfile: str,
    title: Optional[str] = None,
) -> None:
    """
    Plot predicted distribution with confidence intervals for a single sample.

    The `prediction` dictionary should contain keys 'x', 'y', 'f_pred', 'lower', 'upper'.
    """
    if not all(k in prediction for k in ("x", "y", "f_pred", "lower", "upper")):
        raise ValueError("Prediction dictionary must contain keys 'x', 'y', 'f_pred', 'lower', 'upper'")
    x = prediction["x"]
    y = prediction["y"]
    f_pred = prediction["f_pred"]
    lower = prediction["lower"]
    upper = prediction["upper"]
    fig, ax = plt.subplots(figsize=(6, 5))
    # Plot predicted values as scatter with color
    sc = ax.scatter(x, y, c=f_pred, s=20)
    # Overlay error bars as vertical segments connecting lower and upper at each point
    for xi, yi, lo, hi in zip(x, y, lower, upper):
        ax.plot([xi, xi], [lo, hi], color='black', linewidth=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if title:
        ax.set_title(title)
    fig.colorbar(sc, ax=ax, label='Predicted value')
    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def _interpolate_on_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_size: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.column_stack((x, y))
    xi = np.linspace(np.min(x), np.max(x), grid_size)
    yi = np.linspace(np.min(y), np.max(y), grid_size)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata(points, values, (Xi, Yi), method="cubic")
    if np.isnan(Zi).any():
        Zi = griddata(points, values, (Xi, Yi), method="linear")
    if np.isnan(Zi).any():
        Zi = griddata(points, values, (Xi, Yi), method="nearest")
    return Xi, Yi, Zi


def plot_heatmap_triplet(
    x: np.ndarray,
    y: np.ndarray,
    f_true: np.ndarray,
    f_pred: np.ndarray,
    outfile: str,
    title: Optional[str] = None,
    grid_size: int = 100,
    cmap: str = "jet",
    mask_center: Optional[Tuple[float, float]] = None,
    mask_radius: Optional[float] = None,
    boundary_color: str = "black",
) -> None:
    """Plot original, reconstructed, and residual heatmaps constrained to a wafer disk."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if mask_center is None:
        mask_center = (float(np.mean(x)), float(np.mean(y)))
    if mask_radius is None:
        mask_radius = float(
            np.max(np.sqrt((x - mask_center[0]) ** 2 + (y - mask_center[1]) ** 2))
        )
        if mask_radius <= 0:
            mask_radius = 1.0

    residual = f_true - f_pred
    Xi, Yi, Z_true = _interpolate_on_grid(x, y, f_true, grid_size)
    _, _, Z_pred = _interpolate_on_grid(x, y, f_pred, grid_size)
    _, _, Z_res = _interpolate_on_grid(x, y, residual, grid_size)

    mask = (Xi - mask_center[0]) ** 2 + (Yi - mask_center[1]) ** 2 <= (mask_radius ** 2 + 1e-9)
    masked_true = np.ma.array(Z_true, mask=~mask)
    masked_pred = np.ma.array(Z_pred, mask=~mask)
    masked_res = np.ma.array(Z_res, mask=~mask)

    true_vals = masked_true.compressed()
    pred_vals = masked_pred.compressed()
    res_vals = masked_res.compressed()

    if true_vals.size > 0:
        vmin = float(np.nanmin(true_vals))
        vmax = float(np.nanmax(true_vals))
    else:
        vmin, vmax = 0.0, 1.0
    if pred_vals.size > 0:
        vmin = min(vmin, float(np.nanmin(pred_vals)))
        vmax = max(vmax, float(np.nanmax(pred_vals)))
    res_max = float(np.nanmax(np.abs(res_vals))) if res_vals.size > 0 else 1.0

    base_cmap = plt.cm.get_cmap(cmap)
    cmap_data = base_cmap(np.linspace(0, 1, base_cmap.N))
    masked_cmap = mcolors.ListedColormap(cmap_data)
    masked_cmap.set_bad(color=(1, 1, 1, 0))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    extent = [Xi.min(), Xi.max(), Yi.min(), Yi.max()]

    panels = (
        (masked_true, "Original", (vmin, vmax)),
        (masked_pred, "Reconstruction", (vmin, vmax)),
        (masked_res, "Residual", (-res_max, res_max)),
    )

    for ax, (panel_data, panel_title, limits) in zip(axes, panels):
        im = ax.imshow(
            panel_data,
            origin="lower",
            extent=extent,
            cmap=masked_cmap,
            vmin=limits[0],
            vmax=limits[1],
            aspect="equal",
        )
        clip_circle = Circle(mask_center, mask_radius, transform=ax.transData)
        im.set_clip_path(clip_circle)
        boundary = Circle(mask_center, mask_radius, fill=False, edgecolor=boundary_color, linewidth=0.8)
        ax.add_patch(boundary)
        ax.set_title(panel_title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def plot_coefficient_histogram(
    coeff_true: np.ndarray,
    coeff_pred: np.ndarray,
    outfile: str,
    title: Optional[str] = None,
) -> None:
    """Bar histogram comparing true and predicted coefficients."""

    true_arr = np.asarray(coeff_true).ravel()
    pred_arr = np.asarray(coeff_pred).ravel()
    if true_arr.shape != pred_arr.shape:
        raise ValueError("Coefficient arrays must have the same shape")

    indices = np.arange(true_arr.size)
    width = 0.4

    plt.figure(figsize=(max(6, true_arr.size * 0.15), 4))
    plt.bar(indices - width / 2, true_arr, width=width, label="True")
    plt.bar(indices + width / 2, pred_arr, width=width, label="Pred")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Value")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def plot_coefficient_uncertainty_bar(
    variance_mean: np.ndarray,
    variance_std: Optional[np.ndarray],
    outfile: str,
    title: Optional[str] = None,
    coeff_indices: Optional[Iterable[int]] = None,
) -> None:
    """Bar chart of averaged predictive variances per coefficient."""

    variance_mean = np.asarray(variance_mean, dtype=float)
    if coeff_indices is None:
        coeff_indices = np.arange(variance_mean.size)
    labels = [str(idx) for idx in coeff_indices]

    plt.figure(figsize=(max(6, variance_mean.size * 0.4), 4))
    bars = plt.bar(labels, variance_mean, color=sns.color_palette("viridis", len(labels)))
    if variance_std is not None:
        variance_std = np.asarray(variance_std, dtype=float)
        plt.errorbar(labels, variance_mean, yerr=variance_std, fmt='none', ecolor='black', capsize=3)
    plt.ylabel("Variance")
    plt.xlabel("Coefficient Index")
    if title:
        plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def plot_coefficient_relative_uncertainty(
    relative_uncertainty: np.ndarray,
    outfile: str,
    title: Optional[str] = None,
    coeff_indices: Optional[Iterable[int]] = None,
) -> None:
    """Bar plot of relative uncertainty (std/|mean|) per coefficient."""

    relative_uncertainty = np.asarray(relative_uncertainty, dtype=float)
    if coeff_indices is None:
        coeff_indices = np.arange(relative_uncertainty.size)
    labels = [str(idx) for idx in coeff_indices]

    plt.figure(figsize=(max(6, relative_uncertainty.size * 0.4), 4))
    plt.bar(labels, relative_uncertainty, color=sns.color_palette("mako", len(labels)))
    plt.ylabel("Mean Relative Std")
    plt.xlabel("Coefficient Index")
    if title:
        plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def plot_coefficient_histogram_grid(
    coeff_matrix: np.ndarray,
    outfile: str,
    bins: int = 30,
    max_cols: int = 5,
) -> None:
    """Plot histograms for each coefficient across the dataset."""

    coeff_matrix = np.asarray(coeff_matrix, dtype=float)
    if coeff_matrix.ndim != 2:
        raise ValueError("coeff_matrix must be 2D (samples, coefficients)")
    n_samples, n_coeffs = coeff_matrix.shape
    if n_coeffs == 0:
        raise ValueError("No coefficients available for histogram plot")

    n_cols = min(max_cols, n_coeffs)
    n_rows = int(np.ceil(n_coeffs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 2.5))
    axes = np.atleast_1d(axes).flatten()

    for idx in range(n_coeffs):
        ax = axes[idx]
        ax.hist(coeff_matrix[:, idx], bins=bins, color="#1f77b4", alpha=0.75)
        ax.set_title(f"Coeff {idx}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    for idx in range(n_coeffs, len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile)
    plt.close(fig)


def plot_feature_importance_bar(
    feature_names: Iterable[str],
    importance_scores: Iterable[float],
    outfile: str,
    title: Optional[str] = None,
) -> None:
    """Bar chart visualising feature importance scores."""

    names = list(feature_names)
    scores = np.asarray(list(importance_scores), dtype=float)
    plt.figure(figsize=(max(6, len(names) * 0.4), 4))
    palette = sns.color_palette("Blues_d", len(names))
    plt.bar(names, scores, color=palette)
    plt.ylabel("Importance Score")
    plt.xlabel("Feature")
    if title:
        plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def plot_uncertainty_feature_heatmap(
    feature_x: np.ndarray,
    feature_y: np.ndarray,
    uncertainty: np.ndarray,
    outfile: str,
    x_label: str,
    y_label: str,
    cmap: str = "viridis",
) -> None:
    """Scatter-based heatmap of uncertainty over two features."""

    fx = np.asarray(feature_x, dtype=float)
    fy = np.asarray(feature_y, dtype=float)
    unc = np.asarray(uncertainty, dtype=float)
    if fx.size == 0 or fy.size == 0:
        raise ValueError("feature arrays must not be empty")
    if fx.shape[0] != fy.shape[0] or fx.shape[0] != unc.shape[0]:
        raise ValueError("Feature arrays and uncertainty must have the same length")

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(fx, fy, c=unc, cmap=cmap, s=35, alpha=0.7, edgecolor="none")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    cb = plt.colorbar(sc)
    cb.set_label("Mean Coefficient σ")
    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile)
    plt.close()
