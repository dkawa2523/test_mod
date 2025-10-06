#!/usr/bin/env python3
"""
Training script for wafer distribution models
-------------------------------------------

This script reads a configuration YAML file, loads process condition and
distribution data, performs pre‑processing, trains selected models and
outputs evaluation metrics, plots and CSV files. The workflow is:

1. Load configuration via wafer_ml.config.load_config().
2. Read the conditions table and distribution files.
3. Split the data into training and test sets based on `test_size`.
4. Fit a preprocessor on the training condition variables and transform
   both training and test condition variables.
5. For each enabled method, compute the appropriate feature representation
   (e.g. Zernike coefficients) on the training set, fit the regression
   model, evaluate on the test set, and save results.

Usage:
    python train.py --config path/to/config.yaml

The configuration file must specify the data locations, method settings,
output directory and any hyperparameters for the models.
"""

import argparse
import os
import sys
from typing import Dict, Any, Tuple, Optional, List

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata

from wafer_ml.config import load_config, Config
from wafer_ml.data_loader import load_conditions, load_distributions
from wafer_ml.preprocessing import Preprocessor, preprocess_distributions_spline
from wafer_ml.features.zernike import fit_zernike, reconstruct_zernike
from wafer_ml.features.legendre import fit_legendre, reconstruct_legendre
from wafer_ml.features.rbf import (
    compute_normalisation_stats,
    select_centers,
    solve_rbf_weights,
    reconstruct_rbf,
)
from wafer_ml.features.wavelet import wavelet_decompose, wavelet_reconstruct
from wafer_ml.models.linear_regression import LinearRegressionModel
from wafer_ml.models.gpr import GPRModel
from wafer_ml.models.zernike_gpr import ZernikeGPRModel
from wafer_ml.visualization import (
    plot_scatter,
    plot_histogram,
    plot_confidence_intervals,
    plot_heatmap_triplet,
    plot_coefficient_histogram,
    plot_coefficient_uncertainty_bar,
    plot_coefficient_relative_uncertainty,
    plot_coefficient_histogram_grid,
    plot_feature_importance_bar,
    plot_uncertainty_feature_heatmap,
)
from wafer_ml.utils import (
    ensure_dir,
    save_array_to_csv,
    save_dicts_to_csv,
    compute_residual,
    save_coefficients_csv,
)
from wafer_ml.evaluation import compute_r2, compute_rmse, compute_mae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wafer ML training script")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def split_data(
    conditions: pd.DataFrame,
    distributions: Dict[str, pd.DataFrame],
    test_size: float,
    random_seed: int,
    force_test_ids: Optional[List[str]] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
]:
    """
    Split conditions and distributions into training and test sets.

    Returns
    -------
    cond_train, cond_test : pandas.DataFrame
        Condition DataFrames for training and test.
    dist_train, dist_test : dict
        Distribution dictionaries for training and test.
    """
    ids = conditions.index.values
    forced = []
    if force_test_ids:
        seen = set()
        for sid in force_test_ids:
            if sid in ids and sid not in seen:
                forced.append(sid)
                seen.add(sid)

    remaining_ids = [id_ for id_ in ids if id_ not in forced]
    total = len(ids)
    desired_test_count = int(round(test_size * total)) if total > 0 else 0
    if test_size > 0 and desired_test_count == 0 and total > 0:
        desired_test_count = 1
    remaining_test_count = max(desired_test_count - len(forced), 0)

    additional_test: List[str] = []
    if remaining_ids and remaining_test_count > 0:
        test_ratio = min(max(remaining_test_count / len(remaining_ids), 0.0), 1.0)
        if test_ratio <= 0.0:
            additional_train = remaining_ids
        elif test_ratio >= 1.0:
            additional_test = remaining_ids
            additional_train = []
        else:
            additional_train, additional_test = train_test_split(
                remaining_ids,
                test_size=test_ratio,
                random_state=random_seed,
                shuffle=True,
            )
    else:
        additional_train = remaining_ids

    ids_test = forced + list(additional_test)
    ids_train = [id_ for id_ in ids if id_ not in ids_test]

    cond_train = conditions.loc[ids_train]
    cond_test = conditions.loc[ids_test]
    dist_train = {id_: distributions[id_] for id_ in ids_train}
    dist_test = {id_: distributions[id_] for id_ in ids_test}
    return cond_train, cond_test, dist_train, dist_test


def generate_sample_outputs(
    sample_id: str,
    method_name: str,
    method_out_dir: str,
    x: np.ndarray,
    y: np.ndarray,
    f_true: np.ndarray,
    f_pred: np.ndarray,
    r2_value: Optional[float] = None,
    coeff_true: Optional[np.ndarray] = None,
    coeff_pred: Optional[np.ndarray] = None,
    polar_meta: Optional[Dict[str, Any]] = None,
    heatmap_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Produce heatmap, scatter, and optional coefficient diagnostics for a sample."""

    sample_id_str = str(sample_id)
    sample_dir = os.path.join(method_out_dir, "samples", sample_id_str)
    ensure_dir(sample_dir)

    title = f"{method_name} - {sample_id_str}"
    mask_center = None
    mask_radius = None
    if polar_meta:
        mask_center = polar_meta.get("center")
        mask_radius = polar_meta.get("radius") or polar_meta.get("radius_scale")

    heatmap_cfg = heatmap_cfg or {}
    heatmap_grid_size = int(heatmap_cfg.get("grid_size", 100))
    boundary_color = str(heatmap_cfg.get("boundary_color", "black"))

    comparison_path = os.path.join(method_out_dir, f"sample_{sample_id_str}_comparison.png")
    plot_heatmap_triplet(
        x,
        y,
        f_true,
        f_pred,
        outfile=comparison_path,
        title=title,
        cmap="jet",
        mask_center=mask_center,
        mask_radius=mask_radius,
        grid_size=heatmap_grid_size,
        boundary_color=boundary_color,
    )

    plot_heatmap_triplet(
        x,
        y,
        f_true,
        f_pred,
        outfile=os.path.join(sample_dir, "comparison_heatmap.png"),
        title=title,
        cmap="jet",
        mask_center=mask_center,
        mask_radius=mask_radius,
        grid_size=heatmap_grid_size,
        boundary_color=boundary_color,
    )

    scatter_path_root = os.path.join(method_out_dir, f"sample_{sample_id_str}_true_vs_pred.png")
    scatter_path_sample = os.path.join(sample_dir, "true_vs_pred_scatter.png")
    plot_scatter(
        f_true,
        f_pred,
        title=f"{title}: True vs Pred",
        outfile=scatter_path_root,
        r2=r2_value,
    )
    plot_scatter(
        f_true,
        f_pred,
        title=f"{title}: True vs Pred",
        outfile=scatter_path_sample,
        r2=r2_value,
    )

    if coeff_true is not None and coeff_pred is not None:
        coeff_hist_root = os.path.join(method_out_dir, f"sample_{sample_id_str}_coeff_hist.png")
        coeff_hist_sample = os.path.join(sample_dir, "coeff_hist.png")
        plot_coefficient_histogram(
            coeff_true,
            coeff_pred,
            outfile=coeff_hist_root,
            title=f"{title}: Basis Coefficients",
        )
        plot_coefficient_histogram(
            coeff_true,
            coeff_pred,
            outfile=coeff_hist_sample,
            title=f"{title}: Basis Coefficients",
        )
        save_coefficients_csv(
            coeff_true,
            coeff_pred,
            os.path.join(sample_dir, "coefficients.csv"),
        )


def _extract_polar_kwargs(df: pd.DataFrame) -> Dict[str, Any]:
    """Return keyword arguments for polar-aware Zernike utilities."""

    kwargs: Dict[str, Any] = {}
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


def _extract_heatmap_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """Collect centre and radius information for heatmap masking if available."""

    meta: Dict[str, Any] = {}
    if "cx" in df.columns and "cy" in df.columns:
        meta["center"] = (
            float(df["cx"].iloc[0]),
            float(df["cy"].iloc[0]),
        )
    if "radius_max" in df.columns:
        meta["radius"] = float(df["radius_max"].iloc[0])
    elif "r" in df.columns:
        meta["radius"] = float(df["r"].max())
    return meta


def generate_uncertainty_diagnostics(
    method_name: str,
    method_out_dir: str,
    coeff_means: np.ndarray,
    coeff_stds: np.ndarray,
    coeff_samples: np.ndarray,
    feature_df: pd.DataFrame,
    uncertainty_cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Produce uncertainty plots and statistics for GPR-based methods."""

    if not uncertainty_cfg or not uncertainty_cfg.get("enabled", False):
        return None

    if coeff_means is None or coeff_stds is None:
        return None

    coeff_means = np.asarray(coeff_means, dtype=float)
    coeff_stds = np.asarray(coeff_stds, dtype=float)
    coeff_samples = np.asarray(coeff_samples, dtype=float)
    if coeff_means.ndim != 2 or coeff_stds.ndim != 2:
        return None
    if coeff_means.shape != coeff_stds.shape:
        raise ValueError("Coefficient means and stds must have the same shape")
    if coeff_samples.ndim != 2 or coeff_samples.shape[1] != coeff_means.shape[1]:
        raise ValueError("Coefficient samples must match coefficient dimensionality")

    n_samples, n_coeffs = coeff_means.shape
    if n_samples == 0 or n_coeffs == 0:
        return None

    uncertainty_dir = os.path.join(method_out_dir, "uncertainty")
    ensure_dir(uncertainty_dir)

    coeff_indices = np.arange(n_coeffs)
    variances = np.mean(np.square(coeff_stds), axis=0)
    variances_std = np.std(np.square(coeff_stds), axis=0)
    relative_uncertainty = np.mean(
        coeff_stds / (np.abs(coeff_means) + 1e-12), axis=0
    )

    plot_coefficient_uncertainty_bar(
        variances,
        variances_std,
        outfile=os.path.join(uncertainty_dir, "coeff_variance_bar.png"),
        title=f"{method_name}: Coefficient Variance",
        coeff_indices=coeff_indices,
    )

    plot_coefficient_relative_uncertainty(
        relative_uncertainty,
        outfile=os.path.join(uncertainty_dir, "coeff_relative_uncertainty.png"),
        title=f"{method_name}: Relative Uncertainty",
        coeff_indices=coeff_indices,
    )

    stats_records: List[Dict[str, float]] = []
    for idx in coeff_indices:
        stats_records.append(
            {
                "coefficient_index": int(idx),
                "mean_std": float(np.mean(coeff_stds[:, idx])),
                "median_std": float(np.median(coeff_stds[:, idx])),
                "max_std": float(np.max(coeff_stds[:, idx])),
                "mean_variance": float(variances[idx]),
                "relative_uncertainty_mean": float(relative_uncertainty[idx]),
            }
        )
    pd.DataFrame(stats_records).to_csv(
        os.path.join(uncertainty_dir, "coefficient_uncertainty_stats.csv"),
        index=False,
    )
    stats_path = os.path.join(uncertainty_dir, "coefficient_uncertainty_stats.csv")

    # Coefficient histograms across dataset
    coeff_hist_path = os.path.join(uncertainty_dir, "coeff_histograms.png")
    plot_coefficient_histogram_grid(coeff_samples, coeff_hist_path)

    # Feature importance via mean absolute Pearson correlation
    feature_names = list(feature_df.columns)
    importance_scores: List[float] = []
    for feature_name in feature_names:
        values = feature_df[feature_name].values
        if np.std(values) < 1e-12:
            importance_scores.append(0.0)
            continue
        corrs: List[float] = []
        for idx in range(n_coeffs):
            coeff_series = coeff_means[:, idx]
            if np.std(coeff_series) < 1e-12:
                continue
            corr = np.corrcoef(values, coeff_series)[0, 1]
            if np.isnan(corr):
                continue
            corrs.append(abs(float(corr)))
        importance_scores.append(float(np.mean(corrs)) if corrs else 0.0)

    importance_path = os.path.join(uncertainty_dir, "feature_importance.png")
    plot_feature_importance_bar(
        feature_names,
        importance_scores,
        importance_path,
        title=f"{method_name}: Feature Importance",
    )

    description_path = os.path.join(uncertainty_dir, "importance_description.txt")
    with open(description_path, "w", encoding="utf-8") as desc_file:
        desc_file.write(
            "Feature importance is computed as the mean absolute Pearson correlation\n"
            "between each process feature and the predicted coefficient means across\n"
            "the training samples. Higher values indicate a stronger linear\n"
            "relationship between that feature and the basis coefficients."
        )

    # Two-dimensional uncertainty heatmaps for selected feature pairs (GPR only)
    heatmap_features = uncertainty_cfg.get("features")
    if not heatmap_features:
        heatmap_features = feature_names[: min(3, len(feature_names))]
    sample_uncertainty = np.sqrt(np.mean(coeff_stds ** 2, axis=1))
    heatmap_paths: List[str] = []
    if len(heatmap_features) >= 2:
        for fx, fy in itertools.combinations(heatmap_features, 2):
            if fx not in feature_df.columns or fy not in feature_df.columns:
                print(f"[WARN] Feature pair ({fx}, {fy}) not found for uncertainty heatmap")
                continue
            outfile = os.path.join(
                uncertainty_dir,
                f"uncertainty_heatmap_{fx}_vs_{fy}.png",
            )
            plot_uncertainty_feature_heatmap(
                feature_df[fx].values,
                feature_df[fy].values,
                sample_uncertainty,
                outfile,
                x_label=fx,
                y_label=fy,
            )
            heatmap_paths.append(outfile)

    return {
        "method": method_name,
        "directory": uncertainty_dir,
        "stats_csv": stats_path,
        "variance_plot": os.path.join(uncertainty_dir, "coeff_variance_bar.png"),
        "relative_plot": os.path.join(uncertainty_dir, "coeff_relative_uncertainty.png"),
        "histograms": coeff_hist_path,
        "feature_importance": importance_path,
        "importance_description": description_path,
        "uncertainty_heatmaps": heatmap_paths,
    }

def main() -> None:
    args = parse_args()
    cfg: Config = load_config(args.config)

    # Load data
    data_cfg = cfg.data
    cond_file = data_cfg.get("conditions_file")
    dist_dir = data_cfg.get("distribution_dir")
    id_column = data_cfg.get("id_column", "id")
    if cond_file is None or dist_dir is None:
        raise ValueError("Configuration must specify 'conditions_file' and 'distribution_dir' under data")
    conditions_df = load_conditions(cond_file, id_column=id_column)
    distributions = load_distributions(dist_dir, conditions_df.index.values)

    # Optional spline preprocessing for distributions
    spline_cfg = cfg.preprocessing.get("distribution_spline", {})
    if spline_cfg.get("enabled", False):
        distributions = preprocess_distributions_spline(distributions, spline_cfg)

    # Split data
    train_cfg = cfg.training
    test_size = train_cfg.get("test_size", 0.2)
    random_seed = train_cfg.get("random_seed", 0)
    visual_cfg = cfg.visualization if cfg.visualization else {}
    heatmap_cfg = visual_cfg.get("heatmap", {})
    if heatmap_cfg is None:
        heatmap_cfg = {}
    uncertainty_cfg = visual_cfg.get("uncertainty", {})
    if uncertainty_cfg is None:
        uncertainty_cfg = {}
    raw_sample_ids = visual_cfg.get("sample_ids", [])
    if raw_sample_ids is None:
        raw_sample_ids = []
    elif isinstance(raw_sample_ids, (str, int, float)):
        raw_sample_ids = [raw_sample_ids]
    elif not isinstance(raw_sample_ids, (list, tuple, set)):
        raw_sample_ids = list(raw_sample_ids)

    id_lookup = {str(idx): idx for idx in conditions_df.index.tolist()}
    requested_sample_ids: List[Any] = []
    seen_sample_ids = set()
    unresolved_sample_ids: List[str] = []
    for sid in raw_sample_ids:
        sid_key = str(sid)
        matched = id_lookup.get(sid_key)
        if matched is None:
            unresolved_sample_ids.append(sid_key)
            continue
        if matched in seen_sample_ids:
            continue
        requested_sample_ids.append(matched)
        seen_sample_ids.add(matched)
    if unresolved_sample_ids:
        print(f"[WARN] Sample IDs not found in data: {unresolved_sample_ids}")

    cond_train, cond_test, dist_train, dist_test = split_data(
        conditions_df,
        distributions,
        test_size,
        random_seed,
        force_test_ids=requested_sample_ids,
    )

    # Determine test identifiers and requested sample visualisations
    test_ids = list(dist_test.keys())
    sample_ids_set = {sid for sid in test_ids if sid in requested_sample_ids}
    missing_sample_ids = [sid for sid in requested_sample_ids if sid not in dist_test]
    if missing_sample_ids:
        print(f"[WARN] Sample IDs not found in test split: {missing_sample_ids}")

    # Preprocess condition features
    prep_cfg = cfg.preprocessing
    preprocessor = Preprocessor(
        standardize=prep_cfg.get("standardize", False),
        normalize=prep_cfg.get("normalize", False),
        impute=prep_cfg.get("impute", True),
        impute_strategy=prep_cfg.get("impute_strategy", "mean"),
    )
    X_train = preprocessor.fit_transform(cond_train)
    X_test = preprocessor.transform(cond_test)

    # Output directory
    out_dir = cfg.output.get("directory", "results")
    ensure_dir(out_dir)

    # Collect metrics across methods for histogram
    metric_hist: Dict[str, Dict[str, float]] = {}
    uncertainty_reports: List[Dict[str, Any]] = []

    # Run each enabled method
    methods_cfg = cfg.methods
    for method_name, params in methods_cfg.items():
        enabled = params.get("enabled", False)
        if not enabled:
            continue
        print(f"\nRunning method: {method_name}")
        method_out_dir = os.path.join(out_dir, method_name)
        ensure_dir(method_out_dir)
        # Containers for true and predicted values (flattened)
        y_true_all = []
        y_pred_all = []

        if method_name == "zernike_linear":
            # Decompose into Zernike coefficients
            max_order = params.get("max_order", 4)
            polar_reference: Dict[str, Any] = {}
            if dist_train:
                first_df = next(iter(dist_train.values()))
                first_meta = _extract_polar_kwargs(first_df)
                polar_reference = {
                    key: first_meta[key]
                    for key in ("center", "radius_scale")
                    if key in first_meta
                }
            # Compute coefficients for training data
            coeffs_train = []
            nm_pairs = None
            for df in dist_train.values():
                x = df["x"].values
                y = df["y"].values
                f = df["f"].values
                polar_kwargs = _extract_polar_kwargs(df)
                c, nm_pairs = fit_zernike(x, y, f, max_order, **polar_kwargs)
                coeffs_train.append(c)
            coeffs_train = np.array(coeffs_train)
            # Train linear regression model for multi-output
            lr_model = LinearRegressionModel()
            lr_model.fit(X_train, coeffs_train)
            # Compute coefficients for test data (for evaluation)
            coeffs_test = []
            for sample_id in test_ids:
                df = dist_test[sample_id]
                x = df["x"].values
                y = df["y"].values
                f = df["f"].values
                polar_kwargs = _extract_polar_kwargs(df)
                c, _ = fit_zernike(x, y, f, max_order, **polar_kwargs)
                coeffs_test.append(c)
            coeffs_test = np.array(coeffs_test)
            # Predict coefficients for test conditions
            coeffs_pred = lr_model.predict(X_test)
            # Evaluate coefficients prediction
            r2_coeffs, rmse_coeffs, mae_coeffs = lr_model.evaluate(X_test, coeffs_test)
            # Reconstruct distributions and gather metrics
            for idx, sample_id in enumerate(test_ids):
                df = dist_test[sample_id]
                x_sample = df["x"].values
                y_sample = df["y"].values
                f_true = df["f"].values
                polar_kwargs = _extract_polar_kwargs(df)
                heatmap_meta = _extract_heatmap_metadata(df)
                f_recon = reconstruct_zernike(
                    x_sample,
                    y_sample,
                    coeffs_pred[idx],
                    nm_pairs,
                    **polar_kwargs,
                )
                y_true_all.append(f_true)
                y_pred_all.append(f_recon)

                if sample_id in sample_ids_set:
                    try:
                        r2_sample = compute_r2(f_true, f_recon)
                    except Exception:
                        r2_sample = None
                    generate_sample_outputs(
                        sample_id,
                        "Zernike Linear",
                        method_out_dir,
                        x_sample,
                        y_sample,
                        f_true,
                        f_recon,
                        r2_sample,
                        coeff_true=coeffs_test[idx],
                        coeff_pred=coeffs_pred[idx],
                        polar_meta=heatmap_meta,
                        heatmap_cfg=heatmap_cfg,
                    )
            y_true_flat = np.concatenate(y_true_all)
            y_pred_flat = np.concatenate(y_pred_all)
            r2_dist = compute_r2(y_true_flat, y_pred_flat)
            rmse_dist = compute_rmse(y_true_flat, y_pred_flat)
            mae_dist = compute_mae(y_true_flat, y_pred_flat)
            # Save metrics
            metrics_csv = os.path.join(method_out_dir, "metrics.csv")
            save_dicts_to_csv(
                [
                    {
                        "r2_coefficients": r2_coeffs,
                        "rmse_coefficients": rmse_coeffs,
                        "mae_coefficients": mae_coeffs,
                        "r2_distribution": r2_dist,
                        "rmse_distribution": rmse_dist,
                        "mae_distribution": mae_dist,
                    }
                ],
                metrics_csv,
            )
            # Scatter plot coefficients
            plot_scatter(
                coeffs_test.flatten(),
                coeffs_pred.flatten(),
                title="Zernike Linear: Coefficients", outfile=os.path.join(method_out_dir, "coefficients_scatter.png"), r2=r2_coeffs,
            )
            # Scatter plot distributions
            plot_scatter(
                y_true_flat,
                y_pred_flat,
                title="Zernike Linear: Distribution", outfile=os.path.join(method_out_dir, "distribution_scatter.png"), r2=r2_dist,
            )
            # Save model and metadata
            import joblib
            model_path = os.path.join(method_out_dir, "model.pkl")
            joblib.dump({
                "model": lr_model,
                "nm_pairs": nm_pairs,
                "preprocessor": preprocessor,
                "max_order": max_order,
                "polar_metadata": polar_reference,
            }, model_path)
            metric_hist[method_name] = {
                "r2": r2_dist,
                "rmse": rmse_dist,
                "mae": mae_dist,
            }

        elif method_name == "legendre":
            # Legendre decomposition + linear regression
            max_deg_x = params.get("max_degree_x", 3)
            max_deg_y = params.get("max_degree_y", 3)
            # Compute coefficients for training
            coeffs_train = []
            indices = None
            for df in dist_train.values():
                x = df["x"].values
                y = df["y"].values
                f = df["f"].values
                c, idx_pairs = fit_legendre(x, y, f, max_deg_x, max_deg_y)
                coeffs_train.append(c)
                if indices is None:
                    indices = idx_pairs
            coeffs_train = np.array(coeffs_train)
            # Train linear regression
            lr_model = LinearRegressionModel()
            lr_model.fit(X_train, coeffs_train)
            # Compute test coefficients
            coeffs_test = []
            for sample_id in test_ids:
                df = dist_test[sample_id]
                x = df["x"].values
                y = df["y"].values
                f = df["f"].values
                c, _ = fit_legendre(x, y, f, max_deg_x, max_deg_y)
                coeffs_test.append(c)
            coeffs_test = np.array(coeffs_test)
            coeffs_pred = lr_model.predict(X_test)
            r2_coeffs, rmse_coeffs, mae_coeffs = lr_model.evaluate(X_test, coeffs_test)
            # Reconstruct distributions
            for idx, sample_id in enumerate(test_ids):
                df = dist_test[sample_id]
                x_sample = df["x"].values
                y_sample = df["y"].values
                f_true = df["f"].values
                f_recon = reconstruct_legendre(x_sample, y_sample, coeffs_pred[idx], indices)
                y_true_all.append(f_true)
                y_pred_all.append(f_recon)
                heatmap_meta = _extract_heatmap_metadata(df)
                if sample_id in sample_ids_set:
                    try:
                        r2_sample = compute_r2(f_true, f_recon)
                    except Exception:
                        r2_sample = None
                    generate_sample_outputs(
                        sample_id,
                        "Legendre",
                        method_out_dir,
                        x_sample,
                        y_sample,
                        f_true,
                        f_recon,
                        r2_sample,
                        coeff_true=coeffs_test[idx],
                        coeff_pred=coeffs_pred[idx],
                        polar_meta=heatmap_meta,
                        heatmap_cfg=heatmap_cfg,
                    )
            y_true_flat = np.concatenate(y_true_all)
            y_pred_flat = np.concatenate(y_pred_all)
            r2_dist = compute_r2(y_true_flat, y_pred_flat)
            rmse_dist = compute_rmse(y_true_flat, y_pred_flat)
            mae_dist = compute_mae(y_true_flat, y_pred_flat)
            save_dicts_to_csv(
                [
                    {
                        "r2_coefficients": r2_coeffs,
                        "rmse_coefficients": rmse_coeffs,
                        "mae_coefficients": mae_coeffs,
                        "r2_distribution": r2_dist,
                        "rmse_distribution": rmse_dist,
                        "mae_distribution": mae_dist,
                    }
                ],
                os.path.join(method_out_dir, "metrics.csv"),
            )
            plot_scatter(
                coeffs_test.flatten(),
                coeffs_pred.flatten(),
                title="Legendre Linear: Coefficients", outfile=os.path.join(method_out_dir, "coefficients_scatter.png"), r2=r2_coeffs,
            )
            plot_scatter(
                y_true_flat,
                y_pred_flat,
                title="Legendre Linear: Distribution", outfile=os.path.join(method_out_dir, "distribution_scatter.png"), r2=r2_dist,
            )
            import joblib
            joblib.dump({
                "model": lr_model,
                "indices": indices,
                "preprocessor": preprocessor,
                "max_degree_x": max_deg_x,
                "max_degree_y": max_deg_y,
            }, os.path.join(method_out_dir, "model.pkl"))
            metric_hist[method_name] = {
                "r2": r2_dist,
                "rmse": rmse_dist,
                "mae": mae_dist,
            }

        elif method_name == "rbf":
            # RBF decomposition + linear regression or GPR (choose by param)
            n_centers = params.get("n_centers", 20)
            gamma = params.get("gamma", 1.0)
            regression_type = params.get("regression", "linear")
            ridge = float(params.get("ridge", 1e-6))
            max_center_samples = int(params.get("max_center_samples", 50000))
            center_random_state = params.get("random_state", random_seed)

            # Aggregate coordinates from the training wafers
            train_coords = [
                np.column_stack((df["x"].values, df["y"].values)) for df in dist_train.values()
            ]
            all_coords = np.concatenate(train_coords, axis=0)
            coord_mean, coord_std = compute_normalisation_stats(all_coords)
            coords_norm = (all_coords - coord_mean) / coord_std
            if coords_norm.shape[0] > max_center_samples:
                rng = np.random.default_rng(center_random_state)
                idx = rng.choice(coords_norm.shape[0], size=max_center_samples, replace=False)
                coords_norm = coords_norm[idx]
            centers_ref = select_centers(
                coords_norm[:, 0],
                coords_norm[:, 1],
                n_centers,
                random_state=center_random_state,
            )

            # Compute RBF weights for training samples using shared centres
            weights_train = []
            for df in dist_train.values():
                x = df["x"].values
                y = df["y"].values
                f = df["f"].values
                weights = solve_rbf_weights(
                    x,
                    y,
                    f,
                    centers_ref,
                    gamma,
                    coord_mean=coord_mean,
                    coord_std=coord_std,
                    ridge=ridge,
                )
                weights_train.append(weights)
            weights_train = np.array(weights_train)

            # Train regression model for the weight vectors
            if regression_type == "linear":
                model = LinearRegressionModel()
            elif regression_type == "gpr":
                kernel = params.get("kernel", "RBF")
                length_scale = params.get("length_scale", 1.0)
                alpha = params.get("alpha", 1e-10)
                model = GPRModel(kernel_name=kernel, length_scale=length_scale, alpha=alpha)
            else:
                raise ValueError(f"Unsupported regression type '{regression_type}' for RBF")
            model.fit(X_train, weights_train)
            train_pred_mean = None
            train_pred_std = None
            if regression_type == "gpr":
                train_pred_mean, train_pred_std = model.predict_with_std(X_train)
                if uncertainty_cfg.get("enabled", False):
                    report = generate_uncertainty_diagnostics(
                        "RBF GPR",
                        method_out_dir,
                        train_pred_mean,
                        train_pred_std,
                        weights_train,
                        cond_train,
                        uncertainty_cfg,
                    )
                    if report:
                        uncertainty_reports.append(report)
            # Compute weights for test samples
            weights_test = []
            for sample_id in test_ids:
                df = dist_test[sample_id]
                x = df["x"].values
                y = df["y"].values
                f = df["f"].values
                weights = solve_rbf_weights(
                    x,
                    y,
                    f,
                    centers_ref,
                    gamma,
                    coord_mean=coord_mean,
                    coord_std=coord_std,
                    ridge=ridge,
                )
                weights_test.append(weights)
            weights_test = np.array(weights_test)
            # Predict weights
            weights_pred = model.predict(X_test)
            r2_coeffs, rmse_coeffs, mae_coeffs = model.evaluate(X_test, weights_test)
            # Reconstruct distributions
            for idx, sample_id in enumerate(test_ids):
                df = dist_test[sample_id]
                x_sample = df["x"].values
                y_sample = df["y"].values
                f_true = df["f"].values
                f_recon = reconstruct_rbf(
                    x_sample,
                    y_sample,
                    weights_pred[idx],
                    centers_ref,
                    gamma,
                    coord_mean=coord_mean,
                    coord_std=coord_std,
                )
                y_true_all.append(f_true)
                y_pred_all.append(f_recon)
                heatmap_meta = _extract_heatmap_metadata(df)
                if sample_id in sample_ids_set:
                    try:
                        r2_sample = compute_r2(f_true, f_recon)
                    except Exception:
                        r2_sample = None
                    generate_sample_outputs(
                        sample_id,
                        "RBF",
                        method_out_dir,
                        x_sample,
                        y_sample,
                        f_true,
                        f_recon,
                        r2_sample,
                        coeff_true=weights_test[idx],
                        coeff_pred=weights_pred[idx],
                        polar_meta=heatmap_meta,
                        heatmap_cfg=heatmap_cfg,
                    )
            y_true_flat = np.concatenate(y_true_all)
            y_pred_flat = np.concatenate(y_pred_all)
            r2_dist = compute_r2(y_true_flat, y_pred_flat)
            rmse_dist = compute_rmse(y_true_flat, y_pred_flat)
            mae_dist = compute_mae(y_true_flat, y_pred_flat)
            save_dicts_to_csv(
                [
                    {
                        "r2_weights": r2_coeffs,
                        "rmse_weights": rmse_coeffs,
                        "mae_weights": mae_coeffs,
                        "r2_distribution": r2_dist,
                        "rmse_distribution": rmse_dist,
                        "mae_distribution": mae_dist,
                    }
                ],
                os.path.join(method_out_dir, "metrics.csv"),
            )
            plot_scatter(
                weights_test.flatten(),
                weights_pred.flatten(),
                title="RBF Weights Prediction", outfile=os.path.join(method_out_dir, "weights_scatter.png"), r2=r2_coeffs,
            )
            plot_scatter(
                y_true_flat,
                y_pred_flat,
                title="RBF Distribution", outfile=os.path.join(method_out_dir, "distribution_scatter.png"), r2=r2_dist,
            )
            import joblib
            joblib.dump({
                "model": model,
                "centers": centers_ref,
                "preprocessor": preprocessor,
                "gamma": gamma,
                "n_centers": n_centers,
                "coord_mean": coord_mean,
                "coord_std": coord_std,
                "ridge": ridge,
                "max_center_samples": max_center_samples,
                "random_state": center_random_state,
                "regression_type": regression_type,
                "kernel": params.get("kernel", "RBF"),
                "length_scale": params.get("length_scale", 1.0),
                "alpha": params.get("alpha", 1e-10),
            }, os.path.join(method_out_dir, "model.pkl"))
            metric_hist[method_name] = {
                "r2": r2_dist,
                "rmse": rmse_dist,
                "mae": mae_dist,
            }

        elif method_name == "wavelet":
            # Wavelet decomposition + regression
            wavelet_name = params.get("wavelet", "db2")
            level = params.get("level", 2)
            n_coeffs_select = params.get("n_coeffs", 100)
            regression_type = params.get("regression", "linear")
            # Compute wavelet coefficients for training
            coeffs_train = []
            coeffs_structs_train = []
            grid_shapes_train = []
            for df in dist_train.values():
                x = df["x"].values
                y = df["y"].values
                f = df["f"].values
                coeffs_flat, coeffs_struct, grid_shape = wavelet_decompose(x, y, f, wavelet_name, level)
                # Optionally truncate coefficients for dimension reduction
                if n_coeffs_select is not None and n_coeffs_select < len(coeffs_flat):
                    coeffs_flat = coeffs_flat[:n_coeffs_select]
                coeffs_train.append(coeffs_flat)
                coeffs_structs_train.append(coeffs_struct)
                grid_shapes_train.append(grid_shape)
            coeffs_train = np.array(coeffs_train)
            # Choose regression model
            if regression_type == "linear":
                model = LinearRegressionModel()
            elif regression_type == "gpr":
                kernel = params.get("kernel", "RBF")
                length_scale = params.get("length_scale", 1.0)
                alpha = params.get("alpha", 1e-10)
                model = GPRModel(kernel_name=kernel, length_scale=length_scale, alpha=alpha)
            else:
                raise ValueError(f"Unsupported regression type '{regression_type}' for wavelet")
            model.fit(X_train, coeffs_train)
            train_pred_mean = None
            train_pred_std = None
            if regression_type == "gpr":
                train_pred_mean, train_pred_std = model.predict_with_std(X_train)
                if uncertainty_cfg.get("enabled", False):
                    report = generate_uncertainty_diagnostics(
                        "Wavelet GPR",
                        method_out_dir,
                        train_pred_mean,
                        train_pred_std,
                        coeffs_train,
                        cond_train,
                        uncertainty_cfg,
                    )
                    if report:
                        uncertainty_reports.append(report)
            # Compute test coefficients
            coeffs_test = []
            coeffs_structs_test = []
            grid_shapes_test = []
            for sample_id in test_ids:
                df = dist_test[sample_id]
                x = df["x"].values
                y = df["y"].values
                f = df["f"].values
                coeffs_flat, coeffs_struct, grid_shape = wavelet_decompose(x, y, f, wavelet_name, level)
                if n_coeffs_select is not None and n_coeffs_select < len(coeffs_flat):
                    coeffs_flat = coeffs_flat[:n_coeffs_select]
                coeffs_test.append(coeffs_flat)
                coeffs_structs_test.append(coeffs_struct)
                grid_shapes_test.append(grid_shape)
            coeffs_test = np.array(coeffs_test)
            coeffs_pred = model.predict(X_test)
            # Evaluate coefficient prediction
            r2_coeffs, rmse_coeffs, mae_coeffs = model.evaluate(X_test, coeffs_test)
            # Reconstruct distributions
            for idx, sample_id in enumerate(test_ids):
                df = dist_test[sample_id]
                coeff_pred = coeffs_pred[idx]
                coeff_struct = coeffs_structs_test[idx]
                grid_shape = grid_shapes_test[idx]
                # Fill missing coefficients with zeros for those not predicted
                coeff_struct = coeff_struct.copy()
                # Flatten predicted coefficients into structure
                # Here we assume we are only predicting the first coefficients; we replace the corresponding values
                # into the coeff_struct copy
                # Flatten full coefficient structure for replacement
                flat_list = []
                for i, c in enumerate(coeff_struct):
                    if i == 0:
                        flat_list.append(c.flatten())
                    else:
                        for arr in c:
                            flat_list.append(arr.flatten())
                full_flat = np.concatenate(flat_list)
                n_pred = len(coeff_pred)
                full_flat[:n_pred] = coeff_pred
                # Rebuild coeff_struct from flat
                pos = 0
                new_coeff_struct = []
                for i, c in enumerate(coeff_struct):
                    if i == 0:
                        shape = c.shape
                        size = np.prod(shape)
                        new_cA = full_flat[pos:pos + size].reshape(shape)
                        pos += size
                        new_coeff_struct.append(new_cA)
                    else:
                        sublist = []
                        for arr in c:
                            shape = arr.shape
                            size = np.prod(shape)
                            new_arr = full_flat[pos:pos + size].reshape(shape)
                            pos += size
                            sublist.append(new_arr)
                        new_coeff_struct.append(tuple(sublist))
                x_coords = df["x"].values
                y_coords = df["y"].values
                f_true = df["f"].values
                F_recon = wavelet_reconstruct(new_coeff_struct, wavelet_name)
                # Map reconstructed grid to original scattered points via interpolation
                # We assume grid values correspond to unique sorted x and y
                x_unique = np.unique(x_coords)
                y_unique = np.unique(y_coords)
                # Some shapes may differ due to boundary effects; we clip
                F_recon = F_recon[: len(y_unique), : len(x_unique)]
                Xg, Yg = np.meshgrid(x_unique, y_unique)
                grid_points = np.column_stack((Xg.ravel(), Yg.ravel()))
                f_grid = F_recon.ravel()
                # Interpolate grid predictions back onto the original scattered coordinates
                f_recon = griddata(grid_points, f_grid, (x_coords, y_coords), method="linear")
                if np.any(np.isnan(f_recon)):
                    f_nn = griddata(grid_points, f_grid, (x_coords, y_coords), method="nearest")
                    f_recon = np.where(np.isnan(f_recon), f_nn, f_recon)
                y_true_all.append(f_true)
                y_pred_all.append(f_recon)
                heatmap_meta = _extract_heatmap_metadata(df)
                if sample_id in sample_ids_set:
                    try:
                        r2_sample = compute_r2(f_true, f_recon)
                    except Exception:
                        r2_sample = None
                    generate_sample_outputs(
                        sample_id,
                        "Wavelet",
                        method_out_dir,
                        x_coords,
                        y_coords,
                        f_true,
                        f_recon,
                        r2_sample,
                        coeff_true=coeffs_test[idx],
                        coeff_pred=coeffs_pred[idx],
                        polar_meta=heatmap_meta,
                        heatmap_cfg=heatmap_cfg,
                    )
            # Flatten metrics
            y_true_flat = np.concatenate(y_true_all)
            y_pred_flat = np.concatenate(y_pred_all)
            r2_dist = compute_r2(y_true_flat, y_pred_flat)
            rmse_dist = compute_rmse(y_true_flat, y_pred_flat)
            mae_dist = compute_mae(y_true_flat, y_pred_flat)
            save_dicts_to_csv(
                [
                    {
                        "r2_coefficients": r2_coeffs,
                        "rmse_coefficients": rmse_coeffs,
                        "mae_coefficients": mae_coeffs,
                        "r2_distribution": r2_dist,
                        "rmse_distribution": rmse_dist,
                        "mae_distribution": mae_dist,
                    }
                ],
                os.path.join(method_out_dir, "metrics.csv"),
            )
            plot_scatter(
                coeffs_test.flatten(),
                coeffs_pred.flatten(),
                title="Wavelet Coefficients", outfile=os.path.join(method_out_dir, "coefficients_scatter.png"), r2=r2_coeffs,
            )
            plot_scatter(
                y_true_flat,
                y_pred_flat,
                title="Wavelet Distribution", outfile=os.path.join(method_out_dir, "distribution_scatter.png"), r2=r2_dist,
            )
            import joblib
            # For wavelet we save regression model and decomposition parameters
            joblib.dump({
                "model": model,
                "wavelet": wavelet_name,
                "level": level,
                "n_coeffs": n_coeffs_select,
                "preprocessor": preprocessor,
            }, os.path.join(method_out_dir, "model.pkl"))
            metric_hist[method_name] = {
                "r2": r2_dist,
                "rmse": rmse_dist,
                "mae": mae_dist,
            }

        elif method_name == "gpr":
            # Direct GPR on flattened distribution
            # Flatten distributions into vectors
            # Determine number of grid points (assuming same coordinates for all)
            first_df = next(iter(dist_train.values()))
            x_coords_ref = first_df["x"].values
            y_coords_ref = first_df["y"].values
            n_points = len(x_coords_ref)
            # Build y matrices for train and test
            y_train = []
            for df in dist_train.values():
                f = df["f"].values
                y_train.append(f)
            y_train = np.array(y_train)
            y_test = []
            for sample_id in test_ids:
                df = dist_test[sample_id]
                f = df["f"].values
                y_test.append(f)
            y_test = np.array(y_test)
            # Build GPR model
            kernel = params.get("kernel", "RBF")
            length_scale = params.get("length_scale", 1.0)
            alpha = params.get("alpha", 1e-10)
            scale_y = params.get("scale_y", True)
            optimize_hyperparams = params.get("optimize_hyperparams", False)
            gpr_model = GPRModel(
                kernel_name=kernel,
                length_scale=length_scale,
                alpha=alpha,
                scale_y=scale_y,
                optimize_hyperparams=optimize_hyperparams,
            )
            gpr_model.fit(X_train, y_train)
            y_pred = gpr_model.predict(X_test)
            # Evaluate
            r2_dist, rmse_dist, mae_dist = gpr_model.evaluate(X_test, y_test)
            # Save metrics
            save_dicts_to_csv(
                [
                    {
                        "r2_distribution": r2_dist,
                        "rmse_distribution": rmse_dist,
                        "mae_distribution": mae_dist,
                    }
                ],
                os.path.join(method_out_dir, "metrics.csv"),
            )
            # Scatter plot flatten
            plot_scatter(
                y_test.flatten(),
                y_pred.flatten(),
                title="GPR Distribution", outfile=os.path.join(method_out_dir, "distribution_scatter.png"), r2=r2_dist,
            )
            # Compare first few distributions visually
            for idx, sample_id in enumerate(test_ids):
                df = dist_test[sample_id]
                f_true = df["f"].values
                f_pred = y_pred[idx]
                if sample_id in sample_ids_set:
                    x_coords = df["x"].values
                    y_coords = df["y"].values
                    heatmap_meta = _extract_heatmap_metadata(df)
                    try:
                        r2_sample = compute_r2(f_true, f_pred)
                    except Exception:
                        r2_sample = None
                    generate_sample_outputs(
                        sample_id,
                        "GPR",
                        method_out_dir,
                        x_coords,
                        y_coords,
                        f_true,
                        f_pred,
                        r2_sample,
                        polar_meta=heatmap_meta,
                        heatmap_cfg=heatmap_cfg,
                    )
            import joblib
            joblib.dump({
                "model": gpr_model,
                "preprocessor": preprocessor,
            }, os.path.join(method_out_dir, "model.pkl"))
            metric_hist[method_name] = {
                "r2": r2_dist,
                "rmse": rmse_dist,
                "mae": mae_dist,
            }

        elif method_name == "zernike_gpr":
            # Combined Zernike + GPR model
            max_order = params.get("max_order", 4)
            kernel = params.get("kernel", "RBF")
            length_scale = params.get("length_scale", 1.0)
            alpha = params.get("alpha", 1e-10)
            scale_y = params.get("scale_y", True)
            optimize_hyperparams = params.get("optimize_hyperparams", False)
            coeffs_train_matrix = []
            for df in dist_train.values():
                x = df["x"].values
                y = df["y"].values
                f = df["f"].values
                c, _ = fit_zernike(x, y, f, max_order)
                coeffs_train_matrix.append(c)
            coeffs_train_matrix = np.array(coeffs_train_matrix)
            # Build model
            zgpr = ZernikeGPRModel(
                max_order=max_order,
                gpr_kernel_name=kernel,
                gpr_length_scale=length_scale,
                gpr_alpha=alpha,
                gpr_scale_y=scale_y,
                gpr_optimize_hyperparams=optimize_hyperparams,
            )
            zgpr.fit(X_train, dist_train)
            if uncertainty_cfg.get("enabled", False):
                train_mean_coeffs, train_std_coeffs = zgpr.predict_coefficients(X_train)
                report = generate_uncertainty_diagnostics(
                    "Zernike GPR",
                    method_out_dir,
                    train_mean_coeffs,
                    train_std_coeffs,
                    coeffs_train_matrix,
                    cond_train,
                    uncertainty_cfg,
                )
                if report:
                    uncertainty_reports.append(report)
            # Evaluate on test
            # Compute Zernike coefficients for test distributions for ground truth
            coeffs_true = []
            nm_pairs_ref = None
            for sample_id in test_ids:
                df = dist_test[sample_id]
                x = df["x"].values
                y = df["y"].values
                f = df["f"].values
                polar_kwargs = _extract_polar_kwargs(df)
                c, nm_pairs_curr = fit_zernike(x, y, f, max_order, **polar_kwargs)
                coeffs_true.append(c)
                if nm_pairs_ref is None:
                    nm_pairs_ref = nm_pairs_curr
            coeffs_true = np.array(coeffs_true)
            # Predict coefficients and std
            mean_coeffs, std_coeffs = zgpr.predict_coefficients(X_test)
            r2_coeffs = compute_r2(coeffs_true, mean_coeffs)
            rmse_coeffs = compute_rmse(coeffs_true, mean_coeffs)
            mae_coeffs = compute_mae(coeffs_true, mean_coeffs)
            # Predict full distributions with confidence intervals
            preds = zgpr.predict_distribution(X_test, dist_train)
            # Evaluate distribution predictions
            for idx, sample_id in enumerate(test_ids):
                df = dist_test[sample_id]
                pred_dict = preds[str(idx)]
                x_coords = df["x"].values
                y_coords = df["y"].values
                f_true = df["f"].values
                f_pred = pred_dict["f_pred"]
                y_true_all.append(f_true)
                y_pred_all.append(f_pred)
                if sample_id in sample_ids_set:
                    heatmap_meta = _extract_heatmap_metadata(df)
                    try:
                        r2_sample = compute_r2(f_true, f_pred)
                    except Exception:
                        r2_sample = None
                    generate_sample_outputs(
                        sample_id,
                        "Zernike GPR",
                        method_out_dir,
                        x_coords,
                        y_coords,
                        f_true,
                        f_pred,
                        r2_sample,
                        coeff_true=coeffs_true[idx],
                        coeff_pred=mean_coeffs[idx],
                        polar_meta=heatmap_meta,
                        heatmap_cfg=heatmap_cfg,
                    )
                    plot_confidence_intervals(
                        pred_dict,
                        os.path.join(method_out_dir, "samples", sample_id, "confidence.png"),
                        title=f"Zernike GPR Confidence - {sample_id}",
                    )
            y_true_flat = np.concatenate(y_true_all)
            y_pred_flat = np.concatenate(y_pred_all)
            r2_dist = compute_r2(y_true_flat, y_pred_flat)
            rmse_dist = compute_rmse(y_true_flat, y_pred_flat)
            mae_dist = compute_mae(y_true_flat, y_pred_flat)
            save_dicts_to_csv(
                [
                    {
                        "r2_coefficients": r2_coeffs,
                        "rmse_coefficients": rmse_coeffs,
                        "mae_coefficients": mae_coeffs,
                        "r2_distribution": r2_dist,
                        "rmse_distribution": rmse_dist,
                        "mae_distribution": mae_dist,
                    }
                ],
                os.path.join(method_out_dir, "metrics.csv"),
            )
            plot_scatter(
                coeffs_true.flatten(),
                mean_coeffs.flatten(),
                title="Zernike GPR: Coefficients", outfile=os.path.join(method_out_dir, "coefficients_scatter.png"), r2=r2_coeffs,
            )
            plot_scatter(
                y_true_flat,
                y_pred_flat,
                title="Zernike GPR: Distribution", outfile=os.path.join(method_out_dir, "distribution_scatter.png"), r2=r2_dist,
            )
            import joblib
            # Save hybrid model
            joblib.dump({
                "model": zgpr,
                "preprocessor": preprocessor,
            }, os.path.join(method_out_dir, "model.pkl"))
            metric_hist[method_name] = {
                "r2": r2_dist,
                "rmse": rmse_dist,
                "mae": mae_dist,
            }

        else:
            print(f"Unknown method '{method_name}', skipping.")
            continue

    # Aggregate uncertainty reports across methods if available
    if uncertainty_reports:
        summary_dir = os.path.join(out_dir, "uncertainty_summary")
        ensure_dir(summary_dir)
        combined_frames: List[pd.DataFrame] = []
        for report in uncertainty_reports:
            stats_path = report.get("stats_csv")
            if stats_path and os.path.exists(stats_path):
                df_stats = pd.read_csv(stats_path)
                df_stats["method"] = report["method"]
                combined_frames.append(df_stats)
        if combined_frames:
            combined_df = pd.concat(combined_frames, ignore_index=True)
            combined_csv = os.path.join(summary_dir, "coefficient_uncertainty_summary.csv")
            combined_df.to_csv(combined_csv, index=False)

            overview = (
                combined_df.groupby("method")
                [["mean_std", "mean_variance", "relative_uncertainty_mean"]]
                .mean()
                .reset_index()
            )
            plt.figure(figsize=(8, 4))
            x_pos = np.arange(len(overview))
            width = 0.25
            plt.bar(x_pos - width, overview["mean_std"], width=width, label="mean_std")
            plt.bar(x_pos, overview["mean_variance"], width=width, label="mean_variance")
            plt.bar(x_pos + width, overview["relative_uncertainty_mean"], width=width, label="relative_uncertainty_mean")
            plt.xticks(x_pos, overview["method"], rotation=20, ha="right")
            plt.ylabel("Average Metric")
            plt.title("Coefficient Uncertainty Overview")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(summary_dir, "uncertainty_overview.png"))
            plt.close()

    # Create histogram plots across methods
    if metric_hist:
        for metric_key in ["r2", "rmse", "mae"]:
            values = {m: metric_hist[m][metric_key] for m in metric_hist}
            plot_histogram(
                values,
                metric_name=metric_key.upper(),
                outfile=os.path.join(out_dir, f"metrics_{metric_key}.png"),
                title=f"Comparison of {metric_key.upper()} across methods",
            )


if __name__ == "__main__":
    main()
