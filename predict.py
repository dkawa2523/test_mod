#!/usr/bin/env python3
"""
Inference script for wafer distribution models
--------------------------------------------

This script loads trained models saved by `train.py` and applies them to
new process conditions to predict spatial distributions. The models,
preprocessors and any method‑specific metadata are loaded from pickled
objects in the specified models directory. Results are saved as CSV files
and visualisations.

Usage:
    python predict.py --config_pred path/to/config_pred.yaml

The prediction configuration YAML should specify the methods to run,
paths to the new conditions file, a distribution template for
reconstruction coordinates and the directory containing the trained models.
"""

import argparse
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import joblib

from wafer_ml.config import load_config
from wafer_ml.data_loader import load_conditions, load_distribution
from wafer_ml.preprocessing import Preprocessor
from wafer_ml.features.zernike import reconstruct_zernike
from wafer_ml.features.legendre import reconstruct_legendre
from wafer_ml.features.rbf import reconstruct_rbf
from wafer_ml.features.wavelet import wavelet_decompose, wavelet_reconstruct
from wafer_ml.models.gpr import GPRModel
from wafer_ml.models.zernike_gpr import ZernikeGPRModel
from wafer_ml.visualization import (
    plot_distribution_comparison,
    plot_confidence_intervals,
)
from wafer_ml.utils import ensure_dir, save_dicts_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wafer ML inference script")
    parser.add_argument(
        "--config_pred",
        required=True,
        type=str,
        help="Path to YAML prediction configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_pred = load_config(args.config_pred)
    # Load new conditions
    pred_data_cfg = cfg_pred.data
    cond_file = pred_data_cfg.get("conditions_file")
    id_column = pred_data_cfg.get("id_column", "id")
    if cond_file is None:
        raise ValueError("Prediction config must specify 'conditions_file'")
    conditions_df = load_conditions(cond_file, id_column=id_column)
    ids = conditions_df.index.values
    # Load template distribution for coordinates
    template_file = pred_data_cfg.get("distribution_template")
    if template_file is None:
        raise ValueError("Prediction config must specify 'distribution_template' (a CSV file)")
    template_df = load_distribution(template_file)
    x_coords = template_df["x"].values
    y_coords = template_df["y"].values
    # Output directory
    out_dir = cfg_pred.output.get("directory", "predictions")
    ensure_dir(out_dir)
    # Directory containing trained models
    models_dir = cfg_pred.get("models_dir")
    if models_dir is None:
        raise ValueError("Prediction config must specify 'models_dir'")
    # Preallocate results summary
    summary_records: List[Dict[str, Any]] = []
    # For each method
    for method_name, params in cfg_pred.methods.items():
        if not params.get("enabled", False):
            continue
        method_model_dir = os.path.join(models_dir, method_name)
        model_path = os.path.join(method_model_dir, "model.pkl")
        if not os.path.exists(model_path):
            print(f"Model for method '{method_name}' not found at '{model_path}', skipping.")
            continue
        # Load model object
        model_bundle = joblib.load(model_path)
        preprocessor: Preprocessor = model_bundle.get("preprocessor")
        X_new = preprocessor.transform(conditions_df)
        # Prepare output subdirectory
        method_out_dir = os.path.join(out_dir, method_name)
        ensure_dir(method_out_dir)
        print(f"Predicting with method '{method_name}'...")
        if method_name == "zernike_linear":
            lr_model = model_bundle["model"]
            nm_pairs = model_bundle["nm_pairs"]
            max_order = model_bundle["max_order"]
            polar_metadata = model_bundle.get("polar_metadata", {})
            polar_kwargs = {}
            if polar_metadata.get("center") is not None:
                polar_kwargs["center"] = tuple(polar_metadata["center"])
            if polar_metadata.get("radius_scale") is not None:
                polar_kwargs["radius_scale"] = float(polar_metadata["radius_scale"])
            # Predict coefficients
            coeffs_pred = lr_model.predict(X_new)
            for idx, coeff_vec in enumerate(coeffs_pred):
                f_pred = reconstruct_zernike(
                    x_coords,
                    y_coords,
                    coeff_vec,
                    nm_pairs,
                    **polar_kwargs,
                )
                # Save prediction
                df_out = pd.DataFrame({"x": x_coords, "y": y_coords, "f_pred": f_pred})
                pred_file = os.path.join(method_out_dir, f"sample_{ids[idx]}_prediction.csv")
                df_out.to_csv(pred_file, index=False)
                # Save plot
                plot_distribution_comparison(
                    x_coords,
                    y_coords,
                    f_pred,
                    f_pred,
                    os.path.join(method_out_dir, f"sample_{ids[idx]}_plot.png"),
                    title=f"Zernike Linear Prediction (ID {ids[idx]})",
                )
        elif method_name == "legendre":
            lr_model = model_bundle["model"]
            indices = model_bundle["indices"]
            max_deg_x = model_bundle["max_degree_x"]
            max_deg_y = model_bundle["max_degree_y"]
            coeffs_pred = lr_model.predict(X_new)
            for idx, coeff_vec in enumerate(coeffs_pred):
                f_pred = reconstruct_legendre(x_coords, y_coords, coeff_vec, indices)
                df_out = pd.DataFrame({"x": x_coords, "y": y_coords, "f_pred": f_pred})
                pred_file = os.path.join(method_out_dir, f"sample_{ids[idx]}_prediction.csv")
                df_out.to_csv(pred_file, index=False)
                plot_distribution_comparison(
                    x_coords,
                    y_coords,
                    f_pred,
                    f_pred,
                    os.path.join(method_out_dir, f"sample_{ids[idx]}_plot.png"),
                    title=f"Legendre Prediction (ID {ids[idx]})",
                )
        elif method_name == "rbf":
            model = model_bundle["model"]
            centers = model_bundle["centers"]
            gamma = model_bundle["gamma"]
            coord_mean = model_bundle.get("coord_mean")
            coord_std = model_bundle.get("coord_std")
            if coord_mean is not None:
                coord_mean = np.asarray(coord_mean, dtype=float)
            if coord_std is not None:
                coord_std = np.asarray(coord_std, dtype=float)
            weights_pred = model.predict(X_new)
            for idx, weights in enumerate(weights_pred):
                f_pred = reconstruct_rbf(
                    x_coords,
                    y_coords,
                    weights,
                    centers,
                    gamma,
                    coord_mean=coord_mean,
                    coord_std=coord_std,
                )
                df_out = pd.DataFrame({"x": x_coords, "y": y_coords, "f_pred": f_pred})
                pred_file = os.path.join(method_out_dir, f"sample_{ids[idx]}_prediction.csv")
                df_out.to_csv(pred_file, index=False)
                plot_distribution_comparison(
                    x_coords,
                    y_coords,
                    f_pred,
                    f_pred,
                    os.path.join(method_out_dir, f"sample_{ids[idx]}_plot.png"),
                    title=f"RBF Prediction (ID {ids[idx]})",
                )
        elif method_name == "wavelet":
            model = model_bundle["model"]
            wavelet_name = model_bundle["wavelet"]
            level = model_bundle["level"]
            n_coeffs = model_bundle["n_coeffs"]
            # Compute template coefficient structure and grid
            coeffs_flat_template, coeffs_struct_template, grid_shape = wavelet_decompose(
                x_coords, y_coords, template_df["f"].values, wavelet_name, level
            )
            # Predict coefficient vectors
            coeffs_pred = model.predict(X_new)
            for idx, coeff_vec in enumerate(coeffs_pred):
                coeffs_struct = coeffs_struct_template.copy()
                # Fill predicted coeffs into flattened vector
                flat_list = []
                for i, c in enumerate(coeffs_struct):
                    if i == 0:
                        flat_list.append(c.flatten())
                    else:
                        for arr in c:
                            flat_list.append(arr.flatten())
                full_flat = np.concatenate(flat_list)
                # Replace first n_coeffs values
                n_pred = len(coeff_vec)
                full_flat[:n_pred] = coeff_vec
                # Rebuild coeff structure
                pos = 0
                new_struct = []
                for i, c in enumerate(coeffs_struct):
                    if i == 0:
                        shape = c.shape
                        size = np.prod(shape)
                        new_cA = full_flat[pos:pos + size].reshape(shape)
                        pos += size
                        new_struct.append(new_cA)
                    else:
                        sublist = []
                        for arr in c:
                            shape = arr.shape
                            size = np.prod(shape)
                            new_arr = full_flat[pos:pos + size].reshape(shape)
                            pos += size
                            sublist.append(new_arr)
                        new_struct.append(tuple(sublist))
                # Reconstruct on grid
                F_recon = wavelet_reconstruct(new_struct, wavelet_name)
                # Map to x,y; assume x_unique,y_unique
                x_unique = np.unique(x_coords)
                y_unique = np.unique(y_coords)
                F_recon = F_recon[: len(y_unique), : len(x_unique)]
                f_pred = F_recon.flatten()
                df_out = pd.DataFrame({"x": x_coords, "y": y_coords, "f_pred": f_pred})
                pred_file = os.path.join(method_out_dir, f"sample_{ids[idx]}_prediction.csv")
                df_out.to_csv(pred_file, index=False)
                plot_distribution_comparison(
                    x_coords,
                    y_coords,
                    f_pred,
                    f_pred,
                    os.path.join(method_out_dir, f"sample_{ids[idx]}_plot.png"),
                    title=f"Wavelet Prediction (ID {ids[idx]})",
                )
        elif method_name == "gpr":
            gpr_model = model_bundle["model"]
            y_pred, _ = gpr_model.predict_with_std(X_new)
            # Each row corresponds to predicted distribution values
            for idx, f_pred in enumerate(y_pred):
                df_out = pd.DataFrame({"x": x_coords, "y": y_coords, "f_pred": f_pred})
                pred_file = os.path.join(method_out_dir, f"sample_{ids[idx]}_prediction.csv")
                df_out.to_csv(pred_file, index=False)
                plot_distribution_comparison(
                    x_coords,
                    y_coords,
                    f_pred,
                    f_pred,
                    os.path.join(method_out_dir, f"sample_{ids[idx]}_plot.png"),
                    title=f"GPR Prediction (ID {ids[idx]})",
                )
        elif method_name == "zernike_gpr":
            zgpr_model: ZernikeGPRModel = model_bundle["model"]
            # Use preprocessor from bundle to transform
            mean_coeffs, std_coeffs = zgpr_model.predict_coefficients(X_new)
            preds = zgpr_model.predict_distribution(X_new, {"template": template_df})
            # Save each prediction with confidence intervals
            for idx, key in enumerate(preds.keys()):
                pred_dict = preds[key]
                df_out = pd.DataFrame(
                    {
                        "x": pred_dict["x"],
                        "y": pred_dict["y"],
                        "f_pred": pred_dict["f_pred"],
                        "lower": pred_dict.get("lower", np.nan),
                        "upper": pred_dict.get("upper", np.nan),
                    }
                )
                pred_file = os.path.join(method_out_dir, f"sample_{ids[idx]}_prediction.csv")
                df_out.to_csv(pred_file, index=False)
                # Plot confidence intervals
                plot_confidence_intervals(
                    pred_dict,
                    os.path.join(method_out_dir, f"sample_{ids[idx]}_plot.png"),
                    title=f"Zernike GPR Prediction (ID {ids[idx]})",
                )
        else:
            print(f"Unknown method '{method_name}' in prediction config, skipping.")
            continue

    # Optionally save summary of predictions (IDs and files)
    # Save summary to CSV
    # Not implemented, reserved for future


if __name__ == "__main__":
    main()
