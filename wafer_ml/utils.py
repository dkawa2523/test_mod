"""
Utility functions
-----------------

This module provides miscellaneous utility functions such as saving
structured data to CSV files, ensuring directories exist and computing
residual statistics. These helpers centralise repetitive tasks used across
the training and inference pipelines.
"""

import os
from typing import Dict, Iterable, Any
import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)


def save_array_to_csv(array: np.ndarray, header: Iterable[str], outfile: str) -> None:
    """
    Save a 2D array to a CSV file with a header.
    """
    ensure_dir(os.path.dirname(outfile))
    df = pd.DataFrame(array, columns=list(header))
    df.to_csv(outfile, index=False)


def save_dicts_to_csv(records: Iterable[Dict[str, Any]], outfile: str) -> None:
    """
    Save a list of dictionaries to a CSV file.
    """
    ensure_dir(os.path.dirname(outfile))
    df = pd.DataFrame(list(records))
    df.to_csv(outfile, index=False)


def compute_residual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the mean absolute residual between true and predicted values.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def save_coefficients_csv(true: np.ndarray, pred: np.ndarray, outfile: str) -> None:
    """Persist true/predicted coefficients and residuals to CSV."""
    true_arr = np.asarray(true).ravel()
    pred_arr = np.asarray(pred).ravel()
    if true_arr.shape != pred_arr.shape:
        raise ValueError("Coefficient arrays must have the same shape")
    residual = true_arr - pred_arr
    ensure_dir(os.path.dirname(outfile))
    df = pd.DataFrame(
        {
            "index": np.arange(true_arr.size, dtype=int),
            "true": true_arr,
            "pred": pred_arr,
            "residual": residual,
        }
    )
    df.to_csv(outfile, index=False)
