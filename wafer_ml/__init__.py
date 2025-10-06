"""
wafer_ml
========

This package provides a flexible framework for modelling wafer spatial distribution
data using a variety of signal decomposition and machine learning techniques.
It supports multiple methods such as Zernike polynomial expansion with linear
regression, Legendre polynomial decomposition, radial basis function (RBF)
interpolation/networks, wavelet transform, Gaussian process regression (GPR)
and combined Zernike-GPR modelling. The design emphasises configurability
through YAML files, reproducible pre‑processing, and modular extensibility.

Usage of this package is orchestrated via the top‑level scripts `train.py` and
`predict.py` which load configuration files, perform data loading and
pre‑processing, fit selected models and output evaluation metrics, plots and
CSV files.
"""

from . import config
from . import data_loader
from . import preprocessing
from . import features
from . import models
from . import evaluation
from . import visualization
from . import utils

# Expose common functions for convenience
from .config import load_config
from .data_loader import load_conditions, load_distributions
from .preprocessing import Preprocessor
from .evaluation import compute_r2, compute_rmse

__all__ = [
    "load_config",
    "load_conditions",
    "load_distributions",
    "Preprocessor",
    "compute_r2",
    "compute_rmse",
    "features",
    "models",
    "visualization",
]