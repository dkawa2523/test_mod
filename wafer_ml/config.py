"""
Configuration utilities
----------------------

This module provides helper functions for loading YAML configuration files
that drive the behaviour of the training and inference pipelines. The
configuration files define which methods to run, model hyperparameters,
pre‑processing options, file locations and output settings. Using YAML allows
non‑programmers to adjust parameters without modifying code.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


@dataclass
class Config:
    """Dataclass wrapper around a configuration dictionary."""

    data: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    methods: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    visualization: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def load_config(path: str) -> Config:
    """
    Load a configuration YAML file and return a Config object.

    Parameters
    ----------
    path : str
        Path to a YAML configuration file.

    Returns
    -------
    Config
        A dataclass instance with attributes corresponding to top‑level keys
        in the YAML file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file '{path}' does not exist")
    with open(path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f) or {}
    # Flatten config levels into dataclass
    cfg = Config()
    for key, value in config_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            # Unrecognised top‑level keys are stored under data by default
            cfg.data[key] = value
    return cfg
