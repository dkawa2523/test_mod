"""
Data loading utilities
---------------------

This module provides helper functions to read process condition tables and
distribution data from CSV files. The expectation is that one CSV file
contains rows of process conditions keyed by an identifier column (e.g. "id")
and that for each identifier there exists a corresponding distribution file in
a separate directory. Each distribution file should contain columns (x, y, f)
where x and y are coordinates and f is the measured value at that point.

The loader functions return pandas DataFrames and dictionaries of arrays for
subsequent processing.
"""

import os
from typing import Dict, Iterable, List, Tuple
import pandas as pd


def load_conditions(path: str, id_column: str = "id") -> pd.DataFrame:
    """
    Load the process conditions table from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file containing process conditions.
    id_column : str, optional
        Name of the column identifying each wafer/distribution file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with process conditions indexed by the identifier column.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Conditions file '{path}' does not exist")
    df = pd.read_csv(path)
    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found in conditions file")
    df = df.set_index(id_column)
    return df


def load_distribution(path: str) -> pd.DataFrame:
    """
    Load a single distribution file containing columns (x, y, f).

    Parameters
    ----------
    path : str
        Path to the distribution CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns x, y, f.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Distribution file '{path}' does not exist")
    df = pd.read_csv(path)
    # Validate columns
    required_cols = {"x", "y", "f"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in distribution file '{path}'")
    return df[["x", "y", "f"]].copy()


def load_distributions(directory: str, ids: Iterable[str], extension: str = ".csv") -> Dict[str, pd.DataFrame]:
    """
    Load multiple distribution files given a directory and a list of identifiers.

    Parameters
    ----------
    directory : str
        Path to the directory containing distribution files. Each file should be
        named '<id><extension>'.
    ids : Iterable[str]
        Identifiers for which to load distribution files.
    extension : str, optional
        File extension for distribution files (default: '.csv').

    Returns
    -------
    dict
        Mapping from each identifier to its corresponding distribution DataFrame.
    """
    distributions: Dict[str, pd.DataFrame] = {}
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Distribution directory '{directory}' does not exist")
    for id_ in ids:
        file_path = os.path.join(directory, f"{id_}{extension}")
        try:
            distributions[id_] = load_distribution(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Distribution file for id '{id_}' not found in '{directory}'")
    return distributions