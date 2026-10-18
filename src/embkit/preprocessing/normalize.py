"""
Normalization of expression / count matrices.

Before an embedding model can learn a meaningful representation of, say, gene
expression or mutation-count data, the input matrix is almost always dominated by
a handful of high-value features. Normalization rescales the matrix so that every
feature contributes on a comparable scale and the result fits in ``[0, 1]``.

This module provides the transformations embkit applies at that step:

- :func:`quantile_max_norm` — the workhorse "quantile max" normalization used by
  the ``embkit matrix normalize`` command: each sample is divided by a chosen
  quantile of its own values, then clipped to ``[0, 1]``. Robust to outliers
  because it is quantile-based rather than max-based.
- :func:`exp_max_norm` and :class:`ExpMinMaxScaler` — log2-based alternatives for
  data where the dynamic range spans many orders of magnitude.
- :func:`get_dataset_nonzero_mask` — inspects a dataset to find which features
  are meaningfully populated, producing the binary feature masks used downstream.

All functions operate on pandas DataFrames (or a torch :class:`Dataset` for the
mask helper) and return the transformed data as the same type, so they compose
cleanly with the rest of the pipeline.
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
from typing import List
from torch.utils.data import Dataset
import torch

def quantile_max_norm(df: pd.DataFrame, quantile_max: float = 0.9) -> pd.DataFrame:
    """
    Normalizes the DataFrame using Quantile Max normalization.

    Parameters:
        df (DataFrame): The input DataFrame to be normalized.
        quantile_max (float, optional): The maximum quantile value to use for normalization. Default is 0.9.

    Returns:
        DataFrame: The normalized DataFrame.
    """
    norm_df = (df.transpose() / df.quantile(quantile_max, axis=1)).transpose().clip(upper=1.0, lower=0.0).fillna(0.0)
    return norm_df

def exp_max_norm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log2, max-based normalization of an expression matrix.

    Applies a ``log2(x + 1)`` transform to tame the wide dynamic range of
    count/expression data, then divides each sample by its own maximum so that
    the values for every row fall in ``[0, 1]`` (with the sample's largest
    feature rescaled to exactly 1).

    Compared to :func:`quantile_max_norm`, this is max-based rather than
    quantile-based, so a single extreme feature will compress the rest of the
    sample more aggressively. Use it when you want the relative spread preserved
    after a log transform rather than robustness to a single outlier.

    Parameters:
        df: Input DataFrame of shape ``(samples, features)``.

    Returns:
        A DataFrame of the same shape with each sample scaled to ``[0, 1]``.
    """
    t = np.log2(df+1)
    return (t.transpose() / t.max(axis=1)).transpose()

class ExpMinMaxScaler(MinMaxScaler, BaseEstimator):
    """
    A custom Min-Max scaler that applies the log2 transformation before scaling.

    Inherits from sklearn's MinMaxScaler and implements a fit method that
    transforms the input data using the formula np.log2(X+1) before fitting the
    MinMaxScaler to it. This allows for normalization of data with different scales,
    especially when dealing with non-negative values where logarithmic transformation can help.
    """
    def fit(self, X: pd.DataFrame) -> "ExpMinMaxScaler":
        return MinMaxScaler.fit(self, np.log2(X+1))
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return MinMaxScaler.transform(self, np.log2(X+1))
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.exp2(MinMaxScaler.inverse_transform(self, X))-1

def get_dataset_nonzero_mask(d: Dataset, threshold: float) -> List[torch.Tensor]:
    """
    Scan a torch Dataset, identify the elements that are non-zero above the threshold
    percentage. This returns a list of binary masks (one per feature tensor in a row)
    that select those columns.
    """

    s = None
    count = 0
    for row in d:
        if s is None:
            s = []
            for f in row:
                s.append(torch.zeros(f.shape, dtype=torch.int32, device=f.device))
        for i, f in enumerate(row):
            s[i] = s[i] + (f == 0.0).int()
        count += 1
    out = []
    for v in s:
        mask = (v.int() / count) < threshold
        out.append(mask)
    return out
