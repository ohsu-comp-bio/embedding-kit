"""

"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

def quantile_max_norm(df: pd.DataFrame, quantile_max=0.9):
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

def exp_max_norm(df: pd.DataFrame):
    """
    exp_max_norm

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
    def fit(self, X: pd.DataFrame):
        return MinMaxScaler.fit(self, np.log2(X+1))
    def transform(self, X):
        return MinMaxScaler.transform(self, np.log2(X+1))
    def inverse_transform(self, X):
        return np.exp2(MinMaxScaler.inverse_transform(self, X))-1

def get_dataset_nonzero_mask(d: Dataset, threshold: float) -> torch.Tensor:
    """
    Scan a torch Dataset, identify the elements that are non-zero above the threshold
    percentage. This returns a binary mask that selects those columns.
    """

    s = None
    count = 0
    for row in d:
        if s is None:
            s = []
            for f in row:
                s.append( torch.zeros( len(f) ) )
        for i, f in enumerate(row):
            s[i] = s[i] + (f == 0.0).int()
        count += 1
    out = []
    for v in s:
        mask = (v.int() / count) < threshold
        out.append(mask)
    return out
