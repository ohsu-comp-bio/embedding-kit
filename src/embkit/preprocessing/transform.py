"""
Assay-specific transforms for raw count matrices.

Different molecular assays need different preparation before they make good
model inputs. These are the standard single-cell transforms, written for plain
``(samples, features)`` pandas DataFrames (or arrays) so they compose with the
rest of embkit without requiring AnnData or Scanpy:

- :func:`library_size_norm` — RNA: scale each sample to a common total count,
  then ``log1p``. Removes sequencing-depth differences.
- :func:`clr_norm` — surface protein (ADT/CITE-seq): per-sample centred
  log-ratio of ``log1p`` counts. Removes per-cell staining differences.
- :class:`TfidfLSI` — chromatin accessibility (ATAC): TF-IDF weighting
  followed by truncated SVD (latent semantic indexing), fitted on training
  samples and applied to any others.

All functions validate that inputs are finite, non-negative raw counts and
return the same container type they were given.

Ported from UniVII (``univii.preprocessing``).
"""
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler, normalize

MatrixLike = Union[pd.DataFrame, np.ndarray]


def _check_counts(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D (samples, features) matrix, got shape {x.shape}")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("expected finite non-negative raw counts")
    return x


def _wrap(values: np.ndarray, like: MatrixLike, columns=None) -> MatrixLike:
    if isinstance(like, pd.DataFrame):
        return pd.DataFrame(values, index=like.index, columns=like.columns if columns is None else columns)
    return values


def library_size_norm(df: MatrixLike, target_sum: float = 1e4, log1p: bool = True) -> MatrixLike:
    """
    Scale each sample to ``target_sum`` total counts, then optionally ``log1p``.

    Args:
        df: Raw counts, ``(samples, features)``.
        target_sum: Total count each sample is scaled to (1e4 is the common
            "counts per 10k" convention).
        log1p: Apply ``log(1 + x)`` after scaling.

    Returns:
        Normalized matrix of the same type and shape as ``df``.
    """
    x = _check_counts(df)
    totals = x.sum(axis=1, keepdims=True)
    out = x * (float(target_sum) / np.maximum(totals, 1e-12))
    if log1p:
        out = np.log1p(out)
    return _wrap(out, df)


def clr_norm(df: MatrixLike) -> MatrixLike:
    """
    Per-sample centred log-ratio of ``log1p`` counts (the usual ADT transform).

    Each sample's ``log1p`` values are centred on their own mean, so a global
    shift in staining intensity for one cell does not change its profile.

    Args:
        df: Raw (not already CLR-transformed) abundances, ``(samples, features)``.

    Returns:
        CLR-transformed matrix of the same type and shape as ``df``.
    """
    values = np.log1p(_check_counts(df))
    return _wrap(values - values.mean(axis=1, keepdims=True), df)


class TfidfLSI(BaseEstimator, TransformerMixin):
    """
    TF-IDF + truncated SVD (LSI) for peak/bin count matrices, fitted on training samples.

    Three TF-IDF conventions are available:

    - ``"sklearn"`` — scikit-learn's ``TfidfTransformer`` (L2-normalized, smooth IDF).
    - ``"signac"`` — ``log1p(TF * IDF * 1e4)`` with ``IDF = log1p(n / (1 + df))``,
      then L2 normalization.
    - ``"tea"`` — L1 TF times ``log1p(n / (1 + df))``, then L2 normalization.

    Args:
        n_components: Number of SVD components fitted *before* any is dropped.
        method: TF-IDF convention (see above).
        drop_first: Drop the first LSI component, which usually tracks
            sequencing depth rather than biology.
        scale: Standardize the LSI components with a training-fitted scaler.
        min_fraction: Keep features detected in at least this fraction of training samples.
        max_fraction: Keep features detected in at most this fraction of training samples.
        random_state: Seed for the randomized SVD.

    After :meth:`fit`, ``features_`` holds the boolean mask of retained input columns.
    """

    def __init__(self, n_components: int = 100, method: str = "sklearn", drop_first: bool = False,
                 scale: bool = True, min_fraction: float = 0.0, max_fraction: float = 1.0,
                 random_state: Optional[int] = 42):
        self.n_components = n_components
        self.method = method
        self.drop_first = drop_first
        self.scale = scale
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.random_state = random_state

    def _tfidf(self, x: sp.csr_matrix) -> sp.csr_matrix:
        if self.method == "sklearn":
            return self.tfidf_.transform(x)
        tf = normalize(x, norm="l1", axis=1).multiply(self.idf_).tocsr()
        if self.method == "signac":
            tf.data = np.log1p(tf.data * 1e4)
        return normalize(tf, norm="l2", axis=1)

    def fit(self, X: MatrixLike, y=None) -> "TfidfLSI":
        """Fit feature filter, TF-IDF weights, SVD basis and (optionally) the scaler."""
        if self.method not in {"sklearn", "signac", "tea"}:
            raise ValueError(f"method must be 'sklearn', 'signac' or 'tea', got {self.method!r}")
        if not 0 <= self.min_fraction < self.max_fraction <= 1:
            raise ValueError("require 0 <= min_fraction < max_fraction <= 1")
        x = sp.csr_matrix(_check_counts(X))
        df = np.asarray((x > 0).sum(axis=0)).ravel()
        frequency = df / x.shape[0]
        self.features_ = (frequency >= self.min_fraction) & (frequency <= self.max_fraction) & (df > 0)
        x, df = x[:, self.features_], df[self.features_]
        if not 1 <= int(self.n_components) < min(x.shape):
            raise ValueError("n_components must be smaller than the training sample and retained feature counts")
        if self.method == "sklearn":
            self.tfidf_ = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True).fit(x)
        else:
            self.idf_ = np.log1p(x.shape[0] / (1 + df))
        self.svd_ = TruncatedSVD(n_components=int(self.n_components), random_state=self.random_state)
        z = self.svd_.fit_transform(self._tfidf(x))
        if self.drop_first:
            z = z[:, 1:]
        if z.shape[1] == 0:
            raise ValueError("dropping the first component leaves no LSI components; increase n_components")
        if self.scale:
            self.scaler_ = StandardScaler().fit(z)
        return self

    def transform(self, X: MatrixLike) -> MatrixLike:
        """
        Project samples onto the fitted LSI basis.

        Returns:
            ``(samples, n_components - drop_first)``; a DataFrame with
            ``LSI_<i>`` columns when ``X`` is a DataFrame, otherwise an array.
        """
        if not hasattr(self, "svd_"):
            raise RuntimeError("call fit on training samples first")
        x = sp.csr_matrix(_check_counts(X))
        if x.shape[1] != self.features_.shape[0]:
            raise ValueError(f"expected {self.features_.shape[0]} features, got {x.shape[1]}")
        z = self.svd_.transform(self._tfidf(x[:, self.features_]))
        first = int(self.drop_first)
        z = z[:, first:]
        if self.scale:
            z = self.scaler_.transform(z)
        z = z.astype(np.float32)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(z, index=X.index, columns=[f"LSI_{i}" for i in range(first, int(self.n_components))])
        return z
