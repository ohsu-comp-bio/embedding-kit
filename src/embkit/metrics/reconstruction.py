"""
Per-feature reconstruction metrics and outlier scoring.

A single overall reconstruction loss hides which features a model reproduces
and which it ignores. These metrics break reconstruction quality down per
feature (gene, protein, peak), so you can see, for example, that highly
expressed genes are captured while rare ones are flattened:

- :func:`mse_per_feature` — mean squared error of each feature.
- :func:`pearson_corr_per_feature` — correlation of each feature across samples.
- :func:`auc_per_feature_binary` — ROC AUC of each feature for 0/1 data.
- :func:`reconstruction_metrics` — summary dict of the above.
- :func:`deviation_scores` — robust z-score of per-sample losses against a
  reference ("normal") population, to flag samples the model cannot explain.

Ported from UniVII (``univii.evaluation`` and ``univii.reference``).
"""
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def _check_same_shape(x_true: np.ndarray, x_pred: np.ndarray) -> None:
    if x_true.shape != x_pred.shape:
        raise ValueError(f"x_true and x_pred must have the same shape, got {x_true.shape} vs {x_pred.shape}")


def mse_per_feature(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    """Mean squared error of each feature (column), ``(n_features,)``."""
    x_true, x_pred = np.asarray(x_true), np.asarray(x_pred)
    _check_same_shape(x_true, x_pred)
    return np.mean((x_true - x_pred) ** 2, axis=0)


def pearson_corr_per_feature(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    """
    Pearson correlation of each feature between observed and reconstructed values, ``(n_features,)``.

    Constant features give 0 rather than NaN.
    """
    x_true = np.asarray(x_true, dtype=np.float32)
    x_pred = np.asarray(x_pred, dtype=np.float32)
    _check_same_shape(x_true, x_pred)
    x_true_c = x_true - x_true.mean(axis=0, keepdims=True)
    x_pred_c = x_pred - x_pred.mean(axis=0, keepdims=True)
    num = (x_true_c * x_pred_c).sum(axis=0)
    denom = np.sqrt((x_true_c ** 2).sum(axis=0) * (x_pred_c ** 2).sum(axis=0)) + 1e-8
    return num / denom


def auc_per_feature_binary(x_true01: np.ndarray, x_score: np.ndarray) -> np.ndarray:
    """
    ROC AUC of each feature for binary ground truth, ``(n_features,)``.

    ``x_score`` may be probabilities or logits. Features whose ground truth is
    constant get NaN.
    """
    x_true01, x_score = np.asarray(x_true01), np.asarray(x_score)
    _check_same_shape(x_true01, x_score)
    aucs = np.full(x_true01.shape[1], np.nan, dtype=np.float32)
    for j in range(x_true01.shape[1]):
        y = x_true01[:, j]
        if y.min() == y.max():
            continue
        aucs[j] = float(roc_auc_score(y, x_score[:, j]))
    return aucs


def reconstruction_metrics(x_true: np.ndarray, x_pred: np.ndarray, kind: str = "continuous") -> Dict[str, Any]:
    """
    Summarize reconstruction quality.

    Args:
        x_true: Observed values ``(n_samples, n_features)``.
        x_pred: Reconstructed values, same shape.
        kind: ``"continuous"`` (MSE + Pearson) or ``"binary"`` (AUC).

    Returns:
        For ``"continuous"``: ``mse_mean``, ``mse_median``, ``pearson_mean``,
        ``pearson_median``, ``mse_per_feature``, ``pearson_per_feature``.
        For ``"binary"``: ``auc_mean``, ``auc_median``, ``auc_per_feature``.

    Raises:
        ValueError: If ``kind`` is not recognised.
    """
    kind = str(kind).lower().strip()
    if kind == "binary":
        aucs = auc_per_feature_binary(x_true, x_pred)
        return {"auc_mean": float(np.nanmean(aucs)), "auc_median": float(np.nanmedian(aucs)),
                "auc_per_feature": aucs}
    if kind != "continuous":
        raise ValueError(f"kind must be 'continuous' or 'binary', got {kind!r}")
    pf_mse = mse_per_feature(x_true, x_pred)
    pf_r = pearson_corr_per_feature(x_true, x_pred)
    return {
        "mse_mean": float(np.mean(pf_mse)),
        "mse_median": float(np.median(pf_mse)),
        "pearson_mean": float(np.mean(pf_r)),
        "pearson_median": float(np.median(pf_r)),
        "mse_per_feature": pf_mse,
        "pearson_per_feature": pf_r,
    }


def deviation_scores(nll_query, nll_null) -> np.ndarray:
    """
    Robust z-score of query losses against a null population.

    ``(x - median(null)) / (1.4826 * MAD(null))``, where the null is, for
    example, reconstruction losses of held-out healthy samples. Large positive
    values mark samples the model reconstructs much worse than normal.

    Raises:
        ValueError: If the null population has zero spread.
    """
    q = np.asarray(nll_query.detach().cpu() if torch.is_tensor(nll_query) else nll_query, dtype=float)
    n = np.asarray(nll_null.detach().cpu() if torch.is_tensor(nll_null) else nll_null, dtype=float)
    med = np.median(n)
    mad = 1.4826 * np.median(np.abs(n - med))
    if not mad > 0:
        raise ValueError("null population has zero spread; cannot standardise")
    return (q - med) / mad
