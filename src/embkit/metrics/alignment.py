"""
Metrics for how well two embeddings of the same samples line up.

When the same samples are embedded from two data types (e.g. RNA and protein
from the same cells) into a shared latent space, a good integration puts each
sample's two embeddings next to each other, and mixes the data types locally
rather than keeping them in separate islands. These metrics quantify both:

- :func:`foscttm` — Fraction Of Samples Closer Than the True Match (lower is
  better; 0 = every sample's partner is its nearest neighbour, ~0.5 = random).
- :func:`match_recall_at_k` — fraction of samples whose true partner is among
  their ``k`` nearest cross-view neighbours (higher is better).
- :func:`modality_mixing` — mean fraction of each sample's neighbours that come
  from a *different* view/batch.
- :func:`modality_entropy` — normalized entropy of view/batch labels among each
  sample's neighbours (1 = perfectly mixed).

Every function returns the mean by default. ``return_sem=True`` adds the
standard error of the mean and ``return_per_sample=True`` adds the per-sample
values, in that order.

Ported from UniVII (``univii.evaluation``).
"""
from typing import Tuple, Union

import numpy as np
from sklearn.neighbors import NearestNeighbors

MetricResult = Union[float, Tuple[float, float], Tuple[float, np.ndarray], Tuple[float, float, np.ndarray]]


def _mean_sem(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0, 0.0
    if x.size == 1:
        return float(x.mean()), 0.0
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))


def _package(values: np.ndarray, return_sem: bool, return_per_sample: bool) -> MetricResult:
    """Return the mean, plus SEM and/or the per-sample values when asked."""
    m, s = _mean_sem(values)
    out = [m]
    if return_sem:
        out.append(s)
    if return_per_sample:
        out.append(values)
    return out[0] if len(out) == 1 else tuple(out)


def _cross_distance_blocks(Z1: np.ndarray, Z2: np.ndarray, metric: str, block_size: int):
    """Yield ``(start, stop, distance block)`` of Z1 rows against all Z2 rows."""
    metric = str(metric).lower().strip()
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be 'euclidean' or 'cosine'")
    n = Z1.shape[0]
    Z2_T = Z2.T
    if metric == "euclidean":
        n2 = np.sum(Z2 * Z2, axis=1)
    else:
        n2 = np.linalg.norm(Z2, axis=1) + 1e-8
    for i0 in range(0, n, int(block_size)):
        i1 = min(i0 + int(block_size), n)
        A = Z1[i0:i1]
        if metric == "euclidean":
            d = np.sum(A * A, axis=1)[:, None] + n2[None, :] - 2.0 * (A @ Z2_T)
        else:
            n1 = np.linalg.norm(A, axis=1) + 1e-8
            d = 1.0 - (A @ Z2_T) / (n1[:, None] * n2[None, :])
        yield i0, i1, d


def _paired(Z1, Z2) -> Tuple[np.ndarray, np.ndarray]:
    Z1 = np.asarray(Z1, dtype=np.float32)
    Z2 = np.asarray(Z2, dtype=np.float32)
    if Z1.shape != Z2.shape:
        raise ValueError(f"Z1 and Z2 must have the same shape (row i = same sample), got {Z1.shape} vs {Z2.shape}")
    return Z1, Z2


def foscttm(Z1: np.ndarray, Z2: np.ndarray, metric: str = "euclidean", block_size: int = 512,
            return_sem: bool = False, return_per_sample: bool = False) -> MetricResult:
    """
    Fraction Of Samples Closer Than the True Match.

    For each sample ``i``, the fraction of samples in ``Z2`` that are closer to
    ``Z1[i]`` than its true partner ``Z2[i]``.

    Args:
        Z1: Embedding from view 1, ``(n, d)``.
        Z2: Embedding from view 2, ``(n, d)``; row ``i`` is the same sample as ``Z1[i]``.
        metric: ``"euclidean"`` or ``"cosine"``.
        block_size: Rows processed at once (bounds memory).
        return_sem: Also return the standard error of the mean.
        return_per_sample: Also return the per-sample values.

    Returns:
        Mean FOSCTTM (lower is better), optionally with SEM / per-sample values.
    """
    Z1, Z2 = _paired(Z1, Z2)
    n = int(Z1.shape[0])
    if n <= 1:
        return _package(np.zeros(n, dtype=np.float32), return_sem, return_per_sample)
    fos = np.empty(n, dtype=np.float32)
    for i0, i1, d in _cross_distance_blocks(Z1, Z2, metric, block_size):
        true = d[np.arange(i1 - i0), np.arange(i0, i1)]
        fos[i0:i1] = (d < true[:, None]).sum(axis=1) / (n - 1)
    return _package(fos, return_sem, return_per_sample)


def match_recall_at_k(Z1: np.ndarray, Z2: np.ndarray, k: int = 10, metric: str = "euclidean",
                      block_size: int = 512, return_sem: bool = False,
                      return_per_sample: bool = False) -> MetricResult:
    """
    Fraction of samples whose true partner is among their ``k`` nearest cross-view neighbours.

    Args:
        Z1: Embedding from view 1, ``(n, d)``.
        Z2: Embedding from view 2, ``(n, d)``; row ``i`` is the same sample as ``Z1[i]``.
        k: Neighbourhood size (clipped to ``[1, n]``).
        metric: ``"euclidean"`` or ``"cosine"``.
        block_size: Rows processed at once (bounds memory).
        return_sem: Also return the standard error of the mean.
        return_per_sample: Also return the per-sample hits.

    Returns:
        Mean recall@k (higher is better), optionally with SEM / per-sample hits.

    Raises:
        ValueError: On empty or mismatched inputs.
    """
    Z1, Z2 = _paired(Z1, Z2)
    n = int(Z1.shape[0])
    if n == 0:
        raise ValueError("empty inputs")
    if n == 1:
        return _package(np.array([1.0], dtype=np.float32), return_sem, return_per_sample)
    k = int(max(1, min(int(k), n)))
    hits = np.empty(n, dtype=np.float32)
    for i0, i1, d in _cross_distance_blocks(Z1, Z2, metric, block_size):
        topk = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
        for r in range(i1 - i0):
            hits[i0 + r] = 1.0 if (i0 + r) in topk[r] else 0.0
    return _package(hits, return_sem, return_per_sample)


def _self_neighbors(Z: np.ndarray, k: int, metric: str) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1, metric=str(metric)).fit(Z)
    return nn.kneighbors(Z, return_distance=False)[:, 1:]


def modality_mixing(Z: np.ndarray, modality_labels: np.ndarray, k: int = 20, metric: str = "euclidean",
                    return_sem: bool = False, return_per_sample: bool = False) -> MetricResult:
    """
    Mean fraction of each sample's ``k`` nearest neighbours carrying a *different* label.

    Typically ``Z`` is the stacked embeddings of all views and
    ``modality_labels`` names the view (or batch) of each row.

    Returns:
        Mean mixing fraction (higher = more mixed), optionally with SEM / per-sample values.

    Raises:
        ValueError: If ``Z`` and ``modality_labels`` differ in length.
    """
    Z = np.asarray(Z, dtype=np.float32)
    modality_labels = np.asarray(modality_labels)
    if Z.shape[0] != modality_labels.shape[0]:
        raise ValueError("Z and modality_labels must have the same number of rows")
    n = int(Z.shape[0])
    if n <= 1:
        return _package(np.zeros(n, dtype=np.float32), return_sem, return_per_sample)
    neigh_idx = _self_neighbors(Z, int(min(max(int(k), 1), n - 1)), metric)
    frac_other = (modality_labels[neigh_idx] != modality_labels[:, None]).mean(axis=1).astype(np.float32)
    return _package(frac_other, return_sem, return_per_sample)


def modality_entropy(Z: np.ndarray, modality_labels: np.ndarray, k: int = 30, metric: str = "euclidean",
                     return_sem: bool = False, return_per_sample: bool = False) -> MetricResult:
    """
    Normalized entropy of the labels among each sample's ``k`` nearest neighbours.

    Entropy is divided by ``log(n_labels)`` so 1 means the neighbourhood holds
    every label equally and 0 means it holds a single label.

    Returns:
        Mean normalized entropy, optionally with SEM / per-sample values.
    """
    Z = np.asarray(Z, dtype=np.float32)
    modality_labels = np.asarray(modality_labels)
    if Z.shape[0] != modality_labels.shape[0]:
        raise ValueError("Z and modality_labels must have the same number of rows")
    n = int(Z.shape[0])
    uniq, codes = np.unique(modality_labels, return_inverse=True)
    if n <= 1 or uniq.size <= 1:
        return _package(np.zeros(n, dtype=np.float32), return_sem, return_per_sample)
    neigh_idx = _self_neighbors(Z, int(min(max(int(k), 1), n - 1)), metric)
    mcount = int(uniq.size)
    denom = np.log(float(mcount) + 1e-12)
    ent = np.empty(n, dtype=np.float32)
    for i in range(n):
        p = np.bincount(codes[neigh_idx[i]], minlength=mcount).astype(np.float32)
        p = p / (p.sum() + 1e-8)
        ent[i] = float(-(p * np.log(p + 1e-12)).sum() / (denom + 1e-12))
    return _package(ent, return_sem, return_per_sample)
