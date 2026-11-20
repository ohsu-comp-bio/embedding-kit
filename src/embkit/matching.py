"""
Sample matching between two datasets that share an embedding space.

When two datasets are not paired sample-by-sample (e.g. scRNA-seq and
scATAC-seq from different cells of the same tissue), an integration model needs
pseudo-pairs or anchors to learn from. Once both are projected into a
comparable space (shared PCA, CCA, or a jointly trained latent), these
functions find them:

- :func:`bipartite_match` — one-to-one optimal assignment (Hungarian algorithm)
  minimizing total distance.
- :func:`mnn_anchors` — mutual nearest neighbours: pairs that are each among the
  other's ``k`` nearest neighbours, a conservative set of high-confidence anchors.

Both take plain ``(n, d)`` arrays and return positional indices into the inputs.
See :mod:`embkit.align` for transforms (Procrustes, etc.) that map one space
onto another.

Ported from UniVII (``univii.matching``).
"""
import warnings
from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def _subsample_indices(n: int, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Up to ``max_samples`` indices from ``range(n)``, without replacement."""
    idx_full = np.arange(n)
    if n <= max_samples:
        return idx_full
    return rng.choice(idx_full, size=max_samples, replace=False)


def bipartite_match(XA: np.ndarray, XB: np.ndarray, metric: str = "euclidean", max_samples: int = 20000,
                    random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-to-one matching between rows of ``XA`` and ``XB`` minimizing total distance.

    The Hungarian algorithm is ``O(n^3)``, so both sides are randomly
    subsampled to ``min(len(XA), len(XB), max_samples)`` rows first.

    Args:
        XA: Embedding of dataset A, ``(n_a, d)``.
        XB: Embedding of dataset B in the same space, ``(n_b, d)``.
        metric: Distance metric passed to ``sklearn.metrics.pairwise_distances``.
        max_samples: Upper bound on rows matched per side.
        random_state: Seed for subsampling.

    Returns:
        Tuple ``(idx_a, idx_b)`` of equal length: ``XA[idx_a[i]]`` is matched to ``XB[idx_b[i]]``.
    """
    rng = np.random.default_rng(random_state)
    XA, XB = np.asarray(XA), np.asarray(XB)
    n = min(XA.shape[0], XB.shape[0], int(max_samples))
    idx_a = _subsample_indices(XA.shape[0], n, rng)
    idx_b = _subsample_indices(XB.shape[0], n, rng)
    D = pairwise_distances(XA[idx_a], XB[idx_b], metric=metric)
    row_ind, col_ind = linear_sum_assignment(D)
    return idx_a[row_ind], idx_b[col_ind]


def mnn_anchors(XA: np.ndarray, XB: np.ndarray, k: int = 20, max_samples: int = 20000,
                random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mutual nearest neighbour anchors between rows of ``XA`` and ``XB``.

    A pair ``(i, j)`` is an anchor when ``XB[j]`` is among the ``k`` nearest
    B-neighbours of ``XA[i]`` *and* ``XA[i]`` is among the ``k`` nearest
    A-neighbours of ``XB[j]``.

    Args:
        XA: Embedding of dataset A, ``(n_a, d)``.
        XB: Embedding of dataset B in the same space, ``(n_b, d)``.
        k: Neighbourhood size on each side.
        max_samples: Each side is randomly subsampled to at most this many rows.
        random_state: Seed for subsampling.

    Returns:
        Tuple ``(idx_a, idx_b)`` of anchor pairs (empty arrays, with a warning,
        if none are found).
    """
    rng = np.random.default_rng(random_state)
    XA, XB = np.asarray(XA), np.asarray(XB)
    idx_a = _subsample_indices(XA.shape[0], int(max_samples), rng)
    idx_b = _subsample_indices(XB.shape[0], int(max_samples), rng)
    XA_sub, XB_sub = XA[idx_a], XB[idx_b]
    ind_a2b = NearestNeighbors(n_neighbors=min(int(k), idx_b.size)).fit(XB_sub).kneighbors(
        XA_sub, return_distance=False)
    ind_b2a = NearestNeighbors(n_neighbors=min(int(k), idx_a.size)).fit(XA_sub).kneighbors(
        XB_sub, return_distance=False)
    neighbors_b2a = [set(row.tolist()) for row in ind_b2a]
    anchors_a, anchors_b = [], []
    for i in range(idx_a.size):
        for j in ind_a2b[i]:
            if i in neighbors_b2a[j]:
                anchors_a.append(idx_a[i])
                anchors_b.append(idx_b[j])
    if not anchors_a:  # pragma: no cover - defensive; the closest A/B pair is always mutual for k >= 1
        warnings.warn("mnn_anchors: no mutual nearest neighbours found")
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.asarray(anchors_a, dtype=int), np.asarray(anchors_b, dtype=int)
