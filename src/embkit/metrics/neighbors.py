"""
Neighbourhood-based metrics for labelled embeddings.

A useful embedding keeps samples of the same biological label together while
mixing technical groups (batches, cohorts, sites) that should not matter. Both
properties are visible in each sample's nearest neighbours:

- :func:`label_transfer_knn` — majority-vote label transfer from a labelled
  source embedding to a target embedding, with accuracy / F1 / confusion
  matrix when target labels are known.
- :func:`transfer_labels` — the same vote, returning per-sample confidence and
  a *novelty* score that flags query samples the reference does not cover.
- :func:`label_purity` — share of each sample's neighbours carrying its own label.
- :func:`batch_mixing` — how mixed the batches are, relative to what their
  sizes allow (0 = separate, ~1 = fully mixed). Read it next to
  :func:`label_purity`: a collapsed latent mixes batches perfectly and is useless.

:func:`transfer_labels`, :func:`label_purity` and :func:`batch_mixing` accept a
``geometry`` argument (see :mod:`embkit.geometry`) and use geodesic distances
for hyperbolic or product latent spaces.

Ported from UniVII (``univii.evaluation`` and ``univii.atlas``).
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import NearestNeighbors

from ..geometry import pairwise_latent_distance, _rows_per_chunk


def _np(z) -> np.ndarray:
    return np.asarray(z.detach().cpu() if torch.is_tensor(z) else z, dtype=float)


def _knn(query: np.ndarray, ref: np.ndarray, k: int, geometry: str, hyperbolic_dim: Optional[int],
         curvature: float, exclude_self: bool, chunk: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """
    Indices and distances of the ``k`` nearest reference rows for every query row.

    Euclidean spaces use scikit-learn; other geometries compute geodesic
    distances in bounded-memory blocks. With ``exclude_self`` the query must be
    the reference itself and each row's own index is dropped.
    """
    k = min(int(k), ref.shape[0] - (1 if exclude_self else 0))
    if k < 1:
        raise ValueError("not enough samples for a neighbour search")
    if str(geometry).lower().strip() == "euclidean":
        dist, idx = NearestNeighbors(n_neighbors=k + int(exclude_self)).fit(ref).kneighbors(query)
        if exclude_self:  # drop each row's own index; if absent (duplicates), drop the farthest
            own = idx == np.arange(query.shape[0])[:, None]
            order = np.argsort(own, axis=1, kind="stable")
            idx, dist = np.take_along_axis(idx, order, 1)[:, :k], np.take_along_axis(dist, order, 1)[:, :k]
        return idx, dist
    step = _rows_per_chunk(ref.shape[0], chunk)
    idx_out, dist_out = [], []
    for start in range(0, query.shape[0], step):
        D = pairwise_latent_distance(query[start:start + step], ref, geometry, hyperbolic_dim, curvature)
        if exclude_self:
            D[np.arange(D.shape[0]), np.arange(start, start + D.shape[0])] = np.inf
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        idx_out.append(idx)
        dist_out.append(np.take_along_axis(D, idx, 1))
    return np.vstack(idx_out), np.vstack(dist_out)


def label_transfer_knn(Z_source: np.ndarray, labels_source: np.ndarray, Z_target: np.ndarray,
                       labels_target: Optional[np.ndarray] = None, k: int = 15, metric: str = "euclidean",
                       return_label_order: bool = False, return_f1: bool = False) -> tuple:
    """
    Transfer labels from a source embedding to a target embedding by kNN majority vote.

    Args:
        Z_source: Labelled embedding ``(n_source, d)``.
        labels_source: One label per source row.
        Z_target: Embedding to label ``(n_target, d)``.
        labels_target: Optional true target labels for scoring.
        k: Number of neighbours voting.
        metric: Distance metric passed to scikit-learn.
        return_label_order: Also return the label order used for the confusion matrix.
        return_f1: Also return ``{"macro_f1", "weighted_f1"}``.

    Returns:
        ``(pred_labels, accuracy, confusion_matrix[, label_order][, f1])``.
        Without ``labels_target`` returns ``(pred_labels, None, empty array)``.

    Raises:
        ValueError: If ``Z_source`` is empty.
    """
    Z_source = np.asarray(Z_source, dtype=np.float32)
    Z_target = np.asarray(Z_target, dtype=np.float32)
    labels_source = np.asarray(labels_source)
    if Z_source.shape[0] == 0:
        raise ValueError("Z_source is empty")
    k_eff = int(min(max(int(k), 1), Z_source.shape[0]))
    neigh_idx = NearestNeighbors(n_neighbors=k_eff, metric=metric).fit(Z_source).kneighbors(
        Z_target, return_distance=False)
    uniq_src, src_codes = np.unique(labels_source, return_inverse=True)
    pred_codes = np.array([np.bincount(src_codes[row], minlength=len(uniq_src)).argmax() for row in neigh_idx],
                          dtype=np.int64)
    pred_labels = uniq_src[pred_codes]
    if labels_target is None:
        return pred_labels, None, np.array([])
    labels_target = np.asarray(labels_target)
    label_order = np.unique(np.concatenate([labels_target, pred_labels]))
    acc = float(accuracy_score(labels_target, pred_labels))
    cm = confusion_matrix(labels_target, pred_labels, labels=label_order)
    extras = []
    if return_label_order:
        extras.append(label_order)
    if return_f1:
        extras.append({
            "macro_f1": float(f1_score(labels_target, pred_labels, average="macro")),
            "weighted_f1": float(f1_score(labels_target, pred_labels, average="weighted")),
        })
    return (pred_labels, acc, cm, *extras)


def transfer_labels(z_ref, labels_ref, z_query, k: int = 15, geometry: str = "euclidean",
                    hyperbolic_dim: Optional[int] = None, curvature: float = 1.0) -> pd.DataFrame:
    """
    Label query samples from a labelled reference embedded in the same latent space.

    The returned columns are:

    - ``label`` — most common label among the ``k`` nearest reference samples.
    - ``confidence`` — share of those neighbours carrying that label.
    - ``novelty`` — mean distance to those neighbours divided by the median of
      the same quantity for reference samples among themselves. Values near 1
      mean the query sits as close to the reference as reference samples sit
      to each other; values well above 1 flag samples the reference does not
      cover, whatever their ``confidence``.

    Returns:
        DataFrame with one row per query sample and columns ``label``,
        ``confidence`` and ``novelty``.
    """
    z_ref, z_query, labels_ref = _np(z_ref), _np(z_query), np.asarray(labels_ref)
    idx, dist = _knn(z_query, z_ref, k, geometry, hyperbolic_dim, curvature, exclude_self=False)
    names, codes = np.unique(labels_ref, return_inverse=True)
    votes = np.zeros((z_query.shape[0], len(names)))
    np.add.at(votes, (np.repeat(np.arange(z_query.shape[0]), idx.shape[1]), codes[idx].ravel()), 1.0)
    best = votes.argmax(1)
    _, ref_dist = _knn(z_ref, z_ref, k, geometry, hyperbolic_dim, curvature, exclude_self=True)
    typical = max(float(np.median(ref_dist.mean(1))), 1e-12)
    return pd.DataFrame({"label": names[best], "confidence": votes.max(1) / idx.shape[1],
                         "novelty": dist.mean(1) / typical})


def label_purity(z, labels, k: int = 15, geometry: str = "euclidean", hyperbolic_dim: Optional[int] = None,
                 curvature: float = 1.0) -> float:
    """Average share of each sample's ``k`` nearest neighbours that carry its own label."""
    z, labels = _np(z), np.asarray(labels)
    idx, _ = _knn(z, z, k, geometry, hyperbolic_dim, curvature, exclude_self=True)
    return float((labels[idx] == labels[:, None]).mean())


def batch_mixing(z, batch, k: int = 15, geometry: str = "euclidean", hyperbolic_dim: Optional[int] = None,
                 curvature: float = 1.0) -> float:
    """
    How well batches are mixed, from 0 (separate) to about 1 (as mixed as their sizes allow).

    For each sample, the share of its ``k`` nearest neighbours from a
    *different* batch is divided by the share expected under perfect mixing
    (``1 - size of its own batch / n``), then averaged. Values can slightly
    exceed 1 by chance.

    Raises:
        ValueError: If fewer than two batches are present.
    """
    z, batch = _np(z), np.asarray(batch)
    names, counts = np.unique(batch, return_counts=True)
    if len(names) < 2:
        raise ValueError("batch_mixing needs at least two batches")
    idx, _ = _knn(z, z, k, geometry, hyperbolic_dim, curvature, exclude_self=True)
    other = (batch[idx] != batch[:, None]).mean(1)
    share = dict(zip(names.tolist(), (counts / counts.sum()).tolist()))
    expected = np.array([1.0 - share[c] for c in batch.tolist()])
    return float(np.mean(other / expected))
