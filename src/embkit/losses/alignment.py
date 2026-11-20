"""
Losses that align latent representations with each other.

When two views of the same samples are embedded separately — RNA and protein
from the same cells, two assays from the same tumours, an image tile and its
expression profile — the encoders have to be pushed to put matching samples in
the same place. These losses provide that push, and the divergence they often
sit beside:

- :func:`kl_diag_gaussians` — KL between two diagonal Gaussians (generalizes the
  standard-normal KL used by :class:`~embkit.losses.base.VAELoss`).
- :func:`pairwise_alignment_loss` — mean squared distance between the latent
  means of every pair of views (paired samples).
- :func:`info_nce_loss` — symmetric contrastive loss: each sample should be
  closer to its own partner than to every other sample in the batch.
- :func:`mmd_loss` — kernel two-sample statistic between two *unpaired* sets
  (e.g. two cohorts), multi-scale Gaussian kernel.

All return per-sample tensors (``(batch,)``) except :func:`mmd_loss`, which is a
set-level scalar. Reduce with ``.mean()`` before ``backward``.

Ported from UniVII (``univii.objectives`` and the alignment terms of
``UniVIMultiModalVAE``).
"""
from typing import Mapping, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def _sq_cdist(a: Tensor, b: Tensor) -> Tensor:
    """All-pairs squared Euclidean distance ``(n, m)`` from inner products."""
    return ((a ** 2).sum(-1, keepdim=True) + (b ** 2).sum(-1).unsqueeze(0)
            - 2.0 * a @ b.transpose(0, 1)).clamp_min(0.0)


def kl_diag_gaussians(mu_q: Tensor, logvar_q: Tensor, mu_p: Tensor, logvar_p: Tensor) -> Tensor:
    """
    ``KL(q || p)`` for diagonal Gaussians, summed over the latent dimension.

    Args:
        mu_q: Mean of ``q``, ``(batch, latent_dim)``.
        logvar_q: Log-variance of ``q``, ``(batch, latent_dim)``.
        mu_p: Mean of ``p`` (broadcastable).
        logvar_p: Log-variance of ``p`` (broadcastable).

    Returns:
        KL per sample, ``(batch,)``.
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0
    return 0.5 * kl.sum(dim=-1)


def pairwise_alignment_loss(mus: Union[Sequence[Tensor], Mapping[str, Tensor]]) -> Tensor:
    """
    Mean pairwise squared L2 distance between the latent means of paired views.

    Args:
        mus: Latent means of the same samples from each view, either a list or
            a ``{name: tensor}`` mapping; every tensor is ``(batch, latent_dim)``.

    Returns:
        Loss per sample, ``(batch,)``. Zero when only one view is given.

    Raises:
        ValueError: If no views are given.
    """
    views = list(mus.values()) if isinstance(mus, Mapping) else list(mus)
    if len(views) == 0:
        raise ValueError("pairwise_alignment_loss needs at least one view")
    if len(views) < 2:
        return torch.zeros(views[0].size(0), device=views[0].device, dtype=views[0].dtype)
    losses = []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            losses.append(((views[i] - views[j]) ** 2).sum(dim=-1))
    return torch.stack(losses, dim=0).mean(dim=0)


def info_nce_loss(a: Tensor, b: Tensor, temperature: float = 1.0) -> Tensor:
    """
    Symmetric InfoNCE between two paired views of the same batch.

    Row ``i`` of ``a`` is the positive for row ``i`` of ``b``; every other row in
    the batch is a negative. Similarity is negative squared Euclidean distance
    divided by ``temperature``. The loss is the average of the ``a -> b`` and
    ``b -> a`` cross-entropies.

    Args:
        a: First view, ``(batch, latent_dim)``.
        b: Second view, ``(batch, latent_dim)``.
        temperature: Softmax temperature (> 0).

    Returns:
        Loss per sample, ``(batch,)``.

    Raises:
        ValueError: If shapes differ, the batch has fewer than two samples, or
            ``temperature`` is not positive.
    """
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[0] < 2:
        raise ValueError("info_nce_loss needs at least two samples per batch")
    if not float(temperature) > 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    target = torch.arange(a.shape[0], device=a.device)
    logits = -_sq_cdist(a, b) / float(temperature)
    return 0.5 * (F.cross_entropy(logits, target, reduction="none")
                  + F.cross_entropy(logits.t(), target, reduction="none"))


def mmd_loss(a: Tensor, b: Tensor,
             scales: Sequence[float] = (1.0 / 256, 1.0 / 64, 1.0 / 16, 0.25, 1.0, 4.0)) -> Tensor:
    """
    Maximum mean discrepancy between two unpaired sets of latent points.

    Uses Gaussian kernels at several widths set relative to the median cross
    distance (so both a global shift and a small offset inside tight clusters
    are seen). Zero when the two sets have the same distribution.

    Args:
        a: First set, ``(n, latent_dim)``.
        b: Second set, ``(m, latent_dim)``.
        scales: Kernel widths as fractions of the median squared cross distance.

    Returns:
        Scalar MMD estimate.
    """
    d_ab, d_aa, d_bb = _sq_cdist(a, b), _sq_cdist(a, a), _sq_cdist(b, b)
    base = d_ab.detach().median().clamp_min(1e-6)
    total = 0.0
    for scale in scales:
        w = 2.0 * float(scale) * base
        total = total + torch.exp(-d_aa / w).mean() + torch.exp(-d_bb / w).mean() - 2.0 * torch.exp(-d_ab / w).mean()
    return total / len(scales)
