"""
Fusion of Gaussian posteriors from several encoders.

A multi-view model has one encoder per data type, each producing its own latent
Gaussian ``(mu, logvar)`` for the same sample. To get a single joint embedding,
the posteriors are combined by a product of experts: each expert's mean is
weighted by its precision (inverse variance), so a confident view dominates and
an uncertain one contributes little. Views missing for a sample can simply be
left out of the list.

- :func:`product_of_experts` — precision-weighted fusion, optionally with
  per-sample expert weights (e.g. from a gating network).

Ported from UniVII (``UniVIMultiModalVAE.mixture_of_experts``).
"""
from typing import Optional, Sequence, Tuple

import torch

_EPS = 1e-8


def product_of_experts(mus: Sequence[torch.Tensor], logvars: Sequence[torch.Tensor],
                       weights: Optional[torch.Tensor] = None,
                       weight_eps: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine diagonal Gaussian posteriors by precision weighting.

    For experts ``k`` with precision ``p_k = exp(-logvar_k)``::

        var  = 1 / sum_k p_k
        mu   = var * sum_k p_k * mu_k

    Args:
        mus: One ``(batch, latent_dim)`` mean per expert.
        logvars: One ``(batch, latent_dim)`` log-variance per expert.
        weights: Optional ``(batch, n_experts)`` non-negative weights (e.g. a
            softmax over gate logits). Each expert's precision is scaled by
            ``weight + weight_eps`` before fusion.
        weight_eps: Floor added to ``weights`` so no expert is dropped entirely.

    Returns:
        Tuple ``(mu, logvar)`` of the fused posterior.

    Raises:
        ValueError: If no experts are given, the lists differ in length, or
            ``weights`` has the wrong shape.
    """
    mus, logvars = list(mus), list(logvars)
    if len(mus) == 0:
        raise ValueError("product_of_experts requires at least one expert")
    if len(mus) != len(logvars):
        raise ValueError(f"got {len(mus)} means and {len(logvars)} log-variances")
    if len(mus) == 1:
        # a single expert is its own product, whatever its weight
        return mus[0], logvars[0]
    precisions = [torch.exp(-lv) for lv in logvars]
    if weights is not None:
        if weights.shape != (mus[0].shape[0], len(mus)):
            raise ValueError(f"weights must be (batch, n_experts) = ({mus[0].shape[0]}, {len(mus)}), "
                             f"got {tuple(weights.shape)}")
        precisions = [(weights[:, i:i + 1] + weight_eps) * p for i, p in enumerate(precisions)]
    precision_sum = torch.stack(precisions, dim=0).sum(dim=0).clamp_min(_EPS)
    mu = torch.stack([m * p for m, p in zip(mus, precisions)], dim=0).sum(dim=0) / precision_sum
    logvar = torch.log((1.0 / precision_sum).clamp_min(_EPS))
    return mu, logvar
