"""
Likelihood-based reconstruction losses for count and bounded data.

The MSE/BCE losses in :mod:`embkit.losses.vae_loss` assume the input has been
normalized into ``[0, 1]`` or onto a Gaussian-like scale. Raw molecular data is
often better described by its native distribution: UMI counts are overdispersed
(negative binomial), sparse counts can be zero-inflated, methylation is a
count of methylated reads out of a coverage (binomial / beta-binomial), and
fractions live in ``(0, 1)`` (beta). Modelling the right likelihood lets the
VAE learn from raw values instead of a lossy transform.

This module provides:

- Element-wise negative log-likelihoods (same shape as the input; the caller
  reduces): :func:`gaussian_nll`, :func:`poisson_nll`, :func:`nb_nll`,
  :func:`zinb_nll`, :func:`beta_nll`, :func:`binomial_nll`,
  :func:`beta_binomial_nll`.
- Drop-in :class:`~embkit.losses.base.VAELoss` subclasses for count data that
  work with :func:`embkit.optimize.fit_vae`: :class:`PoissonVAELoss` and
  :class:`NBVAELoss`.

Ported from UniVII (the likelihood terms of ``UniVIMultiModalVAE``).
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .base import VAELoss, VAELossOutput

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Element-wise negative log-likelihoods
# ---------------------------------------------------------------------------

def gaussian_nll(x: Tensor, mean: Tensor, logvar: Optional[Tensor] = None,
                 logvar_min: float = -10.0, logvar_max: float = 10.0) -> Tensor:
    """
    Gaussian negative log-likelihood, element-wise, up to an additive constant.

    With ``logvar=None`` this is ``0.5 * (x - mean)^2`` (unit variance). With a
    predicted ``logvar`` it is ``0.5 * (logvar + (x - mean)^2 / var)``; the
    log-variance is clamped to ``[logvar_min, logvar_max]`` for stability.
    """
    if logvar is None:
        return 0.5 * (x - mean) ** 2
    logvar = logvar.clamp(logvar_min, logvar_max)
    return 0.5 * (logvar + (x - mean) ** 2 / (torch.exp(logvar) + _EPS))


def poisson_nll(x: Tensor, log_rate: Tensor) -> Tensor:
    """
    Poisson negative log-likelihood from a log-rate, element-wise.

    Omits the ``log(x!)`` term, which does not depend on the model.
    """
    return F.poisson_nll_loss(log_rate, x, log_input=True, full=False, reduction="none")


def nb_nll(x: Tensor, mu: Tensor, theta: Tensor, eps: float = _EPS) -> Tensor:
    """
    Negative binomial negative log-likelihood, element-wise.

    Parameterized by mean ``mu`` and inverse dispersion ``theta`` (variance
    ``mu + mu^2 / theta``), the standard choice for single-cell UMI counts.

    Args:
        x: Observed counts.
        mu: Predicted mean (> 0), same shape as ``x``.
        theta: Inverse dispersion (> 0), broadcastable to ``x``.
        eps: Floor applied to ``mu`` and ``theta``.
    """
    mu = mu.clamp(min=eps)
    theta = theta.clamp(min=eps)
    t1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
    t2 = theta * (torch.log(theta) - torch.log(theta + mu))
    t3 = x * (torch.log(mu) - torch.log(theta + mu))
    return -(t1 + t2 + t3)


def zinb_nll(x: Tensor, mu: Tensor, theta: Tensor, logit_pi: Tensor, eps: float = _EPS) -> Tensor:
    """
    Zero-inflated negative binomial negative log-likelihood, element-wise.

    A mixture of a point mass at zero (probability ``sigmoid(logit_pi)``) and a
    negative binomial ``(mu, theta)``.
    """
    mu = mu.clamp(min=eps)
    theta = theta.clamp(min=eps)
    pi = torch.sigmoid(logit_pi)
    t1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
    t2 = theta * (torch.log(theta) - torch.log(theta + mu))
    t3 = x * (torch.log(mu) - torch.log(theta + mu))
    log_nb = t1 + t2 + t3
    is_zero = x < eps
    log_prob_pos = torch.log1p(-pi + eps) + log_nb
    log_nb_zero = theta * (torch.log(theta) - torch.log(theta + mu))
    log_prob_zero = torch.log(pi + (1.0 - pi) * torch.exp(log_nb_zero) + eps)
    return -torch.where(is_zero, log_prob_zero, log_prob_pos)


def beta_nll(x: Tensor, alpha: Tensor, beta: Tensor, eps: float = _EPS) -> Tensor:
    """
    Beta negative log-likelihood for fractions in ``(0, 1)``, element-wise.

    ``x`` is clamped to ``[eps, 1 - eps]`` so exact 0/1 values stay finite.
    """
    x = x.clamp(min=eps, max=1.0 - eps)
    alpha = alpha.clamp(min=eps)
    beta = beta.clamp(min=eps)
    log_norm = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    return -((alpha - 1.0) * torch.log(x) + (beta - 1.0) * torch.log(1.0 - x) - log_norm)


def binomial_nll(successes: Tensor, total_count: Tensor, logits: Tensor) -> Tensor:
    """
    Binomial negative log-likelihood (successes out of ``total_count`` trials), element-wise.

    Includes the ``log(n choose k)`` term, so the value is a true NLL.
    """
    log_prob = -F.binary_cross_entropy_with_logits(
        logits, successes / total_count.clamp_min(1.0), reduction="none") * total_count
    comb = (torch.lgamma(total_count + 1.0)
            - torch.lgamma(successes + 1.0)
            - torch.lgamma((total_count - successes) + 1.0))
    return -(comb + log_prob)


def beta_binomial_nll(successes: Tensor, total_count: Tensor, alpha: Tensor, beta: Tensor,
                      eps: float = _EPS) -> Tensor:
    """
    Beta-binomial negative log-likelihood (overdispersed binomial), element-wise.

    Suited to count-with-coverage assays such as bisulfite methylation, where
    the per-site rate itself varies beyond binomial noise.
    """
    successes = successes.clamp_min(0.0)
    total_count = total_count.clamp_min(0.0)
    alpha = alpha.clamp_min(eps)
    beta = beta.clamp_min(eps)
    failures = (total_count - successes).clamp_min(0.0)
    log_comb = (torch.lgamma(total_count + 1.0)
                - torch.lgamma(successes + 1.0)
                - torch.lgamma(failures + 1.0))
    log_beta_term = (torch.lgamma(successes + alpha)
                     + torch.lgamma(failures + beta)
                     - torch.lgamma(total_count + alpha + beta)
                     + torch.lgamma(alpha + beta)
                     - torch.lgamma(alpha)
                     - torch.lgamma(beta))
    return -(log_comb + log_beta_term)


# ---------------------------------------------------------------------------
# VAELoss subclasses for count data
# ---------------------------------------------------------------------------

class PoissonVAELoss(VAELoss):
    """
    VAE loss with a Poisson reconstruction term, for raw count matrices.

    The decoder output is interpreted as the **log-rate** of each feature (no
    final activation). The reconstruction term is the Poisson NLL summed over
    features; the KL term is the standard closed-form Gaussian KL.

    Args:
        beta (float): Initial weight for the KL-divergence term.
    """

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__(beta=beta)

    def forward(
        self,
        recon: Tensor,
        x: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        recon_loss = poisson_nll(x, recon).sum(dim=1)
        kl_loss = self._kl_divergence(mu, logvar)
        total = (recon_loss + self.beta * kl_loss).mean()
        return VAELossOutput(total=total, recon=recon_loss.mean(), kl=kl_loss.mean())


class NBVAELoss(VAELoss):
    """
    VAE loss with a negative binomial reconstruction term, for overdispersed counts.

    The decoder output is mapped to the NB mean with ``softplus(recon) + eps``.
    The inverse dispersion ``theta`` is held by the loss as ``log_theta``:
    one value per feature when ``n_features`` is given (gene-wise dispersion),
    otherwise a single shared value.

    Note:
        When ``learn_dispersion=True`` the dispersion is an ``nn.Parameter`` of
        the loss, not of the model. To train it, pass an optimizer that covers
        both, e.g. ``fit_vae(model, X, loss=loss_fn,
        optimizer=Adam(list(model.parameters()) + list(loss_fn.parameters())))``.
        Otherwise it stays at its initial value and acts as a fixed dispersion.

    Args:
        beta (float): Initial weight for the KL-divergence term.
        n_features (int, optional): Number of features for gene-wise dispersion.
        theta (float): Initial inverse dispersion.
        learn_dispersion (bool): Make ``log_theta`` trainable.
    """

    def __init__(self, beta: float = 1.0, n_features: Optional[int] = None,
                 theta: float = 1.0, learn_dispersion: bool = True) -> None:
        super().__init__(beta=beta)
        if not float(theta) > 0.0:
            raise ValueError(f"theta must be > 0, got {theta}")
        shape = (int(n_features),) if n_features is not None else (1,)
        log_theta = torch.full(shape, float(torch.log(torch.tensor(float(theta)))))
        if learn_dispersion:
            self.log_theta = nn.Parameter(log_theta)
        else:
            self.register_buffer("log_theta", log_theta)

    def forward(
        self,
        recon: Tensor,
        x: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        mean = F.softplus(recon) + _EPS
        theta = torch.exp(self.log_theta.to(device=recon.device, dtype=recon.dtype))
        recon_loss = nb_nll(x, mean, theta.expand_as(mean)).sum(dim=1)
        kl_loss = self._kl_divergence(mu, logvar)
        total = (recon_loss + self.beta * kl_loss).mean()
        return VAELossOutput(total=total, recon=recon_loss.mean(), kl=kl_loss.mean())
