
"""
VAE loss functions implemented as ``nn.Module`` subclasses.

All concrete loss classes share a common base (:class:`VAELoss`) and expose:

* ``forward(recon, x, mu, logvar) -> (total, recon_loss, kl_loss)``
* ``step_beta(kappa, max_beta)`` — linear KL-warmup helper
* ``beta`` attribute — weight of the KL term, updatable at any time

Legacy module-level functions are kept as deprecated thin wrappers so that
existing call-sites continue to work without modification.  They will be
removed in a future release.
"""

import warnings
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import VAELoss


# ---------------------------------------------------------------------------
# Concrete loss classes
# ---------------------------------------------------------------------------

class MSEVAELoss(VAELoss):
    """VAE loss using Mean-Squared-Error reconstruction (regression VAE).

    Args:
        beta (float): Initial weight for the KL-divergence term.
        reduction (str): Reduction mode passed to ``F.mse_loss``
            (``"mean"`` or ``"sum"``).
    """

    def __init__(self, beta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__(beta=beta)
        self.reduction = reduction

    def forward(
        self,
        recon: Tensor,
        x: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        recon_loss = F.mse_loss(recon, x, reduction=self.reduction)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + self.beta * kl
        return total, recon_loss, kl


class BCEVAELoss(VAELoss):
    """VAE loss using Binary Cross-Entropy on probabilities.

    The decoder output must be in ``[0, 1]`` (i.e. after a sigmoid).

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
        bce_per_sample = F.binary_cross_entropy(recon, x, reduction="none").mean(dim=1)
        recon_loss = x.size(1) * bce_per_sample
        kl_loss = self._kl_divergence(mu, logvar)
        total = (recon_loss + self.beta * kl_loss).mean()
        return total, recon_loss.mean(), kl_loss.mean()


class BCEWithLogitsVAELoss(VAELoss):
    """VAE loss using Binary Cross-Entropy with logits (numerically stable).

    Use this when the decoder produces raw logits (no final sigmoid).

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
        bce_per_sample = F.binary_cross_entropy_with_logits(recon, x, reduction="none").mean(dim=1)
        recon_loss = x.size(1) * bce_per_sample
        kl_loss = self._kl_divergence(mu, logvar)
        total = (recon_loss + self.beta * kl_loss).mean()
        return total, recon_loss.mean(), kl_loss.mean()


class BCEKLWeightedVAELoss(VAELoss):
    """VAE loss with a separate KL weight multiplier (RNA-VAE variant).

    The effective KL coefficient is ``kl_weight * beta``, allowing fine-grained
    control of the KL term independently of the warmup schedule.

    Args:
        beta (float): Initial warmup factor (0.0 → 1.0).
        kl_weight (float): Fixed scaling factor for the KL term (default 1.0;
            RNA VAE typically uses 5.0).
    """

    def __init__(self, beta: float = 1.0, kl_weight: float = 1.0) -> None:
        super().__init__(beta=beta)
        self.kl_weight = kl_weight

    def forward(
        self,
        recon: Tensor,
        x: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        bce_per_sample = F.binary_cross_entropy(recon, x, reduction="none").mean(dim=1)
        recon_loss = x.size(1) * bce_per_sample
        kl_loss = self._kl_divergence(mu, logvar)
        total = (recon_loss + self.kl_weight * self.beta * kl_loss).mean()
        return total, recon_loss.mean(), kl_loss.mean()


# ---------------------------------------------------------------------------
# Registry + factory helper
# ---------------------------------------------------------------------------

VAE_LOSS_REGISTRY: dict = {
    "mse": MSEVAELoss,
    "bce": BCEVAELoss,
    "bce-logit": BCEWithLogitsVAELoss,
}


def get_vae_loss(name: str, **kwargs) -> VAELoss:
    """Instantiate a :class:`VAELoss` by registry name.

    Args:
        name: One of ``"mse"``, ``"bce"``, ``"bce-logit"``.
        **kwargs: Forwarded to the loss class constructor (e.g. ``beta=0.0``).

    Returns:
        A configured :class:`VAELoss` instance.

    Raises:
        KeyError: If *name* is not in :data:`VAE_LOSS_REGISTRY`.
    """
    if name not in VAE_LOSS_REGISTRY:
        raise KeyError(
            f"Unknown loss '{name}'. Valid choices: {sorted(VAE_LOSS_REGISTRY)}"
        )
    return VAE_LOSS_REGISTRY[name](**kwargs)


# Backwards-compatible aliases
MSELoss = MSEVAELoss
BCELoss = BCEVAELoss
BCEWithLogitsLoss = BCEWithLogitsVAELoss
BCEKLWeightedLoss = BCEKLWeightedVAELoss
LOSS_REGISTRY = VAE_LOSS_REGISTRY


def get_loss(name: str, **kwargs) -> VAELoss:
    """Backward-compatible alias for :func:`get_vae_loss`."""
    return get_vae_loss(name, **kwargs)


# ---------------------------------------------------------------------------
# Legacy free-function API (deprecated)
# ---------------------------------------------------------------------------

def _deprecated(fn_name: str) -> None:
    class_name_by_fn = {
        "mse": "MSEVAELoss",
        "bce": "BCEVAELoss",
        "bce_with_logits": "BCEWithLogitsVAELoss",
        "bce_kl_weighted": "BCEKLWeightedVAELoss",
        "net_vae_loss": "VAELoss subclass",
    }
    class_name = class_name_by_fn.get(fn_name, "VAELoss subclass")
    warnings.warn(
        f"embkit.losses.{fn_name} is deprecated and will be removed in a future "
        "release. Use the corresponding nn.Module class instead "
        f"(e.g. embkit.losses.{class_name} or "
        "embkit.losses.get_loss()).",
        DeprecationWarning,
        stacklevel=3,
    )


def mse(recon, x, mu, logvar, beta=1.0, reduction="mean") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deprecated — use :class:`MSEVAELoss` instead."""
    _deprecated("mse")
    return MSEVAELoss(beta=beta, reduction=reduction)(recon, x, mu, logvar)


def bce(recon_x, x, mu, logvar, beta=1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deprecated — use :class:`BCEVAELoss` instead."""
    _deprecated("bce")
    return BCEVAELoss(beta=beta)(recon_x, x, mu, logvar)


def bce_with_logits(recon_logits, x, mu, logvar, beta=1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deprecated — use :class:`BCEWithLogitsVAELoss` instead."""
    _deprecated("bce_with_logits")
    return BCEWithLogitsVAELoss(beta=beta)(recon_logits, x, mu, logvar)


def bce_kl_weighted(recon_x, x, mu, logvar, beta=1.0, kl_weight=1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deprecated — use :class:`BCEKLWeightedVAELoss` instead."""
    _deprecated("bce_kl_weighted")
    return BCEKLWeightedVAELoss(beta=beta, kl_weight=kl_weight)(recon_x, x, mu, logvar)


def net_vae_loss(model, x: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deprecated — run the forward pass outside the loss function and call a
    :class:`VAELoss` subclass directly instead."""
    _deprecated("net_vae_loss")
    mu, logvar, z = model.encoder(x)
    reconstruction = model.decoder(z)
    recon_min = float(reconstruction.detach().min())
    recon_max = float(reconstruction.detach().max())
    if recon_min < 0.0 or recon_max > 1.0:
        loss_fn = BCEWithLogitsVAELoss(beta=beta)
    else:
        loss_fn = BCEVAELoss(beta=beta)
    return loss_fn(reconstruction, x, mu, logvar)
