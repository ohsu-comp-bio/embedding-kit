"""
Base classes for VAE loss functions.
"""

from abc import abstractmethod
from typing import Tuple, NamedTuple

import torch
from torch import nn, Tensor


class VAELoss(nn.Module):
    """Abstract base class for all VAE loss functions.

    Subclasses must implement :meth:`forward`, which receives the decoder
    reconstruction, the original input, and the encoder's latent parameters,
    and returns a ``(total, recon_loss, kl_loss)`` tuple.

    The ``beta`` attribute controls the weight of the KL-divergence term.
    It can be updated directly (``loss_fn.beta = new_value``) or via
    :meth:`step_beta` for warmup schedules.
    """

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta

    @abstractmethod
    def forward(
        self,
        recon: Tensor,
        x: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the VAE loss.

        Args:
            recon: Decoder output.  Shape must match ``x``.
            x: Original input.
            mu: Encoder mean of the latent distribution.
            logvar: Encoder log-variance of the latent distribution.

        Returns:
            Tuple of ``(total_loss, reconstruction_loss, kl_loss)`` — all
            scalar tensors.
        """

    def step_beta(self, kappa: float = 0.01, max_beta: float = 1.0) -> None:
        """Increment ``beta`` by *kappa*, clamped to *max_beta*.

        Call once per epoch to implement a linear KL warmup schedule.

        Args:
            kappa: Amount to increase ``beta`` per step.
            max_beta: Upper bound for ``beta``.
        """
        self.beta = min(self.beta + kappa, max_beta)

    @staticmethod
    def _kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
        """Standard closed-form KL( q(z|x) || N(0, I) ), per sample.

        Returns a 1-D tensor of shape ``(batch_size,)``.
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


class VAELossOutput(NamedTuple):
    """Named output of :meth:`VAELoss.forward`."""
    total: Tensor
    recon: Tensor
    kl: Tensor

