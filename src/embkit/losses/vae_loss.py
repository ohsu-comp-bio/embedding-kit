
"""
VAE loss functions

"""

import torch
import torch.nn.functional as F

from ..models.vae.base_vae import BaseVAE


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """Calculate the VAE loss.

    Args:
        recon_x (Tensor): Reconstructed input data. Shape should match `x`.
        x (Tensor): Input data.
        mu (Tensor): Mean values of the latent space distribution.
        logvar (Tensor): Log variance values of the latent space distribution.

    Returns:
        tuple: Total loss, reconstruction loss and KL divergence loss respectively as floats.
    """
    bce_per_sample = F.binary_cross_entropy(recon_x, x, reduction="none").mean(dim=1)
    reconstruction_loss = x.size(1) * bce_per_sample
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return (reconstruction_loss + beta * kl_loss).mean(), reconstruction_loss.mean(), kl_loss.mean()


def net_vae_loss(model: BaseVAE, x: torch.Tensor, beta: float = 1.0):
    """Calculate the VAE loss for a given model and input data.

    Args:
        model (BaseVAE): The Variational Autoencoder model used to calculate the losses.
        x (torch.Tensor): Input data.

    Returns:
        tuple: Total loss, reconstruction loss and KL divergence loss respectively as floats.
    """
    mu, logvar, z = model.encoder(x)
    reconstruction = model.decoder(z)
    # keras: x.shape[1] * binary_crossentropy(x, reconstruction)
    bce_per_sample = F.binary_cross_entropy(reconstruction, x, reduction="none").mean(dim=1)
    reconstruction_loss = x.size(1) * bce_per_sample
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss.mean(), reconstruction_loss.mean(), kl_loss.mean()