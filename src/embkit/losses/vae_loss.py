
"""
VAE loss functions

"""

import torch
import torch.nn.functional as F

# ---------- regression VAE loss (MSE + β·KL) ----------
def mse(recon, x, mu, logvar, beta=1.0, reduction="mean"):
    """
    Calculate the VAE loss.
    Uses Mean Squared Error

    """
    recon_loss = F.mse_loss(recon, x, reduction=reduction)
    # KL( q(z|x) || N(0, I) )
    # mean over batch for stability; fit() averages per-epoch across batches
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl
    return total, recon_loss, kl


def bce(recon_x, x, mu, logvar, beta=1.0):
    """Calculate the VAE loss.
    Used for classification of class probabilities (e.g., MNIST).

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

def bce_with_logits(recon_logits, x, mu, logvar, beta=1.0):
    """
    Calculate the VAE loss using binary cross-entropy with logits.
    This is useful when the decoder outputs raw distributions instead of probabilities.

    Args:
        recon_logits (Tensor): Raw decoder output (logits). Shape should match `x`.
        x (Tensor): Input data.
        mu (Tensor): Mean values of the latent space distribution.
        logvar (Tensor): Log variance values of the latent space distribution.
        beta (float): Weighting factor for the KL divergence term.

    Returns:
        tuple: Total loss, reconstruction loss and KL divergence loss respectively as floats.

    """
    # recon_logits: raw decoder output (no sigmoid)
    bce_per_sample = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="none").mean(dim=1)
    recon_loss = x.size(1) * bce_per_sample
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    total = (recon_loss + beta * kl_loss).mean()
    return total, recon_loss.mean(), kl_loss.mean()


def bce_kl_weighted(recon_x, x, mu, logvar, beta=1.0, kl_weight=1.0):
    """Calculate the VAE loss with configurable KL weight.
    
    This is useful for RNA VAE and other variants that scale the KL term differently.
    Standard VAE uses kl_weight=1.0, while RNA VAE uses kl_weight=5.0.

    Args:
        recon_x (Tensor): Reconstructed input data. Shape should match `x`.
        x (Tensor): Input data.
        mu (Tensor): Mean values of the latent space distribution.
        logvar (Tensor): Log variance values of the latent space distribution.
        beta (float): Beta warmup factor for the KL divergence term (0.0 to 1.0).
        kl_weight (float): Additional scaling factor for KL term (default 1.0).

    Returns:
        tuple: Total loss, reconstruction loss and KL divergence loss respectively as floats.
    """
    bce_per_sample = F.binary_cross_entropy(recon_x, x, reduction="none").mean(dim=1)
    reconstruction_loss = x.size(1) * bce_per_sample
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    total_loss = (reconstruction_loss + kl_weight * beta * kl_loss).mean()
    return total_loss, reconstruction_loss.mean(), kl_loss.mean()


def net_vae_loss(model: "BaseVAE", x: torch.Tensor, beta: float = 1.0):
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