import torch
from src.embkit.models import VAE
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar):
    bce_per_sample = F.binary_cross_entropy(recon_x, x, reduction="none").mean(dim=1)
    reconstruction_loss = x.size(1) * bce_per_sample
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return (reconstruction_loss + kl_loss).mean(), reconstruction_loss.mean(), kl_loss.mean()


def net_vae_loss(model: VAE, x: torch.Tensor):
    mu, logvar, z = model.encoder(x)
    reconstruction = model.decoder(z)
    # keras: x.shape[1] * binary_crossentropy(x, reconstruction)
    bce_per_sample = F.binary_cross_entropy(reconstruction, x, reduction="none").mean(dim=1)
    reconstruction_loss = x.size(1) * bce_per_sample
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    total_loss = reconstruction_loss + kl_loss
    return total_loss.mean(), reconstruction_loss.mean(), kl_loss.mean()