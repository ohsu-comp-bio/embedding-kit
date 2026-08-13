from .base import VAELoss
from .vae_loss import (
    # Concrete nn.Module loss classes
    MSEVAELoss,
    BCEVAELoss,
    BCEWithLogitsVAELoss,
    BCEKLWeightedVAELoss,
    MSELoss,
    BCELoss,
    BCEWithLogitsLoss,
    BCEKLWeightedLoss,
    # Registry + factory
    VAE_LOSS_REGISTRY,
    get_vae_loss,
    LOSS_REGISTRY,
    get_loss,
    # Deprecated free-function API — kept for backwards compatibility
    net_vae_loss,
    bce_with_logits,
    bce,
    mse,
    bce_kl_weighted,
)
