from .base import VAELoss
from .vae_loss import (
    # Concrete nn.Module loss classes
    MSELoss,
    BCELoss,
    BCEWithLogitsLoss,
    BCEKLWeightedLoss,
    # Registry + factory
    LOSS_REGISTRY,
    get_loss,
    # Deprecated free-function API — kept for backwards compatibility
    net_vae_loss,
    bce_with_logits,
    bce,
    mse,
    bce_kl_weighted,
)
