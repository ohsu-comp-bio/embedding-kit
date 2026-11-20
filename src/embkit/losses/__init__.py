from .base import VAELoss, VAELossOutput
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
from .likelihood import (
    # Element-wise negative log-likelihoods
    gaussian_nll,
    poisson_nll,
    nb_nll,
    zinb_nll,
    beta_nll,
    binomial_nll,
    beta_binomial_nll,
    # Count-data VAE losses
    PoissonVAELoss,
    NBVAELoss,
)
from .alignment import (
    kl_diag_gaussians,
    pairwise_alignment_loss,
    info_nce_loss,
    mmd_loss,
)
