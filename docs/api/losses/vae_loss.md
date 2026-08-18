# Losses · VAE

VAE loss modules:

- `MSEVAELoss`
- `BCEVAELoss`
- `BCEWithLogitsVAELoss`
- `BCEKLWeightedVAELoss`

Factory helpers:

- `get_vae_loss(name, **kwargs)` (preferred)
- `get_loss(name, **kwargs)` (compatibility alias)

::: embkit.losses.vae_loss