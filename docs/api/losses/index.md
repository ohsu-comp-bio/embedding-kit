# Losses

`embkit.losses` provides the VAE loss functions used by `optimize.fit_vae` and the CLI training commands.

## VAE loss modules

All share the call signature `loss(recon, x, mu, logvar) -> (total, recon_loss, kl_loss)`:

| Module | When to use |
|--------|-------------|
| `BCEWithLogitsVAELoss` | Decoder outputs raw logits. Most stable; default for `train-vae`. |
| `BCEVAELoss` | Decoder outputs values in (0, 1) via sigmoid. |
| `MSEVAELoss` | Continuous, unbounded targets (default when no loss is passed). |
| `BCEKLWeightedVAELoss` | BCE reconstruction with a weighted KL term. |

```python
from embkit.losses import BCEWithLogitsVAELoss, get_vae_loss

loss = BCEWithLogitsVAELoss(beta=1.0)
# or dispatch by name (same as the CLI --loss flag):
loss = get_vae_loss("bce-logit")   # "mse" | "bce" | "bce-logit"
```

- `get_vae_loss(name, **kwargs)` — preferred name-based factory.
- `get_loss(name, **kwargs)` — compatibility alias.
- Non-VAE base classes (`MSELoss`, `BCELoss`, `BCEWithLogitsLoss`) are thin wrappers around `nn` losses.

## API

::: embkit.losses
    options:
      show_source: false
      filters:
        - "!^_"
