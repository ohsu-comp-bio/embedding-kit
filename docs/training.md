# Training Guide

This page covers everything you need to know about training VAE models with Embedding Kit: the training functions, loss choices, beta scheduling, GPU selection, and inspecting results.

---

## Quick start

```python
from embkit import dataframe_loader
from embkit.models.vae import VAE
from embkit.factory.layers import Layer
from embkit.losses import bce_with_logits
from embkit import optimize

dataloader = dataframe_loader(df_norm, batch_size=256)

vae = VAE(features=list(df_norm.columns), latent_dim=128)
optimize.fit_vae(vae, X=dataloader, epochs=50, lr=1e-3, loss=bce_with_logits)
```

---

## fit_vae

```python
optimize.fit_vae(
    model,
    X,                     # DataFrame, Tensor, or DataLoader
    epochs=20,
    lr=1e-3,
    beta=1.0,
    beta_schedule=None,    # list of (beta, n_epochs) pairs
    loss=None,             # required: loss function from embkit.losses
    optimizer=None,        # optional: custom torch.optim.Optimizer
    device=None,           # auto-detected if None
    progress=True,
    accumulate_steps=1,
    shuffle=True,
    batch_size=256,
)
```

`X` can be:

| Type | Behaviour |
|------|-----------|
| `pd.DataFrame` | columns must match `model.features`; converted to a `DataLoader` internally |
| `torch.Tensor` | wrapped in `TensorDataset` then `DataLoader` |
| `DataLoader` | used directly |

`loss` is **required**. Pass one of the functions from `embkit.losses`.

### Returned value

`fit_vae` returns the final batch loss as a float. Training history is stored on the model at `model.history`.

---

## Loss functions

All loss functions have the signature:

```python
def loss_fn(recon, x, mu, logvar, beta=1.0) -> (total, recon_loss, kl_loss)
```

| Function | When to use |
|----------|-------------|
| `bce_with_logits` | Decoder outputs raw logits (no final activation). Most stable. Default for `train-vae` CLI. |
| `bce` | Decoder outputs values in (0, 1) via sigmoid. Use when `final_activation="sigmoid"`. |
| `mse` | Continuous targets that aren't bounded to [0, 1]. Use with un-normalized data. |

```python
from embkit.losses import bce_with_logits, bce, mse
```

---

## Beta scheduling

Training starts with `beta=0` (reconstruction-only) and gradually raises it to prevent posterior collapse.

### Format

`beta_schedule` is a list of `(beta_value, n_epochs)` tuples:

```python
schedule = [
    (0.0,  20),   # first 20 epochs: β = 0
    (0.1,  20),   # next  20 epochs: β = 0.1
    (0.3,  40),   # next  40 epochs: β = 0.3
    (0.4,  40),   # final 40 epochs: β = 0.4
]

optimize.fit_vae(vae, X=loader, beta_schedule=schedule, loss=bce_with_logits)
```

Total epochs = sum of all `n_epochs` values. The `epochs` argument is ignored when `beta_schedule` is provided.

When a schedule is used, a **single optimizer** is created once and reused across all phases so Adam momentum is preserved across phase transitions.

### CLI syntax

```bash
--schedule "20:0,20:0.1,40:0.3,40:0.4"
```

Each segment is `n_epochs:beta_value`.

---

## Gradient accumulation

For large models on memory-constrained GPUs, accumulate gradients over multiple batches before stepping:

```python
optimize.fit_vae(vae, X=loader, accumulate_steps=4, batch_size=64, ...)
```

This simulates an effective batch size of `4 × 64 = 256` without requiring more GPU memory.

---

## Device selection

`get_device()` scans for CUDA, then MPS (Apple Silicon), then falls back to CPU:

```python
from embkit import get_device
device = get_device()
```

`fit_vae` calls `get_device()` automatically if `device=None`. To force a specific device:

```python
import torch
optimize.fit_vae(vae, X=loader, device=torch.device("cuda:1"), ...)
```

---

## Inspecting training history

After training, `model.history` is a dict of per-epoch means:

```python
history = vae.history
# Keys: "loss", "recon", "kl", "beta"

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(history["loss"],  label="total")
axes[1].plot(history["recon"], label="recon")
axes[2].plot(history["kl"],    label="kl")
for ax in axes:
    ax.legend()
plt.tight_layout()
plt.show()
```

---

## General supervised training: fit

For non-VAE models (e.g., `FFNN`), use `optimize.fit`:

```python
optimize.fit(
    model,
    X,          # Tensor (paired with y), Dataset, or DataLoader
    y=None,     # required when X is a Tensor
    epochs=20,
    lr=1e-3,
    loss=None,  # defaults to nn.MSELoss() if None
    ...
)
```

---

## Training layer configurations

The `Layer` / `LayerList` API lets you define encoder and decoder stacks before passing them to `VAE`:

```python
from embkit.factory.layers import Layer, LayerList

encoder_layers = LayerList([
    Layer(1024, activation="relu"),
    Layer(512,  activation="relu"),
    Layer(256,  activation="relu"),
])
decoder_layers = LayerList([
    Layer(512,  activation="relu"),
    Layer(1024, activation="relu"),
])

vae = VAE(
    features=feature_list,
    latent_dim=128,
    encoder_layers=encoder_layers,
    decoder_layers=decoder_layers,
)
```

The final linear projection to `latent_dim` (encoder) and to `feature_dim` (decoder) is added automatically by `LayerList.build()`. You do not need to include them.

See the [Concepts page](concepts.md) for details on the factory and layer system.

---

## Saving and loading

```python
from embkit.factory import save, load

save(vae, "model.file")
vae2 = load("model.file")
```

The model file stores both weights and architecture. See [Factory API](api/factory/index.md) for details.
