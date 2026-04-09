# Core Concepts

This page explains the key ideas behind Embedding Kit: how VAEs work, how beta-KL annealing controls the latent space, and how the factory system lets you save and reload any model.

---

## Variational Autoencoders (VAEs)

A VAE is a neural network that learns a compressed, continuous representation of your data called a **latent space**.

### Architecture

```
Input x  ──►  Encoder  ──►  μ, σ  ──►  z  ──►  Decoder  ──►  x̂
                                  ▲
                         reparameterization trick
```

The **Encoder** maps each input sample to two vectors:

- **μ (mu)** — the mean of a Gaussian distribution in latent space
- **log σ² (logvar)** — the log variance of that distribution

A latent sample **z** is then drawn by:

```
z = μ + ε * exp(0.5 * logvar),   ε ~ N(0, I)
```

This is the **reparameterization trick**: the randomness lives in ε, so gradients flow through μ and logvar to the encoder weights.

The **Decoder** takes z and reconstructs x̂.

### Loss function

VAE training minimizes:

```
L = reconstruction_loss(x̂, x)  +  β * KL(q(z|x) || p(z))
```

- **Reconstruction loss** measures how faithfully x̂ matches x (MSE, BCE, or BCE-with-logits).
- **KL divergence** regularizes the latent space toward a standard normal prior N(0, I), preventing the encoder from memorizing inputs and forcing the latent space to be smooth and interpolatable.
- **β (beta)** controls the trade-off. β=0 gives a plain autoencoder; β=1 gives the standard VAE; β>1 applies stronger regularization (β-VAE).

---

## Beta-KL Annealing

Training with β=1 from the start can cause the KL term to dominate before the decoder learns anything useful, leading to **posterior collapse** where z is ignored.

Embedding Kit solves this with a **beta schedule**: a sequence of `(beta_value, n_epochs)` phases. Beta starts at 0 (or a small value) and is gradually increased:

```python
schedule = [
    (0.0, 20),   # warm up: reconstruction only
    (0.1, 20),   # introduce a small KL penalty
    (0.3, 40),   # increase regularization
    (0.4, 40),   # settle at final beta
]
```

A single optimizer is reused across all phases so Adam's momentum is preserved. The `model.history` dict records `loss`, `recon`, `kl`, and `beta` for every epoch.

---

## Model Classes

### VAE

The standard model. Takes a feature list, latent dimension, and optional `encoder_layers` / `decoder_layers` built from `Layer` objects.

```
VAE(features, latent_dim, encoder_layers, decoder_layers)
  forward(x) -> recon, mu, logvar, z
  encode(x)  -> z   (no_grad)
```

### RNAVAE

A VAE tuned for RNA-seq data. The encoder applies `BatchNorm + ReLU` to the latent heads (mu, logvar), constraining them to non-negative values. It has a built-in `fit()` method with beta warmup and early stopping.

### NetVAE

A pathway-constrained VAE. The encoder uses `MaskedLinear` layers so each latent dimension corresponds to a biological pathway or transcription factor group. Connections between features and groups not in the pathway are forced to zero.

---

## The Factory System

The factory system handles **model serialization** so that you can save a model and reload it on any machine without re-defining the architecture.

### How it works

Every registered model class implements two methods:

- `to_dict()` — returns a plain Python dict describing the architecture
- `from_dict(desc)` — reconstructs the model from that dict

When you call `factory.save(model, path)`, two things are written into the file:

1. The PyTorch `state_dict` (all weights)
2. A `__model__` key containing the architecture description dict

`factory.load(path)` reads the `__model__` key, calls `factory.build()` to reconstruct the architecture, then loads the weights.

```python
from embkit.factory import save, load

save(vae, "my_model.model")
vae2 = load("my_model.model")
```

### Layer and LayerList

Layers are described as **configuration objects**, not `nn.Module` instances, before training. This separation means the architecture can be serialized cleanly.

```python
from embkit.factory.layers import Layer, LayerList

# Layer(units, activation, op, batch_norm, bias)
encoder_layers = LayerList([
    Layer(512, activation="relu"),
    Layer(256, activation="relu"),
])
```

A `LayerList` is a list of `Layer` configs. When the `VAE` is constructed it calls `LayerList.build(input_dim, latent_dim)` to produce an actual `nn.Sequential`.

You can also pass a comma-separated string of sizes to the CLI; the `train-vae` command parses this into a `LayerList` automatically.

### Supported activations

`"relu"`, `"sigmoid"`, `"tanh"`, `"leaky_relu"`, `"none"`

### Pathway-masked layers

For `NetVAE`, layers use `op="masked_linear"` with a pathway constraint object that describes which features connect to which pathway groups:

```python
from embkit.factory.layers import Layer
from embkit.constraints import PathwayConstraintInfo

Layer(
    units=n_groups,
    op="masked_linear",
    constraint=PathwayConstraintInfo("features-to-group", feature_map),
)
```

`PathwayConstraintInfo` generates a binary mask at build time; weights at masked positions are set to zero and kept at zero throughout training.
