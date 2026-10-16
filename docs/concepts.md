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




