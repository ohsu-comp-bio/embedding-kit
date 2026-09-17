# Models

`embkit.models` provides the trainable model classes. The core building blocks are the VAE family, plus general-purpose components.

## VAE family (`embkit.models.vae`)

| Class | Description |
|-------|-------------|
| `VAE` | Base class: holds an `Encoder` and `Decoder`, provides `forward(x)`, `encode(x)`, serialization. |
| `BaseVAE` | The standard model to build directly — `BaseVAE(features, latent_dim, encoder_layers, decoder_layers)`. |
| `RNAVAE` | VAE tuned for RNA-seq with `BatchNorm + ReLU` latent heads. |
| `NetVAE` | Pathway-constrained VAE (maintenance mode) — latent dims map to pathway groups. |
| `Encoder` / `Decoder` | Modular encoder/decoder stacks built from `LayerList` configs. |

See the [VAE page](vae.md) for the full class reference and the [NetVAE page](net_vae.md) for the constrained variant.

## General-purpose

- **[FFNN](ffnn.md)** (`embkit.models.ffnn`) — a multi-layer feed-forward network for supervised tasks, trained with `optimize.fit`.
- **[PairPredictor](pair.md)** (`embkit.models.pair`) — predicts a scalar interaction from a pair of items and a context.

## Building blocks

- [Factory](../factory/index.md) — layer configuration (`Layer`, `LayerList`), `build`/`save`/`load`.
- [Layers](../layers/layers.md) — reusable `nn.Module` building blocks (`MaskedLinear`, `PairwiseComparison`, `TSPLayer`).
