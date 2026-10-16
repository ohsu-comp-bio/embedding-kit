
# Embedding Kit

Methods for data normalization, embedding, synthesis and transformation in computational biology workflows.

Embedding Kit (embkit) is a toolkit for building and applying embedding models. It combines:

- CLI commands for repeatable model training and encoding
- PyTorch VAE model components for custom pipelines
- Utilities for loading, normalizing, and aligning large molecular datasets
- ESM2-based protein sequence embeddings

Use Embedding Kit when you want to move from tabular molecular data (RNA-seq, proteomics, methylation) to trainable latent representations that can be reused for downstream analysis.

## Features

- **VAE & NetVAE training** — Train variational autoencoders, including pathway-constrained NetVAE models, from tabular or HDF5 matrices.
- **Beta-KL scheduling** — Schedule the KL-regularization weight across training epochs for stable convergence.
- **Protein embeddings** — Generate sequence embeddings from FASTA files using ESM2 models.
- **Normalization utilities** — Min-max and exponential min-max normalization for expression matrices.
- **Device auto-detection** — Runs on CPU, CUDA, or Apple Metal GPUs.
- **Python API** — Composable layers, losses, and model factories for custom pipelines beyond the CLI.

## Installation

```bash
pip install embkit
```


## Training in Python

```python
from embkit import dataframe_loader
from embkit.models.vae import VAE
from embkit.factory.layers import Layer
from embkit.losses import BCEWithLogitsVAELoss
from embkit.factory import save, load
from embkit import optimize

loader = dataframe_loader(df_norm, batch_size=256)

vae = VAE(
    features=list(df_norm.columns),
    latent_dim=128,
    encoder_layers=[Layer(512, activation="relu"), Layer(256, activation="relu")],
    decoder_layers=[Layer(512, activation="relu")],
)

optimize.fit_vae(vae, X=loader, epochs=60, lr=1e-3, loss=BCEWithLogitsVAELoss())

save(vae, "vae.model")
```

## Documentation

Full documentation, including core concepts, the training guide, the CLI reference, and API docs, is available in the [docs folder](docs/). To build it locally, see [DEV.md](DEV.md).

## License

MIT

## Authors

- Kyle Ellrott — ellrott@ohsu.edu
- Raphael Kirchgaessner — kirchgae@ohsu.edu

