
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

## Quickstart

### 1) Inspect available commands

```bash
embkit --help
embkit model --help
```

### 2) Normalize a matrix

```bash
embkit matrix normalize data/raw.tsv --out data/normalized.tsv
```

### 3) Train a VAE

```bash
embkit model train-vae data/normalized.tsv \
    --epochs 120 \
    --latent 256 \
    --schedule "20:0,20:0.1,40:0.3,40:0.4" \
    --out vae.model
```

### 4) Encode samples into latent space

```bash
embkit model encode data/normalized.tsv vae.model --out embedding.tsv
```

### 5) Encode protein sequences

```bash
embkit protein encode sequences.fasta --model t33 --output protein_embeddings.tsv
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

## Development

To install the library locally:

```bash
pip install -e .
```

### Running tests

```bash
coverage run --source=embkit -m unittest discover -s tests
```

### Coverage report

To generate an HTML coverage report:

```bash
coverage html
```

To open the report in a browser:

**macOS**

```bash
open htmlcov/index.html
```

**Linux**

```bash
xdg-open htmlcov/index.html
```

**Windows**

```bash
start htmlcov\index.html
```

## Documentation

Full documentation, including core concepts, the training guide, the CLI reference, and API docs, is available in the [docs folder](docs/). To build it locally, see [DEV.md](DEV.md).

## License

MIT

## Authors

- Kyle Ellrott — ellrott@ohsu.edu
- Raphael Kirchgaessner — kirchgae@ohsu.edu

