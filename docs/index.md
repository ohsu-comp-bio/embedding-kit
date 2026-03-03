# Embedding Kit

Embedding Kit is a toolkit for building and applying embedding models in computational biology workflows.

It combines:

- CLI commands for repeatable model training and encoding
- PyTorch VAE model components for custom pipelines
- Utilities for loading, normalizing, and aligning large molecular datasets
- ESM2-based protein sequence embeddings

Use Embedding Kit when you want to move from tabular molecular data (RNA-seq, proteomics, methylation) to trainable latent representations that can be reused for downstream analysis.

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
from embkit.losses import bce_with_logits
from embkit.factory import save, load
from embkit import optimize

loader = dataframe_loader(df_norm, batch_size=256)

vae = VAE(
    features=list(df_norm.columns),
    latent_dim=128,
    encoder_layers=[Layer(512, activation="relu"), Layer(256, activation="relu")],
    decoder_layers=[Layer(512, activation="relu")],
)

optimize.fit_vae(vae, X=loader, epochs=60, lr=1e-3, loss=bce_with_logits)

save(vae, "vae.model")
```

## Next steps

- Read [Core Concepts](concepts.md) to understand how VAEs and beta-KL annealing work.
- Follow the [Training Guide](training.md) for a full reference on `fit_vae` options and loss functions.
- See the [CLI Reference](cli.md) for all command options.
- Browse the [GTEx example](examples/gtex.md) for an end-to-end RNA expression workflow.
- Explore the [API reference](api/index.md) for Python-level documentation.
