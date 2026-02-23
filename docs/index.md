# Embedding Kit

Embedding Kit is a toolkit for building and applying embedding models in computational biology workflows.

It combines:

- CLI commands for repeatable model training and encoding
- PyTorch model components for custom pipelines
- Utilities for loading and normalizing large matrix-style datasets

Use Embedding Kit when you want to move from tabular molecular data to trainable latent representations that can be reused for downstream analysis.

## Installation

```bash
pip install embkit
```

## Quickstart

### 1) Inspect available commands

```bash
embkit --help
embkit help model
```

### 2) Train a VAE model from a TSV matrix

```bash
embkit model train-vae ./data/train.tsv --epochs 120 --latent 256 --out vae.model
```

### 3) Encode samples into latent space

```bash
embkit model encode ./data/train.tsv vae.model --out embedding.tsv
```

## Next Steps

- Follow the GTEx workflow in the examples section for an end-to-end RNA expression case.
- See the HLA2Vec example for peptide embedding + pairwise training.
- Explore API docs if you want to customize models in Python.

