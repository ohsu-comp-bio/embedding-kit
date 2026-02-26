# Project Overview

Embedding Kit (`embkit`) is a toolkit for building latent-space representations of molecular data using Variational Autoencoders (VAEs). It is designed for computational biology workflows where the input is a large tabular matrix of measurements — gene expression, protein levels, methylation — and the goal is a compact, continuous embedding that captures biological variation.

## Architecture overview

```
Raw data
   │
   ▼
Normalization (ExpMinMaxScaler, quantile_max_norm)
   │
   ▼
VAE training (VAE / RNAVAE / NetVAE)
   │   ├─ Encoder:  features → hidden layers → μ, σ
   │   └─ Decoder:  z → hidden layers → reconstruction
   │
   ▼
Latent embeddings
   │
   ├─ Downstream analysis (clustering, UMAP, classification)
   └─ Alignment across datasets (align pair)
```

## Features

- **VAE models** — `VAE` for general tabular data, `RNAVAE` for RNA-seq with biology-specific architecture choices, `NetVAE` for pathway-constrained embeddings
- **Modular layer system** — build encoder/decoder stacks with `Layer` and `LayerList`; factory serialization stores architecture + weights in a single file
- **Beta-KL scheduling** — multi-phase annealing to prevent posterior collapse
- **Normalization** — log2+1 min-max scaling (`ExpMinMaxScaler`), quantile normalization, zero-masking
- **Protein embeddings** — wraps Meta's ESM2 models for per-sequence or per-residue embeddings from FASTA files
- **Embedding alignment** — Spearman-based optimal pairing of two embedding spaces via `align pair`
- **Data loaders** — GTEx (gene TPM and transcript TPM), cBioPortal studies, HUGO gene nomenclature, pathway SIF files

## Who should use this

- Computational biologists building multi-modal or pan-cancer embeddings
- Data scientists integrating embedding-based analysis into genomics pipelines
- Developers extending the toolkit with custom models via the factory registration system

## Contributing

Open an issue or pull request at [GitHub](https://github.com/ohsu-comp-bio/embedding-kit).

## Contact

**Maintainers**: Kyle Ellrott (ellrott@ohsu.edu), Raphael Kirchgaessner (kirchgae@ohsu.edu)

**Issue tracker**: [GitHub Issues](https://github.com/ohsu-comp-bio/embedding-kit/issues)
