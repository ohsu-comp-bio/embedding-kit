# API Reference

The `embkit` Python API is organized into focused modules. Use the sidebar to navigate.

## Core

| Module | What it provides |
|--------|------------------|
| [Helpers & Data Adapters](helpers.md) | Top-level `embkit` helpers (`dataframe_loader`, `get_device`, …) and `embkit.datasets` adapters. |
| [Factory](factory/index.md) | Layer configuration (`Layer`, `LayerList`), and `build` / `save` / `load` for serialization. |
| [Models](models/index.md) | `VAE`, `BaseVAE`, `RNAVAE`, `NetVAE`, `FFNN`, `PairPredictor`, and modular encoder/decoder. |
| [Training](optimize.md) | `fit`, `fit_vae`, `fit_net_vae` training loops. |
| [Losses](losses/index.md) | `MSEVAELoss`, `BCEVAELoss`, `BCEWithLogitsVAELoss`, `BCEKLWeightedVAELoss`. |
| [Estimator](estimator/index.md) | scikit-learn style `VAEEstimator`. |

## Building blocks

| Module | What it provides |
|--------|------------------|
| [Modules](layers/layers.md) | Reusable `nn.Module`s: `MaskedLinear`, `PairwiseComparison`, `TSPLayer`, attention blocks. |
| [Constraints](constraints/index.md) | `PathwayConstraintInfo` for masked-layer connectivity. |
| [Metrics](metrics/distance.md) | Pairwise distance helpers (`eucpair`, `mhat`). |
| [Preprocessing](preprocessing.md) | Normalization scalers and data loaders. |

## Data & integrations

| Module | What it provides |
|--------|------------------|
| [Resources](resources/index.md) | Downloaders for GTEx, HUGO, SIF, cBioPortal datasets. |
| [Files](files/index.md) | CSV/HDF5 readers and writers, GCT/HUGO loaders. |
| [Encoding](encoding/index.md) | Protein (ESM2) and genome embedding encoders, one-hot encoders. |
| [Commands](commands/index.md) | The `click` CLI command objects. |
| [Align](align.md) | Embedding-space alignment (Spearman, Procrustes). |
| [Utilities](utilities/index.md) | `run_pca`, `run_kmeans` for large matrices. |

## Standalone utilities

- [BMEG](bmeg.md) — BMEG graph connection helpers.
- [Pathway](pathway.md) — SIF parsing and mask-building helpers.
