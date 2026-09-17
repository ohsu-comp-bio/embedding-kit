# Commands

`embkit` groups its CLI by domain. The full option reference is in the [CLI Reference](../../cli.md); this page maps each command group to its CLI entry point.

| Command group | CLI | Purpose | Key modules |
|---------------|-----|---------|-------------|
| `model` | `embkit model` | Train VAE/NetVAE, encode, verify | `embkit.commands.model` |
| `matrix` | `embkit matrix` | Normalize / PCA feature matrices | `embkit.commands.matrix` |
| `protein` | `embkit protein` | ESM2 protein embeddings | `embkit.commands.protein` |
| `resources` | `embkit resources` | Download GTEx / HUGO / SIF | `embkit.commands.resources` |
| `cbio` | `embkit cbio` | cBioPortal studies & download | [cbio](cbio.md) |
| `align` | `embkit align` | Align two embedding spaces | `embkit.commands.align` |

::: embkit.commands
    options:
      show_root_heading: true
      show_root_full_path: true
      show_source: false
