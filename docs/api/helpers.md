# Helpers & Data Adapters

The top-level `embkit` module provides small utilities used throughout the Python API, and `embkit.datasets` provides PyTorch dataset adapters for custom training loops.

## Top-level helpers (`embkit`)

| Name | Purpose |
|------|---------|
| `get_device` | Return the best available device (CUDA → MPS → CPU) |
| `dataframe_tensor` | Convert a `pd.DataFrame` to a `torch.Tensor` (on the chosen device) |
| `tensor_dataframe` | Convert a `torch.Tensor` back to a `pd.DataFrame` |
| `dataframe_dataset` | Wrap a `pd.DataFrame` in a PyTorch `Dataset` |
| `dataframe_loader` | Wrap a `pd.DataFrame` in a `DataLoader` (batching + shuffling) |

```python
import pandas as pd, torch
from embkit import dataframe_loader, dataframe_tensor, tensor_dataframe, get_device

device = get_device()
loader = dataframe_loader(df, batch_size=256)
for (x,) in loader:            # each x is on the right device
    print(x.shape, x.device)

# round-trip tensors <-> dataframes (e.g. to save embeddings)
z = vae.encode(dataframe_tensor(df_norm))
embedding_df = tensor_dataframe(z, index=df_norm.index)
```

## Dataset adapters (`embkit.datasets`)

| Name | Purpose |
|------|---------|
| `BalancedMixer` | Interleave multiple datasets in a balanced fashion |
| `DatasetMask` | Expose only a subset of a dataset via a boolean mask |
| `DataFrameMapper` | Apply mappers on columns of a DataFrame as a dataset |
| `ConstantLabel` | Pair a dataset with a constant label |
| `ZipDataset`, `ChainDataset` | Combine datasets element-wise or sequentially |

::: embkit
    options:
      show_source: false
      members:
        - get_device
        - dataframe_tensor
        - tensor_dataframe
        - dataframe_dataset
        - dataframe_loader
      filters:
        - "!^_"

## API

::: embkit.datasets
    options:
      show_source: false
      filters:
        - "!^_"
