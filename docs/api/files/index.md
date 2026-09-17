# Files

I/O helpers for large tabular and HDF5 data, plus loaders for common bioinformatics formats. These back the `.h5` input path of `embkit model train-vae` and the `matrix` commands.

| Name | Purpose |
|------|---------|
| `CsvReader`, `LargeCsvReader` | Read CSV/TSV with an index column; `LargeCsvReader` supports streaming via an external index |
| `H5Reader`, `H5Writer` | Read/write HDF5 groups |
| `load_gct`, `load_raw_hugo`, `load_gtex_hugo` | Load GCT / HUGO / GTEx-HUGO files into DataFrames |

```python
import torch
from embkit.files import H5Reader, load_gct

reader = H5Reader("matrix.h5", group="rna", device="cpu")
print(reader.shape)              # (n_samples, n_features)
x, = reader[0]                    # a torch tensor for the first sample
dataloader = torch.utils.data.DataLoader(reader, batch_size=64)

df = load_gct("gtex.hugo.tsv")
```

## API

::: embkit.files
    options:
      show_source: false
      filters:
        - "!^_"
