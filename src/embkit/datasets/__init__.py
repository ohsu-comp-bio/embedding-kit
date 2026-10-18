
"""
Dataset adapters for training.

PyTorch expects a :class:`~torch.utils.data.Dataset` that yields samples, but
embkit's inputs come in many shapes — a list of matrices to draw from in
balanced fashion, a matrix filtered down to a feature mask, a pandas DataFrame
with several columns to be mapped to separate tensors, or several datasets that
share the same sample indices. These adapters turn each of those cases into a
usable dataset so the same training loop works across all of them.

- :class:`BalancedMixer` — interleave samples from multiple datasets.
- :class:`DatasetMask` — filter each row to a subset of its features.
- :class:`DataFrameMapper` — build a multi-tensor sample from a DataFrame row.
- :class:`ConstantLabel` — attach a fixed label to every sample.
- :class:`ZipDataset` / :class:`ChainDataset` — combine datasets that share the
  same index.
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader, Dataset




class BalancedMixer(IterableDataset):
    """
    Interleave samples from multiple datasets.

    Used when training on several matrices at once and you want roughly balanced
    coverage of each, rather than exhausting one before starting the next. It
    is an :class:`~torch.utils.data.IterableDataset`: each iteration picks a
    random source dataset and yields one of its samples, restarting that source
    when it runs out. Because selection is per-sample and stream-based, it works
    well with multi-worker loading (each worker draws its own stream).
    """
    def __init__(self, datasets, seed: int = 0):
        self.datasets = datasets
        self.seed = seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0))
        iters = [iter(ds) for ds in self.datasets]
        while True:
            i = int(rng.integers(low=0, high=len(iters)))
            try:
                yield next(iters[i])
            except StopIteration:
                iters[i] = iter(self.datasets[i])
                yield next(iters[i])


class DatasetMask(Dataset):
    """
    DatasetMask

    Given an input dataset, return a subset of the values of a row based on a mask
    """
    def __init__(self, dataset, mask, device=None):
        self.dataset = dataset
        self.mask = mask
        self.device = device
        self.dim = len(mask)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        o = []
        x = self.dataset[idx]
        for i in range( self.dim ):
            if self.device is not None:
                o.append( x[i][self.mask[i]].to(self.device) )
            else:
                o.append( x[i][self.mask[i]] )
        return o


class DataFrameMapper(Dataset):
    """
    Build a multi-part sample from a single DataFrame row.

    Wraps a pandas :class:`~pandas.DataFrame` so that each index yields a list
    of tensors — one per ``mappers`` entry. This is how heterogeneous inputs
    (e.g. expression + a label, or two matrices) are fed to a model from one
    table. Each mapper is applied to its column and the result moved to the
    requested device/dtype.
    """
    def __init__(self, data, mappers, device=None, dtype=None):
        self.data = data
        self.mappers = mappers
        self.device = device
        self.dtype = dtype

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        out = []
        for k, v in self.mappers:
            out.append( v(row[k]).to(self.device, dtype=self.dtype) )
        return out

class ConstantLabel(Dataset):
    """
    Attach the same fixed label to every sample.

    Useful for the "input" side of a multi-input model where every sample in a
    batch shares one label (e.g. a tissue type applied to a whole dataset).
    Each index yields ``[sample, label]``.
    """
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return [self.data[idx], self.label]

class ZipDataset(Dataset):
    """
    Combine datasets that share the same index into one multi-input sample.

    All datasets must have the same length. Index ``i`` returns a tuple of the
    ``i``-th element from each dataset, so the training loop receives them
    together as a single sample (e.g. an expression row plus a matched feature
    row).
    """
    def __init__(self, *datasets):
        if len(datasets) == 0:
            raise ValueError("ZipDataset requires at least one dataset.")
        base_len = len(datasets[0])
        if not all(len(d) == base_len for d in datasets):
            raise ValueError("All datasets passed to ZipDataset must have equal length.")
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datasets)

class ChainDataset(Dataset):
    """
    Flatten several multi-part datasets into one flat sample per index.

    Like :class:`ZipDataset`, all datasets must share the same length, but
    instead of nesting them in a tuple it concatenates each dataset's parts for
    index ``i`` into a single flat tuple. Use this when a model expects a single
    ordered list of tensors per sample rather than per-dataset grouping.
    """
    def __init__(self, *datasets):
        if len(datasets) == 0:
            raise ValueError("ChainDataset requires at least one dataset.")
        base_len = len(datasets[0])
        if not all(len(d) == base_len for d in datasets):
            raise ValueError("All datasets passed to ChainDataset must have equal length.")
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        out = []
        for d in self.datasets:
            out.extend(d[idx])
        return tuple(out)