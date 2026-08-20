
import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader, Dataset




class BalancedMixer(IterableDataset):
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
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return [self.data[idx], self.label]

class ZipDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        # Ensure all zipped datasets are of equal length
        assert all(len(d) == len(datasets[0]) for d in datasets)

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datasets)

class ChainDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        # Ensure all chained datasets are of equal length
        assert all(len(d) == len(datasets[0]) for d in datasets)

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        out = []
        for d in self.datasets:
            out.extend(d[idx])
        return tuple(out)