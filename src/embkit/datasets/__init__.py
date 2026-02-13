
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
