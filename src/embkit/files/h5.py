import h5py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class H5Reader(Dataset):
    def __init__(self, filename, group, device="cpu"):
        self.hfile = h5py.File(filename)
        self.group = group
        self.data = self.hfile[self.group]["X"]
        self.index = pd.Index(i.decode("utf-8") for i in self.hfile[self.group]["obs/_index"])
        self.columns = pd.Index(i.decode("utf-8") for i in self.hfile[self.group]["var/_index"])
        self.shape = self.data.shape
        self.dest_device = device

    def __len__(self):
        return self.shape[0]
    
    def to(self, dev):
        self.dest_device = dev

    def __getitem__(self, idx):
        X_sample = np.nan_to_num( self.data[idx] )
        X_tensor = torch.from_numpy(X_sample).float()
        return X_tensor.to(self.dest_device)

class H5Writer:
    def __init__(self, filename, group, index, columns):
        self.h5f = h5py.File(filename, "w")

        self.index = pd.Index(index)
        self.columns = pd.Index(columns)

        self.h5f.attrs['encoding-type'] = 'anndata'.encode('utf8')
        self.h5f.attrs['encoding-version'] = '0.1.0'.encode('utf8')

        base_group = self.h5f.create_group(group)

        obs_group = base_group.create_group("obs")
        obs_group.create_dataset("_index",
                                data=index,
                                dtype=h5py.string_dtype())

        var_group = base_group.create_group("var")
        var_group.create_dataset("_index",
                                data=columns,
                                dtype=h5py.string_dtype())

        self.dataset = base_group.create_dataset("X", (len(index),len(columns)), dtype='f', compression='gzip')

    def set_row(self, name, row):
        i = self.index.get_loc(name)
        self.dataset[i] = row

    def set_irow(self, i, row):
        self.dataset[i] = row
    
    def close(self):
        self.h5f.close()