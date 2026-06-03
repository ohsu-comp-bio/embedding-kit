import h5py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _decode_index(values):
    return pd.Index(v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v) for v in values)


class _H5BaseReader(Dataset):
    def __init__(self, filename, group, device="cpu"):
        self.hfile = h5py.File(filename)
        self.group = group
        self.data = self.hfile[self.group]["X"]
        self.index = _decode_index(self.hfile[self.group]["obs/_index"])
        self.shape = self.data.shape
        self.dest_device = device

    def __len__(self):
        return self.shape[0]

    def to(self, dev):
        self.dest_device = dev

    def __getitem__(self, idx):
        x_sample = np.nan_to_num(self.data[idx])
        x_tensor = torch.from_numpy(x_sample).float()
        return (x_tensor.to(self.dest_device),)


class H5Reader(_H5BaseReader):
    def __init__(self, filename, group, device="cpu"):
        super().__init__(filename, group, device=device)
        self.columns = _decode_index(self.hfile[self.group]["var/_index"])

class H5Writer:
    def __init__(self, filename, group, index, columns):
        self.h5f = h5py.File(filename, "w")

        base_group = self.h5f.create_group(group)

        self.index = pd.Index(index)
        obs_group = base_group.create_group("obs")
        obs_group.create_dataset("_index",
                                data=index,
                                dtype=h5py.string_dtype())


        if isinstance(columns, int):
            self.columns = pd.RangeIndex(columns)
            var_group = base_group.create_group("var")
            var_group.create_dataset("_index",
                                    data=columns,
                                    dtype=int)
            self.dataset = base_group.create_dataset("X", (len(index),columns), dtype='f')
        else:
            self.columns = pd.Index(columns)
            var_group = base_group.create_group("var")
            var_group.create_dataset("_index",
                                    data=columns,
                                    dtype=h5py.string_dtype())
            self.dataset = base_group.create_dataset("X", (len(index),len(columns)), dtype='f')

        self.h5f.attrs['encoding-type'] = 'anndata'.encode('utf8')
        self.h5f.attrs['encoding-version'] = '0.1.0'.encode('utf8')


    def set_row(self, name, row):
        i = self.index.get_loc(name)
        self.dataset[i] = row

    def set_irow(self, i, row):
        self.dataset[i] = row
    
    def close(self):
        self.h5f.close()

class H5CubeWriter:
    """
    H5CubeWriter

    Specialized wrapper for storing datasets of matrices. Example usage:
    storing a dataset of encoded sequences
    """
    def __init__(self, filename, group, index, xsize:int, ysize:int):
        self.h5f = h5py.File(filename, "w")

        self.index = pd.Index(index)
        self.shape = (len(index), xsize, ysize)

        base_group = self.h5f.create_group(group)

        obs_group = base_group.create_group("obs")
        obs_group.create_dataset("_index",
                                data=index,
                                dtype=h5py.string_dtype())

        self.dataset = base_group.create_dataset("X", (len(index), xsize, ysize), dtype='f')

    def set_row(self, name, row):
        i = self.index.get_loc(name)
        self.dataset[i] = row

    def set_irow(self, i, row):
        self.dataset[i] = row
    
    def close(self):
        self.h5f.close()

class H5CubeReader(_H5BaseReader):
    """
    Store indexed set of 2d matrices
    """

    def __init__(self, filename, group, device="cpu"):
        super().__init__(filename, group, device=device)
    
    def get_loc(self, name):
        return self.index.get_loc(name)
