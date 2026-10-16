"""
Readers and writers for HDF5-backed matrices.

HDF5 lets embkit store and stream large sample matrices on disk without loading
the whole thing into memory — important for expression matrices with many
samples and features. The readers implement the PyTorch
:class:`~torch.utils.data.Dataset` interface so they drop straight into a
:class:`~torch.utils.data.DataLoader`, and the writers produce an
anndata-compatible layout (``obs``/``var`` groups with a ``X`` matrix) so the
data can be consumed by other tools too.

- :class:`H5Reader` / :class:`H5Writer` — 2D (samples x features) matrices.
- :class:`H5CubeReader` / :class:`H5CubeWriter` — 3D cubes (samples x
  position x feature), e.g. encoded protein sequences.
"""
import h5py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _decode_index(values):
    """Decode raw HDF5 index bytes into a pandas Index of strings."""
    return pd.Index(v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v) for v in values)


class _H5BaseReader(Dataset):
    """
    Base H5 matrix reader: a dataset that yields one row at a time.

    Opens an HDF5 group laid out as an anndata file (``X`` matrix plus
    ``obs/_index`` sample names) and serves rows as float32 tensors on the
    requested device. NaN/Inf values are converted to zero to keep tensors
    clean. Subclasses add feature (``var``) metadata or a 3D layout.
    """
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
    """
    Read a 2D (samples x features) H5 matrix, exposing its feature names.

    Extends the base reader with the ``var/_index`` feature names as
    :attr:`columns`, so callers know which feature each column corresponds to.
    """
    def __init__(self, filename, group, device="cpu"):
        super().__init__(filename, group, device=device)
        self.columns = _decode_index(self.hfile[self.group]["var/_index"])

class H5Writer:
    """
    Write a 2D (samples x features) H5 matrix, row by row.

    Creates an anndata-style HDF5 group up front with a pre-allocated ``X``
    matrix sized to the given number of rows and columns, then lets rows be
    written incrementally by name (:meth:`set_row`) or by index
    (:meth:`set_irow`). The row/column names are stored in the ``obs``/``var``
    groups so a reader can recover them.
    """
    def __init__(self, filename, group, index, columns):
        """
        Create the HDF5 file and allocate the matrix to hold ``len(index)``
        rows.

        Parameters:
            filename: Path of the HDF5 file to create.
            group: Name of the group to store the matrix under.
            index: Row (sample) names; stored in ``obs/_index``.
            columns: Column (feature) names, or an int for the feature count;
                stored in ``var/_index``.
        """
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
        """Write one row's values, locating the row by its sample name."""
        i = self.index.get_loc(name)
        self.dataset[i] = row

    def set_irow(self, i, row):
        """Write one row's values at the given row index."""
        self.dataset[i] = row

    def close(self):
        """Flush and close the underlying HDF5 file."""
        self.h5f.close()

class H5CubeWriter:
    """
    Write a 3D (samples x position x feature) H5 cube, row by row.

    Specialized for storing a dataset of matrices, e.g. a set of encoded
    sequences where each sample is a ``(position, feature)`` matrix. The full
    cube is allocated up front and written one sample at a time by name or
    index, mirroring the 2D :class:`H5Writer`.
    """
    def __init__(self, filename, group, index, xsize:int, ysize:int):
        """
        Create the HDF5 file and allocate the cube for ``len(index)`` samples,
        each of shape ``(xsize, ysize)``.
        """
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
    Read a 3D (samples x position x feature) H5 cube, indexed by sample name.

    Each sample is a matrix (e.g. an encoded sequence). Yields one matrix per
    index, and :meth:`get_loc` resolves a sample name to its row position.
    """

    def __init__(self, filename, group, device="cpu"):
        super().__init__(filename, group, device=device)

    def get_loc(self, name):
        """Resolve a sample name to its row position in the cube."""
        return self.index.get_loc(name)
