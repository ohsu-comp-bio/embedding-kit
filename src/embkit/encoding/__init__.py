
import math

import numpy as np

import torch
import torch.nn.functional as F
from .. import factory
from ..files import CsvReader

@factory.nn_module
class OneHotEncoder:
    def __init__(self, classes, device=None, dtype=None):
        self.classes = sorted(classes)
        self.num_classes = len(self.classes)
        self.mapping = {}
        self.class_idx = {}
        self.device = device
        self.shape = (self.num_classes,)
        for i, n in enumerate(self.classes):
            self.mapping[n] = F.one_hot( torch.tensor(i), self.num_classes ).to(device=device, dtype=dtype)
            self.class_idx[n] = i

    def __call__(self, x):
        # Support single label or batch of labels
        # Single label (string/key): return precomputed one-hot tensor
        if not isinstance(x, (list, tuple, torch.Tensor, np.ndarray)):
            return self.mapping[x]

        # Batch input: convert each item to its index and return stacked one-hots
        # Accept lists/tuples/ndarrays/tensors of class keys or numeric indices
        indices = []
        for item in x:
            if torch.is_tensor(item):
                # handle scalar tensors
                if item.numel() == 1:
                    item = int(item.item())
                else:
                    item = item.tolist()
            if isinstance(item, int):
                idx = int(item)
            else:
                idx = self.class_idx[item]
            indices.append(idx)

        idx_tensor = torch.tensor(indices, device=self.device)
        return F.one_hot(idx_tensor, num_classes=self.num_classes).to(self.device)

    def __len__(self):
        return self.num_classes

    def __iter__(self):
        return iter(self.classes)

    def to_dict(self):
        return {
            "classes": self.classes,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(classes=data["classes"])

amino_acids = 'ARNDCEQGHILKMFPSTWYV'

@factory.nn_module
class ProteinOneHotEncoder:
    """
    Converts an amino acid sequence string into a one-hot encoded matrix.

    The encoding uses 21 dimensions: 20 for standard amino acids,
    and 1 for 'X' (representing any unknown or non-standard amino acid)
    if encode_x is True.

    An additional dimension is added to encode the position in the chain
    if 'encode_pos' is True

    Args:
        sequence: A string of the amino acid sequence (e.g., "MGLSD").

    Returns:
        A torch tensor (matrix) of shape (L, oneof[20,21,22)), where L is the sequence length.
    """

    def __init__(self, full_len=None, encode_x=True, encode_pos=False, pe_dim=2, device=None, dtype=torch.float32, backend='torch'):
        self.full_len = full_len
        self.encode_x = encode_x
        self.encode_pos = encode_pos
        self.pe_dim = pe_dim
        self.device = device
        self.dtype = dtype
        self.backend = backend
        if encode_pos and pe_dim <= 0:
            raise ValueError(f"pe_dim must be > 0 when encode_pos=True, got pe_dim={pe_dim}")
        # determine torch and numpy dtype representations
        self.np_dtype = None
        if dtype is None:
            self.torch_dtype = torch.get_default_dtype()
        elif dtype == np.float32:
            self.torch_dtype = torch.float32
            self.np_dtype = np.float32
        elif dtype == np.float64:
            self.torch_dtype = torch.float64
            self.np_dtype = np.float64
        elif isinstance(dtype, torch.dtype):
            self.torch_dtype = dtype
        else:
            # fallback to float32
            self.torch_dtype = torch.float32
            self.np_dtype = np.float32
        # 1. Define the standard 20 amino acid alphabet
        # Ordered alphabetically (common practice, though order doesn't affect correctness)
        # 2. Add 'X' for unknown/non-standard residues
        # This makes the dimension 21
        if self.encode_x:
            self.alphabet = amino_acids + 'X'
        else:
            self.alphabet = amino_acids
        
        if self.full_len is not None:
            self.shape = (self.full_len, len(self.alphabet) + (self.pe_dim if self.encode_pos else 0))
        else:
            self.shape = (len(self.alphabet) + (self.pe_dim if self.encode_pos else 0),)

        # 3. Create a mapping dictionary for quick lookup
        # e.g., {'A': 0, 'R': 1, ..., 'V': 19, 'X': 20}
        self.aa_to_index = {aa: i for i, aa in enumerate(self.alphabet)}
        if not self.encode_x:
            self.aa_to_index["X"] = None

    # Backend selection helper
    def _use_numpy_backend(self):
        if self.backend == 'numpy':
            return True
        if self.backend == 'torch':
            return False
        raise ValueError(f"Unsupported backend: {self.backend}. Use 'torch' or 'numpy'.")

    def __call__(self, sequence):

        is_single = False
        # Fast path for single sequence
        if isinstance(sequence, str):
            L = len(sequence)
            FL = self.full_len if self.full_len is not None else L
            seqs = [sequence]
            is_single = True
        else:
            seqs = list(sequence)
            FL = self.full_len if self.full_len is not None else (max(len(s) for s in seqs) if len(seqs) > 0 else 0)
        dim = len(self.alphabet) + (self.pe_dim if self.encode_pos else 0)
        batch_size = len(seqs)
        if self._use_numpy_backend():
            np_dtype = self.np_dtype or np.float32
            one_hot_matrix = np.zeros((batch_size, FL, dim), dtype=np_dtype)
        else:
            one_hot_matrix = torch.zeros((batch_size, FL, dim), device=self.device, dtype=self.torch_dtype)

        for b, seq in enumerate(seqs):
            L = len(seq)
            for i, aa in enumerate(seq.upper()):
                if i >= FL:
                    break
                index = self.aa_to_index.get(aa, self.aa_to_index['X'])
                if index is not None:
                    one_hot_matrix[b, i, index] = 1.0
            if self.encode_pos and self.pe_dim > 0 and not self._use_numpy_backend():
                # Fill PE channels for this sequence
                pe = torch.stack([position_sin_cos_tensor(i, self.pe_dim, device=self.device, dtype=self.torch_dtype) for i in range(FL)], dim=0)
                one_hot_matrix[b, :, -self.pe_dim:] = pe
            elif self.encode_pos and self.pe_dim > 0 and self._use_numpy_backend():
                # numpy fallback: compute via torch then convert
                pe = np.stack([position_sin_cos(i, self.pe_dim) for i in range(FL)], axis=0)
                one_hot_matrix[b, :, -self.pe_dim:] = pe
            for i in range(L, FL):
                index = self.aa_to_index['X']
                if index is not None:
                    one_hot_matrix[b, i, index] = 1.0
        if is_single:
            return one_hot_matrix[0]
        return one_hot_matrix

    def to_dict(self):
        # Serialize dtype as a canonical string for reliable round-tripping
        if self.dtype == np.float32 or self.torch_dtype == torch.float32:
            dtype_str = "float32"
        elif self.dtype == np.float64 or self.torch_dtype == torch.float64:
            dtype_str = "float64"
        else:
            dtype_str = "float32"
        return {
            "full_len": self.full_len,
            "encode_x": self.encode_x,
            "encode_pos": self.encode_pos,
            "pe_dim": self.pe_dim,
            "device": self.device,
            "dtype": dtype_str,
            "backend": self.backend
        }

    @classmethod
    def from_dict(cls, data):
        dtype_str = data.get("dtype", "float32")
        if dtype_str == "float32":
            dtype = np.float32
        elif dtype_str == "float64":
            dtype = np.float64
        else:
            dtype = None  # default to torch's default dtype
        return cls(
            full_len=data.get("full_len"),
            encode_x=data.get("encode_x", True),
            encode_pos=data.get("encode_pos", False),
            pe_dim=data.get("pe_dim", 2),
            device=data.get("device"),
            dtype=dtype,
            backend=data.get("backend", 'torch')
        )


### Positional Encoding Functions

def position_fractional(pos: int, full_len: int) -> float:
    """Fraction-based positional encoding for position `pos`."""
    if full_len is None or full_len == 0:
        return float(pos)
    return float(pos) / float(full_len)


def position_sin_cos(pos: int, pe_dim: int, log_base: float = 10000.0) -> np.ndarray:
    """Sinusoidal positional encoding for a single position."""
    if pe_dim == 0:
        return np.array([])
    
    # Ensure even dimension
    dim = pe_dim if pe_dim % 2 == 0 else pe_dim + 1
    vec = np.zeros(dim, dtype=np.float64)
    
    for i in range(0, dim, 2):
        freq = np.exp(i * -(np.log(log_base) / dim))
        vec[i] = np.sin(pos * freq)
        if i + 1 < dim:
            vec[i + 1] = np.cos(pos * freq)
    
    return vec[:pe_dim]


def position_sin_cos_tensor(pos: int, pe_dim: int, log_base: float = 10000.0, device=None, dtype=torch.float32) -> torch.Tensor:
    """Sinusoidal positional encoding for a single position (torch version)."""
    if pe_dim == 0:
        return torch.tensor([], device=device, dtype=dtype)
    
    dim = pe_dim if pe_dim % 2 == 0 else pe_dim + 1
    vec = torch.zeros(dim, device=device, dtype=dtype)
    
    for i in range(0, dim, 2):
        freq = torch.exp(torch.tensor(i * -(math.log(log_base) / dim), dtype=dtype, device=device))
        vec[i] = torch.sin(pos * freq)
        if i + 1 < dim:
            vec[i + 1] = torch.cos(pos * freq)
    
    return vec[:pe_dim]



class PreEncoded:
    def __init__(self, path, backend="numpy"):
        self.path = path
        self.backend = backend
        reader = CsvReader(path, index_column=0, header=None, sep="\t")
        self.cache = {}
        dim = None
        count = 0
        for k, v in reader:
            if backend == "numpy":
                self.cache[k] = np.array(v, dtype=np.float32)
            elif backend == "torch":
                self.cache[k] = torch.tensor(np.array(v, dtype=np.float32), dtype=torch.float32)
            if dim is None:
                dim = self.cache[k].shape[0]
            count += 1
        self.shape = (count, dim)

    def __call__(self, names):
        if isinstance(names, str):
            return self.cache[names]
        if self.backend == "numpy":
            return np.array([self.cache[n] for n in names], dtype=np.float32)
        elif self.backend == "torch":
            return torch.stack([self.cache[n] for n in names])