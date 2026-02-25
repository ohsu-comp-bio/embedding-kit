
import numpy as np

import torch
import torch.nn.functional as F

class OneHotEncoder:
    def __init__(self, labels, device=None):
        self.classes = sorted(labels)
        self.num_classes = len(self.classes)
        self.mapping = {}
        self.class_idx = {}
        self.device = device
        for i, n in enumerate(self.classes):
            self.mapping[n] = F.one_hot( torch.tensor(i), self.num_classes ).to(device)
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
        return F.one_hot(idx_tensor, num_classes=self.num_classes).to(self.mapping[self.classes[0]].device)

    def __len__(self):
        return self.num_classes

    def __iter__(self):
        return iter(self.classes)

amino_acids = 'ARNDCEQGHILKMFPSTWYV'

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

    def __init__(self, full_len=None, encode_x=True, encode_pos=False, device=None, dtype=torch.float32, backend='torch'):
        self.full_len = full_len
        self.encode_x = encode_x
        self.encode_pos = encode_pos
        self.device = device
        self.dtype = dtype
        self.backend = backend
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

        dim = len(self.alphabet) + (1 if self.encode_pos else 0)
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
                if self.encode_pos:
                    if self.full_len is not None:
                        one_hot_matrix[b, i, len(self.alphabet)] = float(i) / float(self.full_len)
                    else:
                        one_hot_matrix[b, i, len(self.alphabet)] = float(i)
            for i in range(L, FL):
                index = self.aa_to_index['X']
                if index is not None:
                    one_hot_matrix[b, i, index] = 1.0
        if is_single:
            return one_hot_matrix[0]
        return one_hot_matrix
