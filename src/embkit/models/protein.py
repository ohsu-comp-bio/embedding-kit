
from typing import Literal
import esm
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


def aa_to_one_hot(sequence: str, full_len=None, encode_x=True, encode_pos=False) -> np.ndarray:
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
        A NumPy array (matrix) of shape (L, oneof[20,21,22)), where L is the sequence length.
    """
    # 1. Define the standard 20 amino acid alphabet
    # Ordered alphabetically (common practice, though order doesn't affect correctness)
    amino_acids = 'ARNDCEQGHILKMFPSTWYV'
    # 2. Add 'X' for unknown/non-standard residues
    # This makes the dimension 21
    if encode_x:
        alphabet = amino_acids + 'X'
    else:
        alphabet = amino_acids

    # 3. Create a mapping dictionary for quick lookup
    # e.g., {'A': 0, 'R': 1, ..., 'V': 19, 'X': 20}
    aa_to_index = {aa: i for i, aa in enumerate(alphabet)}
    if not encode_x:
        aa_to_index["X"] = None

    # 4. Initialize the one-hot matrix
    # Shape is (sequence length, 21)
    L = len(sequence)
    if full_len is None:
        FL = L
    else:
        FL = full_len
    if encode_pos:
        one_hot_matrix = np.zeros((FL, len(alphabet)+1), dtype=np.float32)
    else:
        one_hot_matrix = np.zeros((FL, len(alphabet)), dtype=np.float32)

    # 5. Iterate through the sequence and set the corresponding '1'
    for i, aa in enumerate(sequence.upper()):
        # Get the index of the amino acid. If it's not in the alphabet (e.g., 'Z'),
        # map it to the 'X' (unknown) index.
        index = aa_to_index.get(aa, aa_to_index['X'])
        if index is not None:
            # Set the value at the correct position to 1
            one_hot_matrix[i, index] = 1
        if encode_pos:
            if full_len is not None:
                one_hot_matrix[i, len(alphabet)] = i / full_len
            else:
                one_hot_matrix[i, len(alphabet)] = i
    # Fill remaining buffer with unknown
    for i in range(L,FL):
        index = aa_to_index['X']
        if index is not None:
            one_hot_matrix[i, index] = 1
        #if encode_pos:
        #    one_hot_matrix[i, len(alphabet)] = -1
    return one_hot_matrix


def batch_iterable(iterable, batch_size):
    """
    Converts an iterable into a sequence of batched lists.

    Args:
        iterable: The input iterable (e.g., list, tuple, generator).
        batch_size: The desired size of each batch.

    Yields:
        list: A batch (list) of elements from the iterable.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    # Get an iterator from the iterable
    iterator = iter(iterable)

    # Loop indefinitely, or until StopIteration is raised
    while True:
        # Create a generator expression that yields up to batch_size elements
        # from the iterator. 'next(iterator, _SENTINEL)' attempts to get the 
        # next item; if it fails (StopIteration), it yields _SENTINEL.
        _SENTINEL = object()  # Unique sentinel value
        batch_gen = (next(iterator, _SENTINEL) for _ in range(batch_size))
        
        # Filter out the sentinel value(s) and create the list
        current_batch = [item for item in batch_gen if item is not _SENTINEL]

        # If the batch is empty, we've exhausted the iterable
        if not current_batch:
            return  # Stop the generator

        yield current_batch



class ProteinEncoder:
    """
    ProteinEncoder wrapper class for Fair-ESM methods ( https://pypi.org/project/fair-esm/ )
    """

    MODELS = ["t6", "t12", "t30", "t33", "t36", "t48"]

    def __init__(self, model="t33", batch_size=100, device=None):
        if model == "t48":
            model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
            self.out_layer = 48
        elif model == "t36":
            model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            self.out_layer = 36
        elif model == "t33":
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.out_layer = 33
        elif model == "t30":
            model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            self.out_layer = 30
        elif model == "t12":
            model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            self.out_layer = 12
        elif model == "t6":
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.out_layer = 6
        else:
            raise Exception(f"unknown model {model}")

        batch_converter = alphabet.get_batch_converter()

        model.eval()  # disables dropout for deterministic results

        if device is not None:
            model.to(device)
        self.device = device

        self.model = model
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.batch_converter = batch_converter
    
    def to(self, device):
        self.device = device
        self.model.to(self.device)
    
    def get_embed_dim(self):
        return self.model.embed_dim
    
    def encode(self, data, output:Literal["vector","mean-pool","sum-pool"] = "sum-pool", fix_len=None, verbose=False):
        if verbose:
            iter = tqdm( batch_iterable(data, self.batch_size) )
        else:
            iter = batch_iterable(data, self.batch_size)
        for block in iter:
            _, _, batch_tokens = self.batch_converter(block)
            if fix_len is not None:
                #if they have defined that the tokenization will be a fixed length
                batch_tokens = self.pad(batch_tokens, fix_len+1) # length plus start token
            if self.device is not None:
                batch_tokens = batch_tokens.to(device=self.device, non_blocking=True)
            if fix_len is None:
                batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
            else:
                batch_lens = [fix_len] * len(block)
            with torch.no_grad():
                results = self.model(batch_tokens,
                                     repr_layers=[self.out_layer],
                                     return_contacts=True)
            token_representations = results["representations"][self.out_layer]
            for i, tokens_len in enumerate(batch_lens):
                if fix_len:
                    vec = token_representations[i, 1 : fix_len+2] # include fix_len + start_token
                else:
                    vec = token_representations[i, 1 : tokens_len - 1] # remove padding
                if output == "mean-pool":
                    yield block[i][0], vec.mean(0).to(device="cpu")
                elif output == "sum-pool":
                    yield block[i][0], vec.sum(0).to(device="cpu")
                else:
                    yield block[i][0], vec.to(device="cpu")

    def pad(self, tokens, fix_len):
        padded_tokens = F.pad(tokens, (0, fix_len - len(tokens[0]) ), value=self.alphabet.padding_idx)
        return padded_tokens