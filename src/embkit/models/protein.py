
import esm
import torch
from tqdm import tqdm

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
    MODELS = ["t6", "t12", "t30", "t33", "t36", "t48"]

    def __init__(self, batch_size, model="t33"):
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
            raise Exception(f"Unknown model {model}")

        batch_converter = alphabet.get_batch_converter()

        model.eval()  # disables dropout for deterministic results

        if torch.cuda.is_available():
            model = model.cuda()

        self.model = model
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.batch_converter = batch_converter
    
    def encode(self, data):
        for block in tqdm( batch_iterable(data, self.batch_size) ):
            batch_labels, batch_strs, batch_tokens = self.batch_converter(block)
            if torch.cuda.is_available():
                batch_tokens = batch_tokens.to(device="cuda", non_blocking=True)
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
            results = self.model(batch_tokens, repr_layers=[self.out_layer], return_contacts=True)
            token_representations = results["representations"][self.out_layer]
            for i, tokens_len in enumerate(batch_lens):
                vec = token_representations[i, 1 : tokens_len - 1].mean(0).to(device="cpu")
                yield block[i][1], vec
