
from typing import Literal
import esm
import torch
from tqdm import tqdm
import numpy as np

from ..preprocessing import batch_iterable

import torch.nn.functional as F



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