from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import torch

class NetworkConstraint:
    """
    PyTorch-native mask manager that mirrors the Keras constraint logic.

    feature_index: list of feature ids (len = input_dim)
    latent_index:  list of latent ids (len = latent_dim)
    latent_membership: dict mapping latent id -> list of feature ids allowed to connect
                       If None or inactive, mask is all ones (no constraint).
    """

    def __init__(self,
                 feature_index: List[str],
                 latent_index: List[str],
                 latent_membership: Optional[Dict[str, List[str]]] = None):
        self.feature_index = list(feature_index)
        self.latent_index = list(latent_index)
        self.latent_membership = latent_membership
        self.active = True
        self._mask_np = None
        self.update()

    def update_membership(self, latent_membership: Dict[str, List[str]]):
        self.latent_membership = latent_membership
        self.update()

    def set_active(self, a: bool):
        self.active = a
        self.update()

    def update(self):
        in_dim = len(self.feature_index)
        out_dim = len(self.latent_index)
        if not self.active or self.latent_membership is None:
            self._mask_np = np.ones((out_dim, in_dim), dtype=np.float32)
            return
        mask = np.zeros((out_dim, in_dim), dtype=np.float32)
        fi = pd.Index(self.feature_index)
        li = pd.Index(self.latent_index)
        for latent in self.latent_index:
            if latent not in self.latent_membership:
                continue
            latent_pos = li.get_loc(latent)
            for f in self.latent_membership[latent]:
                if f in fi:
                    mask[latent_pos, fi.get_loc(f)] = 1.0
        self._mask_np = mask

    def as_torch(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self._mask_np, device=device)