"""
Feed Forward Neural Network Model
"""

import logging
from typing import Dict, Optional, List, Union
from collections.abc import Callable

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from tqdm.autonotebook import tqdm

from ..layers import LayerInfo, convert_activation
from .. import get_device, dataframe_loader

logger = logging.getLogger(__name__)


class FFNN(nn.Module):
    """
    FeedForward Neural Network
    """

    def __init__(self, input_dim: int, output_dim: int,
                 layers: Optional[List[LayerInfo]] = None,
                 batch_norm: bool = False,
                 lr: float = 1e-3):
        # super(FFNN).__init__()
        super().__init__()

        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.history: Dict[str, list] = {"loss": []}

        in_features = input_dim

        # Optional global BN on input
        if batch_norm:
            self.layers.append(nn.BatchNorm1d(input_dim))

        if layers:
            logger.info("Building encoder with %d layers", len(layers))
            for li in layers:
                out_features = li.units

                layer = li.gen_layer(in_features)
                self.layers.append(layer)

                if li.activation is not None:
                    act = convert_activation(li.activation)
                    if act is not None:
                        self.layers.append(act)

                if li.batch_norm:
                    self.layers.append(nn.BatchNorm1d(out_features))

                in_features = out_features
        else:
            pass

        if in_features != self.output_dim:
            raise Exception(f"Layer issue")

    def fit(self, X: Union[torch.Tensor],
            y: Union[torch.Tensor], **kwargs):

        epochs: int = int(kwargs.pop("epochs", 20))
        lr: Optional[float] = kwargs.pop("lr", None)
        beta: float = float(kwargs.pop("beta", 1.0))
        optimizer: Optional[torch.optim.Optimizer] = kwargs.pop("optimizer", None)
        # loss: Optional[Callable] = kwargs.pop("loss", None)
        reset_optimizer: bool = bool(kwargs.pop("reset_optimizer", False))
        device: Optional[torch.device] = kwargs.pop("device", None)
        progress: bool = bool(kwargs.pop("progress", True))
        pin_memory: bool = bool(kwargs.pop("pin_memory", True))

        # --- setup ---
        if lr is None:
            lr = self.lr
        if device is None:
            device = get_device()

        self.to(device)
        self.train()

        # --- persistent optimizer (reuse momentum/velocity across phases) ---
        if optimizer is not None:
            self._optimizer = optimizer
        elif reset_optimizer or not hasattr(self, "_optimizer") or self._optimizer is None:
            self._optimizer = Adam(self.parameters(), lr=lr)
        else:
            # Reuse existing optimizer but refresh LR if changed
            for g in self._optimizer.param_groups:
                g["lr"] = lr
        opt = self._optimizer
        critereon = nn.MSELoss()

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=pin_memory)

        # --- epoch runner (epoch-only progress) ---
        def run_epochs(n_epochs: int, beta_value: float) -> float:
            last_loss = 0.0
            epoch_batches = 0
            epoch_bar = tqdm(range(n_epochs), disable=not progress, desc=f"β={beta_value:.2f}")
            for epoch_idx in epoch_bar:
                epoch_loss_sum = 0.0
                epoch_batches = 0

                for inputs, outputs in dataloader:
                    inputs = inputs.to(device, non_blocking=pin_memory)
                    outputs = outputs.to(device, non_blocking=pin_memory)

                    predictions = self(inputs)
                    loss = critereon(predictions, outputs)

                    # Backprop
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    # Accumulate epoch stats
                    tl = float(loss.detach().cpu())
                    epoch_loss_sum += tl
                    epoch_batches += 1
                    last_loss = tl

                # Compute epoch means
                if epoch_batches > 0:
                    ep_loss = epoch_loss_sum / epoch_batches
                    self.history["loss"].append(ep_loss)

                    # Update the epoch progress bar once per epoch (no jitter)
                    if progress:
                        epoch_bar.set_postfix(loss=f"{ep_loss:.3f}")

            return last_loss

        return run_epochs(epochs, beta)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return h
